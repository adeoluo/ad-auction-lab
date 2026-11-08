//! Second-order factorization machine (Rendle, 2010) on hashed binary features, trained with
//! AdaGrad on the log loss.
//!
//! `score = w0 + sum_i w_i + 0.5 * sum_f ((sum_i v_if)^2 - sum_i v_if^2)` over the active buckets
//! `i` of a row. Per row the factor sums `s_f` are computed once, then every active parameter is
//! updated in field order with `g = p - y`:
//! `w_i -= lr * (g + l2_linear w_i) / sqrt(G)`, `v_if -= lr * (g (s_f - v_if) + l2_factor v_if) / sqrt(G)`,
//! where `G` is the running sum of squared gradients of that parameter (plus 1e-8).
//! Factors start uniform in `[-init_scale, init_scale]` from a SplitMix64 stream of `seed`.

use rayon::prelude::*;

use crate::hashing::{Features, mix64};
use crate::metrics::{log_loss_one, non_negative, positive, sigmoid};

#[derive(Clone, Copy, Debug)]
pub struct FmParams {
    pub k: usize,
    pub bits: u32,
    pub lr: f64,
    pub l2_linear: f64,
    pub l2_factor: f64,
    pub init_scale: f64,
    pub seed: u64,
}

impl FmParams {
    pub fn validate(&self) -> Result<(), String> {
        if self.k == 0 || self.k > 256 {
            return Err("k must be between 1 and 256".into());
        }
        if !(1..=28).contains(&self.bits) {
            return Err("bits must be between 1 and 28".into());
        }
        if !positive(self.lr)
            || ![self.l2_linear, self.l2_factor, self.init_scale]
                .into_iter()
                .all(non_negative)
        {
            return Err("lr must be positive and l2 and init_scale non-negative".into());
        }
        Ok(())
    }
}

/// Uniform value in [-scale, scale] for factor slot `j` (shared with the numpy reference).
pub fn init_value(seed: u64, j: u64, scale: f64) -> f32 {
    let u = (mix64(seed ^ j.wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 11) as f64
        * (1.0 / (1u64 << 53) as f64);
    ((2.0 * u - 1.0) * scale) as f32
}

pub struct Fm {
    pub params: FmParams,
    pub w0: f64,
    pub g0: f64,
    pub w: Vec<f32>,
    pub gw: Vec<f32>,
    pub v: Vec<f32>,
    pub gv: Vec<f32>,
    mask: u64,
}

const EPS: f64 = 1e-8;

impl Fm {
    pub fn new(params: FmParams) -> Result<Self, String> {
        params.validate()?;
        let size = 1usize << params.bits;
        let k = params.k;
        let v: Vec<f32> = (0..size * k)
            .into_par_iter()
            .map(|j| init_value(params.seed, j as u64, params.init_scale))
            .collect();
        Ok(Self {
            params,
            w0: 0.0,
            g0: 0.0,
            w: vec![0.0; size],
            gw: vec![0.0; size],
            v,
            gv: vec![0.0; size * k],
            mask: (size as u64) - 1,
        })
    }

    fn score(&self, idx: &[usize], s: &mut [f64]) -> f64 {
        let k = self.params.k;
        s.iter_mut().for_each(|x| *x = 0.0);
        let mut lin = self.w0;
        let mut sq = 0.0;
        for &i in idx {
            lin += self.w[i] as f64;
            let row = &self.v[i * k..(i + 1) * k];
            for (sf, &x) in s.iter_mut().zip(row) {
                let x = x as f64;
                *sf += x;
                sq += x * x;
            }
        }
        let inter: f64 = s.iter().map(|x| x * x).sum::<f64>() - sq;
        lin + 0.5 * inter
    }

    /// One pass over `rows` in order; returns the progressive mean log loss.
    pub fn fit(&mut self, x: &Features, labels: &[u8], rows: &[u64]) -> Result<f64, String> {
        let n_rows = x.n_rows();
        if labels.len() != n_rows {
            return Err("labels and features disagree on the row count".into());
        }
        let FmParams {
            k,
            lr,
            l2_linear,
            l2_factor,
            ..
        } = self.params;
        let mut idx = vec![0usize; x.n_fields()];
        let mut s = vec![0f64; k];
        let mut total = 0.0;
        for &r in rows {
            let r = r as usize;
            if r >= n_rows {
                return Err(format!("row {r} out of range"));
            }
            x.buckets(r, self.mask, &mut idx);
            let p = sigmoid(self.score(&idx, &mut s));
            let y = labels[r] as f64;
            total += log_loss_one(p, y);
            let g = p - y;
            self.g0 += g * g;
            self.w0 -= lr * g / (self.g0 + EPS).sqrt();
            for &i in &idx {
                let wi = self.w[i] as f64;
                let gi = g + l2_linear * wi;
                let acc = self.gw[i] as f64 + gi * gi;
                self.gw[i] = acc as f32;
                self.w[i] = (wi - lr * gi / (acc + EPS).sqrt()) as f32;
                let v = &mut self.v[i * k..(i + 1) * k];
                let gv = &mut self.gv[i * k..(i + 1) * k];
                for ((vf, gvf), &sf) in v.iter_mut().zip(gv.iter_mut()).zip(&s) {
                    let vif = *vf as f64;
                    let gf = g * (sf - vif) + l2_factor * vif;
                    let acc = *gvf as f64 + gf * gf;
                    *gvf = acc as f32;
                    *vf = (vif - lr * gf / (acc + EPS).sqrt()) as f32;
                }
            }
        }
        Ok(if rows.is_empty() {
            0.0
        } else {
            total / rows.len() as f64
        })
    }

    pub fn predict(&self, x: &Features, rows: &[u64]) -> Result<Vec<f32>, String> {
        let n_rows = x.n_rows();
        if rows.iter().any(|&r| r as usize >= n_rows) {
            return Err("row index out of range".into());
        }
        let k = self.params.k;
        Ok(rows
            .par_chunks(4096)
            .flat_map_iter(|chunk| {
                let mut idx = vec![0usize; x.n_fields()];
                let mut s = vec![0f64; k];
                chunk
                    .iter()
                    .map(|&r| {
                        x.buckets(r as usize, self.mask, &mut idx);
                        sigmoid(self.score(&idx, &mut s)) as f32
                    })
                    .collect::<Vec<_>>()
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_an_interaction() {
        // y = a XOR b is not linearly separable; the factorization captures it.
        let mut small = Vec::new();
        let mut y = Vec::new();
        for i in 0..4000u32 {
            let a = (i % 2) as u8;
            let b = ((i / 2) % 2) as u8;
            small.extend([a, b]);
            y.push(a ^ b);
        }
        let x = Features::new(&small, 2, &[], 0).unwrap();
        let params = FmParams {
            k: 4,
            bits: 10,
            lr: 0.1,
            l2_linear: 0.0,
            l2_factor: 0.0,
            init_scale: 0.1,
            seed: 1,
        };
        let mut m = Fm::new(params).unwrap();
        let rows: Vec<u64> = (0..4000).collect();
        let mut loss = 1.0;
        for _ in 0..20 {
            loss = m.fit(&x, &y, &rows).unwrap();
        }
        assert!(loss < 0.2, "loss {loss}");
        let p = m.predict(&x, &[0, 1, 2, 3]).unwrap();
        assert!(
            p[0] < 0.5 && p[1] > 0.5 && p[2] > 0.5 && p[3] < 0.5,
            "{p:?}"
        );
    }

    #[test]
    fn init_is_bounded_and_seeded() {
        for j in 0..1000 {
            assert!(init_value(3, j, 0.01).abs() <= 0.01);
        }
        assert_ne!(init_value(3, 5, 0.1), init_value(4, 5, 0.1));
    }

    #[test]
    fn rejects_bad_input() {
        let p = FmParams {
            k: 0,
            bits: 10,
            lr: 0.1,
            l2_linear: 0.0,
            l2_factor: 0.0,
            init_scale: 0.1,
            seed: 1,
        };
        assert!(Fm::new(p).is_err());
        let p = FmParams {
            k: 2,
            bits: 10,
            lr: 0.0,
            ..p
        };
        assert!(Fm::new(p).is_err());
        let p = FmParams { lr: 0.1, ..p };
        let mut m = Fm::new(p).unwrap();
        let small = [0u8, 1];
        let x = Features::new(&small, 2, &[], 0).unwrap();
        assert!(m.fit(&x, &[1], &[3]).is_err());
        assert!(m.fit(&x, &[1, 0], &[0]).is_err());
        assert!(m.predict(&x, &[3]).is_err());
    }
}
