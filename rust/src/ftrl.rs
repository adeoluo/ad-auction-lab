//! Logistic regression on hashed features trained with FTRL-Proximal (McMahan et al., 2013).
//!
//! Per coordinate the learner keeps `z` and `n` (sum of squared gradients); the weight is
//! `w = 0` when `|z| <= l1`, otherwise `-(z - sign(z) l1) / ((beta + sqrt(n)) / alpha + l2)`.
//! Features are binary, so the gradient of every active coordinate is `p - y`. The bias lives in
//! the extra slot `2^bits` and is not L1 or L2 regularized. Training is sequential (the update
//! order matters and is fixed by the caller's row order); prediction is parallel.

use rayon::prelude::*;

use crate::hashing::Features;
use crate::metrics::{log_loss_one, non_negative, positive, sigmoid};

#[derive(Clone, Copy, Debug)]
pub struct FtrlParams {
    pub alpha: f64,
    pub beta: f64,
    pub l1: f64,
    pub l2: f64,
    pub bits: u32,
}

impl FtrlParams {
    pub fn validate(&self) -> Result<(), String> {
        if !positive(self.alpha) || ![self.beta, self.l1, self.l2].into_iter().all(non_negative) {
            return Err("alpha must be positive and beta, l1, l2 non-negative".into());
        }
        if !(1..=30).contains(&self.bits) {
            return Err("bits must be between 1 and 30".into());
        }
        Ok(())
    }
}

pub struct Ftrl {
    pub params: FtrlParams,
    pub z: Vec<f32>,
    pub n: Vec<f32>,
    mask: u64,
}

impl Ftrl {
    pub fn new(params: FtrlParams) -> Result<Self, String> {
        params.validate()?;
        let size = (1usize << params.bits) + 1;
        Ok(Self {
            params,
            z: vec![0.0; size],
            n: vec![0.0; size],
            mask: (1u64 << params.bits) - 1,
        })
    }

    pub fn bias_index(&self) -> usize {
        1usize << self.params.bits
    }

    #[inline]
    fn weight(&self, i: usize, l1: f64, l2: f64) -> f64 {
        let z = self.z[i] as f64;
        if z.abs() <= l1 {
            0.0
        } else {
            let p = &self.params;
            -(z - z.signum() * l1) / ((p.beta + (self.n[i] as f64).sqrt()) / p.alpha + l2)
        }
    }

    #[inline]
    fn update(&mut self, i: usize, wi: f64, g: f64, g2: f64, alpha: f64) {
        let n_old = self.n[i] as f64;
        let sigma = ((n_old + g2).sqrt() - n_old.sqrt()) / alpha;
        self.z[i] = (self.z[i] as f64 + g - sigma * wi) as f32;
        self.n[i] = (n_old + g2) as f32;
    }

    /// One pass over `rows` in order. Returns the progressive (predict then update) mean log loss.
    pub fn fit(&mut self, x: &Features, labels: &[u8], rows: &[u64]) -> Result<f64, String> {
        let n_rows = x.n_rows();
        if labels.len() != n_rows {
            return Err("labels and features disagree on the row count".into());
        }
        let (l1, l2, alpha) = (self.params.l1, self.params.l2, self.params.alpha);
        let bias = self.bias_index();
        let mut idx = vec![0usize; x.n_fields()];
        let mut w = vec![0f64; x.n_fields()];
        let mut total = 0.0;
        for &r in rows {
            let r = r as usize;
            if r >= n_rows {
                return Err(format!("row {r} out of range"));
            }
            x.buckets(r, self.mask, &mut idx);
            let wb = self.weight(bias, 0.0, 0.0);
            let mut s = wb;
            for (k, &i) in idx.iter().enumerate() {
                w[k] = self.weight(i, l1, l2);
                s += w[k];
            }
            let p = sigmoid(s);
            let y = labels[r] as f64;
            total += log_loss_one(p, y);
            let g = p - y;
            let g2 = g * g;
            for (k, &i) in idx.iter().enumerate() {
                self.update(i, w[k], g, g2, alpha);
            }
            self.update(bias, wb, g, g2, alpha);
        }
        Ok(if rows.is_empty() {
            0.0
        } else {
            total / rows.len() as f64
        })
    }

    /// Final weights, `2^bits + 1` values (the last one is the bias).
    pub fn weights(&self) -> Vec<f32> {
        let (l1, l2) = (self.params.l1, self.params.l2);
        let bias = self.bias_index();
        (0..self.z.len())
            .into_par_iter()
            .map(|i| if i == bias { self.weight(i, 0.0, 0.0) } else { self.weight(i, l1, l2) } as f32)
            .collect()
    }

    /// Click probabilities for `rows`, in order.
    pub fn predict(&self, x: &Features, rows: &[u64]) -> Result<Vec<f32>, String> {
        predict_linear(&self.weights(), self.mask, x, rows)
    }
}

/// Probabilities of a linear model with weights `w` (`2^bits + 1`, bias last).
pub fn predict_linear(
    w: &[f32],
    mask: u64,
    x: &Features,
    rows: &[u64],
) -> Result<Vec<f32>, String> {
    let n_rows = x.n_rows();
    if rows.iter().any(|&r| r as usize >= n_rows) {
        return Err("row index out of range".into());
    }
    let bias = w[w.len() - 1] as f64;
    Ok(rows
        .par_chunks(4096)
        .flat_map_iter(|chunk| {
            let mut idx = vec![0usize; x.n_fields()];
            chunk
                .iter()
                .map(|&r| {
                    x.buckets(r as usize, mask, &mut idx);
                    let s = bias + idx.iter().map(|&i| w[i] as f64).sum::<f64>();
                    sigmoid(s) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy() -> (Vec<u8>, Vec<u8>) {
        // Field 0 decides the label; field 1 is noise.
        let mut small = Vec::new();
        let mut y = Vec::new();
        for i in 0..2000u32 {
            let a = (i % 2) as u8;
            let b = ((i / 2) % 7) as u8;
            small.extend([a, b]);
            y.push(a);
        }
        (small, y)
    }

    #[test]
    fn learns_a_separable_signal() {
        let (small, y) = toy();
        let x = Features::new(&small, 2, &[], 0).unwrap();
        let params = FtrlParams {
            alpha: 0.1,
            beta: 1.0,
            l1: 0.0,
            l2: 0.0,
            bits: 12,
        };
        let mut m = Ftrl::new(params).unwrap();
        let rows: Vec<u64> = (0..2000).collect();
        let first = m.fit(&x, &y, &rows).unwrap();
        let second = m.fit(&x, &y, &rows).unwrap();
        assert!(second < first);
        let p = m.predict(&x, &[0, 1]).unwrap();
        assert!(p[0] < 0.2 && p[1] > 0.8);
    }

    #[test]
    fn strong_l1_gives_sparse_weights() {
        let (small, y) = toy();
        let x = Features::new(&small, 2, &[], 0).unwrap();
        let params = FtrlParams {
            alpha: 0.1,
            beta: 1.0,
            l1: 1e6,
            l2: 0.0,
            bits: 12,
        };
        let mut m = Ftrl::new(params).unwrap();
        m.fit(&x, &y, &(0..2000).collect::<Vec<_>>()).unwrap();
        let w = m.weights();
        assert!(w[..w.len() - 1].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rejects_bad_input() {
        let bad = FtrlParams {
            alpha: 0.0,
            beta: 1.0,
            l1: 0.0,
            l2: 0.0,
            bits: 12,
        };
        assert!(Ftrl::new(bad).is_err());
        let bad = FtrlParams {
            alpha: 0.1,
            beta: 1.0,
            l1: 0.0,
            l2: 0.0,
            bits: 31,
        };
        assert!(Ftrl::new(bad).is_err());
        let (small, y) = toy();
        let x = Features::new(&small, 2, &[], 0).unwrap();
        let mut m = Ftrl::new(FtrlParams {
            alpha: 0.1,
            beta: 1.0,
            l1: 0.0,
            l2: 0.0,
            bits: 8,
        })
        .unwrap();
        assert!(m.fit(&x, &y, &[5000]).is_err());
        assert!(m.fit(&x, &y[..10], &[0]).is_err());
        assert!(m.predict(&x, &[5000]).is_err());
        assert_eq!(m.fit(&x, &y, &[]).unwrap(), 0.0);
    }
}
