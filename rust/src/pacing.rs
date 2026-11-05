//! Multi-advertiser budget pacing simulator: repeated single-slot second-price auctions on
//! expected cost per impression (eCPM = bid per click x predicted CTR).
//!
//! In auction `t`, advertiser `i` bids `b_i = v_i m_i` per click, where `m_i` comes from its pacer:
//! no pacing (`m = 1`), a PID controller on the spend error updated every `interval` auctions, or
//! dual descent (Balseiro and Gur, 2019) with `m = 1 / (1 + mu)` and
//! `mu <- max(0, mu - step (rho - z_t) / rho)` after every auction, where `rho = B / T` is the
//! target spend per auction and `z_t` the spend in auction `t`. An advertiser takes part only if
//! its remaining budget covers its own eCPM bid. The highest eCPM wins if it beats both the
//! external market eCPM and the reserve, and pays the larger of the second eCPM, the market and
//! the reserve. Value and spend are expectations (value `v_i ctr`, payment the price eCPM).

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Pacer {
    None,
    Pid {
        kp: f64,
        ki: f64,
        kd: f64,
        interval: usize,
        m_min: f64,
    },
    Dual {
        step: f64,
    },
}

#[derive(Clone, Debug, Default)]
pub struct Outcome {
    pub value: Vec<f64>,
    pub spend: Vec<f64>,
    pub wins: Vec<u64>,
    /// Auction index at which the budget fell below the advertiser's own next bid (T if never).
    pub exhausted_at: Vec<u64>,
    /// Cumulative spend per advertiser at `checkpoints` evenly spaced times, row-major (c, n_adv).
    pub trajectory: Vec<f64>,
    pub revenue: f64,
    pub market_wins: u64,
    pub unsold: u64,
}

pub struct Market<'a> {
    pub values: &'a [f64],
    pub budgets: &'a [f64],
    /// Row-major (T, n_adv) predicted CTRs.
    pub ctr: &'a [f32],
    /// (T,) external highest competing eCPM.
    pub market: &'a [f32],
    pub reserve: f64,
}

impl Market<'_> {
    fn check(&self) -> Result<(usize, usize), String> {
        let n = self.values.len();
        if n == 0 || self.budgets.len() != n {
            return Err("values and budgets must be non-empty and the same length".into());
        }
        if !self.ctr.len().is_multiple_of(n) || self.ctr.len() / n != self.market.len() {
            return Err("ctr must be (T, n_adv) and market (T,)".into());
        }
        if self
            .values
            .iter()
            .chain(self.budgets)
            .any(|x| !x.is_finite() || *x < 0.0)
        {
            return Err("values and budgets must be finite and non-negative".into());
        }
        if !crate::metrics::non_negative(self.reserve) {
            return Err("reserve must be non-negative".into());
        }
        Ok((n, self.market.len()))
    }
}

pub fn simulate(m: &Market, pacer: Pacer, checkpoints: usize) -> Result<Outcome, String> {
    let (n, t_len) = m.check()?;
    if let Pacer::Pid { interval, .. } = pacer {
        if interval == 0 {
            return Err("PID interval must be positive".into());
        }
    }
    let checkpoints = checkpoints.max(1);
    let mut out = Outcome {
        value: vec![0.0; n],
        spend: vec![0.0; n],
        wins: vec![0; n],
        exhausted_at: vec![t_len as u64; n],
        trajectory: vec![0.0; checkpoints * n],
        ..Default::default()
    };
    let mut mult = vec![1.0f64; n];
    let mut mu = vec![0.0f64; n];
    let mut integral = vec![0.0f64; n];
    let mut prev_err = vec![0.0f64; n];
    let rho: Vec<f64> = m.budgets.iter().map(|b| b / t_len.max(1) as f64).collect();
    let mut next_cp = 0usize;
    for t in 0..t_len {
        let ctr = &m.ctr[t * n..(t + 1) * n];
        let (mut best, mut second, mut winner) = (0.0f64, 0.0f64, usize::MAX);
        for i in 0..n {
            let factor = match pacer {
                Pacer::None => 1.0,
                Pacer::Pid { .. } => mult[i],
                Pacer::Dual { .. } => 1.0 / (1.0 + mu[i]),
            };
            let ecpm = m.values[i] * factor * ctr[i] as f64;
            let remaining = m.budgets[i] - out.spend[i];
            if ecpm <= 0.0 || remaining < ecpm {
                if ecpm > 0.0 && out.exhausted_at[i] == t_len as u64 {
                    out.exhausted_at[i] = t as u64;
                }
                continue;
            }
            if ecpm > best {
                second = best;
                best = ecpm;
                winner = i;
            } else if ecpm > second {
                second = ecpm;
            }
        }
        let floor = (m.market[t] as f64).max(m.reserve);
        let mut paid = usize::MAX;
        let mut price = 0.0;
        if winner != usize::MAX && best > floor {
            price = second.max(floor);
            out.spend[winner] += price;
            out.value[winner] += m.values[winner] * ctr[winner] as f64;
            out.wins[winner] += 1;
            out.revenue += price;
            paid = winner;
        } else if m.market[t] as f64 >= m.reserve && m.market[t] > 0.0 {
            out.market_wins += 1;
        } else {
            out.unsold += 1;
        }
        match pacer {
            Pacer::None => {}
            Pacer::Dual { step } => {
                for i in 0..n {
                    if rho[i] <= 0.0 {
                        continue;
                    }
                    let z = if i == paid { price } else { 0.0 };
                    mu[i] = (mu[i] - step * (rho[i] - z) / rho[i]).max(0.0);
                }
            }
            Pacer::Pid {
                kp,
                ki,
                kd,
                interval,
                m_min,
            } => {
                if (t + 1) % interval == 0 {
                    for i in 0..n {
                        if m.budgets[i] <= 0.0 {
                            continue;
                        }
                        let target = m.budgets[i] * (t + 1) as f64 / t_len as f64;
                        let err = (target - out.spend[i]) / m.budgets[i];
                        integral[i] += err;
                        let u = kp * err + ki * integral[i] + kd * (err - prev_err[i]);
                        prev_err[i] = err;
                        mult[i] = u.exp().clamp(m_min, 1.0);
                    }
                }
            }
        }
        while next_cp < checkpoints && (t + 1) * checkpoints >= (next_cp + 1) * t_len {
            out.trajectory[next_cp * n..(next_cp + 1) * n].copy_from_slice(&out.spend);
            next_cp += 1;
        }
    }
    while next_cp < checkpoints {
        out.trajectory[next_cp * n..(next_cp + 1) * n].copy_from_slice(&out.spend);
        next_cp += 1;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn market(t: usize) -> (Vec<f64>, Vec<f64>, Vec<f32>, Vec<f32>) {
        let values = vec![1.0, 0.8];
        let budgets = vec![0.02 * t as f64, 0.02 * t as f64];
        let ctr: Vec<f32> = (0..t * 2)
            .map(|j| 0.05 + 0.1 * ((j * 7919 % 97) as f32 / 97.0))
            .collect();
        let mkt: Vec<f32> = (0..t)
            .map(|j| 0.03 * ((j * 104729 % 89) as f32 / 89.0))
            .collect();
        (values, budgets, ctr, mkt)
    }

    #[test]
    fn spend_never_exceeds_budget_and_accounts_add_up() {
        let (v, b, c, mk) = market(5000);
        let m = Market {
            values: &v,
            budgets: &b,
            ctr: &c,
            market: &mk,
            reserve: 0.001,
        };
        for pacer in [
            Pacer::None,
            Pacer::Dual { step: 0.01 },
            Pacer::Pid {
                kp: 5.0,
                ki: 0.5,
                kd: 0.0,
                interval: 100,
                m_min: 0.05,
            },
        ] {
            let o = simulate(&m, pacer, 10).unwrap();
            for (spend, budget) in o.spend.iter().zip(&b) {
                assert!(*spend <= budget + 1e-9);
            }
            let wins: u64 = o.wins.iter().sum();
            assert_eq!(wins + o.market_wins + o.unsold, 5000);
            assert!((o.revenue - o.spend.iter().sum::<f64>()).abs() < 1e-9);
            assert_eq!(&o.trajectory[18..20], &o.spend[..]);
        }
    }

    #[test]
    fn pacing_delays_exhaustion() {
        let (v, b, c, mk) = market(20000);
        let b: Vec<f64> = b.iter().map(|x| x * 0.2).collect();
        let m = Market {
            values: &v,
            budgets: &b,
            ctr: &c,
            market: &mk,
            reserve: 0.0,
        };
        let none = simulate(&m, Pacer::None, 4).unwrap();
        let dual = simulate(&m, Pacer::Dual { step: 0.005 }, 4).unwrap();
        assert!(none.exhausted_at[0] < 20000);
        assert!(dual.exhausted_at[0] > none.exhausted_at[0]);
    }

    #[test]
    fn rejects_bad_markets() {
        let m = Market {
            values: &[],
            budgets: &[],
            ctr: &[],
            market: &[],
            reserve: 0.0,
        };
        assert!(simulate(&m, Pacer::None, 1).is_err());
        let m = Market {
            values: &[1.0],
            budgets: &[1.0],
            ctr: &[0.1, 0.2],
            market: &[0.0],
            reserve: 0.0,
        };
        assert!(simulate(&m, Pacer::None, 1).is_err());
        let m = Market {
            values: &[1.0],
            budgets: &[-1.0],
            ctr: &[0.1],
            market: &[0.0],
            reserve: 0.0,
        };
        assert!(simulate(&m, Pacer::None, 1).is_err());
        let m = Market {
            values: &[1.0],
            budgets: &[1.0],
            ctr: &[0.1],
            market: &[0.0],
            reserve: -1.0,
        };
        assert!(simulate(&m, Pacer::None, 1).is_err());
        let m = Market {
            values: &[1.0],
            budgets: &[1.0],
            ctr: &[0.1],
            market: &[0.0],
            reserve: 0.0,
        };
        let pid = Pacer::Pid {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
            interval: 0,
            m_min: 0.1,
        };
        assert!(simulate(&m, pid, 1).is_err());
    }
}
