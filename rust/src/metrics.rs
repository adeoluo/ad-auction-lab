//! Small numeric helpers shared by the learners.

/// Whether `x` is finite and strictly positive.
#[inline]
pub fn positive(x: f64) -> bool {
    x.is_finite() && x > 0.0
}

/// Whether `x` is finite and non-negative.
#[inline]
pub fn non_negative(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

/// Logistic function with the argument clipped to [-35, 35].
#[inline]
pub fn sigmoid(s: f64) -> f64 {
    1.0 / (1.0 + (-s.clamp(-35.0, 35.0)).exp())
}

/// Log loss of one prediction, with `p` clipped to [1e-15, 1 - 1e-15] as in scikit-learn.
#[inline]
pub fn log_loss_one(p: f64, y: f64) -> f64 {
    let p = p.clamp(1e-15, 1.0 - 1e-15);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_symmetric_and_bounded() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-15);
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-15);
        assert!(sigmoid(1e9) < 1.0 && sigmoid(-1e9) > 0.0);
    }

    #[test]
    fn log_loss_is_finite() {
        assert!((log_loss_one(0.5, 1.0) - std::f64::consts::LN_2).abs() < 1e-12);
        assert!(log_loss_one(0.0, 1.0).is_finite());
        assert!(log_loss_one(1.0, 0.0).is_finite());
    }
}
