//! Rust core of ad-auction-lab.
//!
//! * [`hashing`]: SplitMix64 feature hashing over two row-major categorical blocks.
//! * [`ftrl`]: logistic regression trained with FTRL-Proximal.
//! * [`fm`]: second-order factorization machine trained with AdaGrad.
//! * [`pacing`]: multi-advertiser budget pacing simulator (no pacing, PID, dual descent).
//!
//! The Python package keeps numpy reference implementations of every model; the parity tests
//! check that both produce the same numbers.

pub mod fm;
pub mod ftrl;
pub mod hashing;
pub mod metrics;
pub mod pacing;

#[cfg(feature = "python")]
mod python;
