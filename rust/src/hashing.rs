//! Feature hashing shared by every model.
//!
//! A row is a set of `(field, value)` tokens. Field `f` with value `v` maps to bucket
//! `mix64(((f + 1) << 32) | v) & (2^bits - 1)`, where `mix64` is the SplitMix64 finalizer.
//! `ad_auction_lab.hashing` computes the same function with numpy.

/// SplitMix64 finalizer: a fast, well-mixed 64-bit bijection.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Bucket of the token `(field, value)` in a table of `2^bits` weights.
#[inline]
pub fn bucket(field: usize, value: u32, mask: u64) -> usize {
    (mix64((((field as u64) + 1) << 32) | value as u64) & mask) as usize
}

/// Row-major categorical features in two blocks: `small` holds `n_small` bytes per row (binned
/// integers) and `wide` holds `n_wide` 32-bit values per row (hashed categories). Field indices
/// run over the small block first, then the wide block.
#[derive(Clone, Copy)]
pub struct Features<'a> {
    pub small: &'a [u8],
    pub n_small: usize,
    pub wide: &'a [u32],
    pub n_wide: usize,
}

impl<'a> Features<'a> {
    pub fn new(
        small: &'a [u8],
        n_small: usize,
        wide: &'a [u32],
        n_wide: usize,
    ) -> Result<Self, String> {
        if n_small == 0 && n_wide == 0 {
            return Err("at least one feature field is required".into());
        }
        if n_small > 0 && !small.len().is_multiple_of(n_small) {
            return Err("small block length is not a multiple of its field count".into());
        }
        if n_wide > 0 && !wide.len().is_multiple_of(n_wide) {
            return Err("wide block length is not a multiple of its field count".into());
        }
        if let (Some(a), Some(b)) = (
            small.len().checked_div(n_small),
            wide.len().checked_div(n_wide),
        ) {
            if a != b {
                return Err(format!("blocks disagree on the row count ({a} vs {b})"));
            }
        }
        Ok(Self {
            small,
            n_small,
            wide,
            n_wide,
        })
    }

    pub fn n_rows(&self) -> usize {
        self.small
            .len()
            .checked_div(self.n_small)
            .or_else(|| self.wide.len().checked_div(self.n_wide))
            .unwrap_or(0)
    }

    pub fn n_fields(&self) -> usize {
        self.n_small + self.n_wide
    }

    /// Write the hashed buckets of `row` into `out` (length `n_fields`).
    #[inline]
    pub fn buckets(&self, row: usize, mask: u64, out: &mut [usize]) {
        let s = &self.small[row * self.n_small..(row + 1) * self.n_small];
        for (f, &v) in s.iter().enumerate() {
            out[f] = bucket(f, v as u32, mask);
        }
        let w = &self.wide[row * self.n_wide..(row + 1) * self.n_wide];
        for (j, &v) in w.iter().enumerate() {
            out[self.n_small + j] = bucket(self.n_small + j, v, mask);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mix_is_deterministic_and_spreads() {
        assert_eq!(mix64(0), 0);
        assert_ne!(mix64(1), mix64(2));
        assert_eq!(mix64(12345), mix64(12345));
    }

    #[test]
    fn buckets_respect_mask_and_field() {
        let mask = (1u64 << 10) - 1;
        for f in 0..5 {
            for v in 0..100u32 {
                assert!(bucket(f, v, mask) < 1024);
            }
        }
        assert_ne!(bucket(0, 7, u64::MAX), bucket(1, 7, u64::MAX));
    }

    #[test]
    fn features_validate_shapes() {
        let small = [1u8, 2, 3, 4];
        let wide = [5u32, 6];
        let f = Features::new(&small, 2, &wide, 1).unwrap();
        assert_eq!(f.n_rows(), 2);
        assert_eq!(f.n_fields(), 3);
        assert!(Features::new(&small, 3, &wide, 1).is_err());
        assert!(Features::new(&small, 2, &wide, 2).is_err());
        assert!(Features::new(&[], 0, &[], 0).is_err());
        let mut out = [0usize; 3];
        f.buckets(1, u64::MAX, &mut out);
        assert_eq!(out[2], bucket(2, 6, u64::MAX));
    }
}
