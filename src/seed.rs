//! Deriving the seeds of a run's parts (threads, generations, ...) from the run's seed.

/// SplitMix64's output function: a bijection on `u64` in which every input bit affects every
/// output bit.
const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed of part `index` of something seeded `base`, scrambled rather than added: with
/// `base + index`, part 1 of `base` was part 0 of `base + 1`, so a thread of one generation
/// replayed another generation's games. For a fixed `index` distinct bases give distinct
/// seeds; any other two `(base, index)` pairs (nested ones too) meet only by chance.
pub const fn derive_seed(base: u64, index: u64) -> u64 {
    splitmix64(base ^ splitmix64(index))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_seeds_do_not_repeat() {
        let mut seen = std::collections::HashSet::new();
        for generation in 0..64 {
            let base = derive_seed(7, generation);
            for thread in 0..64 {
                assert!(seen.insert(derive_seed(base, thread)));
            }
        }
        assert_ne!(derive_seed(0, 1), derive_seed(1, 0));
    }
}
