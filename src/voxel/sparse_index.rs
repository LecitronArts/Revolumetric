use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

pub type U64IndexMap<V> = HashMap<u64, V, BuildHasherDefault<U64MixHasher>>;

#[derive(Default)]
pub struct U64MixHasher {
    state: u64,
}

impl Hasher for U64MixHasher {
    fn finish(&self) -> u64 {
        self.state
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut state = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            state ^= u64::from(*byte);
            state = state.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.state = state;
    }

    fn write_u64(&mut self, value: u64) {
        self.state = mix_u64(value);
    }
}

fn mix_u64(mut value: u64) -> u64 {
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

pub fn new_u64_index_map<V>() -> U64IndexMap<V> {
    U64IndexMap::default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{Hash, Hasher};

    #[test]
    fn u64_mix_hasher_hashes_integer_keys_with_avalanche_mix() {
        let mut hasher = U64MixHasher::default();
        42u64.hash(&mut hasher);

        assert_ne!(hasher.finish(), 42);
        assert_eq!(hasher.finish(), mix_u64(42));
    }

    #[test]
    fn u64_index_map_round_trips_sparse_keys() {
        let mut map = new_u64_index_map();
        map.insert(18_874_367, 7usize);

        assert_eq!(map.get(&18_874_367), Some(&7));
        assert_eq!(map.get(&1), None);
    }
}
