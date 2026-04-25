/// Encodes a 3D position within an 8^3 brick to a Morton (Z-curve) index.
/// Each coordinate must be in [0, 7]. Returns index in [0, 511].
pub fn encode(x: u32, y: u32, z: u32) -> u32 {
    debug_assert!(x < 8 && y < 8 && z < 8, "coordinates must be in [0, 7]");
    let mut m = 0u32;
    for bit in 0..3u32 {
        m |= ((x >> bit) & 1) << (3 * bit);
        m |= ((y >> bit) & 1) << (3 * bit + 1);
        m |= ((z >> bit) & 1) << (3 * bit + 2);
    }
    m
}

pub fn decode(index: u32) -> (u32, u32, u32) {
    debug_assert!(index < 512, "index must be in [0, 511]");
    let (mut x, mut y, mut z) = (0u32, 0u32, 0u32);
    for bit in 0..3u32 {
        x |= ((index >> (3 * bit)) & 1) << bit;
        y |= ((index >> (3 * bit + 1)) & 1) << bit;
        z |= ((index >> (3 * bit + 2)) & 1) << bit;
    }
    (x, y, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn origin_is_zero() {
        assert_eq!(encode(0, 0, 0), 0);
    }

    #[test]
    fn unit_axes() {
        assert_eq!(encode(1, 0, 0), 0b001);
        assert_eq!(encode(0, 1, 0), 0b010);
        assert_eq!(encode(0, 0, 1), 0b100);
    }

    #[test]
    fn max_corner() {
        assert_eq!(encode(7, 7, 7), 511);
    }

    #[test]
    fn roundtrip_all_512() {
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    let m = encode(x, y, z);
                    assert_eq!(decode(m), (x, y, z));
                }
            }
        }
    }

    #[test]
    fn unique_indices() {
        let mut seen = std::collections::HashSet::new();
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    assert!(seen.insert(encode(x, y, z)));
                }
            }
        }
        assert_eq!(seen.len(), 512);
    }
}
