//! The eight symmetries of the board (rotations and reflections of the square), applied to
//! the big board and to every small board alike. A position and its image under a symmetry
//! are the same game, so they make free training data.

use super::{Move, Square};

/// A symmetry of the 3x3 grid. The same mapping is applied to board coordinates and to
/// square coordinates within each board.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Symmetry {
    Identity,
    /// Quarter turn clockwise.
    Rotate90,
    Rotate180,
    /// Quarter turn counter-clockwise.
    Rotate270,
    /// Mirror left-right.
    FlipHorizontal,
    /// Mirror top-bottom.
    FlipVertical,
    /// Transpose across the main diagonal (top-left to bottom-right).
    FlipDiagonal,
    /// Transpose across the anti-diagonal (top-right to bottom-left).
    FlipAntiDiagonal,
}

impl Symmetry {
    pub const ALL: [Symmetry; 8] = [
        Symmetry::Identity,
        Symmetry::Rotate90,
        Symmetry::Rotate180,
        Symmetry::Rotate270,
        Symmetry::FlipHorizontal,
        Symmetry::FlipVertical,
        Symmetry::FlipDiagonal,
        Symmetry::FlipAntiDiagonal,
    ];

    /// The symmetries other than the identity.
    pub const NON_TRIVIAL: [Symmetry; 7] = [
        Symmetry::Rotate90,
        Symmetry::Rotate180,
        Symmetry::Rotate270,
        Symmetry::FlipHorizontal,
        Symmetry::FlipVertical,
        Symmetry::FlipDiagonal,
        Symmetry::FlipAntiDiagonal,
    ];

    pub const fn apply(self, square: Square) -> Square {
        let (r, c) = (square.row, square.col);
        let (row, col) = match self {
            Symmetry::Identity => (r, c),
            Symmetry::Rotate90 => (c, 2 - r),
            Symmetry::Rotate180 => (2 - r, 2 - c),
            Symmetry::Rotate270 => (2 - c, r),
            Symmetry::FlipHorizontal => (r, 2 - c),
            Symmetry::FlipVertical => (2 - r, c),
            Symmetry::FlipDiagonal => (c, r),
            Symmetry::FlipAntiDiagonal => (2 - c, 2 - r),
        };
        Square { row, col }
    }

    /// Applies the symmetry to both coordinates of a move.
    pub const fn apply_move(self, mv: Move) -> Move {
        Move::new(self.apply(mv.board), self.apply(mv.square))
    }

    pub const fn inverse(self) -> Symmetry {
        match self {
            Symmetry::Rotate90 => Symmetry::Rotate270,
            Symmetry::Rotate270 => Symmetry::Rotate90,
            other => other,
        }
    }

    /// Permutation of a 9-bit square mask (bit `i` = `Square::from_index(i)`), for
    /// implementations that store boards as masks.
    pub(crate) const fn permute_mask(self, mask: u16) -> u16 {
        PERMUTATIONS[self as usize][mask as usize]
    }
}

/// `PERMUTATIONS[symmetry][mask]`: the 9-bit mask with every set bit moved to its image.
const PERMUTATIONS: [[u16; 512]; 8] = {
    let mut tables = [[0u16; 512]; 8];
    let mut s = 0;
    while s < 8 {
        let sym = Symmetry::ALL[s];
        let mut mask = 0;
        while mask < 512 {
            let mut out = 0u16;
            let mut i = 0;
            while i < 9 {
                if mask >> i & 1 != 0 {
                    out |= 1 << sym.apply(Square::from_index(i)).index();
                }
                i += 1;
            }
            tables[s][mask] = out;
            mask += 1;
        }
        s += 1;
    }
    tables
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symmetries_permute_squares_and_invert() {
        for sym in Symmetry::ALL {
            let mut seen = [false; 9];
            for sq in Square::ALL {
                let image = sym.apply(sq);
                assert!(!seen[image.index()], "{sym:?} is not a bijection");
                seen[image.index()] = true;
                assert_eq!(sym.inverse().apply(image), sq, "{sym:?} inverse");
            }
        }
        // All eight are distinct maps.
        for (i, a) in Symmetry::ALL.iter().enumerate() {
            for b in &Symmetry::ALL[i + 1..] {
                assert!(Square::ALL.iter().any(|s| a.apply(*s) != b.apply(*s)));
            }
        }
        assert_eq!(
            Symmetry::ALL[Symmetry::Rotate90 as usize],
            Symmetry::Rotate90
        );
    }

    #[test]
    fn specific_images() {
        let top_left = Square::from_index(0);
        assert_eq!(Symmetry::Rotate90.apply(top_left), Square::from_index(2));
        assert_eq!(Symmetry::Rotate180.apply(top_left), Square::from_index(8));
        assert_eq!(
            Symmetry::FlipHorizontal.apply(top_left),
            Square::from_index(2)
        );
        assert_eq!(
            Symmetry::FlipVertical.apply(top_left),
            Square::from_index(6)
        );
        assert_eq!(
            Symmetry::FlipDiagonal.apply(Square::from_index(1)),
            Square::from_index(3)
        );
        assert_eq!(
            Symmetry::FlipAntiDiagonal.apply(top_left),
            Square::from_index(8)
        );
        for sym in Symmetry::ALL {
            assert_eq!(
                sym.apply(Square::from_index(4)),
                Square::from_index(4),
                "centre is fixed"
            );
        }
    }

    #[test]
    fn mask_permutation_matches_square_mapping() {
        for sym in Symmetry::ALL {
            for i in 0..9 {
                assert_eq!(
                    sym.permute_mask(1 << i),
                    1 << sym.apply(Square::from_index(i)).index()
                );
            }
            assert_eq!(sym.permute_mask(0x1FF), 0x1FF);
            assert_eq!(sym.permute_mask(0), 0);
            assert_eq!(sym.permute_mask(0b101_000_101).count_ones(), 4);
        }
    }
}
