use std::fmt::Display;

fn main() {
    let mut grid = FullGrid::default();
    println!("{grid:?}");
    println!("{:?}", grid.to_occupancy());
    let (b, f) = (3, 4);
    grid.play(b, f, Player::X);
    println!("{grid:?}");
    println!("{:?}", grid.to_occupancy());
    grid.play(7, 1, Player::O);
    println!("{grid:?}");
    println!("{:?}", grid.to_occupancy());
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Grid<T, const N: usize>([T; N]);

impl<T, const N: usize> Grid<T, N> {
    fn map<V>(self, f: impl FnMut(T) -> V) -> Grid<V, N> {
        Grid(self.0.map(f))
    }
}

type Grid9<T> = Grid<T, 9>;

impl<T> Grid9<T> {
    fn rows(self) -> [[T; 3]; 3] {
        let [s0, s1, s2, s3, s4, s5, s6, s7, s8] = self.0;
        [
            [s0, s1, s2], //
            [s3, s4, s5], //
            [s6, s7, s8], //
        ]
    }
}

impl<T: Default> Default for Grid9<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(i8)]
enum Player {
    X = 1,
    O = -1,
}

type SmallGrid = Grid9<Option<Player>>;

const LINES: [[usize; 3]; 8] = [
    // rows
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    // cols
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    // diags
    [0, 4, 8],
    [2, 4, 6],
];

const fn lines_for(i: usize) -> [Option<[usize; 3]>; 4] {
    let row = LINES[i / 3];
    let col = LINES[(i % 3) + 3];
    let diag1 = match i {
        0 | 4 | 8 => Some(LINES[6]),
        _ => None,
    };
    let diag2 = match i {
        2 | 4 | 6 => Some(LINES[7]),
        _ => None,
    };
    [Some(row), Some(col), diag1, diag2]
}
impl Display for SmallGrid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let [r1, r2, r3] = self.repr_lines();
        writeln!(f, "{r1}")?;
        writeln!(f, "{r2}")?;
        writeln!(f, "{r3}")?;
        Ok(())
    }
}

impl SmallGrid {
    fn state(&self) -> PlayState {
        for l in LINES {
            if l.into_iter().all(|x| self.0[x] == Some(Player::X)) {
                return PlayState::X;
            }
            if l.into_iter().all(|x| self.0[x] == Some(Player::O)) {
                return PlayState::O;
            }
        }
        if self.0.iter().all(|x| x.is_some()) {
            PlayState::Full
        } else {
            PlayState::Open
        }
    }

    fn repr_lines(self) -> [impl Display; 3] {
        #[repr(transparent)]
        struct D([Option<Player>; 3]);
        fn p(player: Option<Player>) -> char {
            match player {
                Some(Player::X) => 'X',
                Some(Player::O) => 'O',
                None => ' ',
            }
        }
        impl Display for D {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                let [a, b, c] = self.0;
                write!(f, "{}{}{}", p(a), p(b), p(c))
            }
        }

        self.rows().map(D)
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
struct BigCell {
    grid: SmallGrid,
    state: PlayState,
}

impl BigCell {
    fn play(&mut self, field: usize, p: Player) {
        self.grid.0[field] = Some(p);
        let won = lines_for(field)
            .into_iter()
            .flatten()
            .any(|l| l.into_iter().all(|i| self.grid.0[i] == Some(Player::X)));
        if won {
            self.state = PlayState::Won(p);
            return;
        }
        let done = self.grid.0.iter().all(|x| x.is_some());
        if done {
            self.state = PlayState::Full;
        }
        debug_assert!(self.state == self.grid.state())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
enum PlayState {
    #[default]
    Open,
    Full,
    Won(Player),
}

impl PlayState {
    fn is_done(self) -> bool {
        match self {
            PlayState::Open => false,
            PlayState::Full | PlayState::Won(_) => true,
        }
    }
}

impl PlayState {
    const X: Self = PlayState::Won(Player::X);
    const O: Self = PlayState::Won(Player::O);
}

type FullGrid = Grid<BigCell, 9>;

impl FullGrid {
    fn state(&self) -> PlayState {
        for l in LINES {
            if l.into_iter().all(|x| self.0[x].state == PlayState::X) {
                return PlayState::X;
            }
            if l.into_iter().all(|x| self.0[x].state == PlayState::O) {
                return PlayState::O;
            }
        }
        if self.0.iter().all(|x| x.state.is_done()) {
            PlayState::Full
        } else {
            PlayState::Open
        }
    }

    fn play(&mut self, board: usize, field: usize, p: Player) {
        self.0[board].play(field, p);
    }
    pub fn to_occupancy(self) -> Occupancy {
        let xs = self.map(|x| x.grid.map(|x| x == Some(Player::X)));
        let os = self.map(|x| x.grid.map(|x| x == Some(Player::O)));
        let xwin = self.map(|x| x.state == PlayState::X);
        let owin = self.map(|x| x.state == PlayState::O);
        let done = self.map(|x| x.state.is_done());

        Occupancy {
            xs,
            os,
            xwin,
            owin,
            done,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
struct Occupancy {
    xs: Grid9<Grid9<bool>>,
    os: Grid9<Grid9<bool>>,
    xwin: Grid9<bool>,
    owin: Grid9<bool>,
    done: Grid9<bool>,
}
