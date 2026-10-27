//! `FixedVec.hs`: the fixed-width vector family, for exactly the shapes the generated
//! network needs plus the ones hand-written code asks for (`export -- <flags>` or
//! `--vec-extra "<flags>"`).
//!
//! A flat `VecN` is a strict record of `N` unpacked `Float` (or `Int32`/`Int16`). A nested
//! `VecN` (`N = K x M`) is a strict record of `K` boxed `VecM` chunks whose helpers delegate
//! to the `VecM` ones: GHC's native code generator spills badly on functions that bind more
//! than ~64 floats at once. Matrices (`RowsM_N`) are records of row vectors, one row per
//! output.

use std::collections::{BTreeMap, BTreeSet};

use super::haskell::{BODY_BEGIN, BODY_END, IMPORTS_BEGIN, IMPORTS_END};

/// Columns per pass of a sparse layer unless `--sparse M:N:K` says otherwise (K = 4 measured
/// the same, K = 1 slower than the dense layer).
pub const SPARSE_K: usize = 8;

/// What to generate, in the terms of the export's FixedVec flags.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VecSpec {
    pub floats: BTreeSet<usize>,
    pub int32s: BTreeSet<usize>,
    pub int16s: BTreeSet<usize>,
    /// Nested widths: `n -> (k, m)`, `Vec{n}` as `k` chunks of `Vec{m}`.
    pub nest: BTreeMap<usize, (usize, usize)>,
    /// `RowsM_N` containers, as `(m, n)`.
    pub rows: Vec<(usize, usize)>,
    /// Dense layers `(m, n)`: `linearM_N`, `denseM_N` (and `residualM_M`), adding `RowsM_N`.
    pub dense: Vec<(usize, usize)>,
    /// `concatK_N`.
    pub concats: Vec<(usize, usize)>,
    /// `sumK_N`.
    pub sums: Vec<(usize, usize)>,
    /// `axpyK_N`, as `(k, n)`: `acc + s0 * c0 + .. + s(K-1) * c(K-1)`.
    pub axpys: Vec<(usize, usize)>,
    /// Column containers `(m, n)` of a layer with `m` outputs and `n` inputs: `ColsN_M` and
    /// `columnsM_N`, adding `RowsM_N`.
    pub cols: Vec<(usize, usize)>,
    /// Sparse layers `(m, n, k)`: `sparseDenseM_N` (and `sparseResidualM`), computed by
    /// columns over the nonzero inputs, `k` columns per pass; adds the columns and axpys.
    pub sparse: Vec<(usize, usize, usize)>,
    /// `type NAME = VecN`.
    pub aliases: Vec<(String, usize)>,
    pub module: String,
}

impl VecSpec {
    pub fn new(module: &str) -> Self {
        Self {
            module: module.to_string(),
            ..Default::default()
        }
    }

    /// Parses the export's FixedVec flags: `--float N..`, `--int32 N..`, `--int16 N..`,
    /// `--nest N=KxM,..`, `--rows M:N,..`, `--dense M:N,..`, `--concat K:N,..`, `--sum K:N,..`,
    /// `--alias NAME=N,..`, `--module NAME`.
    pub fn parse_args(words: &[&str]) -> Result<Self, String> {
        let mut spec = Self::new("GenVec");
        let mut i = 0;
        let num = |w: &str| {
            w.parse::<usize>()
                .map_err(|_| format!("FixedVec flags: expected a number, got {w:?}"))
        };
        let pairs = |w: &str| -> Result<Vec<(usize, usize)>, String> {
            w.split(',')
                .filter(|p| !p.is_empty())
                .map(|p| {
                    let (a, b) = p
                        .split_once(':')
                        .ok_or_else(|| format!("FixedVec flags: expected M:N, got {p:?}"))?;
                    Ok((num(a)?, num(b)?))
                })
                .collect()
        };
        while i < words.len() {
            let flag = words[i];
            let value = |i: usize| {
                words
                    .get(i + 1)
                    .copied()
                    .ok_or_else(|| format!("FixedVec flags: {flag} needs a value"))
            };
            match flag {
                "--float" | "--int32" | "--int16" => {
                    let mut j = i + 1;
                    while j < words.len() && !words[j].starts_with("--") {
                        let n = num(words[j])?;
                        match flag {
                            "--float" => spec.floats.insert(n),
                            "--int32" => spec.int32s.insert(n),
                            _ => spec.int16s.insert(n),
                        };
                        j += 1;
                    }
                    if j == i + 1 {
                        return Err(format!("FixedVec flags: {flag} needs at least one width"));
                    }
                    i = j;
                    continue;
                }
                "--nest" => {
                    for part in value(i)?.split(',').filter(|p| !p.is_empty()) {
                        let (n, km) = part.split_once('=').ok_or_else(|| {
                            format!("FixedVec flags: expected N=KxM, got {part:?}")
                        })?;
                        let (k, m) = km.split_once('x').ok_or_else(|| {
                            format!("FixedVec flags: expected N=KxM, got {part:?}")
                        })?;
                        let (n, k, m) = (num(n)?, num(k)?, num(m)?);
                        if k * m != n {
                            return Err(format!("FixedVec flags: --nest {part}: {k}x{m} != {n}"));
                        }
                        spec.nest.insert(n, (k, m));
                    }
                }
                "--rows" => spec.rows.extend(pairs(value(i)?)?),
                "--axpy" => spec.axpys.extend(pairs(value(i)?)?),
                "--cols" => spec.cols.extend(pairs(value(i)?)?),
                "--sparse" => {
                    for part in value(i)?.split(',').filter(|p| !p.is_empty()) {
                        let nums: Vec<usize> =
                            part.split(':').map(num).collect::<Result<_, _>>()?;
                        match nums[..] {
                            [m, n] => spec.sparse.push((m, n, SPARSE_K)),
                            [m, n, k] if k > 0 => spec.sparse.push((m, n, k)),
                            _ => {
                                return Err(format!(
                                    "FixedVec flags: expected M:N or M:N:K, got {part:?}"
                                ));
                            }
                        }
                    }
                }
                "--dense" => spec.dense.extend(pairs(value(i)?)?),
                "--concat" => spec.concats.extend(pairs(value(i)?)?),
                "--sum" => spec.sums.extend(pairs(value(i)?)?),
                "--alias" => {
                    for part in value(i)?.split(',').filter(|p| !p.is_empty()) {
                        let (name, w) = part.split_once('=').ok_or_else(|| {
                            format!("FixedVec flags: expected NAME=N, got {part:?}")
                        })?;
                        spec.aliases.push((name.to_string(), num(w)?));
                    }
                }
                "--module" => spec.module = value(i)?.to_string(),
                other => return Err(format!("FixedVec flags: unknown flag {other:?}")),
            }
            i += 2;
        }
        Ok(spec)
    }

    /// Adds everything in `other` (its module name is ignored).
    pub fn merge(&mut self, other: &VecSpec) {
        self.floats.extend(&other.floats);
        self.int32s.extend(&other.int32s);
        self.int16s.extend(&other.int16s);
        for (&n, &km) in &other.nest {
            self.nest.insert(n, km);
        }
        for (dst, src) in [
            (&mut self.rows, &other.rows),
            (&mut self.dense, &other.dense),
            (&mut self.concats, &other.concats),
            (&mut self.sums, &other.sums),
            (&mut self.axpys, &other.axpys),
            (&mut self.cols, &other.cols),
        ] {
            for p in src {
                if !dst.contains(p) {
                    dst.push(*p);
                }
            }
        }
        for p in &other.sparse {
            if !self.sparse.contains(p) {
                self.sparse.push(*p);
            }
        }
        for a in &other.aliases {
            if !self.aliases.contains(a) {
                self.aliases.push(a.clone());
            }
        }
    }

    /// The shapes as the export's FixedVec flags: what makes the same module on its own.
    pub fn flags(&self) -> String {
        let nums = |s: &BTreeSet<usize>| {
            s.iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        };
        let pairs = |v: &[(usize, usize)]| {
            v.iter()
                .map(|(a, b)| format!("{a}:{b}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let mut out = Vec::new();
        if !self.floats.is_empty() {
            out.push(format!("--float {}", nums(&self.floats)));
        }
        if !self.int32s.is_empty() {
            out.push(format!("--int32 {}", nums(&self.int32s)));
        }
        if !self.int16s.is_empty() {
            out.push(format!("--int16 {}", nums(&self.int16s)));
        }
        if !self.nest.is_empty() {
            let nest: Vec<String> = self
                .nest
                .iter()
                .map(|(n, (k, m))| format!("{n}={k}x{m}"))
                .collect();
            out.push(format!("--nest {}", nest.join(",")));
        }
        for (flag, v) in [
            ("--rows", &self.rows),
            ("--dense", &self.dense),
            ("--concat", &self.concats),
            ("--sum", &self.sums),
            ("--axpy", &self.axpys),
            ("--cols", &self.cols),
        ] {
            if !v.is_empty() {
                out.push(format!("{flag} {}", pairs(v)));
            }
        }
        if !self.sparse.is_empty() {
            let sparse: Vec<String> = self
                .sparse
                .iter()
                .map(|&(m, n, k)| {
                    if k == SPARSE_K {
                        format!("{m}:{n}")
                    } else {
                        format!("{m}:{n}:{k}")
                    }
                })
                .collect();
            out.push(format!("--sparse {}", sparse.join(",")));
        }
        if !self.aliases.is_empty() {
            let aliases: Vec<String> = self
                .aliases
                .iter()
                .map(|(n, w)| format!("{n}={w}"))
                .collect();
            out.push(format!("--alias {}", aliases.join(",")));
        }
        out.join(" ")
    }

    /// The module's text, without the bundler's markers.
    pub fn generate(&self) -> String {
        Gen::new(self).generate(false)
    }

    /// The module's text with its imports and declarations inside the bundler's markers
    /// (otherwise the same as [`VecSpec::generate`]).
    pub fn generate_bundled(&self) -> String {
        Gen::new(self).generate(true)
    }
}

// ---------------------------------------------------------------- formatting helpers

/// Index `i` among `count` names of one role, displayed zero-padded to the width of the
/// largest index, so that lists of bindings line up (`a00 .. a63`).
pub(super) fn pad_index(i: usize, count: usize) -> PaddedIndex {
    let largest = count.saturating_sub(1);
    PaddedIndex {
        i,
        width: largest.checked_ilog10().map_or(1, |d| d as usize + 1),
    }
}

/// See [`pad_index`].
pub(super) struct PaddedIndex {
    i: usize,
    width: usize,
}

impl std::fmt::Display for PaddedIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0width$}", self.i, width = self.width)
    }
}

fn names(prefix: &str, n: usize) -> Vec<String> {
    (0..n)
        .map(|i| format!("{prefix}{}", pad_index(i, n)))
        .collect()
}

/// `items` on one line at `indent`: a pattern or an expression is never broken across lines
/// (they read best whole, however long).
pub(super) fn line(items: &[String], indent: usize) -> String {
    format!("{}{}", " ".repeat(indent), items.join(" "))
}

/// A generated expression, for layout: an atom, written as given, or an application whose
/// children may go on lines of their own.
pub(super) enum Expr {
    Atom(String),
    App(String, Vec<Expr>),
}

/// A child longer than this always puts an application's children one per line.
const LONG_CHILD: usize = 30;
/// On a line longer than `WIDE_LINE`, a child longer than `SHORT_CHILD` already does.
const SHORT_CHILD: usize = 16;
const WIDE_LINE: usize = 300;

impl Expr {
    /// A flat pattern or application given as words, `head c0 c1 ..` (no parentheses).
    pub(super) fn words(text: &str) -> Self {
        let mut words = text.split(' ').map(str::to_string);
        let head = words.next().unwrap_or_default();
        Expr::App(head, words.map(Expr::Atom).collect())
    }

    /// `head c0 c1 ..` with atom children.
    pub(super) fn app(head: impl Into<String>, children: &[String]) -> Self {
        Expr::App(
            head.into(),
            children.iter().cloned().map(Expr::Atom).collect(),
        )
    }

    /// The expression on one line.
    fn inline(&self) -> String {
        match self {
            Expr::Atom(a) => a.clone(),
            Expr::App(head, children) => {
                let mut out = head.clone();
                for c in children {
                    if !out.is_empty() {
                        out.push(' ');
                    }
                    out.push_str(&c.as_child());
                }
                out
            }
        }
    }

    /// As a child: an application in parentheses.
    fn as_child(&self) -> String {
        match self {
            Expr::Atom(a) => a.clone(),
            Expr::App(..) => format!("({})", self.inline()),
        }
    }

    /// The expression at `indent`: on one line unless its children are long (see
    /// `LONG_CHILD`), then the head on its line and every child on its own two deeper, laid
    /// out the same way. Nothing is ever broken inside an atom.
    pub(super) fn layout(&self, indent: usize) -> Vec<String> {
        let pad = " ".repeat(indent);
        let flat = self.inline();
        let Expr::App(head, children) = self else {
            return vec![format!("{pad}{flat}")];
        };
        let longest = children
            .iter()
            .map(|c| c.as_child().len())
            .max()
            .unwrap_or(0);
        let split =
            longest > LONG_CHILD || (indent + flat.len() > WIDE_LINE && longest > SHORT_CHILD);
        if !split {
            if indent + flat.len() >= RECT_LINE {
                let cells: Vec<String> = children.iter().map(Expr::as_child).collect();
                return rectangle(head, &cells, indent);
            }
            return vec![format!("{pad}{flat}")];
        }
        let mut out = vec![format!("{pad}{head}")];
        for c in children {
            out.extend(c.child_layout(indent + 2));
        }
        out
    }

    /// As a child at `indent`: an application in parentheses, opened on its first line (its
    /// text one column in, so that its own rows stay aligned) and closed on its last.
    fn child_layout(&self, indent: usize) -> Vec<String> {
        match self {
            Expr::Atom(a) => vec![format!("{}{a}", " ".repeat(indent))],
            Expr::App(..) => {
                let mut lines = self.layout(indent + 1);
                lines[0].replace_range(indent..indent + 1, "(");
                lines.last_mut().expect("a line").push(')');
                lines
            }
        }
    }
}

/// The function head `f (x0:x1:..:tail) =` as an application's items, one cell per element.
fn cons_head(f: &str, xs: &[String], tail: &str) -> Vec<String> {
    let mut items = vec![f.to_string()];
    for (i, x) in xs.iter().enumerate() {
        let open = if i == 0 { "(" } else { "" };
        let close = if i + 1 == xs.len() {
            format!(":{tail}) =")
        } else {
            ":".to_string()
        };
        items.push(format!("{open}{x}{close}"));
    }
    items
}

/// From this length on, an application of short children goes into rows ([`rectangle`]).
const RECT_LINE: usize = 250;
/// The width rows of children keep within (with room for closing parentheses and commas).
const ROW_LINE: usize = 94;

/// `head` and short `cells` in rows of equal length: under the first cell, or failing that
/// on lines of their own two deeper, `k` cells per row with `k` a divisor of the cell count
/// (powers of two first, then the others; the widest that fits in `ROW_LINE`, at least 2);
/// if no divisor fits, rows filled as far as they go.
fn rectangle(head: &str, cells: &[String], indent: usize) -> Vec<String> {
    let n = cells.len();
    // A bracket opened by the first cell only (`(x0:`, `[a00,`): the later rows start one
    // column further in, under the first element.
    let opens = |c: &String| c.starts_with('(') || c.starts_with('[');
    let shift = usize::from(opens(&cells[0]) && !cells[1..].iter().any(opens));
    let fits = |start: usize, k: usize| {
        cells.chunks(k).enumerate().all(|(i, row)| {
            let lead = if i == 0 { 0 } else { shift };
            start + lead + row.iter().map(String::len).sum::<usize>() + row.len() <= ROW_LINE
        })
    };
    let divisors: Vec<usize> = (2..=n).rev().filter(|k| n % k == 0).collect();
    let best = |start: usize| {
        let powers = divisors.iter().copied().filter(|k| k.is_power_of_two());
        let others = divisors.iter().copied().filter(|k| !k.is_power_of_two());
        powers
            .clone()
            .find(|&k| fits(start, k))
            .or_else(|| others.clone().find(|&k| fits(start, k)))
    };
    let after_head = if head.is_empty() {
        indent
    } else {
        indent + head.len() + 1
    };
    let below = indent + 2;
    let rows = |start: usize, sizes: &[usize]| -> Vec<String> {
        let mut out = Vec::new();
        let mut rest = cells;
        for &size in sizes {
            let (row, tail) = rest.split_at(size);
            rest = tail;
            let lead = if out.is_empty() { 0 } else { shift };
            out.push(format!("{}{}", " ".repeat(start + lead), row.join(" ")));
        }
        out
    };
    let (start, sizes) = if let Some(k) = best(after_head) {
        (after_head, vec![k; n / k])
    } else if let Some(k) = best(below) {
        (below, vec![k; n / k])
    } else {
        // Ragged: fill each row as far as it goes.
        let start = if after_head <= ROW_LINE / 2 {
            after_head
        } else {
            below
        };
        let mut sizes = Vec::new();
        let (mut count, mut width) = (0, start);
        for cell in cells {
            if count > 0 && width + 1 + cell.len() > ROW_LINE {
                sizes.push(count);
                (count, width) = (0, start + shift);
            }
            width += cell.len() + usize::from(count > 0);
            count += 1;
        }
        sizes.push(count);
        (start, sizes)
    };
    let mut lines = rows(start, &sizes);
    if head.is_empty() {
        // Nothing to put first: the rows are the expression.
    } else if start == after_head {
        // The first row goes on the head's line.
        lines[0].replace_range(indent..indent + head.len(), head);
    } else {
        lines.insert(0, format!("{}{head}", " ".repeat(indent)));
    }
    lines
}

/// Sums `terms` as a balanced tree, `((t0 + t1) + (t2 + t3)) + ..`: GHC never reassociates
/// `Float` additions, so the expression's shape is the dependency graph, and a tree of depth
/// log2 n leaves the most additions independent (measured: 25% less time per multiply-add in
/// a 64-wide dot than four strided partial sums).
fn sum_tree(terms: &[String]) -> Sum {
    let mut level: Vec<Sum> = terms.iter().cloned().map(Sum::Term).collect();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut items = level.into_iter();
        while let Some(a) = items.next() {
            next.push(match items.next() {
                Some(b) => Sum::Add(Box::new(a), Box::new(b)),
                None => a,
            });
        }
        level = next;
    }
    level.pop().unwrap_or_else(|| Sum::Term("0".to_string()))
}

/// Columns a sum's line may take before its tree splits.
const SUM_LINE: usize = 100;

/// A tree of additions, laid out along its own boundaries.
enum Sum {
    Term(String),
    Add(Box<Sum>, Box<Sum>),
}

impl Sum {
    fn inline(&self) -> String {
        match self {
            Sum::Term(t) => t.clone(),
            Sum::Add(l, r) => format!("{} + {}", l.as_child(), r.as_child()),
        }
    }

    fn as_child(&self) -> String {
        match self {
            Sum::Term(t) => t.clone(),
            Sum::Add(..) => format!("({})", self.inline()),
        }
    }

    /// The sum at `indent`: on one line if it fits in `SUM_LINE` columns, else its two
    /// operands each on their own lines (the right one led by `+`), laid out the same way:
    /// the two children of a node are both inline or both split, never half and half.
    fn layout(&self, indent: usize) -> Vec<String> {
        let flat = self.inline();
        match self {
            Sum::Add(l, r) if indent + flat.len() > SUM_LINE => {
                let mut out = l.child_layout(indent);
                let mut right = r.child_layout(indent + 2);
                right[0].replace_range(indent..indent + 2, "+ ");
                out.extend(right);
                out
            }
            _ => vec![format!("{}{flat}", " ".repeat(indent))],
        }
    }

    /// As an operand: a sum in parentheses, opened on its first line and closed on its last.
    fn child_layout(&self, indent: usize) -> Vec<String> {
        match self {
            Sum::Term(t) => vec![format!("{}{t}", " ".repeat(indent))],
            Sum::Add(..) => {
                let mut lines = self.layout(indent + 1);
                lines[0].replace_range(indent..indent + 1, "(");
                lines.last_mut().expect("a line").push(')');
                lines
            }
        }
    }
}

fn ty_prefix(elem: &str) -> (&'static str, &'static str) {
    match elem {
        "Float" => ("Vec", ""),
        "Int32" => ("I32Vec", "i32"),
        "Int16" => ("I16Vec", "i16"),
        other => panic!("no vector element type {other}"),
    }
}

fn tname(elem: &str, n: usize) -> String {
    format!("{}{n}", ty_prefix(elem).0)
}

/// Helper name for an element type: `dot9`, `i32Dot9`, `i16Dot9`.
fn fname(elem: &str, base: &str, n: usize) -> String {
    let pre = ty_prefix(elem).1;
    if pre.is_empty() {
        format!("{base}{n}")
    } else {
        let mut chars = base.chars();
        let first = chars.next().expect("name").to_ascii_uppercase();
        format!("{pre}{first}{}{n}", chars.as_str())
    }
}

struct Gen<'a> {
    spec: &'a VecSpec,
    out: Vec<String>,
}

impl<'a> Gen<'a> {
    fn new(spec: &'a VecSpec) -> Self {
        Self {
            spec,
            out: Vec::new(),
        }
    }

    fn push(&mut self, line: impl Into<String>) {
        self.out.push(line.into());
    }

    fn line(&mut self, items: &[String], indent: usize) {
        self.out.push(line(items, indent));
    }

    /// An application `items[0] items[1] ..` laid out by [`Expr::layout`].
    fn apply(&mut self, items: &[String], indent: usize) {
        self.layout(&Expr::app(items[0].clone(), &items[1..]), indent);
    }

    /// A definition `head = sum`: the sum on the head's line if the head is one line and the
    /// whole fits in `SUM_LINE` columns, else below it, laid out along its tree.
    fn definition(&mut self, head: Vec<String>, body: &Sum) {
        let flat = body.inline();
        match head.as_slice() {
            [line] if line.len() + 1 + flat.len() <= SUM_LINE => {
                self.push(format!("{line} {flat}"));
            }
            _ => {
                self.out.extend(head);
                self.out.extend(body.layout(2));
            }
        }
    }

    fn layout(&mut self, e: &Expr, indent: usize) {
        self.out.extend(e.layout(indent));
    }

    fn chunks(&self, n: usize) -> Option<(usize, usize)> {
        self.spec.nest.get(&n).copied()
    }

    /// `case (s0, s1, ..) of` at indent `d`, its one alternative matching every scrutinee
    /// against its pattern, one pattern per line however long (they read best whole); the
    /// body goes at indent `d + 4` (continuations `d + 6`). GHC never builds the tuple.
    fn case_match(&mut self, scruts: &[&str], pats: &[String], d: usize) {
        let pats: Vec<Expr> = pats.iter().map(|p| Expr::words(p)).collect();
        self.case_match_exprs(scruts, &pats, d);
    }

    /// [`Gen::case_match`] with the patterns as expressions: each laid out like a
    /// construction (a nested pattern with long parts puts each part on its own line).
    fn case_match_exprs(&mut self, scruts: &[&str], pats: &[Expr], d: usize) {
        self.push(format!("{}case ({}) of", " ".repeat(d), scruts.join(", ")));
        for (i, pat) in pats.iter().enumerate() {
            let mut lines = pat.layout(d + 4);
            if i == 0 {
                lines[0].replace_range(d + 2..d + 4, "( ");
            }
            let close = if i + 1 == pats.len() { " ) ->" } else { "," };
            lines.last_mut().expect("a line").push_str(close);
            self.out.extend(lines);
        }
    }

    // ------------------------------------------------------------ vectors

    fn emit_vec(&mut self, n: usize, elem: &str) {
        let t = tname(elem, n);
        let f = |base: &str| fname(elem, base, n);
        let int_vec = elem != "Float";
        if !int_vec {
            if let Some(nest) = self.chunks(n) {
                self.emit_vec_nested(n, nest, elem);
                return;
            }
        }
        let (a, b) = (names("a", n), names("b", n));
        let last = n - 1;
        self.push(format!(
            "{{- A fixed-width vector of {n} {elem} values, one strict unpacked field per"
        ));
        self.push("   coordinate (coordinate i is field i, counting from 0). -}");
        self.apply(
            &std::iter::once(format!("data {t} = {t}"))
                .chain(std::iter::repeat_n(format!("{{-# UNPACK #-}} !{elem}"), n))
                .collect::<Vec<_>>(),
            0,
        );
        self.push("");
        let (fl, tl, ix) = (f("fromList"), f("toList"), f("index"));
        self.push(format!("{{- {fl} xs"));
        self.push(format!("   RETURNS: the {t} whose coordinate i is xs !! i"));
        self.push(format!(
            "   PRE: length xs >= {n}; extra elements are ignored -}}"
        ));
        self.push(format!("{fl} :: [{elem}] -> {t}"));
        self.apply(&cons_head(&fl, &a, "_"), 0);
        let mut items = vec![t.clone()];
        items.extend(a.iter().cloned());
        self.apply(&items, 2);
        self.push(format!("{fl} _ = error \"{fl}: list too short\""));
        self.push("");
        let fr = f("fromListRest");
        self.push(format!(
            "-- {fr} xs: the {t} of the first {n} elements of xs, and the rest of xs"
        ));
        self.push(format!("{fr} :: [{elem}] -> ({t}, [{elem}])"));
        self.apply(&cons_head(&fr, &a, "rest"), 0);
        let mut items = vec![format!("({t}")];
        items.extend(a[..last].iter().cloned());
        items.push(format!("{}, rest)", a[last]));
        self.apply(&items, 2);
        self.push(format!("{fr} _ = error \"{fr}: list too short\""));
        self.push("");
        self.push(format!("-- {tl} v: the coordinates of v in order"));
        self.push(format!("{tl} :: {t} -> [{elem}]"));
        self.apply(&self.pattern_head(&format!("{tl} ({t}"), &a, ") ="), 0);
        let mut items = vec![
            String::new(),
            format!("[{}{}", a[0], if n > 1 { "," } else { "" }),
        ];
        for x in &a[1..last.max(1)] {
            items.push(format!("{x},"));
        }
        if n > 1 {
            items.push(format!("{}]", a[last]));
        } else {
            items.push("]".to_string());
        }
        self.apply(&items, 2);
        self.push("");
        self.push(format!(
            "-- {ix} i v: coordinate i of v (a jump table).  PRE: 0 <= i < {n}"
        ));
        self.push(format!("{ix} :: Int -> {t} -> {elem}"));
        self.apply(&self.pattern_head(&format!("{ix} i ({t}"), &a, ") ="), 0);
        self.push("  case i of");
        for (i, x) in a.iter().enumerate() {
            self.push(format!("    {i} -> {x}"));
        }
        self.push(format!("    _ -> error \"{ix}: index out of range\""));
        self.push("");
        let (rp, mp, cb) = (f("replicate"), f("map"), f("combine"));
        self.push(format!(
            "-- {rp} x: the vector with every coordinate equal to x"
        ));
        self.push(format!("{rp} :: {elem} -> {t}"));
        let mut items = vec![format!("{rp} x = {t}")];
        items.extend(std::iter::repeat_n("x".to_string(), n));
        self.apply(&items, 0);
        self.push("");
        self.push(format!(
            "-- {mp} f v: f applied to every coordinate (INLINE so f is known at the call site)"
        ));
        self.push(format!("{{-# INLINE {mp} #-}}"));
        self.push(format!("{mp} :: ({elem} -> {elem}) -> {t} -> {t}"));
        self.apply(&self.pattern_head(&format!("{mp} f ({t}"), &a, ") ="), 0);
        let mut items = vec![t.clone()];
        items.extend(a.iter().map(|x| format!("(f {x})")));
        self.apply(&items, 2);
        self.push("");
        self.push(format!(
            "-- {cb} f u v: f applied coordinatewise to u and v (both forced before matching)"
        ));
        self.push(format!("{{-# INLINE {cb} #-}}"));
        self.push(format!(
            "{cb} :: ({elem} -> {elem} -> {elem}) -> {t} -> {t} -> {t}"
        ));
        self.push(format!("{cb} f !u !v ="));
        self.case_match(
            &["u", "v"],
            &[
                format!("{t} {}", a.join(" ")),
                format!("{t} {}", b.join(" ")),
            ],
            2,
        );
        let mut items = vec![t.clone()];
        items.extend(a.iter().zip(&b).map(|(x, y)| format!("(f {x} {y})")));
        self.apply(&items, 6);
        self.push("");
        self.emit_arith(n, elem, &t, &mp, &cb);
        let (sm, dt) = (f("sum"), f("dot"));
        self.push(format!("-- {sm} v: the sum of the coordinates of v"));
        self.push(format!("{sm} :: {t} -> {elem}"));
        let items = self.pattern_head(&format!("{sm} ({t}"), &a, ") =");
        let head = Expr::app(items[0].clone(), &items[1..]);
        self.definition(head.layout(0), &sum_tree(&a));
        if n >= 32 {
            self.push(format!("{{-# NOINLINE {sm} #-}}"));
        }
        self.push("");
        self.push(format!("{{- {dt} u v"));
        self.push("   RETURNS: the dot product of u and v, its products summed as a balanced tree");
        self.push("   so that the additions do not form long dependency chains -}");
        self.push(format!("{dt} :: {t} -> {t} -> {elem}"));
        self.push(format!("{dt} !u !v ="));
        self.case_match(
            &["u", "v"],
            &[
                format!("{t} {}", a.join(" ")),
                format!("{t} {}", b.join(" ")),
            ],
            2,
        );
        let products: Vec<String> = a
            .iter()
            .zip(&b)
            .map(|(x, y)| format!("{x} * {y}"))
            .collect();
        self.out.extend(sum_tree(&products).layout(6));
        self.push(format!("{{-# NOINLINE {dt} #-}}"));
        self.push("");
        if !int_vec && n <= 64 {
            self.push(format!("{{- maskedMax{n} mask v acc"));
            self.push(
                "   RETURNS: the maximum of acc and the coordinates v_i for which bit i of mask is clear -}",
            );
            self.push(format!("maskedMax{n} :: Int -> {t} -> Float -> Float"));
            self.apply(
                &self.pattern_head(
                    &format!("maskedMax{n} m ({t}"),
                    &a,
                    &format!(") acc{} =", pad_index(0, n + 1)),
                ),
                0,
            );
            self.push("  let");
            for (i, x) in a.iter().enumerate() {
                let prev = format!("acc{}", pad_index(i, n + 1));
                self.push(format!(
                    "    acc{} = if testBit m {i} then {prev} else max {prev} {x}",
                    pad_index(i + 1, n + 1)
                ));
            }
            self.push(format!("  in acc{}", pad_index(n, n + 1)));
            self.push("");
            self.push(format!("{{- maskedExp{n} mask c v"));
            self.push(
                "   RETURNS: the vector with coordinate i = exp (v_i - c) where bit i of mask is clear, 0 where set -}",
            );
            self.push(format!("maskedExp{n} :: Int -> Float -> {t} -> {t}"));
            self.apply(
                &self.pattern_head(&format!("maskedExp{n} m c ({t}"), &a, ") ="),
                0,
            );
            let mut items = vec![t.clone()];
            items.extend(
                a.iter()
                    .enumerate()
                    .map(|(i, x)| format!("(if testBit m {i} then 0 else exp ({x} - c))")),
            );
            self.apply(&items, 2);
            self.push("");
        }
    }

    /// `[head, a0, .., a_last + tail]`: a function head over a flat pattern.
    fn pattern_head(&self, head: &str, a: &[String], tail: &str) -> Vec<String> {
        let last = a.len() - 1;
        let mut items = vec![head.to_string()];
        items.extend(a[..last].iter().cloned());
        items.push(format!("{}{tail}", a[last]));
        items
    }

    /// add/sub/scale/relu for a flat width. Small widths inline; from 32 up they are
    /// NOINLINE so that a nested caller makes one call per chunk.
    fn emit_arith(&mut self, n: usize, elem: &str, t: &str, mp: &str, cb: &str) {
        let prag = if n < 32 { "INLINE" } else { "NOINLINE" };
        let (ad, sb) = (fname(elem, "add", n), fname(elem, "sub", n));
        self.push(format!("{ad}, {sb} :: {t} -> {t} -> {t}"));
        self.push(format!("{ad} u v = {cb} (+) u v"));
        self.push(format!("{sb} u v = {cb} (-) u v"));
        self.push(format!("{{-# {prag} {ad} #-}}"));
        self.push(format!("{{-# {prag} {sb} #-}}"));
        self.push("");
        if elem == "Float" {
            self.push(format!("scale{n} :: Float -> {t} -> {t}"));
            self.push(format!("scale{n} s v = {mp} (s *) v"));
            self.push(format!("{{-# {prag} scale{n} #-}}"));
            self.push("");
            self.push(format!("relu{n} :: {t} -> {t}"));
            self.push(format!("relu{n} v = {mp} (max 0) v"));
            self.push(format!("{{-# {prag} relu{n} #-}}"));
            self.push("");
        }
    }

    fn emit_vec_nested(&mut self, n: usize, (k, m): (usize, usize), elem: &str) {
        let (t, c_ty) = (tname(elem, n), tname(elem, m));
        let f = |base: &str| fname(elem, base, n);
        let g = |base: &str| fname(elem, base, m);
        let (c, d) = (names("c", k), names("d", k));
        self.push(format!(
            "{{- A vector of {n} {elem} values stored as {k} strict chunks of {c_ty} (coordinate i is"
        ));
        self.push(format!(
            "   coordinate i mod {m} of chunk i div {m}).  Chunked because GHC generates much better"
        ));
        self.push(format!(
            "   code for {m}-wide straight-line functions than for {n}-wide ones. -}}"
        ));
        self.apply(
            &std::iter::once(format!("data {t} = {t}"))
                .chain(std::iter::repeat_n(format!("!{c_ty}"), k))
                .collect::<Vec<_>>(),
            0,
        );
        self.push("");
        let (fl, tl, ix) = (f("fromList"), f("toList"), f("index"));
        let fr = f("fromListRest");
        self.push(format!(
            "-- {fr} xs: the {t} of the first {n} elements of xs, and the rest of xs"
        ));
        self.push(format!("{fr} :: [{elem}] -> ({t}, [{elem}])"));
        self.push(format!("{fr} xs0 ="));
        self.push("  let");
        for i in 0..k {
            let src = if i == 0 {
                "xs0".to_string()
            } else {
                format!("r{}", i - 1)
            };
            self.push(format!(
                "    ({}, r{i}) = {} {src}",
                c[i],
                g("fromListRest")
            ));
        }
        self.push(format!("  in ({t} {}, r{})", c.join(" "), k - 1));
        self.push("");
        self.push(format!(
            "-- {fl} xs: the {t} whose coordinate i is xs !! i.  PRE: length xs >= {n}"
        ));
        self.push(format!("{fl} :: [{elem}] -> {t}"));
        self.push(format!("{fl} xs = fst ({fr} xs)"));
        self.push("");
        self.push(format!("-- {tl} v: the coordinates of v in order"));
        self.push(format!("{tl} :: {t} -> [{elem}]"));
        let parts: Vec<String> = c.iter().map(|ci| format!("{} {ci}", g("toList"))).collect();
        self.push(format!(
            "{tl} ({t} {}) = {}",
            c.join(" "),
            parts.join(" ++ ")
        ));
        self.push("");
        self.push(format!(
            "-- {ix} i v: coordinate i of v.  PRE: 0 <= i < {n}"
        ));
        self.push(format!("{ix} :: Int -> {t} -> {elem}"));
        self.push(format!("{ix} i ({t} {}) =", c.join(" ")));
        self.push(format!("  case i `quot` {m} of"));
        for i in 0..k {
            self.push(format!(
                "    {i} -> {} (i - {}) {}",
                g("index"),
                i * m,
                c[i]
            ));
        }
        self.push(format!("    _ -> error \"{ix}: index out of range\""));
        self.push("");
        let (rp, mp, cb) = (f("replicate"), f("map"), f("combine"));
        self.push(format!("{rp} :: {elem} -> {t}"));
        self.push(format!(
            "{rp} x = let !ch = {} x in {t} {}",
            g("replicate"),
            vec!["ch"; k].join(" ")
        ));
        self.push("");
        self.push(format!("{{-# INLINE {mp} #-}}"));
        self.push(format!("{mp} :: ({elem} -> {elem}) -> {t} -> {t}"));
        self.push(format!("{mp} f ({t} {}) =", c.join(" ")));
        let mut items = vec![t.clone()];
        items.extend(c.iter().map(|ci| format!("({} f {ci})", g("map"))));
        self.apply(&items, 2);
        self.push("");
        self.push(format!("{{-# INLINE {cb} #-}}"));
        self.push(format!(
            "{cb} :: ({elem} -> {elem} -> {elem}) -> {t} -> {t} -> {t}"
        ));
        self.push(format!("{cb} f !u !v ="));
        self.case_match(
            &["u", "v"],
            &[
                format!("{t} {}", c.join(" ")),
                format!("{t} {}", d.join(" ")),
            ],
            2,
        );
        let mut items = vec![t.clone()];
        items.extend(
            c.iter()
                .zip(&d)
                .map(|(ci, di)| format!("({} f {ci} {di})", g("combine"))),
        );
        self.apply(&items, 6);
        self.push("");
        // Arithmetic as one call per chunk (the chunk functions are NOINLINE).
        for base in ["add", "sub"] {
            let fun = f(base);
            self.push(format!("{fun} :: {t} -> {t} -> {t}"));
            self.push(format!("{fun} !u !v ="));
            self.case_match(
                &["u", "v"],
                &[
                    format!("{t} {}", c.join(" ")),
                    format!("{t} {}", d.join(" ")),
                ],
                2,
            );
            let mut items = vec![t.clone()];
            items.extend(
                c.iter()
                    .zip(&d)
                    .map(|(ci, di)| format!("({} {ci} {di})", g(base))),
            );
            self.apply(&items, 6);
            self.push("");
        }
        if elem == "Float" {
            self.push(format!("scale{n} :: Float -> {t} -> {t}"));
            self.push(format!("scale{n} s ({t} {}) =", c.join(" ")));
            let mut items = vec![t.clone()];
            items.extend(c.iter().map(|ci| format!("(scale{m} s {ci})")));
            self.apply(&items, 2);
            self.push("");
            self.push(format!("relu{n} :: {t} -> {t}"));
            self.push(format!("relu{n} ({t} {}) =", c.join(" ")));
            let mut items = vec![t.clone()];
            items.extend(c.iter().map(|ci| format!("(relu{m} {ci})")));
            self.apply(&items, 2);
            self.push("");
        }
        let (sm, dt) = (f("sum"), f("dot"));
        self.push(format!("{sm} :: {t} -> {elem}"));
        let sums: Vec<String> = c.iter().map(|ci| format!("{} {ci}", g("sum"))).collect();
        self.definition(
            vec![format!("{sm} ({t} {}) =", c.join(" "))],
            &sum_tree(&sums),
        );
        self.push("");
        self.push(format!(
            "-- {dt} u v: the dot product, as the sum of the chunk dot products"
        ));
        self.push(format!("{dt} :: {t} -> {t} -> {elem}"));
        self.push(format!("{dt} !u !v ="));
        self.case_match(
            &["u", "v"],
            &[
                format!("{t} {}", c.join(" ")),
                format!("{t} {}", d.join(" ")),
            ],
            2,
        );
        let dots: Vec<String> = c
            .iter()
            .zip(&d)
            .map(|(ci, di)| format!("{} {ci} {di}", g("dot")))
            .collect();
        self.out.extend(sum_tree(&dots).layout(6));
        self.push(format!("{{-# NOINLINE {dt} #-}}"));
        self.push("");
    }

    // ------------------------------------------------------------ matrices and layers

    /// Pattern binding the `n` coordinates of a Float `Vec{n}` to `vs` (nested aware).
    fn pat(&self, n: usize, vs: &[String]) -> Expr {
        match self.chunks(n) {
            None => Expr::app(format!("Vec{n}"), vs),
            Some((k, m)) => Expr::App(
                format!("Vec{n}"),
                (0..k)
                    .map(|i| self.pat(m, &vs[i * m..(i + 1) * m]))
                    .collect(),
            ),
        }
    }

    /// Construction of a Float `Vec{n}` from `n` coordinate expressions (nested aware).
    fn con(&self, n: usize, es: &[String]) -> Expr {
        match self.chunks(n) {
            None => Expr::app(format!("Vec{n}"), es),
            Some((k, m)) => Expr::App(
                format!("Vec{n}"),
                (0..k)
                    .map(|i| self.con(m, &es[i * m..(i + 1) * m]))
                    .collect(),
            ),
        }
    }

    fn emit_rows(&mut self, m: usize, n: usize) {
        let (r_ty, v_ty) = (format!("Rows{m}_{n}"), format!("Vec{n}"));
        let r = names("r", m);
        self.push(format!(
            "{{- {m} rows of width {n}: a matrix stored one row per output, row i = field i. -}}"
        ));
        self.apply(
            &std::iter::once(format!("data {r_ty} = {r_ty}"))
                .chain(std::iter::repeat_n(format!("!{v_ty}"), m))
                .collect::<Vec<_>>(),
            0,
        );
        self.push("");
        self.push(format!(
            "-- sel{m}_{n} i rows: row i (a jump table).  PRE: 0 <= i < {m}"
        ));
        self.push(format!("sel{m}_{n} :: Int -> {r_ty} -> {v_ty}"));
        self.apply(
            &self.pattern_head(&format!("sel{m}_{n} i ({r_ty}"), &r, ") ="),
            0,
        );
        self.push("  case i of");
        for (i, ri) in r.iter().enumerate() {
            self.push(format!("    {i} -> {ri}"));
        }
        self.push(format!("    _ -> error \"sel{m}_{n}: index out of range\""));
        self.push("");
        self.push(format!(
            "-- fromListRows{m}_{n} rs: the matrix whose row i is rs !! i.  PRE: length rs >= {m}"
        ));
        self.push(format!("fromListRows{m}_{n} :: [{v_ty}] -> {r_ty}"));
        self.apply(&cons_head(&format!("fromListRows{m}_{n}"), &r, "_"), 0);
        let mut items = vec![r_ty.clone()];
        items.extend(r.iter().cloned());
        self.apply(&items, 2);
        self.push(format!(
            "fromListRows{m}_{n} _ = error \"fromListRows{m}_{n}: list too short\""
        ));
        self.push("");
    }

    /// `linearM_N rows bias x = bias + rows . x`; `denseM_N` = relu of that;
    /// `residualM_M rows bias x = relu (x + bias + rows . x)`.
    fn emit_dense(&mut self, m: usize, n: usize) {
        let (r_ty, vm, vn) = (format!("Rows{m}_{n}"), format!("Vec{m}"), format!("Vec{n}"));
        let (r, bb, xx) = (names("r", m), names("b", m), names("x", m));
        let mut kinds = vec!["linear", "dense"];
        if m == n {
            kinds.push("residual");
        }
        for kind in kinds {
            let fun = format!("{kind}{m}_{n}");
            let what = match kind {
                "linear" => "bias + rows . x",
                "dense" => "relu (bias + rows . x)",
                _ => "relu (x + bias + rows . x)",
            };
            self.push(format!("{{- {fun} rows bias x"));
            self.push(format!(
                "   RETURNS: {what}, with rows . x the vector of dot products row_i . x"
            ));
            self.push(format!("   ({m} outputs of width-{n} rows) -}}"));
            self.push(format!("{fun} :: {r_ty} -> {vm} -> {vn} -> {vm}"));
            self.push(format!(
                "{fun} !rows !bias !{} =",
                if kind == "residual" { "xin" } else { "x" }
            ));
            let mut scruts = vec!["rows", "bias"];
            let mut pats = vec![Expr::app(r_ty.clone(), &r), self.pat(m, &bb)];
            if kind == "residual" {
                scruts.push("xin");
                pats.push(self.pat(m, &xx));
            }
            self.case_match_exprs(&scruts, &pats, 2);
            let bd = 6;
            let cells: Vec<String> = (0..m)
                .map(|i| match kind {
                    "linear" => format!("({} + dot{n} {} x)", bb[i], r[i]),
                    "dense" => format!("(max 0 ({} + dot{n} {} x))", bb[i], r[i]),
                    _ => format!("(max 0 ({} + {} + dot{n} {} xin))", xx[i], bb[i], r[i]),
                })
                .collect();
            let con = self.con(m, &cells);
            self.layout(&con, bd);
            self.push(format!("{{-# NOINLINE {fun} #-}}"));
            self.push("");
        }
    }

    /// `concatK_N v0 .. v(K-1)`: the width-K*N vector made of the K pieces in order.
    fn emit_concat(&mut self, k: usize, n: usize) {
        let (v, w) = (format!("Vec{n}"), format!("Vec{}", k * n));
        let fun = format!("concat{k}_{n}");
        self.push(format!(
            "-- {fun} v0 .. v{}: the {w} whose coordinate i*{n}+j is coordinate j of v_i",
            k - 1
        ));
        let mut sig = vec![v.clone(); k];
        sig.push(w.clone());
        self.push(format!("{fun} :: {}", sig.join(" -> ")));
        if self.chunks(k * n) == Some((k, n)) {
            self.push(format!("{fun} = {w}"));
            self.push("");
            return;
        }
        let args: Vec<String> = (0..k).map(|i| format!("!u{}", pad_index(i, k))).collect();
        self.push(format!("{fun} {} =", args.join(" ")));
        let vs: Vec<String> = (0..k)
            .flat_map(|i| (0..n).map(move |j| format!("v{}_{}", pad_index(i, k), pad_index(j, n))))
            .collect();
        let names: Vec<String> = (0..k).map(|i| format!("u{}", pad_index(i, k))).collect();
        let scruts: Vec<&str> = names.iter().map(String::as_str).collect();
        let pats: Vec<Expr> = (0..k)
            .map(|i| self.pat(n, &vs[i * n..(i + 1) * n]))
            .collect();
        self.case_match_exprs(&scruts, &pats, 2);
        let con = self.con(k * n, &vs);
        self.layout(&con, 6);
        self.push("");
    }

    /// `sumK_N v0 .. v(K-1)`: componentwise sum of K vectors (chunkwise if nested).
    fn emit_sum(&mut self, k: usize, n: usize) {
        let v = format!("Vec{n}");
        let fun = format!("sum{k}_{n}");
        self.push(format!(
            "-- {fun} v0 .. v{}: the componentwise sum of the {k} arguments",
            k - 1
        ));
        self.push(format!("{fun} :: {}", vec![v.clone(); k + 1].join(" -> ")));
        let args: Vec<String> = (0..k).map(|i| format!("!u{}", pad_index(i, k))).collect();
        self.push(format!("{fun} {} =", args.join(" ")));
        let names: Vec<String> = (0..k).map(|i| format!("u{}", pad_index(i, k))).collect();
        let scruts: Vec<&str> = names.iter().map(String::as_str).collect();
        match self.chunks(n) {
            Some((kk, m)) => {
                let pats: Vec<String> = (0..k)
                    .map(|i| {
                        let fields: Vec<String> = (0..kk)
                            .map(|j| format!("c{}_{}", pad_index(i, k), pad_index(j, kk)))
                            .collect();
                        format!("{v} {}", fields.join(" "))
                    })
                    .collect();
                self.case_match(&scruts, &pats, 2);
                let mut items = vec![v.clone()];
                for j in 0..kk {
                    let terms: Vec<String> = (0..k)
                        .map(|i| format!("c{}_{}", pad_index(i, k), pad_index(j, kk)))
                        .collect();
                    items.push(format!("(sum{k}_{m} {})", terms.join(" ")));
                }
                self.apply(&items, 6);
            }
            None => {
                let pats: Vec<String> = (0..k)
                    .map(|i| {
                        let fields: Vec<String> = (0..n)
                            .map(|j| format!("v{}_{}", pad_index(i, k), pad_index(j, n)))
                            .collect();
                        format!("{v} {}", fields.join(" "))
                    })
                    .collect();
                self.case_match(&scruts, &pats, 2);
                let mut items = vec![v.clone()];
                for j in 0..n {
                    let terms: Vec<String> = (0..k)
                        .map(|i| format!("v{}_{}", pad_index(i, k), pad_index(j, n)))
                        .collect();
                    items.push(format!("({})", terms.join(" + ")));
                }
                self.apply(&items, 6);
            }
        }
        self.push(format!("{{-# NOINLINE {fun} #-}}"));
        self.push("");
    }

    // ------------------------------------------------------------ sparse layers

    /// The input chunking of a layer with `n` inputs: `(g, Some(k))` for `k` chunks of width
    /// `g`, `(n, None)` for a flat input.
    fn input_groups(&self, n: usize) -> (usize, Option<usize>) {
        match self.chunks(n) {
            Some((k, g)) => (g, Some(k)),
            None => (n, None),
        }
    }

    /// `axpyK_N s0 c0 .. acc`: `acc + s0 * c0 + .. + s(K-1) * c(K-1)` componentwise. Flat it is
    /// one `NOINLINE` constructor of `N` sums; nested, an `INLINE` call per chunk.
    fn emit_axpy(&mut self, k: usize, n: usize) {
        let v = format!("Vec{n}");
        let fun = format!("axpy{k}_{n}");
        let (ss, cs) = (names("s", k), names("c", k));
        self.push(format!(
            "-- {fun} s0 c0 .. acc: acc + s0 * c0 + .. componentwise ({k} scaled columns)"
        ));
        let mut sig: Vec<String> = (0..k)
            .flat_map(|_| ["Float".to_string(), v.clone()])
            .collect();
        sig.extend([v.clone(), v.clone()]);
        self.push(format!("{fun} :: {}", sig.join(" -> ")));
        let args: Vec<String> = ss
            .iter()
            .zip(&cs)
            .map(|(s, c)| format!("!{s} !{c}"))
            .collect();
        self.push(format!("{fun} {} !acc =", args.join(" ")));
        let mut scruts: Vec<&str> = cs.iter().map(String::as_str).collect();
        scruts.push("acc");
        let (width, chunk) = match self.chunks(n) {
            Some((kk, m)) => (kk, Some(m)),
            None => (n, None),
        };
        // Fields: chunks of a nested vector, lanes of a flat one.
        let fields: Vec<Vec<String>> = (0..k)
            .map(|i| {
                (0..width)
                    .map(|j| format!("x{}_{}", pad_index(i, k), pad_index(j, width)))
                    .collect()
            })
            .collect();
        let acc = names("a", width);
        let mut pats: Vec<String> = fields
            .iter()
            .map(|f| format!("{v} {}", f.join(" ")))
            .collect();
        pats.push(format!("{v} {}", acc.join(" ")));
        self.case_match(&scruts, &pats, 2);
        let mut items = vec![v.clone()];
        items.extend((0..width).map(|j| match chunk {
            Some(m) => {
                let pairs: Vec<String> = (0..k)
                    .map(|i| format!("{} {}", ss[i], fields[i][j]))
                    .collect();
                format!("(axpy{k}_{m} {} {})", pairs.join(" "), acc[j])
            }
            None => {
                let terms: Vec<String> = std::iter::once(acc[j].clone())
                    .chain((0..k).map(|i| format!("{} * {}", ss[i], fields[i][j])))
                    .collect();
                format!("({})", terms.join(" + "))
            }
        }));
        self.apply(&items, 6);
        let prag = if chunk.is_some() {
            "INLINE"
        } else {
            "NOINLINE"
        };
        self.push(format!("{{-# {prag} {fun} #-}}"));
        self.push("");
    }

    /// `ColsG_M`: `g` columns of width `m`, the group a layer's columns come in per input chunk.
    fn emit_column_group(&mut self, g: usize, m: usize) {
        let ty = format!("Cols{g}_{m}");
        self.push(format!(
            "{{- {g} columns of width {m}: a layer's columns (one per input) for {g} inputs. -}}"
        ));
        self.apply(
            &std::iter::once(format!("data {ty} = {ty}"))
                .chain(std::iter::repeat_n(format!("!Vec{m}"), g))
                .collect::<Vec<_>>(),
            0,
        );
        self.push("");
    }

    /// `ColsN_M` (for a nested input) and `columnsM_N`, the transposition from the rows.
    fn emit_cols(&mut self, m: usize, n: usize) {
        let ty = format!("Cols{n}_{m}");
        let (g, groups) = self.input_groups(n);
        if let Some(k) = groups {
            self.push(format!(
                "{{- The columns of a layer with {m} outputs and {n} inputs, {k} groups of {g}. -}}"
            ));
            self.apply(
                &std::iter::once(format!("data {ty} = {ty}"))
                    .chain(std::iter::repeat_n(format!("!Cols{g}_{m}"), k))
                    .collect::<Vec<_>>(),
                0,
            );
            self.push("");
        }
        let fun = format!("columns{m}_{n}");
        self.push(format!(
            "-- {fun} rows: the layer's columns (one per input) from its rows (one per output);"
        ));
        self.push("-- a transposition, meant to run once, at load");
        self.push(format!("{fun} :: Rows{m}_{n} -> {ty}"));
        self.push(format!("{fun} rows ="));
        self.push(format!(
            "  let rs = [sel{m}_{n} i rows | i <- [0 .. {}]]",
            m - 1
        ));
        self.push(format!(
            "      col j = fromList{m} [index{n} j r | r <- rs]"
        ));
        let mut items = vec![ty.clone()];
        match groups {
            Some(k) => {
                let mut group = vec![format!("group g = Cols{g}_{m}")];
                group.extend((0..g).map(|j| format!("(col ({g} * g + {j}))")));
                self.apply(&group, 6);
                items.extend((0..k).map(|i| format!("(group {i})")));
            }
            None => items.extend((0..n).map(|j| format!("(col {j})"))),
        }
        let mut body = vec!["in".to_string()];
        body.extend(items);
        self.line(&body, 3);
        self.push("");
    }

    /// `gatherG_M x cols rest`: the `(x_j, column j)` pairs of the nonzero `x_j`, in order,
    /// consed onto `rest`, all strict (`rest` too, so a chain of gathers allocates no thunks):
    /// only the list cells and pairs allocate.
    fn emit_gather(&mut self, g: usize, m: usize) {
        let fun = format!("gather{g}_{m}");
        let (a, k, r) = (names("a", g), names("k", g), names("r", g));
        self.push(format!(
            "-- {fun} x cols rest: the (x_j, column j) pairs of the nonzero x_j, consed onto rest"
        ));
        self.push(format!(
            "{fun} :: Vec{g} -> Cols{g}_{m} -> [(Float, Vec{m})] -> [(Float, Vec{m})]"
        ));
        self.push(format!("{fun} !x !cols !rest ="));
        self.case_match(
            &["x", "cols"],
            &[
                format!("Vec{g} {}", a.join(" ")),
                format!("Cols{g}_{m} {}", k.join(" ")),
            ],
            2,
        );
        for j in (0..g).rev() {
            let next = if j + 1 == g { "rest" } else { &r[j + 1] };
            let lead = if j + 1 == g { "let" } else { "   " };
            self.push(format!(
                "      {lead} !{} = if {} == 0 then {next} else ({}, {}) : {next}",
                r[j], a[j], a[j], k[j]
            ));
        }
        self.push(format!("       in {}", r[0]));
        self.push(format!("{{-# NOINLINE {fun} #-}}"));
        self.push("");
    }

    /// `addColumnsK_M acc pairs`: `acc` plus `s * column` over the pairs, `k` per `axpyK_M`
    /// call, then one at a time.
    fn emit_add_columns(&mut self, k: usize, m: usize) {
        let fun = format!("addColumns{k}_{m}");
        self.push(format!(
            "-- {fun} acc pairs: acc plus s * column over the (s, column) pairs, {k} per pass"
        ));
        self.push(format!("{fun} :: Vec{m} -> [(Float, Vec{m})] -> Vec{m}"));
        if k > 1 {
            let (ss, cs) = (names("s", k), names("c", k));
            let pairs: Vec<String> = ss
                .iter()
                .zip(&cs)
                .map(|(s, c)| format!("({s}, {c})"))
                .collect();
            let args: Vec<String> = ss
                .iter()
                .zip(&cs)
                .map(|(s, c)| format!("{s} {c}"))
                .collect();
            self.push(format!(
                "{fun} !acc ({} : rest) = {fun} (axpy{k}_{m} {} acc) rest",
                pairs.join(" : "),
                args.join(" ")
            ));
        }
        self.push(format!(
            "{fun} !acc ((s, c) : rest) = {fun} (axpy1_{m} s c acc) rest"
        ));
        self.push(format!("{fun} !acc [] = acc"));
        self.push("");
    }

    /// `sparseDenseM_N` and, for `m == n`, `sparseResidualM`: the dense layers computed by
    /// columns over the nonzero inputs only.
    fn emit_sparse(&mut self, m: usize, n: usize, k: usize) {
        let (g, groups) = self.input_groups(n);
        // The gathered pairs of every input chunk, one gather per line.
        let gathers: Vec<String> = match groups {
            Some(c) => (0..c)
                .map(|i| {
                    let (xi, gi) = (
                        format!("x{}", pad_index(i, c)),
                        format!("g{}", pad_index(i, c)),
                    );
                    if i + 1 == c {
                        format!("gather{g}_{m} {xi} {gi} []")
                    } else {
                        format!("gather{g}_{m} {xi} {gi} $")
                    }
                })
                .collect(),
            None => vec![format!("gather{n}_{m} x cols []")],
        };
        let mut kinds = vec![(
            format!("sparseDense{m}_{n}"),
            "relu (bias + sum over the nonzero x_j of x_j * column j), as dense",
            "bias".to_string(),
        )];
        if m == n {
            kinds.push((
                format!("sparseResidual{m}"),
                "relu (x + bias + sum over the nonzero x_j of x_j * column j), as residual",
                format!("(add{m} x bias)"),
            ));
        }
        for (fun, what, start) in kinds {
            self.push(format!("{{- {fun} cols bias x"));
            self.push(format!(
                "   RETURNS: {what}{m}_{n} up to rounding, adding {k} columns per pass -}}"
            ));
            self.push(format!(
                "{fun} :: Cols{n}_{m} -> Vec{m} -> Vec{n} -> Vec{m}"
            ));
            self.push(format!("{fun} !cols !bias !x ="));
            let d = match groups {
                Some(c) => {
                    self.case_match(
                        &["cols", "x"],
                        &[
                            format!("Cols{n}_{m} {}", names("g", c).join(" ")),
                            format!("Vec{n} {}", names("x", c).join(" ")),
                        ],
                        2,
                    );
                    6
                }
                None => 2,
            };
            let pad = " ".repeat(d);
            self.push(format!("{pad}let pairs ="));
            for gather in &gathers {
                self.push(format!("{pad}      {gather}"));
            }
            self.push(format!(
                "{pad} in relu{m} (addColumns{k}_{m} {start} pairs)"
            ));
            self.push("");
        }
    }

    // ------------------------------------------------------------ driver

    fn generate(mut self, bundle: bool) -> String {
        let spec = self.spec;
        let mut floats: BTreeSet<usize> = spec.floats.clone();
        let dense = spec.dense.clone();
        let mut rows: BTreeSet<(usize, usize)> = spec.rows.iter().copied().collect();
        rows.extend(dense.iter().copied());
        let concats = spec.concats.clone();
        let mut sums = spec.sums.clone();
        // Sparse layers imply their columns, axpys (K and 1 per pass) and rows.
        let mut sparse: BTreeMap<(usize, usize), usize> = BTreeMap::new();
        for &(m, n, k) in &spec.sparse {
            sparse.entry((m, n)).or_insert(k);
        }
        let mut cols: BTreeSet<(usize, usize)> = spec.cols.iter().copied().collect();
        let mut axpys: BTreeSet<(usize, usize)> = spec.axpys.iter().copied().collect();
        for (&(m, n), &k) in &sparse {
            cols.insert((m, n));
            axpys.extend([(k, m), (1, m)]);
        }
        rows.extend(cols.iter().copied());
        floats.extend(cols.iter().flat_map(|&(m, n)| [m, n]));
        floats.extend(axpys.iter().map(|&(_, n)| n));
        floats.extend(rows.iter().map(|&(_, n)| n));
        floats.extend(dense.iter().map(|&(m, _)| m));
        floats.extend(concats.iter().map(|&(_, n)| n));
        floats.extend(concats.iter().map(|&(k, n)| k * n));
        floats.extend(sums.iter().map(|&(_, n)| n));
        floats.extend(spec.aliases.iter().map(|&(_, w)| w));
        // Nested widths need their chunk width; nested sums need the chunk-width sum.
        let mut changed = true;
        while changed {
            changed = false;
            for n in floats.clone() {
                if let Some((_, m)) = self.chunks(n) {
                    if floats.insert(m) {
                        changed = true;
                    }
                }
            }
            for (k, n) in sums.clone() {
                if let Some((_, m)) = self.chunks(n) {
                    if !sums.contains(&(k, m)) {
                        sums.push((k, m));
                        changed = true;
                    }
                }
            }
            for (k, n) in axpys.clone() {
                if let Some((_, m)) = self.chunks(n) {
                    changed |= axpys.insert((k, m));
                }
            }
        }
        // Chunk sums first, deduplicated.
        let mut sums: Vec<(usize, usize)> = {
            let set: BTreeSet<(usize, usize)> = sums.into_iter().collect();
            set.into_iter().collect()
        };
        sums.sort_by_key(|&(k, n)| (n, k));

        self.push("{-# LANGUAGE Safe #-}");
        self.push(format!("module {} where", spec.module));
        self.push("");
        if bundle {
            self.push(IMPORTS_BEGIN);
        }
        if floats.iter().any(|&n| n <= 64 && self.chunks(n).is_none()) {
            self.push("import Data.Bits (testBit)");
        }
        if !spec.int32s.is_empty() || !spec.int16s.is_empty() {
            let mut types = Vec::new();
            if !spec.int16s.is_empty() {
                types.push("Int16");
            }
            if !spec.int32s.is_empty() {
                types.push("Int32");
            }
            self.push(format!("import Data.Int ({})", types.join(", ")));
        }
        if bundle {
            self.push(IMPORTS_END);
        }
        self.push("");
        if bundle {
            self.push("-- Top-level names: the vector families, every name ending in its");
            self.push(
                "-- widths (Vec64, Rows9_128, fromListRest512, dot64, linear9_128, concat8_64, ...).",
            );
            self.push(BODY_BEGIN);
        }
        self.push(concat!(
            "-- BEGIN GENERATED VEC  (",
            env!("CARGO_PKG_NAME"),
            " export; do not edit by hand)"
        ));
        self.push("{- FOURMOLU_DISABLE -}");
        self.push("");
        if !spec.aliases.is_empty() {
            for (name, w) in &spec.aliases {
                self.push(format!("type {name} = Vec{w}"));
            }
            self.push("");
        }
        for &n in &floats {
            self.emit_vec(n, "Float");
        }
        for &n in &spec.int32s {
            self.emit_vec(n, "Int32");
        }
        for &n in &spec.int16s {
            self.emit_vec(n, "Int16");
        }
        for &(m, n) in &rows {
            self.emit_rows(m, n);
        }
        for &(m, n) in &dense {
            self.emit_dense(m, n);
        }
        for &(k, n) in &concats {
            self.emit_concat(k, n);
        }
        for &(k, n) in &sums {
            self.emit_sum(k, n);
        }
        for &(k, n) in &axpys {
            self.emit_axpy(k, n);
        }
        let groups: BTreeSet<(usize, usize)> = cols
            .iter()
            .map(|&(m, n)| (self.input_groups(n).0, m))
            .collect();
        for &(g, m) in &groups {
            self.emit_column_group(g, m);
        }
        for &(m, n) in &cols {
            self.emit_cols(m, n);
        }
        if !sparse.is_empty() {
            for &(g, m) in &groups {
                self.emit_gather(g, m);
            }
            let adds: BTreeSet<(usize, usize)> =
                sparse.iter().map(|(&(m, _), &k)| (k, m)).collect();
            for &(k, m) in &adds {
                self.emit_add_columns(k, m);
            }
            for (&(m, n), &k) in &sparse {
                self.emit_sparse(m, n, k);
            }
        }
        self.push("{- FOURMOLU_ENABLE -}");
        self.push("-- END GENERATED VEC");
        if bundle {
            self.push(BODY_END);
        }
        let mut text = self.out.join("\n");
        text.push('\n');
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_prints_the_command_line() {
        let spec = VecSpec::parse_args(&[
            "--float",
            "3",
            "9",
            "64",
            "128",
            "512",
            "--nest",
            "512=8x64,128=2x64",
            "--rows",
            "10:512",
            "--dense",
            "64:64,128:512",
            "--concat",
            "8:64",
            "--module",
            "FixedVec",
        ])
        .unwrap();
        assert_eq!(spec.nest[&512], (8, 64));
        assert_eq!(
            spec.flags(),
            "--float 3 9 64 128 512 --nest 128=2x64,512=8x64 --rows 10:512 --dense 64:64,128:512 --concat 8:64"
        );
        assert!(VecSpec::parse_args(&["--nest", "512=7x64"]).is_err());
        assert!(VecSpec::parse_args(&["--bogus"]).is_err());
    }

    #[test]
    fn generates_the_expected_declarations() {
        let spec = VecSpec::parse_args(&[
            "--float", "3", "16", "--nest", "16=2x8", "--rows", "4:16", "--dense", "3:16,8:8",
            "--concat", "2:8", "--sum", "4:16", "--int32", "8", "--alias", "VecA=16",
        ])
        .unwrap();
        let text = spec.generate();
        for needle in [
            "module GenVec where",
            "import Data.Bits (testBit)",
            "import Data.Int (Int32)",
            "type VecA = Vec16",
            "data Vec16 = Vec16 !Vec8 !Vec8",
            "fromListRest8 :: [Float] -> (Vec8, [Float])",
            "fromListRest16 xs0 =",
            "dot16 !u !v =",
            "{-# NOINLINE dot8 #-}",
            "{-# INLINE add8 #-}",
            "data I32Vec8",
            "i32Dot8 :: I32Vec8 -> I32Vec8 -> Int32",
            "sel4_16 :: Int -> Rows4_16 -> Vec16",
            "dense3_16 :: Rows3_16 -> Vec3 -> Vec16 -> Vec3",
            "residual8_8 :: Rows8_8 -> Vec8 -> Vec8 -> Vec8",
            "concat2_8 = Vec16",
            "sum4_8 :: Vec8 -> Vec8 -> Vec8 -> Vec8 -> Vec8",
            "sum4_16 !u0 !u1 !u2 !u3 =",
            "maskedExp9",
        ] {
            assert!(
                text.contains(needle) || needle == "maskedExp9",
                "missing {needle}"
            );
        }
        assert!(
            !text.contains("maskedMax16"),
            "nested widths have no masked helpers"
        );
        assert!(text.contains("maskedMax8"));
    }
}
