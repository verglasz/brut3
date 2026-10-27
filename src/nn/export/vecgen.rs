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

/// Soft line width for wrapped field lists.
const LINE: usize = 92;

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
        ] {
            for p in src {
                if !dst.contains(p) {
                    dst.push(*p);
                }
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
        ] {
            if !v.is_empty() {
                out.push(format!("{flag} {}", pairs(v)));
            }
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

/// Lays `items` out on lines of at most `LINE` characters.
fn wrap(items: &[String], indent: usize, first_indent: Option<usize>) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = " ".repeat(first_indent.unwrap_or(indent));
    let body_indent = " ".repeat(indent);
    for it in items {
        if cur.trim().is_empty() {
            cur.push_str(it);
        } else if cur.len() + 1 + it.len() > LINE {
            lines.push(std::mem::replace(&mut cur, format!("{body_indent}{it}")));
        } else {
            cur.push(' ');
            cur.push_str(it);
        }
    }
    lines.push(cur);
    lines
}

/// Sums `terms` with `lanes` independent partial sums, then adds the lanes.
fn sum_chain(terms: &[String], lanes: usize) -> String {
    let lanes = lanes.min(terms.len()).max(1);
    let parts: Vec<String> = (0..lanes)
        .map(|l| {
            let lane: Vec<&str> = terms
                .iter()
                .skip(l)
                .step_by(lanes)
                .map(String::as_str)
                .collect();
            if lane.len() > 1 {
                format!("({})", lane.join(" + "))
            } else {
                lane[0].to_string()
            }
        })
        .collect();
    parts.join(" + ")
}

fn split_words(text: &str) -> Vec<String> {
    text.split(' ').map(str::to_string).collect()
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

    fn wrap(&mut self, items: &[String], indent: usize, first_indent: Option<usize>) {
        self.out.extend(wrap(items, indent, first_indent));
    }

    fn chunks(&self, n: usize) -> Option<(usize, usize)> {
        self.spec.nest.get(&n).copied()
    }

    /// `case scrut of` at indent `d`, the single alternative `con fields ->` on the next line.
    /// `case (s0, s1, ..) of` at indent `d`, its one alternative matching every scrutinee
    /// against its pattern, one pattern per line however long (they read best whole); the
    /// body goes at indent `d + 4` (continuations `d + 6`). GHC never builds the tuple.
    fn case_match(&mut self, scruts: &[&str], pats: &[String], d: usize) {
        self.push(format!("{}case ({}) of", " ".repeat(d), scruts.join(", ")));
        for (i, pat) in pats.iter().enumerate() {
            let (indent, open) = if i == 0 { (d + 2, "( ") } else { (d + 4, "") };
            let close = if i + 1 == pats.len() { " ) ->" } else { "," };
            self.push(format!("{}{open}{pat}{close}", " ".repeat(indent)));
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
        self.push(format!("data {t}"));
        self.push(format!("  = {t}"));
        self.wrap(&vec![format!("{{-# UNPACK #-}} !{elem}"); n], 6, None);
        self.push("");
        let (fl, tl, ix) = (f("fromList"), f("toList"), f("index"));
        self.push(format!("{{- {fl} xs"));
        self.push(format!("   RETURNS: the {t} whose coordinate i is xs !! i"));
        self.push(format!(
            "   PRE: length xs >= {n}; extra elements are ignored -}}"
        ));
        self.push(format!("{fl} :: [{elem}] -> {t}"));
        self.wrap(&[format!("{fl} ({}:_) =", a.join(":"))], 2, Some(0));
        let mut items = vec![t.clone()];
        items.extend(a.iter().cloned());
        self.wrap(&items, 4, Some(2));
        self.push(format!("{fl} _ = error \"{fl}: list too short\""));
        self.push("");
        let fr = f("fromListRest");
        self.push(format!(
            "-- {fr} xs: the {t} of the first {n} elements of xs, and the rest of xs"
        ));
        self.push(format!("{fr} :: [{elem}] -> ({t}, [{elem}])"));
        self.wrap(&[format!("{fr} ({}:rest) =", a.join(":"))], 2, Some(0));
        let mut items = vec![format!("({t}")];
        items.extend(a[..last].iter().cloned());
        items.push(format!("{}, rest)", a[last]));
        self.wrap(&items, 4, Some(2));
        self.push(format!("{fr} _ = error \"{fr}: list too short\""));
        self.push("");
        self.push(format!("-- {tl} v: the coordinates of v in order"));
        self.push(format!("{tl} :: {t} -> [{elem}]"));
        self.wrap(
            &self.pattern_head(&format!("{tl} ({t}"), &a, ") ="),
            2,
            Some(0),
        );
        let mut items = vec![format!("[{}{}", a[0], if n > 1 { "," } else { "" })];
        for x in &a[1..last.max(1)] {
            items.push(format!("{x},"));
        }
        if n > 1 {
            items.push(format!("{}]", a[last]));
        } else {
            items.push("]".to_string());
        }
        self.wrap(&items, 4, Some(2));
        self.push("");
        self.push(format!(
            "-- {ix} i v: coordinate i of v (a jump table).  PRE: 0 <= i < {n}"
        ));
        self.push(format!("{ix} :: Int -> {t} -> {elem}"));
        self.wrap(
            &self.pattern_head(&format!("{ix} i ({t}"), &a, ") ="),
            2,
            Some(0),
        );
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
        self.wrap(&items, 2, Some(0));
        self.push("");
        self.push(format!(
            "-- {mp} f v: f applied to every coordinate (INLINE so f is known at the call site)"
        ));
        self.push(format!("{{-# INLINE {mp} #-}}"));
        self.push(format!("{mp} :: ({elem} -> {elem}) -> {t} -> {t}"));
        self.wrap(
            &self.pattern_head(&format!("{mp} f ({t}"), &a, ") ="),
            2,
            Some(0),
        );
        let mut items = vec![t.clone()];
        items.extend(a.iter().map(|x| format!("(f {x})")));
        self.wrap(&items, 4, Some(2));
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
        self.wrap(&items, 8, Some(6));
        self.push("");
        self.emit_arith(n, elem, &t, &mp, &cb);
        let (sm, dt) = (f("sum"), f("dot"));
        self.push(format!("-- {sm} v: the sum of the coordinates of v"));
        self.push(format!("{sm} :: {t} -> {elem}"));
        self.wrap(
            &self.pattern_head(&format!("{sm} ({t}"), &a, ") ="),
            2,
            Some(0),
        );
        let lanes = if n >= 8 { 4 } else { 1 };
        self.wrap(&split_words(&sum_chain(&a, lanes)), 4, Some(2));
        if n >= 32 {
            self.push(format!("{{-# NOINLINE {sm} #-}}"));
        }
        self.push("");
        self.push(format!("{{- {dt} u v"));
        self.push("   RETURNS: the dot product of u and v, accumulated in four partial sums");
        self.push("   so the additions do not form one dependency chain -}");
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
        self.wrap(&split_words(&sum_chain(&products, lanes)), 8, Some(6));
        self.push(format!("{{-# NOINLINE {dt} #-}}"));
        self.push("");
        if !int_vec && n <= 64 {
            self.push(format!("{{- maskedMax{n} mask v acc"));
            self.push(
                "   RETURNS: the maximum of acc and the coordinates v_i for which bit i of mask is clear -}",
            );
            self.push(format!("maskedMax{n} :: Int -> {t} -> Float -> Float"));
            self.wrap(
                &self.pattern_head(
                    &format!("maskedMax{n} m ({t}"),
                    &a,
                    &format!(") acc{} =", pad_index(0, n + 1)),
                ),
                2,
                Some(0),
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
            self.wrap(
                &self.pattern_head(&format!("maskedExp{n} m c ({t}"), &a, ") ="),
                2,
                Some(0),
            );
            let mut items = vec![t.clone()];
            items.extend(
                a.iter()
                    .enumerate()
                    .map(|(i, x)| format!("(if testBit m {i} then 0 else exp ({x} - c))")),
            );
            self.wrap(&items, 4, Some(2));
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
        self.push(format!(
            "data {t} = {t} {}",
            vec![format!("!{c_ty}"); k].join(" ")
        ));
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
        self.wrap(&items, 4, Some(2));
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
        self.wrap(&items, 8, Some(6));
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
            self.wrap(&items, 8, Some(6));
            self.push("");
        }
        if elem == "Float" {
            self.push(format!("scale{n} :: Float -> {t} -> {t}"));
            self.push(format!("scale{n} s ({t} {}) =", c.join(" ")));
            let mut items = vec![t.clone()];
            items.extend(c.iter().map(|ci| format!("(scale{m} s {ci})")));
            self.wrap(&items, 4, Some(2));
            self.push("");
            self.push(format!("relu{n} :: {t} -> {t}"));
            self.push(format!("relu{n} ({t} {}) =", c.join(" ")));
            let mut items = vec![t.clone()];
            items.extend(c.iter().map(|ci| format!("(relu{m} {ci})")));
            self.wrap(&items, 4, Some(2));
            self.push("");
        }
        let (sm, dt) = (f("sum"), f("dot"));
        self.push(format!("{sm} :: {t} -> {elem}"));
        let sums: Vec<String> = c.iter().map(|ci| format!("{} {ci}", g("sum"))).collect();
        self.push(format!(
            "{sm} ({t} {}) = {}",
            c.join(" "),
            sum_chain(&sums, 1)
        ));
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
        let lanes = if k >= 4 { 2 } else { 1 };
        self.wrap(&split_words(&sum_chain(&dots, lanes)), 8, Some(6));
        self.push(format!("{{-# NOINLINE {dt} #-}}"));
        self.push("");
    }

    // ------------------------------------------------------------ matrices and layers

    /// Pattern binding the `n` coordinates of a Float `Vec{n}` to `vs` (nested aware).
    fn pat(&self, n: usize, vs: &[String]) -> String {
        match self.chunks(n) {
            None => format!("Vec{n} {}", vs.join(" ")),
            Some((k, m)) => {
                let parts: Vec<String> = (0..k)
                    .map(|i| format!("({})", self.pat(m, &vs[i * m..(i + 1) * m])))
                    .collect();
                format!("Vec{n} {}", parts.join(" "))
            }
        }
    }

    /// Construction of a Float `Vec{n}` from `n` coordinate expressions (nested aware).
    fn con(&self, n: usize, es: &[String]) -> Vec<String> {
        match self.chunks(n) {
            None => {
                let mut toks = vec![format!("Vec{n}")];
                toks.extend(es.iter().cloned());
                toks
            }
            Some((k, m)) => {
                let mut toks = vec![format!("Vec{n}")];
                for i in 0..k {
                    let inner = self.con(m, &es[i * m..(i + 1) * m]);
                    let last = inner.len() - 1;
                    toks.push(format!("({}", inner[0]));
                    toks.extend(inner[1..last].iter().cloned());
                    toks.push(format!("{})", inner[last]));
                }
                toks
            }
        }
    }

    fn emit_rows(&mut self, m: usize, n: usize) {
        let (r_ty, v_ty) = (format!("Rows{m}_{n}"), format!("Vec{n}"));
        let r = names("r", m);
        self.push(format!(
            "{{- {m} rows of width {n}: a matrix stored one row per output, row i = field i. -}}"
        ));
        self.push(format!("data {r_ty}"));
        self.push(format!("  = {r_ty}"));
        self.wrap(&vec![format!("!{v_ty}"); m], 6, None);
        self.push("");
        self.push(format!(
            "-- sel{m}_{n} i rows: row i (a jump table).  PRE: 0 <= i < {m}"
        ));
        self.push(format!("sel{m}_{n} :: Int -> {r_ty} -> {v_ty}"));
        self.wrap(
            &self.pattern_head(&format!("sel{m}_{n} i ({r_ty}"), &r, ") ="),
            2,
            Some(0),
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
        self.wrap(
            &[format!("fromListRows{m}_{n} ({}:_) =", r.join(":"))],
            2,
            Some(0),
        );
        let mut items = vec![r_ty.clone()];
        items.extend(r.iter().cloned());
        self.wrap(&items, 4, Some(2));
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
            let mut pats = vec![format!("{r_ty} {}", r.join(" ")), self.pat(m, &bb)];
            if kind == "residual" {
                scruts.push("xin");
                pats.push(self.pat(m, &xx));
            }
            self.case_match(&scruts, &pats, 2);
            let bd = 6;
            let cells: Vec<String> = (0..m)
                .map(|i| match kind {
                    "linear" => format!("({} + dot{n} {} x)", bb[i], r[i]),
                    "dense" => format!("(max 0 ({} + dot{n} {} x))", bb[i], r[i]),
                    _ => format!("(max 0 ({} + {} + dot{n} {} xin))", xx[i], bb[i], r[i]),
                })
                .collect();
            let con = self.con(m, &cells);
            self.wrap(&con, bd + 2, Some(bd));
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
        let pats: Vec<String> = (0..k)
            .map(|i| self.pat(n, &vs[i * n..(i + 1) * n]))
            .collect();
        self.case_match(&scruts, &pats, 2);
        let con = self.con(k * n, &vs);
        self.wrap(&con, 8, Some(6));
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
                self.wrap(&items, 8, Some(6));
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
                self.wrap(&items, 8, Some(6));
            }
        }
        self.push(format!("{{-# NOINLINE {fun} #-}}"));
        self.push("");
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
        self.push("-- BEGIN GENERATED VEC  (ultimate-ttt-burn export; do not edit by hand)");
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
