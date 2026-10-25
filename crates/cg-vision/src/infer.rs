//! Legal-move inference from square diffs.
//!
//! Given the per-square change scores from [`crate::features::square_diff`],
//! every legal move is scored by how well the squares it touches explain the
//! observed change. A move is accepted only when it clearly beats every move
//! that touches a different set of squares, and when the observed change is
//! not explained by squares the move leaves alone.

use shakmaty::{CastlingMode, Chess, File, Move, Position, Role, Square};

/// Thresholds for [`infer_move`].
#[derive(Clone, Copy, Debug)]
pub struct InferConfig {
    /// Floor for the per-square change threshold.
    pub min_change: f32,
    /// Minimum score gap between the best and second-best changed-square set.
    pub min_margin: f32,
    /// Maximum positive excess on squares outside the best move's changed set.
    pub max_residual: f32,
}

impl Default for InferConfig {
    /// Measured on the synthetic renders (noise 3 / 3.5, brightness +8, 32 px squares):
    /// - `min_change` 0.2: unchanged squares have diff at most 0.025 over 40 seeds,
    ///   and a moved piece has diff at least 1.14. The threshold sits well inside
    ///   that gap; the 2.5 × median term only raises it on busy diffs.
    /// - `min_margin` 0.25: a true move beat the runner-up by at least 0.35 in
    ///   every correct accept of the end-to-end run, while a genuine tie (one
    ///   changed square, e.g. e2e3 vs e2e4) gives about 0. 0.25 keeps a 40%
    ///   safety factor over the smallest true margin.
    /// - `max_residual` 0.1: an unchanged square's excess is negative, so a
    ///   correct move has residual 0 in the tests. Above 0.1 means real,
    ///   unexplained change (a hand, a second move, a bump).
    fn default() -> Self {
        InferConfig { min_change: 0.2, min_margin: 0.25, max_residual: 0.1 }
    }
}

/// Result of [`infer_move`].
#[derive(Clone, Debug)]
pub struct Inference {
    /// Best legal move (if any square changed meaningfully).
    pub best: Option<Move>,
    pub best_uci: Option<String>,
    pub score: f32,
    /// best score minus the best score among moves with a DIFFERENT changed-square set.
    pub margin: f32,
    /// Sum of positive excess on squares NOT in best's changed set.
    pub residual: f32,
    /// Number of squares whose diff exceeds the threshold.
    pub changed: u32,
    /// Top 5 (uci, score), best first, one entry per distinct changed-square set.
    pub ranked: Vec<(String, f32)>,
    /// best.is_some() && margin >= min_margin && residual <= max_residual
    pub accept: bool,
    /// When !accept: "no change" | "ambiguous" | "unexplained change" | "no legal moves"
    pub reason: Option<&'static str>,
}

/// Squares whose appearance changes when `mv` is played in `pos`:
/// from, to; castling: king from/to AND rook from/to; en passant: also the
/// captured pawn's square. Sorted, deduplicated.
pub fn changed_squares(_pos: &Chess, mv: &Move) -> Vec<u8> {
    let mut v: Vec<u8> = match *mv {
        Move::Normal { from, to, .. } => vec![u8::from(from), u8::from(to)],
        Move::EnPassant { from, to } => {
            // The captured pawn stands on the mover's rank, on the destination's file.
            let captured = Square::from_coords(to.file(), from.rank());
            vec![u8::from(from), u8::from(to), u8::from(captured)]
        }
        Move::Castle { king, rook } => {
            // shakmaty encodes castling as king → rook; the king and rook land on fixed files.
            let kingside = u8::from(rook.file()) > u8::from(king.file());
            let (king_file, rook_file) = if kingside { (File::G, File::F) } else { (File::C, File::D) };
            vec![
                u8::from(king),
                u8::from(rook),
                u8::from(Square::from_coords(king_file, king.rank())),
                u8::from(Square::from_coords(rook_file, rook.rank())),
            ]
        }
        Move::Put { to, .. } => vec![u8::from(to)],
    };
    v.sort_unstable();
    v.dedup();
    v
}

fn uci(mv: Move) -> String {
    mv.to_uci(CastlingMode::Standard).to_string()
}

/// One distinct changed-square set and the best move that produces it.
struct Group {
    squares: Vec<u8>,
    score: f32,
    mv: Move,
    /// Number of legal moves with this changed-square set (> 1 only for promotions).
    members: u32,
}

/// Infer which legal move of `pos` explains the per-square change `diff`.
pub fn infer_move(pos: &Chess, diff: &[f32; 64], cfg: &InferConfig) -> Inference {
    let (excess, changed, _) = excess_over_threshold(diff, cfg);
    let mut out = Inference {
        best: None,
        best_uci: None,
        score: 0.0,
        margin: 0.0,
        residual: 0.0,
        changed,
        ranked: Vec::new(),
        accept: false,
        reason: None,
    };
    if changed == 0 {
        out.reason = Some("no change");
        return out;
    }

    // Group moves by their changed-square set; within a group, prefer queen promotion.
    let mut groups: Vec<Group> = Vec::new();
    for mv in pos.legal_moves() {
        let squares = changed_squares(pos, &mv);
        if let Some(g) = groups.iter_mut().find(|g| g.squares == squares) {
            g.members += 1;
            if mv.promotion() == Some(Role::Queen) && g.mv.promotion() != Some(Role::Queen) {
                g.mv = mv;
            }
            continue;
        }
        let score = squares.iter().map(|&s| excess[s as usize]).sum();
        groups.push(Group { squares, score, mv, members: 1 });
    }
    groups.sort_by(|a, b| b.score.total_cmp(&a.score));

    let Some(best) = groups.first() else {
        out.reason = Some("no legal moves");
        return out;
    };
    let best_uci = uci(best.mv);
    out.best = Some(best.mv);
    out.best_uci = Some(best_uci);
    out.score = best.score;
    out.margin = match groups.get(1) {
        Some(second) => best.score - second.score,
        None => best.score,
    };

    let mut in_best = [false; 64];
    for &s in &best.squares {
        in_best[s as usize] = true;
    }
    out.residual = (0..64).filter(|&s| !in_best[s]).map(|s| excess[s].max(0.0)).sum();

    out.ranked = groups.iter().take(5).map(|g| (uci(g.mv), g.score)).collect();

    // A promotion group (several moves, one changed-square set) cannot be resolved
    // from the board: the piece is invisible in the diff. Never accept it; the
    // caller must ask which piece was chosen.
    let piece_unknown = best.members > 1;
    out.accept = !piece_unknown && out.margin >= cfg.min_margin && out.residual <= cfg.max_residual;
    if !out.accept {
        let ambiguous = piece_unknown || out.margin < cfg.min_margin;
        out.reason = Some(if ambiguous { "ambiguous" } else { "unexplained change" });
    }
    out
}

/// Per-square excess over the change threshold, the number of changed squares,
/// and the threshold. The median is the typical unchanged square; a change must
/// stand well above it.
fn excess_over_threshold(diff: &[f32; 64], cfg: &InferConfig) -> ([f32; 64], u32, f32) {
    let mut sorted = *diff;
    sorted.sort_unstable_by(f32::total_cmp);
    let median = (sorted[31] + sorted[32]) / 2.0;
    let t = cfg.min_change.max(2.5 * median);
    let excess: [f32; 64] = diff.map(|d| d - t);
    let changed = excess.iter().filter(|&&e| e > 0.0).count() as u32;
    (excess, changed, t)
}

/// Result of [`infer_plies`].
#[derive(Clone, Debug)]
pub struct PlyInference {
    /// Best explanation, in play order (empty when nothing changed or no line fits).
    pub moves: Vec<Move>,
    pub ucis: Vec<String>,
    pub score: f32,
    /// Best score minus the best line whose changed squares are neither the same
    /// set nor a superset of the best line's (see [`infer_plies`]).
    pub margin: f32,
    /// Sum of positive excess on squares the best line leaves alone.
    pub residual: f32,
    /// Number of squares whose diff exceeds the threshold.
    pub changed: u32,
    pub accept: bool,
    /// When !accept: "no change" | "ambiguous" | "unexplained change" | "no legal moves"
    pub reason: Option<&'static str>,
}

/// One distinct changed-square set over move sequences, and its best line.
struct Line {
    mask: u64,
    score: f32,
    moves: Vec<Move>,
    /// Queen promotions in `moves` (prefer the queen when the piece is invisible).
    queens: u32,
    members: u32,
}

/// Squares whose occupant differs between two boards, as a bit mask (a1 = bit 0).
fn board_change_mask(a: &shakmaty::Board, b: &shakmaty::Board) -> u64 {
    (0..64u32).filter(|&s| a.piece_at(Square::new(s)) != b.piece_at(Square::new(s))).fold(0, |m, s| m | 1 << s)
}

fn collect_lines(pos: &Chess, root: &shakmaty::Board, plies: usize, line: &mut Vec<Move>, excess: &[f32; 64], out: &mut Vec<Line>) {
    for mv in pos.legal_moves() {
        line.push(mv);
        let mut next = pos.clone();
        next.play_unchecked(mv);
        if plies > 1 {
            collect_lines(&next, root, plies - 1, line, excess, out);
        } else {
            let mask = board_change_mask(root, next.board());
            let queens = line.iter().filter(|m| m.promotion() == Some(Role::Queen)).count() as u32;
            match out.iter_mut().find(|l| l.mask == mask) {
                Some(l) => {
                    l.members += 1;
                    if queens > l.queens {
                        l.moves.clone_from(line);
                        l.queens = queens;
                    }
                }
                None => {
                    let score = (0..64).filter(|s| mask & (1 << s) != 0).map(|s| excess[s]).sum();
                    out.push(Line { mask, score, moves: line.clone(), queens, members: 1 });
                }
            }
        }
        line.pop();
    }
}

/// Like [`infer_move`], but the diff may span up to `max_plies` moves (1 or 2 in
/// practice: a read that was missed, or moves made while the board was out of
/// view). Tries exactly 1 ply, then exactly 2, … and returns the first accepted
/// line, else the result for `max_plies`.
///
/// A line's changed squares are the squares whose occupant differs between
/// `pos` and the position after the line, so a recapture on the square just
/// moved to counts that square once. Lines are scored like moves in
/// [`infer_move`]. One difference: a line whose squares are a strict superset of
/// the best line's (the best line plus a capture of the piece that just moved,
/// say) differs by one unchanged square only, so it is held to half of
/// `min_margin` (the extra square must still clearly not have changed) and does
/// not count toward `margin`. And a line of two or more plies is rejected if
/// any of its squares clearly did not change (diff under half the threshold),
/// so a resting hand is not read as an extra opponent move.
pub fn infer_plies(pos: &Chess, diff: &[f32; 64], max_plies: usize, cfg: &InferConfig) -> PlyInference {
    let (excess, changed, t) = excess_over_threshold(diff, cfg);
    let mut out = PlyInference {
        moves: Vec::new(),
        ucis: Vec::new(),
        score: 0.0,
        margin: 0.0,
        residual: 0.0,
        changed,
        accept: false,
        reason: Some("no legal moves"),
    };
    if changed == 0 {
        out.reason = Some("no change");
        return out;
    }
    for plies in 1..=max_plies.max(1) {
        let mut lines = Vec::new();
        collect_lines(pos, pos.board(), plies, &mut Vec::with_capacity(plies), &excess, &mut lines);
        lines.sort_by(|a, b| b.score.total_cmp(&a.score));
        let Some(best) = lines.first() else { break };
        let (mut other, mut superset) = (None::<f32>, None::<f32>);
        for l in &lines[1..] {
            let slot = if l.mask & best.mask == best.mask { &mut superset } else { &mut other };
            slot.get_or_insert(l.score);
        }
        let margin = best.score - other.unwrap_or(0.0);
        let superset_gap = superset.map_or(f32::MAX, |s| best.score - s);
        let residual = (0..64).filter(|s| best.mask & (1 << s) == 0).map(|s| excess[s].max(0.0)).sum();
        let piece_unknown = best.members > 1;
        // A line of two or more plies must not claim a square that clearly did
        // not change: otherwise any opponent move "explains" a resting hand at
        // the cost of one quiet square.
        let quiet_square = plies > 1 && (0..64).any(|s| best.mask & (1 << s) != 0 && excess[s] < -t / 2.0);
        let accept = !piece_unknown
            && !quiet_square
            && margin >= cfg.min_margin
            && superset_gap >= cfg.min_margin / 2.0
            && residual <= cfg.max_residual;
        out = PlyInference {
            moves: best.moves.clone(),
            ucis: best.moves.iter().map(|&m| uci(m)).collect(),
            score: best.score,
            margin,
            residual,
            changed,
            accept,
            reason: None,
        };
        if accept {
            return out;
        }
        let ambiguous = piece_unknown || margin < cfg.min_margin || superset_gap < cfg.min_margin / 2.0;
        out.reason = Some(if ambiguous { "ambiguous" } else { "unexplained change" });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{square_diff, square_stats};
    use crate::geometry::{board_to_image, square_to_grid, warp_board};
    use crate::synth::{default_params, render, render_hand, Rng, SynthParams};
    use shakmaty::fen::Fen;
    use shakmaty::EnPassantMode;

    const W: u32 = 640;
    const H: u32 = 480;
    /// Warped square size used by the end-to-end tests.
    const SQ: u32 = 32;

    fn sq(name: &str) -> u8 {
        u8::from(Square::from_ascii(name.as_bytes()).unwrap())
    }

    fn position(fen: &str) -> Chess {
        fen.parse::<Fen>().unwrap().into_position(CastlingMode::Standard).unwrap()
    }

    /// The legal move of `pos` with this UCI string.
    fn find(pos: &Chess, u: &str) -> Move {
        *pos.legal_moves().iter().find(|m| uci(**m) == u).unwrap_or_else(|| panic!("no legal move {u}"))
    }

    /// Square indices as algebraic names, for readable assertions.
    fn named(v: &[u8]) -> Vec<String> {
        v.iter().map(|&s| Square::new(s as u32).to_string()).collect()
    }

    #[test]
    fn changed_squares_normal_and_capture() {
        let start = Chess::default();
        assert_eq!(named(&changed_squares(&start, &find(&start, "e2e4"))), ["e2", "e4"]);

        let pos = position("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
        assert_eq!(named(&changed_squares(&pos, &find(&pos, "e4d5"))), ["e4", "d5"]);
    }

    #[test]
    fn changed_squares_castling() {
        let pos = position("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
        assert_eq!(named(&changed_squares(&pos, &find(&pos, "e1g1"))), ["e1", "f1", "g1", "h1"]);
        assert_eq!(named(&changed_squares(&pos, &find(&pos, "e1c1"))), ["a1", "c1", "d1", "e1"]);
    }

    #[test]
    fn changed_squares_en_passant() {
        let pos = position("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
        let mv = find(&pos, "e5d6");
        let sqs = changed_squares(&pos, &mv);
        assert_eq!(named(&sqs), ["d5", "e5", "d6"]);
        assert_eq!(sqs, vec![sq("d5"), sq("e5"), sq("d6")]);
    }

    #[test]
    fn changed_squares_promotion() {
        let pos = position("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        for u in ["a7a8q", "a7a8r", "a7a8b", "a7a8n"] {
            assert_eq!(named(&changed_squares(&pos, &find(&pos, u))), ["a7", "a8"], "{u}");
        }
    }

    /// Diff that is flat noise except for the given squares.
    fn diff_with(hot: &[(u8, f32)]) -> [f32; 64] {
        let mut d = [0.01f32; 64];
        for &(s, v) in hot {
            d[s as usize] = v;
        }
        d
    }

    #[test]
    fn clean_move_is_accepted() {
        let pos = Chess::default();
        let d = diff_with(&[(sq("e2"), 1.0), (sq("e4"), 1.0)]);
        let inf = infer_move(&pos, &d, &InferConfig::default());
        assert!(inf.accept, "{inf:?}");
        assert_eq!(inf.best_uci.as_deref(), Some("e2e4"));
        assert_eq!(inf.changed, 2);
        assert_eq!(inf.residual, 0.0);
        assert!(inf.margin > 0.5);
        assert_eq!(inf.ranked[0].0, "e2e4");
        assert!(inf.ranked.windows(2).all(|w| w[0].1 >= w[1].1));
        assert!(inf.reason.is_none());
    }

    #[test]
    fn flat_diff_is_no_change() {
        let inf = infer_move(&Chess::default(), &[0.01; 64], &InferConfig::default());
        assert!(!inf.accept);
        assert_eq!(inf.reason, Some("no change"));
        assert_eq!(inf.changed, 0);
        assert!(inf.best.is_none());
    }

    #[test]
    fn single_square_change_is_ambiguous() {
        // Only e2 changed: e2e3 and e2e4 both explain it equally well.
        let inf = infer_move(&Chess::default(), &diff_with(&[(sq("e2"), 1.0)]), &InferConfig::default());
        assert!(!inf.accept);
        assert_eq!(inf.reason, Some("ambiguous"));
    }

    #[test]
    fn stray_change_is_unexplained() {
        let d = diff_with(&[(sq("e2"), 1.0), (sq("e4"), 1.0), (sq("h5"), 1.0)]);
        let inf = infer_move(&Chess::default(), &d, &InferConfig::default());
        assert!(!inf.accept);
        assert_eq!(inf.reason, Some("unexplained change"));
        assert!(inf.residual > 0.5);
    }

    #[test]
    fn promotion_is_one_group_queen_preferred_but_not_accepted() {
        let pos = position("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
        let inf = infer_move(&pos, &diff_with(&[(sq("a7"), 1.0), (sq("a8"), 1.0)]), &InferConfig::default());
        // The four promotions are one group; the piece is invisible on the board, so no accept.
        assert!(!inf.accept, "{inf:?}");
        assert_eq!(inf.reason, Some("ambiguous"));
        assert_eq!(inf.best_uci.as_deref(), Some("a7a8q"));
        assert_eq!(inf.ranked.iter().filter(|(u, _)| u.starts_with("a7a8")).count(), 1);
    }

    /// Image of `board` as seen with the given noise/brightness/seed, rectified at `SQ`.
    fn observe(board: &shakmaty::Board, seed: u64, noise: f32, brightness: f32) -> (Vec<u8>, SynthParams) {
        let p = SynthParams { seed, noise, brightness, ..default_params(W, H) };
        (render(board, &p), p)
    }

    /// Per-square diff between the reference rendering of `before` and the
    /// observed rendering of `after`.
    fn diff_between(before: &shakmaty::Board, after: &shakmaty::Board, seed: u64, hand: Option<(f32, f32, f32)>) -> [f32; 64] {
        let (img_before, p) = observe(before, seed, 3.0, 0.0);
        let (mut img_after, _) = observe(after, seed.wrapping_add(7919), 3.5, 8.0);
        if let Some((cx, cy, r)) = hand {
            render_hand(&mut img_after, W, H, cx, cy, r);
        }
        let wb_before = warp_board(&img_before, W, H, &p.corners, SQ).unwrap();
        let wb_after = warp_board(&img_after, W, H, &p.corners, SQ).unwrap();
        square_diff(&square_stats(&wb_before), &square_stats(&wb_after))
    }

    /// Full synthetic pipeline for one played move.
    fn run_case(pos: &Chess, mv: Move, seed: u64, hand: Option<(f32, f32, f32)>) -> Inference {
        let mut after = pos.clone();
        after.play_unchecked(mv);
        let d = diff_between(pos.board(), after.board(), seed, hand);
        infer_move(pos, &d, &InferConfig::default())
    }

    /// Positions and played moves for the end-to-end checks: a seeded random
    /// walk from the start position, plus every legal move of special positions
    /// (castling, en passant, promotion, captures).
    fn e2e_cases() -> Vec<(Chess, Move)> {
        let mut cases = Vec::new();
        let mut rng = Rng::new(2024);
        let mut pos = Chess::default();
        for _ in 0..40 {
            let moves: Vec<Move> = pos.legal_moves().iter().copied().collect();
            if moves.is_empty() {
                pos = Chess::default();
                continue;
            }
            let mv = moves[rng.below(moves.len())];
            cases.push((pos.clone(), mv));
            pos.play_unchecked(mv);
        }
        for fen in [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
            "4k3/P7/8/8/8/8/8/4K3 w - - 0 1",
            "4k3/8/8/8/8/8/p7/4K3 b - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ] {
            let pos = position(fen);
            for mv in pos.legal_moves() {
                cases.push((pos.clone(), mv));
            }
        }
        cases
    }

    #[test]
    fn e2e_never_accepts_wrong_move() {
        let cases = e2e_cases();
        let (mut correct, mut promotions) = (0usize, 0usize);
        let mut wrong = Vec::new();
        // Margin and residual of the correct accepts, to document the defaults.
        let (mut min_margin, mut max_residual) = (f32::MAX, 0f32);
        for (i, (pos, mv)) in cases.iter().enumerate() {
            let inf = run_case(pos, *mv, 1000 + i as u64, None);
            let played = uci(*mv);
            if mv.promotion().is_some() {
                promotions += 1;
            }
            if inf.accept {
                if inf.best_uci.as_deref() == Some(played.as_str()) {
                    correct += 1;
                    min_margin = min_margin.min(inf.margin);
                    max_residual = max_residual.max(inf.residual);
                } else {
                    let fen = Fen::from_position(pos, EnPassantMode::Legal);
                    wrong.push(format!("{fen} played {played}, accepted {:?}", inf.best_uci));
                }
            }
        }
        println!(
            "e2e: {correct}/{} accepted correctly ({promotions} promotions, never accepted); \
             min margin {min_margin:.3}, max residual {max_residual:.3}",
            cases.len()
        );
        assert!(wrong.is_empty(), "wrong accepts:\n{}", wrong.join("\n"));
        // Promotions are excluded from the rate: the board cannot tell the piece (see `infer_move`).
        let non_promo = cases.len() - promotions;
        assert!(correct * 100 >= non_promo * 95, "only {correct}/{non_promo} non-promotion moves accepted");
    }

    #[test]
    fn e2e_same_position_is_no_change() {
        let mut seen = Vec::new();
        for (pos, _) in e2e_cases() {
            if seen.iter().any(|b: &shakmaty::Board| *b == *pos.board()) {
                continue;
            }
            for seed in 0..2u64 {
                let (a, pa) = observe(pos.board(), seed, 3.0, 0.0);
                let (b, pb) = observe(pos.board(), seed + 99, 4.0, 8.0);
                let wa = warp_board(&a, W, H, &pa.corners, SQ).unwrap();
                let wb = warp_board(&b, W, H, &pb.corners, SQ).unwrap();
                let d = square_diff(&square_stats(&wa), &square_stats(&wb));
                let inf = infer_move(&pos, &d, &InferConfig::default());
                assert!(!inf.accept, "accepted with no change: {inf:?}");
                assert_eq!(inf.reason, Some("no change"));
            }
            seen.push(pos.board().clone());
        }
    }

    #[test]
    fn e2e_hand_never_causes_wrong_accept() {
        let cases = e2e_cases();
        let mut accepted = 0usize;
        for (i, (pos, mv)) in cases.iter().enumerate().step_by(2) {
            // Put a hand over a rotating square (plus a second one for variety), in board coordinates.
            let corners = default_params(W, H).corners;
            let to_img = board_to_image(&corners, 8).unwrap();
            let s = ((i * 7) % 64) as u8;
            let (row, col) = square_to_grid(s);
            let (x, y) = to_img.apply(col as f64 + 0.5, row as f64 + 0.5);
            let r = [18.0f32, 30.0, 45.0][i % 3];
            let inf = run_case(pos, *mv, 5000 + i as u64, Some((x as f32, y as f32, r)));
            if inf.accept {
                accepted += 1;
                assert_eq!(inf.best_uci.as_deref(), Some(uci(*mv).as_str()), "wrong accept with hand on {s}");
            }
        }
        println!("hand: {accepted} accepted, all correct");
    }

    /// Synthetic diff for playing `line` from `pos`.
    fn run_line(pos: &Chess, line: &[Move], seed: u64) -> PlyInference {
        let mut after = pos.clone();
        for &mv in line {
            after.play_unchecked(mv);
        }
        let d = diff_between(pos.board(), after.board(), seed, None);
        infer_plies(pos, &d, 2, &InferConfig::default())
    }

    #[test]
    fn plies_read_two_moves_between_reads() {
        // Two plies from every position of the random walk (and the special
        // positions): the pair must be read, never a wrong line.
        let mut rng = Rng::new(77);
        let (mut correct, mut total) = (0usize, 0usize);
        let mut wrong = Vec::new();
        for (i, (pos, mv)) in e2e_cases().into_iter().enumerate().step_by(3) {
            let mut mid = pos.clone();
            mid.play_unchecked(mv);
            let replies: Vec<Move> = mid.legal_moves().iter().copied().collect();
            if replies.is_empty() {
                continue;
            }
            let line = [mv, replies[rng.below(replies.len())]];
            total += 1;
            let inf = run_line(&pos, &line, 9000 + i as u64);
            let want: Vec<String> = line.iter().map(|&m| uci(m)).collect();
            if inf.accept {
                if inf.ucis == want {
                    correct += 1;
                } else {
                    let fen = Fen::from_position(&pos, EnPassantMode::Legal);
                    wrong.push(format!("{fen} played {want:?}, accepted {:?}", inf.ucis));
                }
            }
        }
        println!("two plies: {correct}/{total} accepted correctly");
        assert!(wrong.is_empty(), "wrong accepts:\n{}", wrong.join("\n"));
        assert!(correct * 100 >= total * 85, "only {correct}/{total} two-ply lines read");
    }

    #[test]
    fn plies_prefer_one_move_and_see_no_change() {
        let pos = Chess::default();
        let inf = run_line(&pos, &[find(&pos, "g1f3")], 3);
        assert!(inf.accept, "{inf:?}");
        assert_eq!(inf.ucis, ["g1f3"]);

        let inf = run_line(&pos, &[], 4);
        assert!(!inf.accept);
        assert_eq!(inf.reason, Some("no change"));
        assert!(inf.moves.is_empty());
    }

    #[test]
    fn plies_read_a_recapture_on_the_square_just_moved_to() {
        // 1. e4 d5 2. exd5 Qxd5: the second pair changes e4, d5 and d8 only.
        let pos = position("rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
        let line = [find(&pos, "e4d5")];
        let mut mid = pos.clone();
        mid.play_unchecked(line[0]);
        let line = [line[0], find(&mid, "d8d5")];
        let inf = run_line(&pos, &line, 11);
        assert!(inf.accept, "{inf:?}");
        assert_eq!(inf.ucis, ["e4d5", "d8d5"]);
    }

    #[test]
    fn plies_hand_never_causes_wrong_accept() {
        let corners = default_params(W, H).corners;
        let to_img = board_to_image(&corners, 8).unwrap();
        for (i, (pos, mv)) in e2e_cases().into_iter().enumerate().step_by(4) {
            let mut after = pos.clone();
            after.play_unchecked(mv);
            let (row, col) = square_to_grid(((i * 11) % 64) as u8);
            let (x, y) = to_img.apply(col as f64 + 0.5, row as f64 + 0.5);
            let r = 30.0;
            let d = diff_between(pos.board(), after.board(), 7000 + i as u64, Some((x as f32, y as f32, r as f32)));
            let inf = infer_plies(&pos, &d, 2, &InferConfig::default());
            if !inf.accept {
                continue;
            }
            assert_eq!(inf.ucis[0], uci(mv), "wrong first ply with a hand");
            // A hand resting exactly over both squares of a legal reply is
            // indistinguishable from that reply. That is the only allowed extra ply.
            if let Some(extra) = inf.moves.get(1) {
                let mut mid = pos.clone();
                mid.play_unchecked(mv);
                for s in changed_squares(&mid, extra) {
                    let (row, col) = square_to_grid(s);
                    let (sx, sy) = to_img.apply(col as f64 + 0.5, row as f64 + 0.5);
                    let dist = ((sx - x).powi(2) + (sy - y).powi(2)).sqrt();
                    assert!(dist < r + 30.0, "extra ply {:?} not under the hand (square {s}, {dist:.0} px)", inf.ucis);
                }
            }
        }
    }

    #[test]
    fn timing_release() {
        let pos = Chess::default();
        let mut after = pos.clone();
        after.play_unchecked(find(&pos, "e2e4"));
        let (img_b, p) = observe(pos.board(), 1, 3.0, 0.0);
        let (img_a, _) = observe(after.board(), 2, 3.5, 8.0);
        let wb_b = warp_board(&img_b, W, H, &p.corners, 32).unwrap();
        let wb_a = warp_board(&img_a, W, H, &p.corners, 32).unwrap();
        let iters = 500u32;

        let t = std::time::Instant::now();
        let mut sink = 0f32;
        for _ in 0..iters {
            let d = square_diff(&square_stats(&wb_b), &square_stats(&wb_a));
            sink += d[0];
        }
        let stats_ms = t.elapsed().as_secs_f64() * 1e3 / (2.0 * iters as f64);
        std::hint::black_box(sink);

        let d = square_diff(&square_stats(&wb_b), &square_stats(&wb_a));
        let t = std::time::Instant::now();
        for _ in 0..iters {
            std::hint::black_box(infer_move(&pos, &d, &InferConfig::default()));
        }
        let infer_ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
        println!("square_stats (2 boards, 256x256): {stats_ms:.4} ms each; infer_move: {infer_ms:.4} ms");

        // Two plies (e2e4 e7e5) checked against every legal pair.
        let mut two = after.clone();
        two.play_unchecked(find(&after, "e7e5"));
        let d2 = diff_between(pos.board(), two.board(), 3, None);
        let t = std::time::Instant::now();
        for _ in 0..20 {
            std::hint::black_box(infer_plies(&pos, &d2, 2, &InferConfig::default()));
        }
        println!("infer_plies (2 plies): {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    }
}
