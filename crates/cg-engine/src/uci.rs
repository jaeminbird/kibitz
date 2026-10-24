//! Parsing of the engine's UCI output lines.

use cg_core::Score;

use crate::EngineError;

/// The fields of an `info` line that the engine uses.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Info {
    pub depth: Option<u32>,
    pub score: Option<Score>,
    /// PV line index; 1 when absent.
    pub multipv: u32,
    /// True for `lowerbound` / `upperbound` scores, which are not exact.
    pub bound: bool,
}

/// Parse an `info ...` line. Returns `None` if it is not an info line.
pub(crate) fn parse_info(line: &str) -> Option<Info> {
    let mut tokens = line.split_whitespace();
    if tokens.next()? != "info" {
        return None;
    }
    let mut info = Info {
        multipv: 1,
        ..Info::default()
    };
    while let Some(tok) = tokens.next() {
        match tok {
            // Free text follows; nothing after it is a field.
            "string" => break,
            "depth" => info.depth = tokens.next().and_then(|v| v.parse().ok()),
            "multipv" => info.multipv = tokens.next().and_then(|v| v.parse().ok()).unwrap_or(1),
            "lowerbound" | "upperbound" => info.bound = true,
            "score" => {
                info.score = match (tokens.next(), tokens.next()) {
                    (Some("cp"), Some(v)) => v.parse().ok().map(Score::Cp),
                    (Some("mate"), Some(v)) => v.parse().ok().map(Score::Mate),
                    _ => None,
                };
            }
            _ => {}
        }
    }
    Some(info)
}

/// Parse the rest of a `bestmove` line (after the keyword): `e2e4 [ponder e7e5]`.
/// Returns `(bestmove, ponder)`.
pub(crate) fn parse_bestmove(rest: &str) -> Result<(String, Option<String>), EngineError> {
    let mut tokens = rest.split_whitespace();
    let best = tokens
        .next()
        .filter(|m| (4..=5).contains(&m.len()))
        .ok_or_else(|| EngineError::Protocol(format!("bad bestmove: {rest}")))?;
    let ponder = match (tokens.next(), tokens.next()) {
        (Some("ponder"), Some(p)) => Some(p.to_string()),
        _ => None,
    };
    Ok((best.to_string(), ponder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cp_score_and_depth() {
        let info = parse_info("info depth 10 seldepth 12 multipv 1 score cp 31 nodes 900 pv e2e4").unwrap();
        assert_eq!(info.depth, Some(10));
        assert_eq!(info.score, Some(Score::Cp(31)));
        assert_eq!(info.multipv, 1);
        assert!(!info.bound);
    }

    #[test]
    fn parses_mate_and_bounds() {
        let info = parse_info("info depth 4 score mate -3 upperbound").unwrap();
        assert_eq!(info.score, Some(Score::Mate(-3)));
        assert!(info.bound);
    }

    #[test]
    fn ignores_string_lines_and_non_info() {
        assert_eq!(parse_info("info string score cp 5").unwrap().score, None);
        assert!(parse_info("readyok").is_none());
    }

    #[test]
    fn bestmove_with_and_without_ponder() {
        assert_eq!(
            parse_bestmove(" e2e4 ponder e7e5").unwrap(),
            ("e2e4".into(), Some("e7e5".into()))
        );
        assert_eq!(parse_bestmove(" a7a8q").unwrap(), ("a7a8q".into(), None));
        assert!(parse_bestmove(" nonsense").is_err());
    }
}
