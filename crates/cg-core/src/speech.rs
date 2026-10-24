//! Turning SAN into words for the earpiece. The vocabulary is deliberately
//! small and closed so the voice crate can pre-render every word as a clip.

/// Every word `spoken_san` can produce, plus fixed announcements.
/// Square names ("e4") are spoken as one token each.
pub fn vocabulary() -> Vec<String> {
    let mut words: Vec<String> = [
        "King", "Queen", "Rook", "Bishop", "Knight", "Pawn", "to", "takes", "castles",
        "kingside", "queenside", "promotes", "check", "checkmate",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    for f in 'a'..='h' {
        for r in '1'..='8' {
            words.push(format!("{f}{r}"));
        }
    }
    words
}

/// "Nf3" → "Knight to f3", "exd5" → "Pawn takes d5", "O-O" → "castles kingside",
/// "e8=Q+" → "Pawn to e8 promotes Queen check". Disambiguation letters are
/// dropped (the destination is what the wearer needs).
pub fn spoken_san(san: &str) -> String {
    let s = san.trim();
    let (core, suffix) = match s.trim_end_matches(['!', '?']) {
        x if x.ends_with('#') => (&x[..x.len() - 1], Some("checkmate")),
        x if x.ends_with('+') => (&x[..x.len() - 1], Some("check")),
        x => (x, None),
    };
    let mut out: Vec<String> = Vec::new();
    if core == "O-O" || core == "0-0" {
        out.push("castles kingside".into());
    } else if core == "O-O-O" || core == "0-0-0" {
        out.push("castles queenside".into());
    } else {
        let (body, promo) = match core.split_once('=') {
            Some((b, p)) => (b, piece_name(p.chars().next().unwrap_or('Q'))),
            None => (core, None),
        };
        let first = body.chars().next().unwrap_or('P');
        let piece = if first.is_ascii_uppercase() { piece_name(first) } else { None };
        out.push(piece.unwrap_or("Pawn").into());
        out.push(if body.contains('x') { "takes" } else { "to" }.into());
        let dest = if body.len() >= 2 { &body[body.len() - 2..] } else { body };
        out.push(dest.to_string());
        if let Some(p) = promo {
            out.push(format!("promotes {p}"));
        }
    }
    if let Some(sfx) = suffix {
        out.push(sfx.into());
    }
    out.join(" ")
}

fn piece_name(c: char) -> Option<&'static str> {
    Some(match c {
        'K' => "King",
        'Q' => "Queen",
        'R' => "Rook",
        'B' => "Bishop",
        'N' => "Knight",
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phrases() {
        assert_eq!(spoken_san("Nf3"), "Knight to f3");
        assert_eq!(spoken_san("Rxd1"), "Rook takes d1");
        assert_eq!(spoken_san("exd5"), "Pawn takes d5");
        assert_eq!(spoken_san("e4"), "Pawn to e4");
        assert_eq!(spoken_san("O-O"), "castles kingside");
        assert_eq!(spoken_san("O-O-O+"), "castles queenside check");
        assert_eq!(spoken_san("Nbd2"), "Knight to d2");
        assert_eq!(spoken_san("e8=Q#"), "Pawn to e8 promotes Queen checkmate");
    }

    #[test]
    fn every_word_is_in_vocabulary() {
        let vocab = vocabulary();
        for san in ["Nf3", "Rxd1", "exd5", "O-O", "O-O-O+", "e8=Q#", "Kxh8", "Bb5+"] {
            for w in spoken_san(san).split(' ') {
                assert!(vocab.iter().any(|v| v == w), "{w} missing from vocabulary");
            }
        }
    }
}
