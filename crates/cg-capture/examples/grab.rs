//! Manual check: `cargo run -p cg-capture --example grab -- <spec>`.
//! With `list` as the spec, prints every capturable window instead.

use std::time::Instant;

fn main() {
    let spec = std::env::args().nth(1).unwrap_or_else(|| "list".into());
    if spec == "list" {
        for (app, title, w, h) in cg_capture::list_windows() {
            println!("{w:>5}x{h:<5} app={app:?} title={title:?}");
        }
        return;
    }

    // Optional second arg: fps (default 5.0). Use 0 to disable pacing so `took` is pure capture time.
    let fps: f32 = std::env::args().nth(2).map_or(5.0, |s| s.parse().expect("fps"));
    let mut source = cg_capture::open_source(&spec, fps).expect("open_source failed");
    println!("source: {}", source.describe());
    for _ in 0..3 {
        let t = Instant::now();
        match source.next_frame() {
            Ok(Some(f)) => println!(
                "seq={} size={}x{} rgb_bytes={} took={:.1} ms",
                f.seq,
                f.width,
                f.height,
                f.rgb.len(),
                t.elapsed().as_secs_f64() * 1000.0
            ),
            Ok(None) => {
                println!("source ended");
                break;
            }
            Err(e) => {
                println!("error: {e}");
                break;
            }
        }
    }
}
