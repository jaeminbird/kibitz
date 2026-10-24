//! Image-directory source: a folder of stills played as frames.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use cg_core::{CaptureError, Frame, FrameSource};

use crate::{pace, period};

/// Every `.png` / `.jpg` in a directory, sorted by file name, one per `1/fps` s.
pub struct ImageDirSource {
    dir: PathBuf,
    paths: Vec<PathBuf>,
    next: usize,
    looping: bool,
    period: Option<Duration>,
    last_start: Option<Instant>,
    seq: u64,
}

impl ImageDirSource {
    /// No pacing if `fps <= 0`. After the last image: end (`Ok(None)`), or restart if `looping`.
    pub fn open(dir: &Path, fps: f32, looping: bool) -> Result<Self, CaptureError> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| CaptureError::Other(format!("cannot read {}: {e}", dir.display())))?;
        let mut paths: Vec<PathBuf> = entries
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| is_still(p))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(CaptureError::Other(format!("no .png/.jpg files in {}", dir.display())));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            paths,
            next: 0,
            looping,
            period: period(fps),
            last_start: None,
            seq: 0,
        })
    }
}

impl FrameSource for ImageDirSource {
    fn next_frame(&mut self) -> Result<Option<Frame>, CaptureError> {
        if self.next == self.paths.len() {
            if !self.looping {
                return Ok(None);
            }
            self.next = 0;
        }
        pace(self.period, &mut self.last_start);
        let path = self.paths[self.next].clone();
        self.next += 1;
        let img = image::open(&path)
            .map_err(|e| CaptureError::Other(format!("cannot decode {}: {e}", path.display())))?
            .to_rgb8();
        self.seq += 1;
        Ok(Some(Frame::from_rgb(self.seq, img.width(), img.height(), img.into_raw())))
    }

    fn describe(&self) -> String {
        format!("dir:{}", self.dir.display())
    }
}

fn is_still(p: &Path) -> bool {
    p.is_file()
        && p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "png" | "jpg"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn write_stills(dir: &Path) {
        // Written out of order; sizes identify them after sorting by name: a=2, b=3, c=4.
        for (name, size) in [("c.png", 4u32), ("a.png", 2), ("b.png", 3)] {
            RgbImage::from_pixel(size, size, Rgb([10, 20, 30])).save(dir.join(name)).unwrap();
        }
    }

    #[test]
    fn reads_pngs_in_name_order_then_ends() {
        let dir = tempfile::tempdir().unwrap();
        write_stills(dir.path());

        let mut src = ImageDirSource::open(dir.path(), 0.0, false).unwrap();
        let mut got = Vec::new();
        while let Some(f) = src.next_frame().unwrap() {
            assert_eq!(f.width, f.height);
            got.push((f.seq, f.width));
        }
        assert_eq!(got, vec![(1, 2), (2, 3), (3, 4)]);
        assert!(src.next_frame().unwrap().is_none());
    }

    #[test]
    fn looping_restarts_and_keeps_counting() {
        let dir = tempfile::tempdir().unwrap();
        write_stills(dir.path());

        let mut src = ImageDirSource::open(dir.path(), 0.0, true).unwrap();
        let widths: Vec<(u64, u32)> = (0..5)
            .map(|_| {
                let f = src.next_frame().unwrap().unwrap();
                (f.seq, f.width)
            })
            .collect();
        assert_eq!(widths, vec![(1, 2), (2, 3), (3, 4), (4, 2), (5, 3)]);
    }

    #[test]
    fn empty_directory_is_an_error() {
        let dir = tempfile::tempdir().unwrap();
        assert!(ImageDirSource::open(dir.path(), 5.0, false).is_err());
    }
}
