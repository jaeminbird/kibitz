//! Finding a Stockfish binary and installing a pinned official build.

use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context};
use futures_util::StreamExt;
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::EngineError;

/// Archive format of a pinned download.
#[derive(Clone, Copy)]
enum Archive {
    Zip,
    TarGz,
}

/// One official Stockfish release asset, pinned by URL and SHA-256.
struct Pin {
    os: &'static str,
    /// `None` means the asset covers every architecture on `os`.
    arch: Option<&'static str>,
    url: &'static str,
    sha256: Option<&'static str>,
    archive: Archive,
    /// Path of the engine binary inside the archive.
    entry: &'static str,
}

/// Official Stockfish 19 (tag `sf_19`) "universal" builds. The release has no
/// `avx2`-only assets; the universal builds pick the best code path at run time.
const PINS: &[Pin] = &[
    Pin {
        os: "windows",
        arch: Some("x86_64"),
        url: "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-windows-x86-64-universal.zip",
        sha256: Some("3c8bf1f9ea66a09350a40df4f632288285ac206d99f33ab5842c408fc30b48a7"),
        archive: Archive::Zip,
        entry: "stockfish/stockfish-windows-x86-64-universal.exe",
    },
    Pin {
        os: "macos",
        arch: None,
        url: "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-macos-universal.tar.gz",
        sha256: Some("a1f0e3bcc5a6927a11fe6fc8e54a779754645f3c2bae2cf13420fd1957adaa77"),
        archive: Archive::TarGz,
        entry: "stockfish/stockfish-macos-universal",
    },
    Pin {
        os: "linux",
        arch: Some("x86_64"),
        url: "https://github.com/official-stockfish/Stockfish/releases/download/sf_19/stockfish-linux-x86-64-universal.tar.gz",
        sha256: Some("9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f"),
        archive: Archive::TarGz,
        entry: "stockfish/stockfish-linux-x86-64-universal",
    },
];

/// The engine file name on this OS.
fn exe_name() -> &'static str {
    if cfg!(windows) {
        "stockfish.exe"
    } else {
        "stockfish"
    }
}

/// Where `download` installs the engine.
pub fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("kibitz")
}

/// Find a Stockfish binary: `KIBITZ_STOCKFISH`, then `PATH`, then the data dir.
pub fn locate() -> Result<PathBuf, EngineError> {
    if let Some(p) = std::env::var_os("KIBITZ_STOCKFISH").map(PathBuf::from) {
        if p.is_file() {
            return Ok(p);
        }
        tracing::warn!(path = %p.display(), "KIBITZ_STOCKFISH does not point to a file");
    }
    if let Some(p) = find_on_path(exe_name()) {
        return Ok(p);
    }
    let installed = data_dir().join("bin").join(exe_name());
    if installed.is_file() {
        return Ok(installed);
    }
    Err(EngineError::NotInstalled)
}

fn find_on_path(name: &str) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

/// The pin for this OS and architecture, if there is one.
fn pin_for_this_platform() -> anyhow::Result<&'static Pin> {
    PINS.iter()
        .find(|p| p.os == std::env::consts::OS && p.arch.is_none_or(|a| a == std::env::consts::ARCH))
        .ok_or_else(|| {
            anyhow!(
                "no pinned Stockfish build for {}/{}",
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        })
}

/// Download the pinned build for this platform, check its SHA-256, and unpack the
/// engine to `data_dir()/bin/stockfish[.exe]` (via a `.part` file and a rename).
/// `progress` gets `(bytes_so_far, total_bytes_if_known)`.
///
/// Refuses to install if the pin has no checksum.
pub async fn download(mut progress: impl FnMut(u64, Option<u64>) + Send) -> anyhow::Result<PathBuf> {
    let pin = pin_for_this_platform()?;
    let expected = pin
        .sha256
        .ok_or_else(|| anyhow!("no pinned checksum for {}; refusing to install", pin.url))?;

    let bin_dir = data_dir().join("bin");
    tokio::fs::create_dir_all(&bin_dir).await?;
    let archive_path = bin_dir.join("stockfish-download.part");

    let response = reqwest::get(pin.url)
        .await
        .and_then(|r| r.error_for_status())
        .with_context(|| format!("downloading {}", pin.url))?;
    let total = response.content_length();
    let mut file = tokio::fs::File::create(&archive_path).await?;
    let mut hasher = Sha256::new();
    let mut done = 0u64;
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        hasher.update(&chunk);
        file.write_all(&chunk).await?;
        done += chunk.len() as u64;
        progress(done, total);
    }
    file.flush().await?;
    drop(file);

    let digest: String = hasher.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if digest != expected {
        let _ = tokio::fs::remove_file(&archive_path).await;
        bail!("checksum mismatch for {}: got {digest}, expected {expected}", pin.url);
    }

    let target = bin_dir.join(exe_name());
    let part = bin_dir.join(format!("{}.part", exe_name()));
    let (src, dst) = (archive_path.clone(), part.clone());
    let (archive, entry) = (pin.archive, pin.entry);
    tokio::task::spawn_blocking(move || extract(&src, &dst, archive, entry))
        .await
        .context("unpacking task failed")??;
    tokio::fs::rename(&part, &target).await?;
    let _ = tokio::fs::remove_file(&archive_path).await;
    make_executable(&target)?;
    Ok(target)
}

/// Copy the engine entry out of the archive to `out`.
fn extract(archive: &Path, out: &Path, kind: Archive, entry: &str) -> anyhow::Result<()> {
    let mut dst = File::create(out)?;
    match kind {
        Archive::Zip => {
            let mut zip = zip::ZipArchive::new(File::open(archive)?)?;
            let mut f = zip
                .by_name(entry)
                .with_context(|| format!("{entry} not in archive"))?;
            io::copy(&mut f, &mut dst)?;
        }
        Archive::TarGz => {
            let gz = flate2::read::GzDecoder::new(File::open(archive)?);
            let mut tar = tar::Archive::new(gz);
            for e in tar.entries()? {
                let mut e = e?;
                if e.path()?.as_ref() == Path::new(entry) {
                    io::copy(&mut e, &mut dst)?;
                    return Ok(());
                }
            }
            bail!("{entry} not in archive");
        }
    }
    Ok(())
}

#[cfg(unix)]
fn make_executable(path: &Path) -> io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn extracts_entry_from_zip_and_tar_gz() {
        let dir = tempfile::tempdir().unwrap();
        let entry = "stockfish/engine-bin";

        let zip_path = dir.path().join("a.zip");
        let mut zw = zip::ZipWriter::new(File::create(&zip_path).unwrap());
        zw.start_file("stockfish/other.txt", zip::write::SimpleFileOptions::default())
            .unwrap();
        zw.write_all(b"nope").unwrap();
        zw.start_file(entry, zip::write::SimpleFileOptions::default()).unwrap();
        zw.write_all(b"ZIPDATA").unwrap();
        zw.finish().unwrap();

        let tgz_path = dir.path().join("b.tar.gz");
        let gz = flate2::write::GzEncoder::new(File::create(&tgz_path).unwrap(), flate2::Compression::fast());
        let mut tb = tar::Builder::new(gz);
        let data = b"TARDATA";
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        tb.append_data(&mut header, entry, &data[..]).unwrap();
        tb.into_inner().unwrap().finish().unwrap();

        let out = dir.path().join("out");
        extract(&zip_path, &out, Archive::Zip, entry).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"ZIPDATA");
        extract(&tgz_path, &out, Archive::TarGz, entry).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"TARDATA");
        assert!(extract(&tgz_path, &out, Archive::TarGz, "missing").is_err());
    }
}
