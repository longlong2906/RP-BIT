"""Explicit, opt-in downloads to .deps; never installs outside the workspace."""
import hashlib
import json
import pathlib
import tarfile
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPENDENCIES = {
    "ompl-1.7.0": "https://codeload.github.com/ompl/ompl/tar.gz/refs/tags/1.7.0",
    "eigen-3.4.0": "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz",
    "boost_1_82_0": "https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz",
}

def main():
    directory = ROOT / ".deps"
    directory.mkdir(exist_ok=True)
    manifest = directory / "downloads.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    for name, url in DEPENDENCIES.items():
        archive = directory / (name + ".tar.gz")
        if not archive.exists():
            print("Downloading", url, flush=True)
            partial = archive.with_suffix(".partial")
            urllib.request.urlretrieve(url, partial)
            partial.replace(archive)
        sha = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        lock=json.loads((ROOT/"configs/dependencies.lock.json").read_text())
        if sha != lock["archives"][name]["sha256"]:
            raise RuntimeError("archive does not match pinned SHA-256: " + name)
        if name in records and records[name]["sha256"] != sha:
            raise RuntimeError("dependency archive changed: " + name)
        records[name] = {"url": url, "sha256": sha}
        manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
        if not (directory / name).exists():
            print("Extracting", name, flush=True)
            with tarfile.open(archive) as tar:
                tar.extractall(directory, filter="data")
    print(json.dumps(records, indent=2))

if __name__ == "__main__":
    main()
