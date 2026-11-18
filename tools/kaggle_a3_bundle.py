"""Checksummed Kaggle source/runtime archives. Standard library only at bootstrap."""
import argparse
import base64
import hashlib
import inspect
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = ("include", "src", "apps", "tools", "tests", "cmake", "patches", "diagnostics", "configs")
DOCS = ("EXPERIMENT_SPEC.md", "IMPLEMENTATION_PLAN.md", "A3_IMPROVEMENTS.md",
        "OMPL_INTEGRATION.md", "STATUS.md", "KAGGLE_A3_PILOT.md")
SKIP = {"__pycache__", ".pytest_cache", ".git", ".venv"}
MANIFEST = "bundle.json"
PORTABLE_CASE_POLICY = "portable_casefold"
LINUX_CASE_POLICY = "linux_case_sensitive"


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def safe_name(name):
    p = PurePosixPath(name)
    if (not name or "\\" in name or ":" in name or p.is_absolute()
            or PureWindowsPath(name).drive or ".." in p.parts or str(p) != name
            or name == "." or name.endswith("/")):
        raise ValueError("unsafe archive path: " + name)
    return name


def archive_path_key(name, policy):
    if policy == PORTABLE_CASE_POLICY:
        return name.casefold()
    if policy == LINUX_CASE_POLICY:
        return name
    raise ValueError("unsupported archive path case policy: " + str(policy))


def manifest_path_policy(manifest):
    # Archives produced before the explicit Linux option retain their contract.
    policy = manifest.get("path_case_policy", PORTABLE_CASE_POLICY)
    archive_path_key("", policy)
    return policy


def validate_extraction_case_policy(manifest, target):
    """Fail before extracting bytes or removing restored runtime aliases."""
    if manifest_path_policy(manifest) != LINUX_CASE_POLICY:
        return
    if not sys.platform.startswith("linux") or os.name == "nt":
        raise ValueError("Linux case-sensitive archive requires a Linux destination")
    parent = Path(target).resolve()
    while not parent.exists():
        parent = parent.parent
    # A Linux mount can itself be case-insensitive; do not assume ext4 semantics.
    with tempfile.TemporaryDirectory(prefix=".archive-case-probe-", dir=parent) as probe:
        Path(probe, "case_probe").write_bytes(b"")
        if Path(probe, "CASE_PROBE").exists():
            raise ValueError("Linux case-sensitive archive requires a case-sensitive filesystem")


def archive(entries, output, kind, metadata=None, *, path_case_policy=PORTABLE_CASE_POLICY):
    """entries=(safe name, file path); Linux runtime names require explicit opt-in."""
    archive_path_key("", path_case_policy)
    output = Path(output)
    if output.exists():
        raise ValueError("archive already exists; retain it and use a new name")
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + ".partial")
    if pending.exists():
        raise ValueError("unfinished archive retained: " + str(pending))
    records, names = [], set()
    with zipfile.ZipFile(pending, "x", allowZip64=True) as z:
        for name, source in sorted(entries, key=lambda item: item[0]):
            safe_name(name)
            key = archive_path_key(name, path_case_policy)
            if key in names or key == archive_path_key(MANIFEST, path_case_policy):
                raise ValueError("duplicate archive member: " + name)
            names.add(key)
            source = Path(source)
            if not source.is_file() or source.is_symlink():
                raise ValueError("archive input must be a regular file: " + str(source))
            mode = 0o755 if os.name != "nt" and source.stat().st_mode & stat.S_IXUSR else 0o644
            records.append({"path": name, "size_bytes": source.stat().st_size,
                            "sha256": sha256(source), "mode": mode})
            compression = zipfile.ZIP_STORED if name.endswith((".zip", ".gz")) else zipfile.ZIP_DEFLATED
            z.write(source, name, compress_type=compression, compresslevel=1)
        manifest = {"schema_version": 1, "kind": kind, "files": records,
                    "files_sha256": digest(records), "metadata": metadata or {}}
        if path_case_policy != PORTABLE_CASE_POLICY:
            manifest["path_case_policy"] = path_case_policy
        z.writestr(MANIFEST, json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False))
    # Verify the full archive before publishing it; retain a failed partial.
    verify(pending)
    pending.rename(output)
    output.with_name(output.name + ".sha256").write_text(sha256(output) + "  " + output.name + "\n")
    return manifest


def verify(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is not None and sha256(path) != expected_sha256:
        raise ValueError("archive SHA-256 mismatch")
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        names = [safe_name(i.filename) for i in infos]
        # Exact duplicates are forbidden before reading the unique manifest.
        if len(set(names)) != len(names):
            raise ValueError("duplicate archive member")
        if any(stat.S_ISLNK(i.external_attr >> 16) for i in infos):
            raise ValueError("archive symlinks are forbidden")
        manifest = json.loads(z.read(MANIFEST))
        policy = manifest_path_policy(manifest)
        if len({archive_path_key(n, policy) for n in names}) != len(names):
            raise ValueError("duplicate archive member")
        records = manifest["files"]
        if manifest.get("schema_version") != 1 or digest(records) != manifest["files_sha256"]:
            raise ValueError("archive inventory mismatch")
        if set(names) != {MANIFEST, *(r["path"] for r in records)} or len(records) + 1 != len(names):
            raise ValueError("archive contains unregistered/missing members")
        for r in records:
            safe_name(r["path"])
            if r["mode"] not in (0o644, 0o755) or z.getinfo(r["path"]).file_size != r["size_bytes"]:
                raise ValueError("archive size/mode mismatch")
            with z.open(r["path"]) as handle:
                checksum = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    checksum.update(chunk)
                if checksum.hexdigest() != r["sha256"]:
                    raise ValueError("archive member checksum mismatch: " + r["path"])
    return manifest


def extract(path, target, expected_sha256=None, allow_identical=False):
    manifest = verify(path, expected_sha256)
    target = Path(target).resolve()
    validate_extraction_case_policy(manifest, target)
    # Preflight every destination before writing anything.
    for r in manifest["files"]:
        dst = target / r["path"]
        if dst.is_symlink() or target not in dst.resolve().parents:
            raise ValueError("extraction destination escapes target")
        if dst.exists() and (not allow_identical or not dst.is_file() or sha256(dst) != r["sha256"]):
            raise ValueError("refusing to overwrite archive destination: " + str(dst))
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as z:
        for r in manifest["files"]:
            dst = target / r["path"]
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(r["path"]) as src, dst.open("xb") as out:
                shutil.copyfileobj(src, out)
            if os.name != "nt":
                dst.chmod(r["mode"])
    return manifest


def source_entries(root=ROOT):
    root = Path(root)
    result = []
    for directory in SOURCE_DIRS:
        for p in (root / directory).rglob("*"):
            if any(part in SKIP for part in p.parts) or p.suffix == ".pyc":
                continue
            if p.is_symlink():
                raise ValueError("source symlink: " + str(p))
            if p.is_file():
                result.append((p.relative_to(root).as_posix(), p))
    for name in ("CMakeLists.txt", "AGENTS.md", "requirements-cloud-benchmark.txt"):
        result.append((name, root / name))
    result += [("docs/" + name, root / "docs" / name) for name in DOCS]
    lock = json.loads((root / "configs/dependencies.lock.json").read_text())
    for name, record in lock["archives"].items():
        p = root / ".deps" / (name + ".tar.gz")
        if sha256(p) != record["sha256"]:
            raise ValueError("pinned dependency archive mismatch: " + name)
        result.append((".deps/" + p.name, p))
    for relative, filename in (("ompl-1.7.0", "LICENSE"), ("boost_1_82_0", "LICENSE_1_0.txt")):
        result.append(("licenses/" + relative + "-" + filename, root / ".deps" / relative / filename))
    for p in (root / ".deps/eigen-3.4.0").glob("COPYING*"):
        result.append(("licenses/Eigen-" + p.name, p))
    return result


def upload_transport(input_archive, output):
    """Keep the inner ZIP opaque when Kaggle expands uploaded ZIP archives.

    The only outer members are the base64 payload and archive()'s small,
    checksummed bundle.json. The temporary encoded file is never a deliverable.
    """
    input_archive, output = Path(input_archive), Path(output)
    verify(input_archive)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"encoding": "base64_rfc4648_no_whitespace", "decoded_name": "a3_kaggle_input.zip",
                "decoded_sha256": sha256(input_archive), "decoded_size_bytes": input_archive.stat().st_size,
                "notebook_hash_is_authoritative": True}
    with tempfile.TemporaryDirectory(prefix="a3_transport_", dir=output.parent) as temporary:
        payload = Path(temporary)/"a3_kaggle_input.b64"
        with input_archive.open("rb") as source, payload.open("xb") as encoded:
            # Every non-final chunk has a multiple-of-three length: no interior padding.
            for chunk in iter(lambda: source.read(3 * 1024 * 1024), b""):
                encoded.write(base64.b64encode(chunk))
        return archive([("a3_kaggle_input.b64", payload)], output, "A3_KAGGLE_BASE64_TRANSPORT", metadata)


def decode_transport(payload, target, expected_sha256, expected_size_bytes=None):
    """Self-contained stdlib decoder, also embedded verbatim in the notebook."""
    import base64
    import hashlib
    from pathlib import Path
    payload, target = Path(payload), Path(target)
    if not payload.is_file() or payload.is_symlink():
        raise ValueError("transport payload must be a regular file")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("invalid pinned input SHA-256")
    if expected_size_bytes is not None and (not isinstance(expected_size_bytes, int) or
            isinstance(expected_size_bytes, bool) or expected_size_bytes <= 0):
        raise ValueError("invalid pinned input size")
    maximum = expected_size_bytes if expected_size_bytes is not None else 4 * 1024**3
    if payload.stat().st_size > 4 * ((maximum + 2) // 3):
        raise ValueError("encoded transport exceeds pinned input size bound")
    pending = target.with_name(target.name + ".partial")
    if target.exists() or pending.exists():
        raise ValueError("transport destination already exists; use a fresh session")
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum, decoded_size, padding_seen = hashlib.sha256(), 0, False
    with payload.open("rb") as encoded, pending.open("xb") as decoded:
        for chunk in iter(lambda: encoded.read(4 * 1024 * 1024), b""):
            if padding_seen:
                raise ValueError("base64 data follows final padding")
            value = base64.b64decode(chunk, validate=True)
            padding_seen = b"=" in chunk
            decoded_size += len(value)
            if decoded_size > maximum:
                raise ValueError("decoded transport exceeds pinned input size bound")
            checksum.update(value)
            decoded.write(value)
    if expected_size_bytes is not None and decoded_size != expected_size_bytes:
        raise ValueError("decoded transport size mismatch")
    if checksum.hexdigest() != expected_sha256:
        raise ValueError("Input checksum mismatch; do not execute another source")
    pending.rename(target)
    return target


def select_input_archive(input_root, decode_root, expected_sha256, expected_size_bytes=None):
    """Prefer one opaque payload; accept one directly mounted ZIP as fallback."""
    import hashlib
    from pathlib import Path
    input_root, decode_root = Path(input_root), Path(decode_root)
    payloads = [path for path in input_root.rglob("a3_kaggle_input.b64")
                if path.is_file() and not path.is_symlink()]
    if payloads:
        if len(payloads) != 1:
            raise ValueError("Attach exactly one A3 upload dataset; multiple base64 payloads found")
        archive = decode_transport(payloads[0], decode_root/"a3_kaggle_input.zip",
                                   expected_sha256, expected_size_bytes)
    else:
        archives = [path for path in input_root.rglob("a3_kaggle_input.zip")
                    if path.is_file() and not path.is_symlink()]
        if len(archives) != 1:
            raise ValueError("Attach the dataset created from a3_kaggle_upload.zip using Add Input")
        archive = archives[0]
    maximum = expected_size_bytes if expected_size_bytes is not None else 4 * 1024**3
    if archive.stat().st_size > maximum or (expected_size_bytes is not None and archive.stat().st_size != maximum):
        raise ValueError("Input archive size mismatch")
    checksum = hashlib.sha256()
    with archive.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != expected_sha256:
        raise ValueError("Input checksum mismatch; do not execute another source")
    return archive


def notebook(expected_hash, expected_size_bytes=None):
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "metadata": {}, "id": "a3-cell-" + str(len(cells)),
                "source": source.splitlines(keepends=True)}
        if kind == "code": cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    add("markdown", "# Pilot A3 — Kaggle CPU\n\n13 cấu hình, 780 trial; cost ở 10 giây và thời gian nghiệm đầu từ solve.\n"
        "Dataset và Notebook đặt **Private**, Accelerator **None**, Internet **On** cho bootstrap.\n"
        "Tạo Dataset từ `a3_kaggle_upload.zip`, rồi gắn Dataset bằng Add Input. Kaggle có thể giải nén ZIP upload; "
        "payload base64 giữ nguyên source bên trong. Chọn **Save Version → Save & Run All** để chạy nền.\n"
        "Sau khi job đã bắt đầu trên Kaggle, có thể tắt máy cá nhân. Kết quả và checkpoint nằm trong Output.\n"
        "Kaggle hoàn tất notebook không đồng nghĩa pilot COMPLETE: đọc `RUN_STATUS.json` và báo cáo.\n")
    add("code", "import time\nNOTEBOOK_STARTED_MONOTONIC = time.monotonic()\n"
        "# Để trống cho lần đầu. Khi resume: gắn Output cũ bằng Add Input rồi nhập đường dẫn checkpoint.\n"
        "RESUME_ARCHIVE = ''\nALLOW_A3_PILOT = True\n")
    add("code", "from pathlib import Path\nimport hashlib, json, os, shutil, stat, subprocess, sys, zipfile\n"
        f"EXPECTED_INPUT_SHA256 = {expected_hash!r}\n"
        f"EXPECTED_INPUT_SIZE_BYTES = {expected_size_bytes!r}\n\n" +
        inspect.getsource(decode_transport) + "\n" + inspect.getsource(select_input_archive) + "\n" +
        "archive = select_input_archive('/kaggle/input', '/kaggle/temp/a3-transport', "
        "EXPECTED_INPUT_SHA256, EXPECTED_INPUT_SIZE_BYTES)\n"
        "workspace = Path('/kaggle/temp/a3-pilot')\n"
        "if workspace.exists() and any(workspace.iterdir()):\n    raise RuntimeError('Cần phiên Kaggle mới; không ghi đè workspace đang có')\n"
        "workspace.mkdir(parents=True, exist_ok=True)\n"
        "with zipfile.ZipFile(archive) as z:\n"
        "    for info in z.infolist():\n"
        "        p = Path(info.filename)\n"
        "        if p.is_absolute() or '..' in p.parts or chr(92) in info.filename or ':' in info.filename or stat.S_ISLNK(info.external_attr >> 16):\n"
        "            raise RuntimeError('Unsafe input archive member')\n"
        "    z.extractall(workspace)\n"
        "cmd = [sys.executable, str(workspace/'tools/kaggle_a3.py'), '--stage', 'bootstrap',\n"
        "       '--input-archive', str(archive), '--input-sha256', EXPECTED_INPUT_SHA256,\n"
        "       '--output', '/kaggle/working/a3-results',\n"
        "       '--notebook-started-monotonic', str(NOTEBOOK_STARTED_MONOTONIC)]\n"
        "if ALLOW_A3_PILOT: cmd.append('--allow-a3-pilot')\n"
        "if RESUME_ARCHIVE: cmd += ['--resume-archive', RESUME_ARCHIVE]\n"
        "# Batch plotting must not inherit Jupyter's inline backend into the isolated Python.\n"
        "run_environment = dict(os.environ, MPLBACKEND='Agg')\n"
        "launcher_receipt = Path('/kaggle/working/a3-results/setup/notebook_launcher.json')\n"
        "launcher_receipt.parent.mkdir(parents=True, exist_ok=True)\n"
        "launcher_receipt.write_text(json.dumps({'input_sha256': EXPECTED_INPUT_SHA256,\n"
        "    'launcher_revision': 'headless_agg_v1', 'child_mplbackend': run_environment['MPLBACKEND'],\n"
        "    'parent_mplbackend': os.environ.get('MPLBACKEND')}, indent=2))\n"
        "completed = subprocess.run(cmd, cwd=workspace, env=run_environment)\n"
        "print('Bootstrap/runner exit code:', completed.returncode)\n")
    add("code", "status_file = Path('/kaggle/working/a3-results/RUN_STATUS.json')\n"
        "if status_file.exists():\n    print(status_file.read_text())\n"
        "else:\n    print('BLOCKED: chưa có RUN_STATUS; đọc output/log cell trước')\n"
        "from IPython.display import display, FileLink\n"
        "for p in sorted(Path('/kaggle/working/a3-results').glob('exports/*')):\n"
        "    if p.is_file(): display(FileLink(str(p)))\n"
        "report = Path('/kaggle/working/a3-results/report/REPORT.md')\n"
        "if report.exists():\n    from IPython.display import Markdown\n    display(Markdown(report.read_text()))\n")
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}}, "cells": cells}


def package(output, root=ROOT):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("delivery directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "a3_kaggle_input.zip"
    archive(source_entries(root), target, "A3_KAGGLE_SOURCE_INPUT",
            {"protocol_id": "a3_improvements_pilot_v1", "copied_historic_trials": 0,
             "copied_planner_binaries": 0})
    upload_transport(target, output / "a3_kaggle_upload.zip")
    (output / "a3_pilot_kaggle.ipynb").write_text(json.dumps(notebook(sha256(target), target.stat().st_size),
                                                          ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(Path(root) / "docs/KAGGLE_A3_PILOT.md", output / "HUONG_DAN.md")
    files = [{"name": p.name, "sha256": sha256(p), "size_bytes": p.stat().st_size}
             for p in sorted(output.iterdir()) if p.is_file()]
    (output / "delivery_manifest.json").write_text(json.dumps({"files": files, "pilot_status": "NOT_RUN"}, indent=2))
    return files


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path)
    p.add_argument("--verify", type=Path)
    args = p.parse_args()
    if args.verify:
        m = verify(args.verify)
        print(json.dumps({"status": "PASS", "kind": m["kind"], "files": len(m["files"])}))
    elif args.output: print(json.dumps(package(args.output), indent=2))
    else: p.error("use --output or --verify")


if __name__ == "__main__": main()
