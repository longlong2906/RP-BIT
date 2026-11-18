"""Package an external verifier correction without changing the frozen campaign."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = "b4e6eb1c6e958117fffcd87fd0ae4c314fbf8116af43c0b05ab1ed4008826691"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            value.update(chunk)
    return value.hexdigest()


def cell(kind, source, number):
    value = dict(cell_type=kind, metadata={}, id=f"recovery-r1-{number:02d}",
                 source=textwrap.dedent(source).lstrip("\n").splitlines(keepends=True))
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def notebook(patch_hash, analyzer_hash, stem="native_paper_v4_recovery_r1", chunk_manifest_sha256=None):
    cells = [cell("markdown", """
        # Tiếp tục benchmark A3-T v4 — sửa kiểm chứng endpoint

        Phiên trước đã hoàn tất 120/1.500 lượt main rồi dừng do so sánh endpoint tuyệt đối.
        Boost.JSON và Python lệch một ULP (3,55×10⁻¹⁵ m). Bản sửa dùng sai số singleton
        10⁻⁹ m đã có trong đặc tả, giữ nguyên cap risk 40 và swept slope 35°.

        Upload ZIP checkpoint đã chuẩn bị hoặc `_output_.zip` vào một Dataset riêng tư,
        rồi Add Input dataset đó và payload `.b64` đi kèm notebook này.
        Nếu Kaggle giải nén ZIP, notebook tự tìm index; nếu ZIP còn nguyên,
        notebook chỉ lấy 21 file checkpoint cần dùng, không giải nén toàn bộ Output.
        Cũng có thể Add Input → Notebook Output của phiên trước.
        Chọn CPU đủ 4 vCPU rồi Run All.
        Không cần tải lại source payload cũ. Internet không cần cho runtime đã restore.
        Notebook giữ source/binary/map/seed/request gốc, kiểm lại smoke và raw main,
        rồi chỉ nhận trial chưa bắt đầu. Không chạy lại 120 lượt đã hoàn tất.
        `ALLOW_MAIN=True` cho phép tiếp tục 1.380 lượt còn lại; `PREPARE_ONLY=True`
        chỉ chuẩn bị và kiểm lại smoke. Giữ toàn bộ Output mới để tiếp tục phiên sau.
        """, 0), cell("code", f"""
        from pathlib import Path
        import time
        SESSION_STARTED = globals().get('SESSION_STARTED', time.monotonic())
        PATCH_SHA256 = {patch_hash!r}
        ANALYZER_SHA256 = {analyzer_hash!r}
        SOURCE_SHA256 = {SOURCE_SHA256!r}
        PATCH_STEM = {stem!r}
        PATCH_PAYLOAD = None
        CHUNK_MANIFEST = None  # Auto-select native_paper_v4_resume_chunks.json under /kaggle/input.
        CHUNK_MANIFEST_SHA256 = {chunk_manifest_sha256!r}
        OUTPUT_ZIP = None  # Optional Path('/kaggle/input/your-dataset/_output_.zip').
        RESUME_INDEX = None
        RESUME_SHA256 = None
        ALLOW_MAIN = True
        PREPARE_ONLY = False
        BASE = Path('/kaggle/working/native_paper_v4')
        ROOT = BASE / 'repo'
        WORKSPACE = BASE / 'campaign'
        PATCH_ROOT = Path('/kaggle/working/native_paper_v4_recovery_r1')
        ZIP_STAGE = Path('/kaggle/working/_native_paper_v4_zip_input')
        CHUNK_ZIP = Path('/kaggle/working/_native_paper_v4_chunk_input.zip')
        """, 1), cell("code", """
        import base64, hashlib, json, os, shutil, stat, sys, zipfile
        from pathlib import PurePosixPath
        if not sys.platform.startswith('linux'):
            raise RuntimeError('Recovery notebook requires Kaggle Linux')
        def checksum(path):
            result = hashlib.sha256()
            with Path(path).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024**2), b''):
                    result.update(chunk)
            return result.hexdigest()
        def safe_member(name):
            p = PurePosixPath(name)
            if not name or p.is_absolute() or '..' in p.parts or ':' in name or chr(92) in name or str(p) != name:
                raise ValueError('Unsafe archive member')
            return p
        def copy_checked(source, destination, size_bytes, expected_sha256):
            partial = destination.with_name(destination.name+'.recovery-partial')
            if partial.is_symlink():
                raise ValueError('Partial copy symlink refused')
            offset = 0
            if partial.exists():
                if not partial.is_file() or partial.stat().st_size > size_bytes:
                    raise ValueError('Invalid retained partial copy')
                with partial.open('rb') as previous:
                    while chunk := previous.read(1024**2):
                        if source.read(len(chunk)) != chunk:
                            raise ValueError('Retained partial differs from verified input')
                        offset += len(chunk)
            with partial.open('ab' if partial.exists() else 'xb') as target:
                shutil.copyfileobj(source, target, length=1024**2)
            if partial.stat().st_size != size_bytes or checksum(partial) != expected_sha256:
                raise ValueError('Copied checksum mismatch')
            partial.replace(destination)
        def extract_verified(archive_path, destination, expected_kind):
            destination.mkdir(parents=True, exist_ok=True)
            if destination.is_symlink():
                raise ValueError('Extraction root must not be a symlink')
            with zipfile.ZipFile(archive_path) as archive:
                infos = archive.infolist()
                names = [i.filename for i in infos]
                if len(names) != len(set(names)):
                    raise ValueError('Duplicate archive members')
                for info in infos:
                    safe_member(info.filename)
                    if stat.S_ISLNK(info.external_attr >> 16):
                        raise ValueError('Source links are forbidden')
                manifest = json.loads(archive.read('bundle.json'))
                if manifest['kind'] != expected_kind:
                    raise ValueError('Wrong archive kind')
                records = manifest['files']
                if set(names) != {'bundle.json', *(r['path'] for r in records)}:
                    raise ValueError('Archive inventory mismatch')
                for record in records:
                    path = destination / safe_member(record['path'])
                    if not path.resolve().is_relative_to(destination.resolve()):
                        raise ValueError('Destination escapes extraction root')
                    for parent in (path, *path.parents):
                        if parent == destination:
                            break
                        if parent.is_symlink():
                            raise ValueError('Destination symlink refused')
                    if path.exists():
                        if not path.is_file() or path.stat().st_size != record['size_bytes'] or checksum(path) != record['sha256']:
                            raise ValueError('Existing source differs: '+str(path))
                    else:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(record['path']) as source:
                            copy_checked(source, path, record['size_bytes'], record['sha256'])
                    path.chmod(record.get('mode', 0o644))
            return manifest
        BASE.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(BASE).free < 6*1024**3:
            raise RuntimeError('Recovery requires at least 6 GiB free before restore')
        patch_archive = Path('/kaggle/working') / (PATCH_STEM+'.zip')
        if not patch_archive.exists():
            if PATCH_PAYLOAD is None:
                found = sorted(Path('/kaggle/input').rglob(PATCH_STEM+'*.b64'))
                if len(found) != 1:
                    raise ValueError('Attach exactly one recovery .b64, or set PATCH_PAYLOAD')
                PATCH_PAYLOAD = found[0]
            partial = patch_archive.with_suffix('.partial')
            with Path(PATCH_PAYLOAD).open('rb') as source, partial.open('xb') as target:
                padded = False
                while chunk := source.read(4*1024**2):
                    if padded:
                        raise ValueError('Data after base64 padding')
                    data = base64.b64decode(chunk, validate=True)
                    padded = b'=' in chunk
                    if target.tell()+len(data) > 16*1024**2:
                        raise ValueError('Recovery payload too large')
                    target.write(data)
            if checksum(partial) != PATCH_SHA256:
                raise ValueError('Recovery checksum mismatch')
            partial.replace(patch_archive)
        if checksum(patch_archive) != PATCH_SHA256:
            raise ValueError('Recovery archive changed')
        patch_manifest = extract_verified(patch_archive, PATCH_ROOT, 'NATIVE_PAPER_V4_EXTERNAL_VERIFIER_RECOVERY_R1')
        if patch_manifest['source_input_sha256'] != SOURCE_SHA256:
            raise ValueError('Recovery targets a different source bundle')
        if checksum(PATCH_ROOT/'native_paper_v4_analysis.py') != ANALYZER_SHA256:
            raise ValueError('Verifier patch changed')
        """, 2), cell("code", """
        import importlib.util
        zip_spec = importlib.util.spec_from_file_location('_v4_zip_input', PATCH_ROOT/'native_paper_v4_zip_input.py')
        zip_input = importlib.util.module_from_spec(zip_spec)
        sys.modules[zip_spec.name] = zip_input
        zip_spec.loader.exec_module(zip_input)
        chunk_restore_receipt = None
        if RESUME_INDEX is None and OUTPUT_ZIP is None:
            if CHUNK_MANIFEST is None:
                manifests = sorted(Path('/kaggle/input').rglob('native_paper_v4_resume_chunks.json'))
                if len(manifests) > 1:
                    raise ValueError('Multiple chunk manifests: set CHUNK_MANIFEST explicitly')
                if manifests:
                    CHUNK_MANIFEST = manifests[0]
            if CHUNK_MANIFEST is not None:
                chunk_info = json.loads(Path(CHUNK_MANIFEST).read_text())
                if chunk_info.get('source_input_sha256') != SOURCE_SHA256:
                    raise ValueError('Chunk input targets a different frozen source')
                chunk_spec = importlib.util.spec_from_file_location('_v4_chunk_input', PATCH_ROOT/'native_paper_v4_chunk_input.py')
                chunk_input = importlib.util.module_from_spec(chunk_spec)
                sys.modules[chunk_spec.name] = chunk_input
                chunk_spec.loader.exec_module(chunk_input)
                chunk_restore_receipt = chunk_input.restore_chunk_input(Path(CHUNK_MANIFEST), CHUNK_ZIP,
                    expected_manifest_sha256=CHUNK_MANIFEST_SHA256)
                OUTPUT_ZIP = Path(chunk_restore_receipt['path'])
                if RESUME_SHA256 is None:
                    RESUME_SHA256 = chunk_info.get('checkpoint_index_sha256')
                print('Reconstructed verified ZIP from base64 parts:', chunk_restore_receipt['parts'])
        zip_input_staged = False
        if RESUME_INDEX is None and OUTPUT_ZIP is None:
            matches = []
            for candidate in Path('/kaggle/input').rglob('checkpoint_index.json'):
                value = json.loads(candidate.read_text())
                if value.get('source_input_sha256') == SOURCE_SHA256 and value.get('protocol_id') == 'a3t_old_safe_sem_kaggle_v4':
                    matches.append(candidate)
            if len(matches) > 1:
                raise ValueError('Multiple checkpoint indexes: set RESUME_INDEX explicitly')
            if matches:
                RESUME_INDEX = matches[0]
        if RESUME_INDEX is None:
            if OUTPUT_ZIP is None:
                archives = sorted({p for pattern in ('_output_*.zip', 'native_paper_v4_resume_input*.zip')
                    for p in Path('/kaggle/input').rglob(pattern)})
                if len(archives) != 1:
                    raise ValueError('Attach the resume dataset or set OUTPUT_ZIP to its path under /kaggle/input')
                OUTPUT_ZIP = archives[0]
            RESUME_INDEX = zip_input.prepare_zip_input(Path(OUTPUT_ZIP), ZIP_STAGE,
                SOURCE_SHA256, expected_index_sha256=RESUME_SHA256)
            zip_input_staged = True
            print('Selected checkpoint input from ZIP:', OUTPUT_ZIP)
        RESUME_INDEX = Path(RESUME_INDEX)
        if RESUME_SHA256 is None:
            RESUME_SHA256 = RESUME_INDEX.with_name(RESUME_INDEX.name+'.sha256').read_text().split()[0]
        if checksum(RESUME_INDEX) != RESUME_SHA256:
            raise ValueError('Checkpoint index checksum mismatch')
        old_base = RESUME_INDEX.parents[2]
        old_source = old_base / 'source_input.zip'
        source_archive = BASE / 'source_input.zip'
        if checksum(old_source) != SOURCE_SHA256:
            raise ValueError('Previous output lacks the exact original source archive')
        if not source_archive.exists():
            with old_source.open('rb') as source:
                copy_checked(source, source_archive, old_source.stat().st_size, SOURCE_SHA256)
        if checksum(source_archive) != SOURCE_SHA256:
            raise ValueError('Local source archive changed')
        extract_verified(source_archive, ROOT, 'TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA13_SOURCE')
        sys.path.insert(0, str(ROOT/'tools'))
        from kaggle_native_paper_v4_bundle import restore_checkpoint
        restored = restore_checkpoint(RESUME_INDEX, BASE, RESUME_SHA256,
            source_sha256=SOURCE_SHA256, root=ROOT, workspace=WORKSPACE)
        if zip_input_staged:
            receipt = zip_input.cleanup_zip_input(ZIP_STAGE)
            receipt_path = WORKSPACE/'setup/zip_input_restore.json'
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2)+'\\n')
            print('Removed verified temporary ZIP input after restore')
            RESUME_INDEX = WORKSPACE/'exports/checkpoint_index.json'
        if chunk_restore_receipt is not None:
            if (CHUNK_ZIP.is_symlink() or CHUNK_ZIP.resolve().parent != BASE.parent.resolve() or
                    Path(chunk_restore_receipt['path']).absolute() != CHUNK_ZIP.absolute() or
                    CHUNK_ZIP.stat().st_size != chunk_restore_receipt['size_bytes'] or
                    checksum(CHUNK_ZIP) != chunk_restore_receipt['sha256']):
                raise ValueError('Temporary reconstructed ZIP ownership/hash changed')
            CHUNK_ZIP.unlink()
            receipt_path = WORKSPACE/'setup/chunk_input_restore.json'
            receipt_path.write_text(json.dumps(dict(chunk_restore_receipt, temporary_removed=True), indent=2)+'\\n')
            OUTPUT_ZIP = None
            print('Removed verified reconstructed ZIP after checkpoint restore')
        print('Restored verified checkpoint:', RESUME_SHA256)
        print('Payloads:', len(restored['payloads']), 'files:', len(restored['inventory']))
        main_manifest = json.loads((WORKSPACE/'manifests/main.json').read_text())
        from collections import Counter
        print('Existing main slots:', Counter(e['execution_status'] for e in main_manifest['entries']))
        print('Free GiB:', round(shutil.disk_usage(BASE).free/1024**3, 2))
        """, 3), cell("code", """
        import kaggle_native_paper_v4 as bootstrap
        if not ALLOW_MAIN and not PREPARE_ONLY:
            raise ValueError('Continuing main requires ALLOW_MAIN=True')
        stage_error = None
        try:
            runtimes, env = bootstrap.isolated_runtimes(WORKSPACE, SESSION_STARTED, restore=True)
            env['LD_LIBRARY_PATH'] = str(ROOT/'.deps/install/lib')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
            argv = [runtimes['controller'], PATCH_ROOT/'native_paper_v4_recovery.py',
                '--root', ROOT, '--workspace', WORKSPACE,
                '--analyzer-patch', PATCH_ROOT/'native_paper_v4_analysis.py',
                '--analyzer-sha256', ANALYZER_SHA256,
                '--manifest', PATCH_ROOT/'recovery.json',
                '--manifest-sha256', checksum(PATCH_ROOT/'recovery.json'),
                '--session-started-monotonic', SESSION_STARTED]
            if PREPARE_ONLY:
                argv += ['--prepare-only']
            if ALLOW_MAIN:
                argv += ['--allow-main']
            bootstrap.command('recovery_r1', argv, WORKSPACE, SESSION_STARTED, env=env)
        except BaseException as error:
            stage_error = error
            raise
        finally:
            bootstrap.retain_checkpoint(WORKSPACE, source_archive, stage_error=stage_error)
        print('Keep the complete Notebook Output, including campaign/exports, campaign/shards and campaign/recovery.')
        """, 4)]
    if chunk_manifest_sha256 is not None:
        cells[0] = cell("markdown", """
            # Chạy tiếp A3-T v4 bằng Input chia phần base64

            Upload toàn bộ 11 file trong thư mục `upload` vào một Dataset Private:
            9 phần `.b64`, manifest JSON và payload `.b64` nhỏ. Không upload ZIP.
            Add Input dataset đó, chọn CPU đủ 4 vCPU rồi Run All, giữ cấu hình mặc định.

            Notebook kiểm hash từng phần và ZIP ghép, phục hồi source/runtime/checkpoint
            đã khóa, kiểm lại smoke rồi tiếp tục 1.380 lượt chưa bắt đầu.
            Giữ 120 lượt main đã hoàn tất; không thay planner, map, seed hoặc budget20s.
            ZIP và bản staging tạm chỉ được thu hồi sau khi restore đã xác minh.
            `ALLOW_MAIN=True`, `PREPARE_ONLY=False`. Internet không cần cho runtime restore.
            """, 0)
    return dict(cells=cells, nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.12"), accelerator="none"))


def build(output, stem="native_paper_v4_recovery_r1", chunk_manifest=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output/(stem+".zip")
    files = [(name, ROOT/"tools"/name) for name in (
        "native_paper_v4_recovery.py", "native_paper_v4_analysis.py", "native_paper_v4_zip_input.py")]
    if chunk_manifest is not None:
        files.append(("native_paper_v4_chunk_input.py", ROOT/"tools/native_paper_v4_chunk_input.py"))
    recovery = dict(policy_id="v4_endpoint_roundoff_recovery_v1",
                    source_input_sha256=SOURCE_SHA256,
                    prepared_source_sha256="4a1ac767f098c4c8bc2c469e0680ec53b5e655df64b346027b75ee0233da1b7b",
                    initial_started_count=120,
                    initial_started_ids_sha256="1a61cf6f592f632e2ccf9201c870f21aaceb80ab3f2d2b63c8c45ce3659b6ec8",
                    runner=dict(path=files[0][0], sha256=sha(files[0][1])),
                    analyzer=dict(path=files[1][0], sha256=sha(files[1][1])))
    recovery_path = output/"recovery.json"
    if recovery_path.exists():
        raise ValueError("Recovery output already exists")
    recovery_path.write_text(json.dumps(recovery, indent=2)+"\n", encoding="utf-8")
    files.append(("recovery.json", recovery_path))
    records = [dict(path=name, size_bytes=path.stat().st_size, sha256=sha(path), mode=0o644)
               for name, path in files]
    manifest = dict(kind="NATIVE_PAPER_V4_EXTERNAL_VERIFIER_RECOVERY_R1", schema_version=13,
                    source_input_sha256=SOURCE_SHA256, files=records)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in files:
            archive.write(path, name)
        archive.writestr("bundle.json", json.dumps(manifest, indent=2)+"\n")
    archive_path.with_suffix(".b64").write_bytes(base64.b64encode(archive_path.read_bytes()))
    archive_path.with_suffix(".zip.sha256").write_text(sha(archive_path)+"  "+archive_path.name+"\n")
    archive_path.with_suffix(".ipynb").write_text(json.dumps(
        notebook(sha(archive_path), sha(ROOT/"tools/native_paper_v4_analysis.py"), stem,
                 sha(chunk_manifest) if chunk_manifest else None),
        indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(archive=str(archive_path), sha256=sha(archive_path),
                         bytes=archive_path.stat().st_size, files=records)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stem", default="native_paper_v4_recovery_r1")
    parser.add_argument("--chunk-manifest", type=Path)
    args = parser.parse_args()
    build(args.output, args.stem, args.chunk_manifest)
