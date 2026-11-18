"""Build an external, hash-bound R2 notebook; never alter a frozen campaign."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import inspect
import json
from pathlib import Path, PurePosixPath
import re
import textwrap
import zipfile

import kaggle_native_paper_v4_recovery_bundle as r1_bundle

ROOT = Path(__file__).resolve().parents[1]
KIND = "NATIVE_PAPER_V4_EXTERNAL_AFFINITY_RECOVERY_R2"
POLICY_ID = "v4_affinity_recovery_v2"
SOURCE_SHA256 = "b4e6eb1c6e958117fffcd87fd0ae4c314fbf8116af43c0b05ab1ed4008826691"
CHECKPOINT_SHA256 = "a849a9e5004f1eaffd50fab2ffba15c938d60044c0f447e3e22a4a232d36984d"
PREPARED_SOURCE_SHA256 = "4a1ac767f098c4c8bc2c469e0680ec53b5e655df64b346027b75ee0233da1b7b"
FROZEN_IDENTITIES = dict(
    policy_id=POLICY_ID, source_input_sha256=SOURCE_SHA256,
    source_checkpoint_index_sha256=CHECKPOINT_SHA256,
    prepared_source_sha256=PREPARED_SOURCE_SHA256,
    initial_started_count=893,
    initial_started_ids_sha256="2b09f5c2d7ac740d16f3d47ce00c05fbd4d7943983915abec7c0d2d9b43cad03",
    initial_pending_count=607,
    initial_pending_ids_sha256="b646b8cf377f705205467da35165854baa2ee6f5e2610d83004aa6382764ecb6",
    initial_missing_count=53,
    initial_missing_ids_sha256="d88bb303d19278d7f89dd06cdcafd5a894da49d1b3218fb57b2b4aae046f0cf4",
    source_checkpoint_inventory_sha256="cd7a36b1753a1967caf539be9b976f317c75ce23626b95ed441e397ebe4252f5",
    base_payloads=[
        dict(path="native_paper_v4_20261107T194140732765Z.payload", size_bytes=801067839,
             sha256="b38d7b54c0a8bba62c0a5eb633c321f2b543b3c59e10843a2485c9f4854506bb"),
        dict(path="native_paper_v4_20261107T194750601068Z.payload", size_bytes=1199469,
             sha256="e457c8ad04a9de8d060180c360ac079664e947939c498fae12010d8dcd7cc3a8"),
        dict(path="native_paper_v4_20261107T195041462491Z.payload", size_bytes=953181,
             sha256="7e4d92c55ab523b93b09a21a8a6cb3550617d1b97e3ee3ea31508c19e6b5f3f6"),
        dict(path="native_paper_v4_20261107T202025570771Z.payload", size_bytes=71646620,
             sha256="c5cf4fb2e9c5605826224af9c120494e666d02ceae6d4b643395b262880e757f"),
        dict(path="native_paper_v4_20261108T215254494570Z.payload", size_bytes=251499434,
             sha256="7c5fb75376897d06a0bb448f447cc2f938aee72e3dd5f18fdc8769cec864d36e"),
    ],
)
PATCH_NAMES = dict(
    runner="native_paper_v4_recovery_r2.py",
    scheduler="terrain_compare_scheduler_r2.py",
    r1_runner="native_paper_v4_recovery.py",
    analyzer="native_paper_v4_analysis.py",
    zip_input="native_paper_v4_zip_input.py",
    chunk_input="native_paper_v4_chunk_input.py",
    r1_manifest="r1_recovery.json",
)
TEST_NAMES = dict(scheduler="test_affinity_launch_race.py", runner="test_native_paper_v4_recovery_r2.py")


def sha(path):
    return r1_bundle.sha(path)


def encoded_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8")


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _sha_text(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(label+" must be a lowercase SHA-256")
    return value


def default_manifest():
    return json.loads(json.dumps(FROZEN_IDENTITIES))


def verify_checkpoint_header(value, actual_sha256, manifest, sidecar_sha256=None):
    """Accept only the pinned base or a hash-checked extension carrying its proof."""
    if (value.get("source_input_sha256") != manifest["source_input_sha256"] or
            value.get("protocol_id") != "a3t_old_safe_sem_kaggle_v4" or value.get("schema_version") != 13):
        raise ValueError("Checkpoint source/protocol differs from R2")
    base = manifest["base_payloads"]
    payloads = value.get("payloads", [])
    if not isinstance(payloads, list) or payloads[:len(base)] != base:
        raise ValueError("Checkpoint does not preserve the exact base payload chain")
    if actual_sha256 == manifest["source_checkpoint_index_sha256"]:
        if len(payloads) != len(base) or value.get("inventory_sha256") != manifest["source_checkpoint_inventory_sha256"]:
            raise ValueError("Base checkpoint metadata differs")
        return False
    if sidecar_sha256 != actual_sha256 or len(payloads) <= len(base):
        raise ValueError("Descendant checkpoint lacks a matching sidecar or extension")
    baseline_path = "campaign/recovery/v4_affinity_recovery_v2/baseline_checkpoint_index.json"
    record = value.get("inventory", {}).get(baseline_path, {})
    if record.get("path") != baseline_path or record.get("sha256") != manifest["source_checkpoint_index_sha256"]:
        raise ValueError("Descendant checkpoint lacks its bound R2 baseline index")
    return True


def _journal_record(index, name, size_bytes):
    record = index.get("inventory", {}).get(name)
    if not isinstance(record, dict) or record.get("path") != name or record.get("size_bytes") != size_bytes:
        raise ValueError("Unindexed or newer physical start journal: "+name)
    return record


def _journal_stream_sha(stream):
    value = hashlib.sha256()
    for data in iter(lambda: stream.read(1024**2), b""):
        value.update(data)
    return value.hexdigest()


def verify_physical_journals(base, index):
    """Refuse an Output whose physical launches are newer than its checkpoint.

    A slim, explicitly prepared checkpoint input may have no physical journals;
    its checked payload chain is authoritative. Never silently discard a journal
    when a full Notebook Output also contains it.
    """
    base = Path(base)
    checked = 0
    for path in sorted((base/"campaign/collection").glob("main*/start_journal.jsonl")):
        name = path.relative_to(base).as_posix()
        if path.is_symlink() or not path.is_file():
            raise ValueError("Physical start journal is not regular: "+name)
        record = _journal_record(index, name, path.stat().st_size)
        with path.open("rb") as stream:
            if _journal_stream_sha(stream) != record.get("sha256"):
                raise ValueError("Unindexed or newer physical start journal: "+name)
        checked += 1
    return checked


def verify_archive_journals(archive, index_name, index):
    """The same launch guard for a full raw Output ZIP, before selective import."""
    suffix = "campaign/exports/checkpoint_index.json"
    if not index_name.endswith(suffix):
        raise ValueError("Unexpected checkpoint archive path")
    prefix = index_name[:-len(suffix)]
    seen = set()
    for info in archive.infolist():
        if not info.filename.startswith(prefix):
            continue
        name = info.filename[len(prefix):]
        parts = PurePosixPath(name).parts
        if len(parts) != 4 or parts[:2] != ("campaign", "collection") or not parts[2].startswith("main") or parts[3] != "start_journal.jsonl":
            continue
        if name in seen or info.is_dir() or (info.external_attr >> 16) & 0o170000 not in (0, 0o100000):
            raise ValueError("Duplicate or nonregular archived start journal: "+name)
        seen.add(name)
        record = _journal_record(index, name, info.file_size)
        with archive.open(info) as stream:
            if _journal_stream_sha(stream) != record.get("sha256"):
                raise ValueError("Unindexed or newer archived start journal: "+name)
    return len(seen)


def _cell(kind, source, number):
    item = r1_bundle.cell(kind, source, number)
    item["id"] = f"recovery-r2-{number:02d}"
    return item


def _restore_utilities():
    # Reuse the reviewed bounded copy/extraction primitives, not R1's stage or
    # its 120-slot checkpoint selection. The generator source is bound below.
    source = "".join(r1_bundle.notebook("unused", "unused")["cells"][2]["source"])
    marker = "BASE.mkdir(parents=True, exist_ok=True)\n"
    if source.count(marker) != 1:
        raise ValueError("R1 extraction template changed; review R2 generator")
    return source.split(marker)[0]


def notebook(patch_bytes, recovery_manifest_sha256, *, stem="native_paper_v4_recovery_r2", chunk_manifest_sha256=None):
    patch_hash = _digest(patch_bytes)
    _sha_text(recovery_manifest_sha256, "manifest hash")
    if chunk_manifest_sha256 is not None:
        _sha_text(chunk_manifest_sha256, "chunk manifest hash")
    config = f"""
        from pathlib import Path
        import time
        SESSION_STARTED = globals().get('SESSION_STARTED', time.monotonic())
        PATCH_SHA256 = {patch_hash!r}
        RECOVERY_MANIFEST_SHA256 = {recovery_manifest_sha256!r}
        SOURCE_SHA256 = {SOURCE_SHA256!r}
        SOURCE_CHECKPOINT_SHA256 = {CHECKPOINT_SHA256!r}
        PATCH_STEM = {stem!r}
        PATCH_PAYLOAD = None  # Optional .b64 path; the default payload is embedded.
        EMBEDDED_PATCH_BASE64 = {base64.b64encode(patch_bytes).decode('ascii')!r}
        RESUME_INDEX = None  # Default: exact checkpoint from Notebook Output version 2.
        OUTPUT_ZIP = None
        CHUNK_MANIFEST = None
        CHUNK_MANIFEST_SHA256 = {chunk_manifest_sha256!r}
        ALLOW_MAIN = True
        PREPARE_ONLY = False
        BASE = Path('/kaggle/working/native_paper_v4')
        ROOT = BASE / 'repo'
        WORKSPACE = BASE / 'campaign'
        PATCH_ROOT = Path('/kaggle/working/native_paper_v4_recovery_r2')
        ZIP_STAGE = Path('/kaggle/working/_native_paper_v4_r2_zip_input')
        CHUNK_ZIP = Path('/kaggle/working/_native_paper_v4_r2_chunk_input.zip')
        """
    unpack = _restore_utilities()+textwrap.dedent("""
        import io
        BASE.mkdir(parents=True, exist_ok=True)
        if PATCH_ROOT.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('R2 patch must remain outside the frozen source')
        if shutil.disk_usage(BASE).free < 6*1024**3:
            raise RuntimeError('Recovery requires at least 6 GiB free before restore')
        patch_archive = BASE.parent / (PATCH_STEM+'.zip')
        if not patch_archive.exists():
            payload = (Path(PATCH_PAYLOAD).read_bytes() if PATCH_PAYLOAD is not None
                       else EMBEDDED_PATCH_BASE64.encode('ascii'))
            if len(payload) > 24*1024**2:
                raise ValueError('Recovery payload exceeds its bound')
            data = base64.b64decode(payload, validate=True)
            if len(data) > 16*1024**2 or hashlib.sha256(data).hexdigest() != PATCH_SHA256:
                raise ValueError('Recovery payload checksum/size mismatch')
            copy_checked(io.BytesIO(data), patch_archive, len(data), PATCH_SHA256)
        if patch_archive.is_symlink() or checksum(patch_archive) != PATCH_SHA256:
            raise ValueError('Recovery archive changed')
        patch_manifest = extract_verified(patch_archive, PATCH_ROOT,
            'NATIVE_PAPER_V4_EXTERNAL_AFFINITY_RECOVERY_R2')
        recovery_manifest_path = PATCH_ROOT/'recovery_r2.json'
        if checksum(recovery_manifest_path) != RECOVERY_MANIFEST_SHA256:
            raise ValueError('Bound R2 manifest changed')
        recovery_manifest = json.loads(recovery_manifest_path.read_text())
        if (recovery_manifest['source_input_sha256'] != SOURCE_SHA256 or
                recovery_manifest['source_checkpoint_index_sha256'] != SOURCE_CHECKPOINT_SHA256):
            raise ValueError('R2 manifest targets a different frozen checkpoint')
        print('Verified external R2 patch:', PATCH_SHA256)
        """)
    restore = "\n".join(inspect.getsource(function) for function in (
        verify_checkpoint_header, _journal_record, _journal_stream_sha,
        verify_physical_journals, verify_archive_journals))+textwrap.dedent(r"""
        import importlib.util
        def load_external(name, filename):
            spec = importlib.util.spec_from_file_location(name, PATCH_ROOT/filename)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
        zip_input = load_external('_v4_r2_zip_input', 'native_paper_v4_zip_input.py')
        def checkpoint_candidate(path):
            value = json.loads(Path(path).read_text())
            actual = checksum(path)
            sidecar = Path(path).with_name(Path(path).name+'.sha256')
            sidecar_hash = sidecar.read_text().split()[0] if sidecar.is_file() else None
            descendant = verify_checkpoint_header(value, actual, recovery_manifest, sidecar_hash)
            return actual, descendant
        # Prefer an attached Notebook Output, with exact ancestry. Multiple
        # eligible inputs are ambiguous: select RESUME_INDEX explicitly.
        if RESUME_INDEX is None and OUTPUT_ZIP is None and CHUNK_MANIFEST is None:
            matching_indexes = []
            for candidate in sorted(Path('/kaggle/input').rglob('checkpoint_index.json')):
                try:
                    checkpoint_candidate(candidate)
                except (ValueError, OSError, KeyError, IndexError):
                    continue
                matching_indexes.append(candidate)
            if len(matching_indexes) > 1:
                raise ValueError('Multiple eligible checkpoints: attach only the latest Output or set RESUME_INDEX')
            if matching_indexes:
                RESUME_INDEX = matching_indexes[0]
        chunk_restore_receipt = None
        if RESUME_INDEX is None and OUTPUT_ZIP is None:
            if CHUNK_MANIFEST is None and CHUNK_MANIFEST_SHA256 is not None:
                matching_chunks = [p for p in sorted(Path('/kaggle/input').rglob('*resume_chunks*.json'))
                                   if p.is_file() and checksum(p) == CHUNK_MANIFEST_SHA256]
                if matching_chunks:
                    CHUNK_MANIFEST = matching_chunks[0]
            if CHUNK_MANIFEST is not None:
                if CHUNK_MANIFEST_SHA256 is None or checksum(CHUNK_MANIFEST) != CHUNK_MANIFEST_SHA256:
                    raise ValueError('Use the R2 notebook bound to this prepared chunk manifest')
                chunk_info = json.loads(Path(CHUNK_MANIFEST).read_text())
                if (chunk_info.get('source_input_sha256') != SOURCE_SHA256 or
                        chunk_info.get('checkpoint_index_sha256') != SOURCE_CHECKPOINT_SHA256):
                    raise ValueError('Chunk input is not the locked 893-attempt checkpoint')
                chunk_input = load_external('_v4_r2_chunk_input', 'native_paper_v4_chunk_input.py')
                chunk_restore_receipt = chunk_input.restore_chunk_input(Path(CHUNK_MANIFEST), CHUNK_ZIP,
                    expected_manifest_sha256=CHUNK_MANIFEST_SHA256)
                OUTPUT_ZIP = Path(chunk_restore_receipt['path'])
                print('Reconstructed verified checkpoint ZIP:', chunk_restore_receipt['parts'], 'parts')
        zip_input_staged = False
        if RESUME_INDEX is None:
            if OUTPUT_ZIP is None:
                raise ValueError('Add Input: long290604/native-paper-v4-recovery-r1-chunks, version 2; '
                                 'or set the prepared R2 checkpoint ZIP/chunks input')
            with zipfile.ZipFile(OUTPUT_ZIP) as input_archive:
                index_name, zip_index_sha, selected_files = zip_input._scan(input_archive, SOURCE_SHA256, None)
                zip_index = json.loads(input_archive.read(index_name))
                # _scan verifies the physical index sidecar before returning.
                verify_checkpoint_header(zip_index, zip_index_sha, recovery_manifest, zip_index_sha)
                checked = verify_archive_journals(input_archive, index_name, zip_index)
                print('Verified outer ZIP start journals:', checked)
            RESUME_INDEX = zip_input.prepare_zip_input(Path(OUTPUT_ZIP), ZIP_STAGE,
                SOURCE_SHA256, expected_index_sha256=zip_index_sha)
            zip_input_staged = True
        RESUME_INDEX = Path(RESUME_INDEX)
        RESTORE_SHA256, RESTORE_IS_DESCENDANT = checkpoint_candidate(RESUME_INDEX)
        input_index = json.loads(RESUME_INDEX.read_text())
        checked = verify_physical_journals(RESUME_INDEX.parents[2], input_index)
        print('Verified physical input start journals:', checked)
        old_source = RESUME_INDEX.parents[2]/'source_input.zip'
        source_archive = BASE/'source_input.zip'
        if checksum(old_source) != SOURCE_SHA256:
            raise ValueError('Input lacks the exact original source archive')
        if not source_archive.exists():
            with old_source.open('rb') as source:
                copy_checked(source, source_archive, old_source.stat().st_size, SOURCE_SHA256)
        if checksum(source_archive) != SOURCE_SHA256:
            raise ValueError('Local original source archive changed')
        extract_verified(source_archive, ROOT, 'TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA13_SOURCE')
        sys.path.insert(0, str(ROOT/'tools'))
        from kaggle_native_paper_v4_bundle import restore_checkpoint
        restored = restore_checkpoint(RESUME_INDEX, BASE, RESTORE_SHA256,
            source_sha256=SOURCE_SHA256, root=ROOT, workspace=WORKSPACE)
        if zip_input_staged:
            receipt = zip_input.cleanup_zip_input(ZIP_STAGE)
            receipt_path = WORKSPACE/'setup/r2_zip_input_restore.json'
            receipt_path.parent.mkdir(parents=True, exist_ok=True)
            receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        if chunk_restore_receipt is not None:
            if (CHUNK_ZIP.is_symlink() or CHUNK_ZIP.resolve().parent != BASE.parent.resolve() or
                    Path(chunk_restore_receipt['path']).absolute() != CHUNK_ZIP.absolute() or
                    CHUNK_ZIP.stat().st_size != chunk_restore_receipt['size_bytes'] or
                    checksum(CHUNK_ZIP) != chunk_restore_receipt['sha256']):
                raise ValueError('Temporary reconstructed ZIP ownership/hash changed')
            CHUNK_ZIP.unlink()
            (WORKSPACE/'setup/r2_chunk_input_restore.json').write_text(
                json.dumps(dict(chunk_restore_receipt, temporary_removed=True), indent=2)+'\n')
        main_manifest = json.loads((WORKSPACE/'manifests/main.json').read_text())
        def ids_digest(ids):
            return hashlib.sha256(json.dumps(sorted(ids), sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        started_ids = [e['trial_id'] for e in main_manifest['entries'] if e['execution_status'] != 'PLANNED']
        pending_ids = [e['trial_id'] for e in main_manifest['entries'] if e['execution_status'] == 'PLANNED']
        if not RESTORE_IS_DESCENDANT:
            for label, ids in [('started', started_ids), ('pending', pending_ids)]:
                if (len(ids) != recovery_manifest['initial_'+label+'_count'] or
                        ids_digest(ids) != recovery_manifest['initial_'+label+'_ids_sha256']):
                    raise ValueError('Restored '+label+' trial identities differ from locked R2 input')
        # R2's bound runner checks descendant origin journals and every retained
        # raw hash before accepting any never-started trial.
        print('Restored verified checkpoint:', RESTORE_SHA256, '| descendant:', RESTORE_IS_DESCENDANT)
        print('Preserved attempts:', len(started_ids), '| eligible never-started slots:', len(pending_ids))
        print('Payloads:', len(restored['payloads']), '| free GiB:', round(shutil.disk_usage(BASE).free/1024**3, 2))
        """)
    stage = """
        import kaggle_native_paper_v4 as bootstrap
        if not ALLOW_MAIN and not PREPARE_ONLY:
            raise ValueError('Continuing main requires ALLOW_MAIN=True')
        stage_error = None
        try:
            runtimes, env = bootstrap.isolated_runtimes(WORKSPACE, SESSION_STARTED, restore=True)
            env['LD_LIBRARY_PATH'] = str(ROOT/'.deps/install/lib')+os.pathsep+env.get('LD_LIBRARY_PATH', '')
            gate_receipts = {}
            def run_gate(stem, argv):
                log_dir = WORKSPACE/'logs/bootstrap'
                before = set(log_dir.glob(stem+'_*.json'))
                bootstrap.command(stem, argv, WORKSPACE, SESSION_STARTED, env=env)
                created = set(log_dir.glob(stem+'_*.json'))-before
                if len(created) != 1:
                    raise ValueError('Expected exactly one fresh regression receipt: '+stem)
                receipt = created.pop()
                if receipt.is_symlink() or not receipt.resolve().is_relative_to(WORKSPACE.resolve()):
                    raise ValueError('Regression receipt escapes workspace')
                gate_receipts[stem] = dict(path=receipt.relative_to(WORKSPACE).as_posix(),
                                           sha256=checksum(receipt))
            # Patch tests import the external module by absolute path. The
            # frozen worker, launcher and original source remain unchanged.
            run_gate('r2_affinity_regression', [runtimes['controller'],
                PATCH_ROOT/'test_affinity_launch_race.py', '--scheduler-path',
                PATCH_ROOT/'terrain_compare_scheduler_r2.py', '--root', ROOT])
            run_gate('r2_preservation_regression', [runtimes['controller'],
                PATCH_ROOT/'test_native_paper_v4_recovery_r2.py', '--runner-path',
                PATCH_ROOT/'native_paper_v4_recovery_r2.py', '--root', ROOT,
                '--patch-dir', PATCH_ROOT])
            bootstrap.write(WORKSPACE/'recovery/v4_affinity_recovery_v2/regression_gates.json',
                dict(manifest_sha256=RECOVERY_MANIFEST_SHA256, receipts=gate_receipts))
            argv = [runtimes['controller'], PATCH_ROOT/'native_paper_v4_recovery_r2.py',
                '--root', ROOT, '--workspace', WORKSPACE, '--patch-dir', PATCH_ROOT,
                '--manifest', recovery_manifest_path, '--manifest-sha256', RECOVERY_MANIFEST_SHA256,
                '--session-started-monotonic', SESSION_STARTED]
            if PREPARE_ONLY:
                argv += ['--prepare-only']
            if ALLOW_MAIN and not PREPARE_ONLY:
                argv += ['--allow-main']
            bootstrap.command('recovery_r2', argv, WORKSPACE, SESSION_STARTED, env=env)
        except BaseException as error:
            stage_error = error
            raise
        finally:
            bootstrap.retain_checkpoint(WORKSPACE, source_archive, stage_error=stage_error)
        print('Keep the complete Notebook Output: exports, shards, collection, and recovery evidence.')
        """
    cells = [_cell("markdown", """
        # Tiếp tục A3-T v4 — R2, giữ 893 lượt đã bắt đầu

        Add Input → Notebook Output → `long290604/native-paper-v4-recovery-r1-chunks`,
        chọn **version 2**. Chọn CPU đủ 4 vCPU rồi Run All. Patch R2 nhỏ đã được
        nhúng và khóa SHA-256 trong notebook; không cần upload lại checkpoint 1 GB.

        Notebook khôi phục đúng source, runtime và checkpoint mới nhất; kiểm thử
        bản sửa điều phối CPU ngoài source gốc, kiểm chứng 53 lượt còn thiếu verdict,
        rồi chỉ cho chạy 607 lượt chưa bắt đầu. Giữ nguyên cả 4 lượt đã bị ngắt;
        không chạy lại seed, không đổi planner, bản đồ, budget 20 giây hoặc ngưỡng an toàn.

        `ALLOW_MAIN=True`, `PREPARE_ONLY=False` là cấu hình tiếp tục đã được yêu cầu.
        Đặt `PREPARE_ONLY=True` để kiểm chứng 53 lượt, khóa bằng chứng bảo toàn 893 lượt
        và xuất checkpoint, không chạy planner. Wave đang dở chỉ đóng gói sau khi hoàn tất.
        Input ZIP/chunks dự phòng phải được chuẩn bị từ đúng checkpoint R2 đã khóa.
        Nếu phiên R2 bị ngắt, thêm Output mới nhất thay Input cũ; notebook chỉ nhận
        checkpoint kế thừa đúng chuỗi payload và bằng chứng R2, không chọn giữa nhiều Input.
        Giữ toàn bộ Output mới, kể cả khi notebook hết thời gian hoặc báo lỗi.
        """, 0), _cell("code", config, 1), _cell("code", unpack, 2),
        _cell("code", restore, 3), _cell("code", stage, 4)]
    result = dict(cells=cells, nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python", version="3.12"), accelerator="none"))
    for i, item in enumerate(cells):
        if item["cell_type"] == "code":
            compile("".join(item["source"]), f"recovery_r2_cell_{i}", "exec")
    return result


def _default_source_files():
    files = {key: ROOT/"tools"/name for key, name in PATCH_NAMES.items()}
    files["r1_manifest"] = ROOT/"artifacts/native_paper_v4_recovery_r1_chunks/recovery.json"
    files.update({"test_"+key: ROOT/"tests"/name for key, name in TEST_NAMES.items()})
    return files


def build(output, *, resume_manifest=None, scheduler_path=None, chunk_manifest=None,
          stem="native_paper_v4_recovery_r2", source_files=None):
    """Create immutable delivery files. ``source_files`` supplies synthetic unit inputs."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", stem):
        raise ValueError("Unsafe output stem")
    identity = default_manifest()
    if resume_manifest is not None:
        supplied = (dict(resume_manifest) if isinstance(resume_manifest, dict)
                    else json.loads(Path(resume_manifest).read_text(encoding="utf-8")))
        for key, expected in FROZEN_IDENTITIES.items():
            if supplied.get(key, expected) != expected:
                raise ValueError("R2 frozen identity differs: "+key)
        identity.update(supplied)
    _sha_text(identity.get("original_scheduler_sha256"), "original_scheduler_sha256")
    files = _default_source_files() if source_files is None else {k: Path(v) for k, v in source_files.items()}
    if scheduler_path is not None:
        files["scheduler"] = Path(scheduler_path)
    names = dict(PATCH_NAMES, **{"test_"+k: v for k, v in TEST_NAMES.items()})
    if set(files) != set(names):
        raise ValueError("Incomplete or unexpected R2 source inventory")
    content = {}
    for key, name in names.items():
        path = files[key]
        if path.is_symlink() or not path.is_file():
            raise ValueError("R2 input must be a regular file: "+str(path))
        content[name] = path.read_bytes()
        if name.endswith(".py"):
            compile(content[name], name, "exec")
        identity[key] = dict(path=name, sha256=_digest(content[name]))
    original_manifest = json.loads(content[PATCH_NAMES["r1_manifest"]])
    if original_manifest.get("source_input_sha256") != SOURCE_SHA256:
        raise ValueError("R1 manifest binds a different original source")
    for key in ("runner", "analyzer"):
        r2_key = "r1_runner" if key == "runner" else key
        if original_manifest[key] != identity[r2_key]:
            raise ValueError("R1 "+key+" bytes differ from its immutable manifest")
    chunk_sha = None
    if chunk_manifest is not None:
        chunk_manifest = Path(chunk_manifest)
        chunk = json.loads(chunk_manifest.read_text(encoding="utf-8"))
        if (chunk.get("source_input_sha256") != SOURCE_SHA256 or
                chunk.get("checkpoint_index_sha256") != CHECKPOINT_SHA256):
            raise ValueError("Fallback chunks target a different checkpoint")
        chunk_sha = sha(chunk_manifest)
        identity["fallback_chunk_manifest_sha256"] = chunk_sha
    identity["schema_version"] = 13
    identity["notebook_output_input"] = dict(owner="long290604", slug="native-paper-v4-recovery-r1-chunks", version=2)
    identity["generator_sources"] = {Path(__file__).name: sha(__file__), Path(r1_bundle.__file__).name: sha(r1_bundle.__file__)}
    manifest_bytes = encoded_json(identity)
    content["recovery_r2.json"] = manifest_bytes
    records = [dict(path=name, size_bytes=len(data), sha256=_digest(data), mode=0o644)
               for name, data in sorted(content.items())]
    bundle = dict(kind=KIND, schema_version=13, source_input_sha256=SOURCE_SHA256,
                  source_checkpoint_index_sha256=CHECKPOINT_SHA256, files=records)
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data in sorted(content.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 9, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
        archive.writestr("bundle.json", encoded_json(bundle))
    patch_bytes = archive_bytes.getvalue()
    if len(patch_bytes) > 16*1024**2:
        raise ValueError("R2 patch exceeds notebook decoder bound")
    book = notebook(patch_bytes, _digest(manifest_bytes), stem=stem, chunk_manifest_sha256=chunk_sha)
    artifacts = {stem+".zip": patch_bytes, stem+".b64": base64.b64encode(patch_bytes),
        stem+".zip.sha256": (_digest(patch_bytes)+"  "+stem+".zip\n").encode(),
        stem+".ipynb": encoded_json(book), "recovery_r2.json": manifest_bytes}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() or (output/name).is_symlink() for name in artifacts):
        raise ValueError("R2 delivery output already exists; refusing overwrite")
    for name, data in artifacts.items():
        with (output/name).open("xb") as stream:
            stream.write(data)
    return dict(archive=str(output/(stem+".zip")), notebook=str(output/(stem+".ipynb")),
                sha256=_digest(patch_bytes), manifest_sha256=_digest(manifest_bytes),
                bytes=len(patch_bytes), files=records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-manifest", type=Path, required=True)
    parser.add_argument("--scheduler-path", type=Path)
    parser.add_argument("--chunk-manifest", type=Path)
    parser.add_argument("--stem", default="native_paper_v4_recovery_r2")
    args = parser.parse_args(argv)
    result = build(args.output, resume_manifest=args.resume_manifest, scheduler_path=args.scheduler_path,
                   chunk_manifest=args.chunk_manifest, stem=args.stem)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
