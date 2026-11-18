"""Lossless, verified map shards for a workspace-owned v4 campaign.

Only active/<wave>/<map> may be removed. Inputs, shared rasters, old shards and
runtime assets are never cleanup targets. A committed shard retains every file
byte and file mode; logical trial paths are restored unchanged on demand.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import uuid
import zipfile


GIB = 1024**3
MANIFEST = "__map_shard__.json"
SCHEMA = 1


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_json(value)).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _link(path):
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) or "__" in value:
        raise ValueError("unsafe/ambiguous wave or map identifier")
    return value


def _member(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name or
            PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or
            str(PurePosixPath(name)) != name):
        raise ValueError("unsafe shard member")
    return name


def _atomic_json(path, value):
    path = Path(path)
    # Keep the temporary component short on hosts with legacy MAX_PATH limits.
    temporary = path.parent/(".tmp-"+uuid.uuid4().hex[:12])
    with temporary.open("xb") as stream:
        stream.write(_json(value)+b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_parent(path)


def _sync_parent(path):
    # POSIX directory fsync makes the verified archive/index rename durable
    # before cleanup. Windows has no portable directory-fsync operation here.
    if os.name != "nt":
        descriptor = os.open(Path(path).parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _sync_file(path):
    # Windows _commit/fsync requires a writable descriptor.
    with Path(path).open("r+b") as stream:
        os.fsync(stream.fileno())


def _tree_size(path):
    path = Path(path)
    if not path.exists():
        return 0
    total = 0
    for folder, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if not _link(Path(folder)/name)]
        for name in files:
            item = Path(folder)/name
            if not _link(item) and item.is_file():
                total += item.stat().st_size
    return total


class DiskBudgetPolicy:
    """Admission only: never kills a trial, deletes data or changes a seed."""
    def __init__(self, *, disk_usage_fn=shutil.disk_usage, usage_fn=_tree_size,
                 output_soft_limit_bytes=16*GIB, workspace_soft_limit_bytes=16*GIB,
                 target_peak_bytes=12*GIB):
        self.disk_usage_fn, self.usage_fn = disk_usage_fn, usage_fn
        self.output_soft_limit_bytes = output_soft_limit_bytes
        self.workspace_soft_limit_bytes = workspace_soft_limit_bytes
        self.target_peak_bytes = target_peak_bytes

    def preflight(self, workspace, *, first_wave=False, raw_budget_bytes=1536*1024**2,
                  pack_budget_bytes=1920*1024**2, projected_output_bytes=None, output_path=None):
        for value in (raw_budget_bytes, pack_budget_bytes):
            if type(value) is not int or value < 0:
                raise ValueError("disk budgets must be nonnegative integer bytes")
        if projected_output_bytes is not None and (type(projected_output_bytes) is not int or projected_output_bytes < 0):
            raise ValueError("invalid projected output size")
        workspace = Path(workspace)
        output_path = Path(output_path) if output_path is not None else workspace/"shards"
        free = int(self.disk_usage_fn(workspace).free)
        used, output_used = int(self.usage_fn(workspace)), int(self.usage_fn(output_path))
        required = max(6*GIB if first_wave else 4*GIB,
                       raw_budget_bytes+pack_budget_bytes+2*GIB)
        projected_peak = used+raw_budget_bytes+pack_budget_bytes
        projected_output = output_used+pack_budget_bytes if projected_output_bytes is None else projected_output_bytes
        reason = ("DISK_FREE_RESERVE" if free < required else
                  "OUTPUT_SOFT_LIMIT" if projected_output > self.output_soft_limit_bytes else
                  "WORKSPACE_SOFT_LIMIT" if projected_peak > self.workspace_soft_limit_bytes else "OK")
        return dict(allowed=reason == "OK", reason=reason, free_bytes=free,
                    workspace_used_bytes=used, output_used_bytes=output_used,
                    required_free_bytes=required, projected_peak_bytes=projected_peak,
                    projected_output_bytes=projected_output, reserve_bytes=2*GIB,
                    target_peak_bytes=self.target_peak_bytes,
                    output_soft_limit_bytes=self.output_soft_limit_bytes,
                    workspace_soft_limit_bytes=self.workspace_soft_limit_bytes)


def preflight(workspace, *, policy=None, **kwargs):
    return (policy or DiskBudgetPolicy()).preflight(workspace, **kwargs)


class ArtifactStore:
    def __init__(self, workspace):
        requested = Path(workspace).absolute()
        if _link(requested):
            raise ValueError("workspace root must not be a link")
        requested.mkdir(parents=True, exist_ok=True)
        self.root = requested.resolve(strict=True)
        for name in ("active", "shards", "shards/.locks", "shards/.transactions", "shards/quarantine"):
            self._mkdir(self.root/name)
        self.index_path = self.root/"shards/index.json"

    def _guard(self, path, *, base=None):
        path = Path(path).absolute()
        base = self.root if base is None else Path(base).absolute()
        if ".." in path.parts or not path.is_relative_to(base) or not base.is_relative_to(self.root):
            raise ValueError("storage path escapes its allowed root")
        current = self.root
        if _link(current) or current.resolve(strict=True) != self.root:
            raise ValueError("workspace root changed")
        for part in path.relative_to(self.root).parts:
            current = current/part
            if _link(current):
                raise ValueError("storage links are forbidden")
            if current.exists() and not current.resolve().is_relative_to(self.root):
                raise ValueError("storage path resolves outside workspace")
        return path

    def _mkdir(self, path):
        self._guard(path)
        Path(path).mkdir(parents=True, exist_ok=True)
        self._guard(path)

    def _paths(self, wave_id, map_id):
        wave_id, map_id = _identifier(wave_id), _identifier(map_id)
        stem = wave_id+"__"+map_id
        return (self._guard(self.root/"active"/wave_id/map_id),
                self._guard(self.root/"shards"/(stem+".zip")),
                self._guard(self.root/"shards/.transactions"/(stem+".json")))

    @contextmanager
    def _lock(self, name):
        path = self._guard(self.root/"shards/.locks"/(name+".lock"))
        with path.open("a+b") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError("invalid storage lock")
            if info.st_size == 0:
                stream.write(b"\0");stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _scan(self, directory):
        self._guard(directory, base=self.root/"active")
        if not directory.is_dir():
            raise ValueError("active map is missing")
        inventory, directories = {}, []
        for folder, dirs, files in os.walk(directory, followlinks=False):
            for name in dirs:
                path = self._guard(Path(folder)/name, base=directory)
                if not stat.S_ISDIR(path.stat().st_mode):
                    raise ValueError("non-directory active member")
                directories.append(_member(path.relative_to(directory).as_posix()))
            for name in files:
                path = self._guard(Path(folder)/name, base=directory)
                info = path.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise ValueError("nonregular/hardlinked active file")
                relative = _member(path.relative_to(directory).as_posix())
                if relative == MANIFEST:
                    raise ValueError("reserved shard manifest name")
                digest = file_hash(path)
                after = path.stat()
                if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
                        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
                    raise ValueError("active file changed during inventory")
                inventory[relative] = dict(sha256=digest, size_bytes=info.st_size,
                                           mode=stat.S_IMODE(info.st_mode))
        return dict(sorted(inventory.items())), sorted(directories)

    def _read_index(self):
        self._guard(self.index_path)
        if not self.index_path.exists():
            return dict(schema_version=SCHEMA, entries={})
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if value.get("schema_version") != SCHEMA or not isinstance(value.get("entries"), dict):
            raise ValueError("invalid shard index")
        return value

    def _commit(self, receipt):
        key = receipt["wave_id"]+"/"+receipt["map_id"]
        with self._lock("index"):
            index = self._read_index()
            existing = index["entries"].get(key)
            if existing is not None:
                if existing != receipt:
                    raise ValueError("committed shard identity differs")
                return
            index["entries"][key] = receipt
            self._guard(self.index_path)
            _atomic_json(self.index_path, index)

    def _verify(self, archive, wave_id, map_id, *, expected_sha256=None):
        self._guard(archive, base=self.root/"shards")
        if not archive.is_file() or archive.stat().st_nlink != 1:
            raise ValueError("missing/nonregular shard archive")
        digest = file_hash(archive)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("shard SHA-256 mismatch")
        with zipfile.ZipFile(archive) as source:
            members = source.infolist()
            names = set()
            for item in members:
                plain = item.filename[:-1] if item.is_dir() else item.filename
                _member(plain)
                if item.filename != item.orig_filename or item.filename in names or item.flag_bits & 1:
                    raise ValueError("duplicate/ambiguous/encrypted shard member")
                names.add(item.filename)
                kind = stat.S_IFMT(item.external_attr >> 16)
                if kind != (stat.S_IFDIR if item.is_dir() else stat.S_IFREG):
                    raise ValueError("nonregular shard entry")
            if MANIFEST not in names or source.getinfo(MANIFEST).file_size > 64*1024**2:
                raise ValueError("missing/oversize shard manifest")
            value = json.loads(source.read(MANIFEST))
            if (value.get("schema_version") != SCHEMA or value.get("wave_id") != wave_id or
                    value.get("map_id") != map_id or not isinstance(value.get("inventory"), dict) or
                    not isinstance(value.get("directories"), list)):
                raise ValueError("wrong shard identity/schema")
            inventory = value["inventory"]
            expected = {MANIFEST, *inventory, *(name+"/" for name in value["directories"])}
            if names != expected:
                raise ValueError("unregistered/missing shard members")
            for name, record in inventory.items():
                _member(name)
                if record.keys() != {"sha256", "size_bytes", "mode"}:
                    raise ValueError("invalid shard file record")
                item = source.getinfo(name)
                if item.is_dir() or item.file_size != record["size_bytes"] or stat.S_IMODE(item.external_attr >> 16) != record["mode"]:
                    raise ValueError("shard size/mode mismatch")
                computed = hashlib.sha256()
                with source.open(item) as stream:
                    for block in iter(lambda: stream.read(1024*1024), b""):
                        computed.update(block)
                if computed.hexdigest() != record["sha256"]:
                    raise ValueError("shard member SHA-256 mismatch")
            if source.testzip() is not None:
                raise ValueError("shard CRC mismatch")
        return value, digest

    def _cleanup(self, directory, inventory, directories):
        """Resume-safe subset removal; every removed byte has a verified shard."""
        self._guard(directory, base=self.root/"active")
        if not directory.exists():
            return
        actual, actual_dirs = self._scan(directory)
        if any(name not in inventory or record != inventory[name] for name, record in actual.items()) or not set(actual_dirs).issubset(directories):
            raise ValueError("refusing cleanup of new or modified active artifacts")
        for name, record in actual.items():
            path = self._guard(directory/name, base=directory)
            if file_hash(path) != record["sha256"] or not stat.S_ISREG(path.stat().st_mode):
                raise ValueError("active artifact changed before cleanup")
            path.unlink()
        for name in sorted(actual_dirs, key=lambda x: (x.count("/"), x), reverse=True):
            self._guard(directory/name, base=directory).rmdir()
        self._guard(directory, base=self.root/"active").rmdir()

    def pack_map(self, wave_id, map_id, cleanup=True):
        directory, archive, journal = self._paths(wave_id, map_id)
        with self._lock(wave_id+"__"+map_id):
            with self._lock("index"):
                prior = self._read_index()["entries"].get(wave_id+"/"+map_id)
            if prior is not None:
                if prior.get("archive") != archive.relative_to(self.root).as_posix():
                    raise ValueError("indexed archive path differs")
                value, digest = self._verify(archive, wave_id, map_id, expected_sha256=prior["sha256"])
                if value["inventory"] != prior["inventory"] or value["directories"] != prior["directories"]:
                    raise ValueError("indexed shard inventory differs")
                if cleanup:
                    self._cleanup(directory, value["inventory"], value["directories"])
                return prior

            inventory, directories = self._scan(directory) if directory.exists() else ({}, [])
            expected = dict(schema_version=SCHEMA, wave_id=wave_id, map_id=map_id,
                            inventory=inventory, directories=directories)
            pending = self._guard(archive.with_name(archive.name+".partial"))
            transaction = json.loads(journal.read_text()) if journal.exists() else None
            if archive.exists():
                value, digest = self._verify(archive, wave_id, map_id)
                if not ((directory.exists() and value == expected) or
                        (not directory.exists() and transaction and transaction.get("inventory_sha256") == _digest(value))):
                    raise ValueError("cannot adopt unproven archive without matching raw inventory")
                expected = value
            else:
                if pending.exists():
                    try:
                        value, digest = self._verify(pending, wave_id, map_id)
                        if not ((directory.exists() and value == expected) or
                                (not directory.exists() and transaction and transaction.get("inventory_sha256") == _digest(value))):
                            raise ValueError("partial inventory differs from retained raw")
                    except (ValueError, OSError, zipfile.BadZipFile, EOFError) as error:
                        quarantine = self._guard(self.root/"shards/quarantine"/("failed-"+uuid.uuid4().hex[:12]+".partial"))
                        os.replace(pending, quarantine)
                        _atomic_json(quarantine.with_name(quarantine.name+".json"), dict(reason=str(error), original=str(pending)))
                        if not directory.exists():
                            raise ValueError("invalid partial quarantined; original raw is unavailable") from error
                    else:
                        expected = value
                        _sync_file(pending)
                        os.replace(pending, archive)
                        _sync_parent(archive)
                if not archive.exists():
                    if not directory.exists():
                        raise ValueError("map has neither raw artifacts nor a recoverable shard")
                    _atomic_json(journal, dict(state="PACKING", inventory_sha256=_digest(expected), wave_id=wave_id, map_id=map_id))
                    with zipfile.ZipFile(pending, "x", allowZip64=True) as packed:
                        for name in directories:
                            item = zipfile.ZipInfo(name+"/")
                            item.create_system = 3;item.external_attr = (stat.S_IFDIR|0o755) << 16
                            packed.writestr(item, b"")
                        for name, record in inventory.items():
                            path = self._guard(directory/name, base=directory)
                            item = zipfile.ZipInfo(name)
                            item.create_system = 3;item.external_attr = (stat.S_IFREG|record["mode"]) << 16
                            item.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in (".npz", ".zip", ".gz") else zipfile.ZIP_DEFLATED
                            item._compresslevel = 1
                            with path.open("rb") as stream, packed.open(item, "w", force_zip64=True) as output:
                                info = os.fstat(stream.fileno())
                                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                                    raise ValueError("active input replaced by nonregular file")
                                shutil.copyfileobj(stream, output, length=1024*1024)
                        item = zipfile.ZipInfo(MANIFEST)
                        item.create_system = 3;item.external_attr = (stat.S_IFREG|0o644) << 16
                        item.compress_type = zipfile.ZIP_DEFLATED
                        packed.writestr(item, _json(expected))
                    value, digest = self._verify(pending, wave_id, map_id)
                    if value != expected or self._scan(directory) != (inventory, directories):
                        raise ValueError("raw artifacts changed during pack; no cleanup performed")
                    _sync_file(pending)
                    os.replace(pending, archive)
                    _sync_parent(archive)
            value, digest = self._verify(archive, wave_id, map_id)
            receipt = dict(status="PACKED", wave_id=wave_id, map_id=map_id,
                           archive=archive.relative_to(self.root).as_posix(), sha256=digest,
                           inventory=value["inventory"], directories=value["directories"],
                           file_count=len(value["inventory"]),
                           uncompressed_bytes=sum(v["size_bytes"] for v in value["inventory"].values()),
                           archive_bytes=archive.stat().st_size)
            _atomic_json(journal, dict(state="VERIFIED", inventory_sha256=_digest(value), archive_sha256=digest,
                                      wave_id=wave_id, map_id=map_id))
            self._commit(receipt)
            if cleanup:
                self._cleanup(directory, value["inventory"], value["directories"])
            return receipt

    @contextmanager
    def materialize_map(self, wave_id, map_id):
        directory, archive, _ = self._paths(wave_id, map_id)
        with self._lock(wave_id+"__"+map_id):
            with self._lock("index"):
                receipt = self._read_index()["entries"].get(wave_id+"/"+map_id)
            if receipt is None:
                self._scan(directory)
                yield directory  # Unsealed active maps may still receive validation outputs.
                return
            value, _ = self._verify(archive, wave_id, map_id, expected_sha256=receipt["sha256"])
            if value["inventory"] != receipt["inventory"] or value["directories"] != receipt["directories"]:
                raise ValueError("indexed materialization inventory differs")
            restored = not directory.exists()
            if directory.exists():
                if self._scan(directory) != (value["inventory"], value["directories"]):
                    raise ValueError("existing active map differs from committed shard")
            else:
                self._mkdir(directory)
                with zipfile.ZipFile(archive) as packed:
                    for name in value["directories"]:
                        self._mkdir(self._guard(directory/name, base=directory))
                    for name, record in value["inventory"].items():
                        target = self._guard(directory/name, base=directory)
                        self._mkdir(target.parent)
                        with packed.open(name) as source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, length=1024*1024)
                        os.chmod(target, record["mode"])
                if self._scan(directory) != (value["inventory"], value["directories"]):
                    raise ValueError("restored map verification failed; retained for diagnosis")
            try:
                yield directory
            finally:
                if restored:
                    if self._scan(directory) != (value["inventory"], value["directories"]):
                        raise ValueError("materialized artifacts changed; preserving the restored directory")
                    self._cleanup(directory, value["inventory"], value["directories"])
