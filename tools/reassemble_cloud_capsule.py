"""Reassemble and verify an immutable Cloud capsule from ordinary-Git chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "a3-vs-all-cloud-capsule-chunks"
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_KEYS = {
    "schema_version",
    "artifact_type",
    "capsule_name",
    "capsule_sha256",
    "capsule_size_bytes",
    "chunk_size_bytes",
    "parts",
}
PART_KEYS = {"index", "name", "sha256", "size_bytes"}


class ChunkIntegrityError(RuntimeError):
    """Raised when chunk metadata or bytes do not match the immutable capsule."""


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ChunkIntegrityError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ChunkIntegrityError(f"non-finite JSON constant is forbidden: {value}")


def _load_manifest(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ChunkIntegrityError(f"manifest is not a regular file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ChunkIntegrityError(f"cannot read chunk manifest: {exc}") from exc
    if not isinstance(value, dict) or set(value) != MANIFEST_KEYS:
        raise ChunkIntegrityError("chunk manifest has an unexpected schema")
    return value


def _positive_integer(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ChunkIntegrityError(f"{label} must be a positive integer")
    return value


def _hash(value, label: str) -> str:
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ChunkIntegrityError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_part_path(root: Path, name: str) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or Path(name).name != name
    ):
        raise ChunkIntegrityError(f"unsafe chunk name: {name!r}")
    path = root / name
    if path.is_symlink() or not path.is_file():
        raise ChunkIntegrityError(f"chunk is not a regular file: {name}")
    return path


def _validated_parts(manifest_path: Path, manifest: dict) -> list[tuple[Path, dict]]:
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ChunkIntegrityError("unsupported chunk manifest schema_version")
    if manifest["artifact_type"] != ARTIFACT_TYPE:
        raise ChunkIntegrityError("unexpected chunk artifact_type")
    capsule_name = manifest["capsule_name"]
    if (
        not isinstance(capsule_name, str)
        or not capsule_name
        or capsule_name in {".", ".."}
        or "/" in capsule_name
        or "\\" in capsule_name
        or ":" in capsule_name
        or Path(capsule_name).name != capsule_name
    ):
        raise ChunkIntegrityError("capsule_name must be a safe basename")
    capsule_size = _positive_integer(manifest["capsule_size_bytes"], "capsule_size_bytes")
    chunk_size = _positive_integer(manifest["chunk_size_bytes"], "chunk_size_bytes")
    _hash(manifest["capsule_sha256"], "capsule_sha256")
    records = manifest["parts"]
    if not isinstance(records, list) or not records:
        raise ChunkIntegrityError("parts must be a non-empty list")
    expected_count = (capsule_size + chunk_size - 1) // chunk_size
    if len(records) != expected_count:
        raise ChunkIntegrityError("part count does not match capsule size and chunk size")

    result: list[tuple[Path, dict]] = []
    names: set[str] = set()
    total = 0
    for expected_index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != PART_KEYS:
            raise ChunkIntegrityError("chunk record has an unexpected schema")
        if record["index"] != expected_index:
            raise ChunkIntegrityError("chunk indices must be contiguous and ordered")
        name = record["name"]
        expected_name = f"{capsule_name}.part-{expected_index:03d}"
        if name != expected_name:
            raise ChunkIntegrityError(
                f"unexpected chunk name at index {expected_index}: {name!r}"
            )
        if not isinstance(name, str) or name.casefold() in names:
            raise ChunkIntegrityError("chunk names must be unique, including by case")
        names.add(name.casefold())
        size = _positive_integer(record["size_bytes"], f"parts[{expected_index}].size_bytes")
        if expected_index < len(records) - 1 and size != chunk_size:
            raise ChunkIntegrityError("every non-final chunk must have chunk_size_bytes bytes")
        if expected_index == len(records) - 1 and size > chunk_size:
            raise ChunkIntegrityError("final chunk exceeds chunk_size_bytes")
        _hash(record["sha256"], f"parts[{expected_index}].sha256")
        path = _safe_part_path(manifest_path.parent, name)
        if path.stat().st_size != size:
            raise ChunkIntegrityError(f"chunk size mismatch: {name}")
        total += size
        result.append((path, record))
    if total != capsule_size:
        raise ChunkIntegrityError("chunk sizes do not sum to capsule_size_bytes")
    return result


def reassemble_chunks(
    manifest_path: Path,
    output: Path,
    *,
    expected_sha256: str,
) -> dict:
    manifest_path = Path(manifest_path)
    output = Path(output)
    expected_sha256 = expected_sha256.lower()
    _hash(expected_sha256, "expected_sha256")
    manifest = _load_manifest(manifest_path)
    manifest_sha256 = _hash(manifest["capsule_sha256"], "capsule_sha256")
    if manifest_sha256 != expected_sha256:
        raise ChunkIntegrityError(
            f"manifest capsule SHA-256 mismatch: expected {expected_sha256}, got {manifest_sha256}"
        )
    parts = _validated_parts(manifest_path, manifest)
    if output.name != manifest["capsule_name"]:
        raise ChunkIntegrityError(
            f"output basename must be {manifest['capsule_name']!r}"
        )
    if manifest_path.parent.name != f"{manifest['capsule_name']}.parts":
        raise ChunkIntegrityError("manifest directory does not match capsule_name")
    if output.parent.is_symlink():
        raise ChunkIntegrityError(f"output parent must not be a symlink: {output.parent}")
    if output.is_symlink():
        raise ChunkIntegrityError(f"output must not be a symlink: {output}")
    if output.exists():
        if not output.is_file():
            raise ChunkIntegrityError(f"existing output is not a regular file: {output}")
        existing_size = output.stat().st_size
        existing_sha256 = _sha256_file(output)
        if (
            existing_size != manifest["capsule_size_bytes"]
            or existing_sha256 != expected_sha256
        ):
            raise ChunkIntegrityError(
                "existing output does not match the immutable capsule; refusing to overwrite"
            )
        for path, record in parts:
            actual_part_hash = _sha256_file(path)
            if actual_part_hash != record["sha256"]:
                raise ChunkIntegrityError(
                    f"chunk SHA-256 mismatch for {record['name']}: "
                    f"expected {record['sha256']}, got {actual_part_hash}"
                )
        return {
            "status": "PASS",
            "output": str(output),
            "capsule_sha256": existing_sha256,
            "capsule_size_bytes": existing_size,
            "parts": len(parts),
            "reused_existing": True,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.parent.is_symlink():
        raise ChunkIntegrityError(f"output parent must not be a symlink: {output.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.tmp-", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    whole_digest = hashlib.sha256()
    written = 0
    try:
        with temporary.open("wb") as destination:
            for path, record in parts:
                part_digest = hashlib.sha256()
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        part_digest.update(block)
                        whole_digest.update(block)
                        destination.write(block)
                        written += len(block)
                actual_part_hash = part_digest.hexdigest()
                if actual_part_hash != record["sha256"]:
                    raise ChunkIntegrityError(
                        f"chunk SHA-256 mismatch for {record['name']}: "
                        f"expected {record['sha256']}, got {actual_part_hash}"
                    )
            destination.flush()
            os.fsync(destination.fileno())
        actual_sha256 = whole_digest.hexdigest()
        if written != manifest["capsule_size_bytes"]:
            raise ChunkIntegrityError("reassembled capsule size mismatch")
        if actual_sha256 != expected_sha256:
            raise ChunkIntegrityError(
                f"reassembled capsule SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        os.link(temporary, output)
        return {
            "status": "PASS",
            "output": str(output),
            "capsule_sha256": actual_sha256,
            "capsule_size_bytes": written,
            "parts": len(parts),
            "reused_existing": False,
        }
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    args = parser.parse_args()
    try:
        result = reassemble_chunks(
            args.manifest, args.output, expected_sha256=args.expected_sha256
        )
    except (ChunkIntegrityError, FileExistsError, OSError) as exc:
        print(f"BLOCKED: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
