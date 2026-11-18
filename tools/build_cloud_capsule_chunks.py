"""Split an immutable Cloud capsule into checksummed ordinary-Git chunks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


SCHEMA_VERSION = 1
ARTIFACT_TYPE = "a3-vs-all-cloud-capsule-chunks"
DEFAULT_CHUNK_SIZE_BYTES = 32 * 1024 * 1024
HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class ChunkBuildError(RuntimeError):
    """Raised when the capsule cannot be split without ambiguity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def build_chunks(
    capsule: Path,
    output_dir: Path,
    *,
    expected_sha256: str,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
) -> dict:
    capsule = Path(capsule)
    output_dir = Path(output_dir)
    expected_sha256 = expected_sha256.lower()
    if not HASH_PATTERN.fullmatch(expected_sha256):
        raise ChunkBuildError("expected_sha256 must be 64 lowercase hexadecimal characters")
    if type(chunk_size_bytes) is not int or chunk_size_bytes <= 0:
        raise ChunkBuildError("chunk_size_bytes must be a positive integer")
    if capsule.is_symlink() or not capsule.is_file():
        raise ChunkBuildError(f"capsule is not a regular file: {capsule}")
    if (
        capsule.name in {"", ".", ".."}
        or "/" in capsule.name
        or "\\" in capsule.name
        or ":" in capsule.name
    ):
        raise ChunkBuildError("capsule name must be a safe basename")
    capsule_size = capsule.stat().st_size
    if capsule_size <= 0:
        raise ChunkBuildError("capsule must not be empty")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    whole_digest = hashlib.sha256()
    parts: list[dict] = []
    try:
        with capsule.open("rb") as source:
            index = 0
            while True:
                payload = source.read(chunk_size_bytes)
                if not payload:
                    break
                name = f"{capsule.name}.part-{index:03d}"
                part_path = temporary / name
                part_path.write_bytes(payload)
                part_digest = hashlib.sha256(payload).hexdigest()
                whole_digest.update(payload)
                parts.append(
                    {
                        "index": index,
                        "name": name,
                        "sha256": part_digest,
                        "size_bytes": len(payload),
                    }
                )
                index += 1

        actual_sha256 = whole_digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ChunkBuildError(
                f"capsule SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
            )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "capsule_name": capsule.name,
            "capsule_sha256": actual_sha256,
            "capsule_size_bytes": capsule_size,
            "chunk_size_bytes": chunk_size_bytes,
            "parts": parts,
        }
        _write_json(temporary / "manifest.json", manifest)
        manifest_sha256 = _sha256_file(temporary / "manifest.json")
        (temporary / "manifest.json.sha256").write_text(
            f"{manifest_sha256}  manifest.json\n", encoding="ascii", newline="\n"
        )
        os.rename(temporary, output_dir)
        return {
            "status": "PASS",
            "output_dir": str(output_dir),
            "capsule_sha256": actual_sha256,
            "capsule_size_bytes": capsule_size,
            "chunk_size_bytes": chunk_size_bytes,
            "parts": len(parts),
            "manifest_sha256": manifest_sha256,
        }
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--chunk-size-bytes", type=int, default=DEFAULT_CHUNK_SIZE_BYTES
    )
    args = parser.parse_args()
    try:
        result = build_chunks(
            args.capsule,
            args.output_dir,
            expected_sha256=args.expected_sha256,
            chunk_size_bytes=args.chunk_size_bytes,
        )
    except (ChunkBuildError, FileExistsError, OSError) as exc:
        print(f"BLOCKED: {exc}")
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
