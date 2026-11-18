"""Regression tests for the Git-LFS-independent Cloud capsule transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_cloud_capsule_chunks as chunk_builder
import reassemble_cloud_capsule as chunk_reader


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class CloudCapsuleChunkTests(unittest.TestCase):
    def _build(self, root: Path, payload: bytes = bytes(range(251)) * 7):
        capsule = root / "capsule.zip"
        capsule.write_bytes(payload)
        chunks = root / f"{capsule.name}.parts"
        result = chunk_builder.build_chunks(
            capsule,
            chunks,
            expected_sha256=sha256_bytes(payload),
            chunk_size_bytes=257,
        )
        return capsule, chunks, result, payload

    def test_round_trip_checks_every_part_and_full_capsule(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_round_trip_") as temporary:
            root = Path(temporary)
            capsule, chunks, result, payload = self._build(root)
            manifest = json.loads((chunks / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(result["parts"], 7)
            self.assertEqual(manifest["capsule_sha256"], sha256_bytes(payload))
            self.assertEqual(sum(item["size_bytes"] for item in manifest["parts"]), len(payload))
            self.assertTrue(all(item["size_bytes"] <= 257 for item in manifest["parts"]))

            capsule.unlink()
            output = root / "capsule.zip"
            rebuilt = chunk_reader.reassemble_chunks(
                chunks / "manifest.json",
                output,
                expected_sha256=sha256_bytes(payload),
            )
            self.assertEqual(rebuilt["status"], "PASS")
            self.assertEqual(rebuilt["parts"], 7)
            self.assertEqual(output.read_bytes(), payload)

    def test_tampered_chunk_is_rejected_without_publishing_output(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_tamper_") as temporary:
            root = Path(temporary)
            _, chunks, _, payload = self._build(root)
            (root / "capsule.zip").unlink()
            manifest = json.loads((chunks / "manifest.json").read_text(encoding="utf-8"))
            part = chunks / manifest["parts"][0]["name"]
            damaged = bytearray(part.read_bytes())
            damaged[0] ^= 0xFF
            part.write_bytes(damaged)
            output = root / "capsule.zip"
            with self.assertRaisesRegex(chunk_reader.ChunkIntegrityError, "chunk SHA-256"):
                chunk_reader.reassemble_chunks(
                    chunks / "manifest.json",
                    output,
                    expected_sha256=sha256_bytes(payload),
                )
            self.assertFalse(output.exists())

    def test_manifest_rejects_unsafe_duplicate_and_out_of_order_parts(self):
        mutations = {
            "unsafe": lambda value: value["parts"][0].update(name="../escape"),
            "duplicate": lambda value: value["parts"][1].update(
                name=value["parts"][0]["name"].upper()
            ),
            "order": lambda value: value["parts"][0].update(index=1),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"cloud_chunks_{label}_"
            ) as temporary:
                root = Path(temporary)
                _, chunks, _, payload = self._build(root)
                (root / "capsule.zip").unlink()
                manifest_path = chunks / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(manifest)
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                with self.assertRaises(chunk_reader.ChunkIntegrityError):
                    chunk_reader.reassemble_chunks(
                        manifest_path,
                        root / "capsule.zip",
                        expected_sha256=sha256_bytes(payload),
                    )

    def test_existing_output_and_chunk_directory_are_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_no_overwrite_") as temporary:
            root = Path(temporary)
            _, chunks, _, payload = self._build(root)
            output = root / "capsule.zip"
            output.write_bytes(b"keep-output")
            with self.assertRaisesRegex(
                chunk_reader.ChunkIntegrityError, "refusing to overwrite"
            ):
                chunk_reader.reassemble_chunks(
                    chunks / "manifest.json",
                    output,
                    expected_sha256=sha256_bytes(payload),
                )
            self.assertEqual(output.read_bytes(), b"keep-output")

            with self.assertRaises(FileExistsError):
                chunk_builder.build_chunks(
                    root / "capsule.zip",
                    chunks,
                    expected_sha256=sha256_bytes(payload),
                    chunk_size_bytes=257,
                )

    def test_expected_hash_mismatch_leaves_no_chunk_directory(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_wrong_hash_") as temporary:
            root = Path(temporary)
            capsule = root / "capsule.zip"
            capsule.write_bytes(b"immutable")
            chunks = root / "chunks"
            with self.assertRaisesRegex(chunk_builder.ChunkBuildError, "SHA-256 mismatch"):
                chunk_builder.build_chunks(
                    capsule,
                    chunks,
                    expected_sha256="0" * 64,
                    chunk_size_bytes=4,
                )
            self.assertFalse(chunks.exists())

    def test_existing_exact_output_is_reused_idempotently(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_idempotent_") as temporary:
            root = Path(temporary)
            capsule, chunks, _, payload = self._build(root)
            before = capsule.stat()
            rebuilt = chunk_reader.reassemble_chunks(
                chunks / "manifest.json",
                capsule,
                expected_sha256=sha256_bytes(payload),
            )
            after = capsule.stat()
            self.assertTrue(rebuilt["reused_existing"])
            self.assertEqual(before.st_mtime_ns, after.st_mtime_ns)
            self.assertEqual(capsule.read_bytes(), payload)

    def test_existing_exact_output_does_not_hide_tampered_chunk(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_cached_tamper_") as temporary:
            root = Path(temporary)
            capsule, chunks, _, payload = self._build(root)
            manifest = json.loads((chunks / "manifest.json").read_text(encoding="utf-8"))
            part = chunks / manifest["parts"][0]["name"]
            damaged = bytearray(part.read_bytes())
            damaged[-1] ^= 0x01
            part.write_bytes(damaged)
            with self.assertRaisesRegex(chunk_reader.ChunkIntegrityError, "chunk SHA-256"):
                chunk_reader.reassemble_chunks(
                    chunks / "manifest.json",
                    capsule,
                    expected_sha256=sha256_bytes(payload),
                )
            self.assertEqual(capsule.read_bytes(), payload)

    def test_rejects_wrong_output_name_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory(prefix="cloud_chunks_manifest_strict_") as temporary:
            root = Path(temporary)
            capsule, chunks, _, payload = self._build(root)
            capsule.unlink()
            with self.assertRaisesRegex(chunk_reader.ChunkIntegrityError, "output basename"):
                chunk_reader.reassemble_chunks(
                    chunks / "manifest.json",
                    root / "other.zip",
                    expected_sha256=sha256_bytes(payload),
                )

            manifest_path = chunks / "manifest.json"
            original = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(
                original.replace(
                    "{",
                    '{"schema_version": 1,',
                    1,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(chunk_reader.ChunkIntegrityError, "duplicate JSON"):
                chunk_reader.reassemble_chunks(
                    manifest_path,
                    root / "capsule.zip",
                    expected_sha256=sha256_bytes(payload),
                )

    def test_missing_or_truncated_part_is_rejected_before_output(self):
        for label in ("missing", "truncated"):
            with self.subTest(label=label), tempfile.TemporaryDirectory(
                prefix=f"cloud_chunks_{label}_"
            ) as temporary:
                root = Path(temporary)
                capsule, chunks, _, payload = self._build(root)
                capsule.unlink()
                manifest = json.loads(
                    (chunks / "manifest.json").read_text(encoding="utf-8")
                )
                part = chunks / manifest["parts"][0]["name"]
                if label == "missing":
                    part.unlink()
                else:
                    part.write_bytes(part.read_bytes()[:-1])
                with self.assertRaises(chunk_reader.ChunkIntegrityError):
                    chunk_reader.reassemble_chunks(
                        chunks / "manifest.json",
                        capsule,
                        expected_sha256=sha256_bytes(payload),
                    )
                self.assertFalse(capsule.exists())

    def test_deployed_reassembler_matches_canonical_tool(self):
        canonical = ROOT / "tools" / "reassemble_cloud_capsule.py"
        deployed = ROOT / "cloud_validation_repo" / "tools" / "reassemble_cloud_capsule.py"
        self.assertEqual(canonical.read_bytes(), deployed.read_bytes())


if __name__ == "__main__":
    unittest.main()
