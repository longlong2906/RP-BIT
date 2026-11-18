"""Checksummed schema-12 native-paper source bundles and retained Linux runtime checkpoints."""
from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from kaggle_a3_bundle import (LINUX_CASE_POLICY, PORTABLE_CASE_POLICY, archive, archive_path_key,
                              digest, extract, manifest_path_policy, safe_name, sha256,
                              validate_extraction_case_policy, verify)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_sem_putn_kaggle_v3"
KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA12_SOURCE"
CHECKPOINT_KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA12_CHECKPOINT"
SOURCE_DIRS = ("include", "src", "apps", "tools", "tests", "cmake", "patches", "diagnostics",
               "configs", "external", "notebooks")
SKIP = {"__pycache__", ".pytest_cache", ".git", ".venv"}


def source_entries(root=ROOT):
    root = Path(root)
    result = []
    for directory in SOURCE_DIRS:
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if any(part in SKIP or part.startswith("synthetic-scheduler-") for part in relative.parts):
                continue
            if directory == "diagnostics" and path.suffix not in (".py", ".cpp", ".hpp", ".h", ".cmake"):
                continue
            if path.is_symlink():
                raise ValueError("source symlinks are forbidden: " + str(relative))
            if path.is_file() and path.suffix not in (".pyc", ".o", ".obj", ".dll", ".exe"):
                result.append((relative.as_posix(), path))
    for name in ("CMakeLists.txt", "AGENTS.md", "requirements-cloud-benchmark.txt",
                 "requirements-terrain-compare-controller.in", "requirements-terrain-compare-controller.txt",
                 "requirements-terrain-compare-sgp.in", "requirements-terrain-compare-sgp.txt",
                 "requirements-native-paper-v3.in", "requirements-native-paper-v3.txt"):
        result.append((name, root / name))
    for name in ("EXPERIMENT_SPEC.md", "IMPLEMENTATION_PLAN.md", "STATUS.md", "KAGGLE_TERRAIN_COMPARE.md",
                 "TERRAIN_COMPARE_PROTOCOL.md", "NATIVE_PAPER_PROTOCOL.md", "KAGGLE_NATIVE_PAPER.md",
                 "KAGGLE_NATIVE_PAPER_V3.md", "NATIVE_PAPER_V3_PROTOCOL.md"):
        result.append(("docs/" + name, root / "docs" / name))
    locked = json.loads((root / "configs/dependencies.lock.json").read_text())
    for name, record in locked["archives"].items():
        path = root / ".deps" / (name + ".tar.gz")
        if sha256(path) != record["sha256"]:
            raise ValueError("pinned dependency archive mismatch: " + name)
        result.append((".deps/" + path.name, path))
    uv_wheel = root / ".deps/uv-0.9.7-linux.whl"
    if sha256(uv_wheel) != "8cf6bc2482d1293cc630f66b862b494c09acda9b7faff7307ef52667a2b3ad49":
        raise ValueError("pinned uv wheel mismatch")
    result.append((".deps/" + uv_wheel.name, uv_wheel))
    for relative in ("ompl-1.7.0/LICENSE", "boost_1_82_0/LICENSE_1_0.txt", "eigen-3.4.0/COPYING.MPL2"):
        path = root / ".deps" / relative
        result.append(("licenses/" + relative.replace("/", "-"), path))
    return result


def build_source(output, root=ROOT):
    output, root = Path(output), Path(root)
    manifest = archive(source_entries(root), output, KIND,
                       {"protocol_id": "a3t_sem_putn_kaggle_v3", "config_schema": 12,
                        "main_requires_explicit_flag": True, "contains_results": False})
    notebook = json.loads((root / "notebooks/native_paper_kaggle_v3.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["source"] = [line.replace("SOURCE_SHA256_PLACEHOLDER", sha256(output)) for line in cell["source"]]
    destination = output.with_suffix(".ipynb")
    if destination.exists():
        raise ValueError("notebook output already exists")
    destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # Kaggle can expand uploaded ZIPs. Upload the opaque .b64 alongside the notebook.
    payload = output.with_suffix(".b64")
    with output.open("rb") as source, payload.open("xb") as encoded:
        for chunk in iter(lambda: source.read(3 * 1024 * 1024), b""):
            encoded.write(base64.b64encode(chunk))
    return manifest


def export_checkpoint(workspace, output, *, root=ROOT, input_archive=None):
    workspace, root, output = Path(workspace).resolve(), Path(root).resolve(), Path(output)
    records, internal_links = [], []
    for path in workspace.rglob("*"):
        relative = path.relative_to(workspace)
        if "exports" in relative.parts or path.name == "controller.lock":
            continue
        if path.is_symlink():
            raise ValueError("campaign symlink refused")
        if path.is_file():
            records.append(("campaign/" + relative.as_posix(), path))
    runtime_dirs = ("build-native-paper-v3", ".deps/install", ".venv-native-paper-v3", ".managed-python")
    for directory in runtime_dirs:
        for path in (root / directory).rglob("*"):
            if path.suffix in (".o", ".obj", ".pyc") or "__pycache__" in path.parts:
                continue
            if path.is_symlink():
                resolved = path.resolve(strict=True)
                if root not in resolved.parents:
                    raise ValueError("runtime link escapes workspace: " + str(path))
                internal_links.append({"path": "repo/" + path.relative_to(root).as_posix(),
                                       "target": "repo/" + resolved.relative_to(root).as_posix(),
                                       "target_is_directory": resolved.is_dir(),
                                       "target_sha256": None if resolved.is_dir() else sha256(resolved)})
            if not path.is_file():
                continue
            resolved = path.resolve()
            if root not in resolved.parents:
                raise ValueError("runtime link escapes workspace: " + str(path))
            # ZIP contains regular bytes only. Verified metadata reconstructs
            # internal symlinks after extraction, preserving venv/$ORIGIN paths.
            records.append(("repo/" + path.relative_to(root).as_posix(), resolved))
    metadata = {"protocol_id": "a3t_sem_putn_kaggle_v3", "root_path": str(root),
                "workspace_path": str(workspace), "requires_same_runtime_identity": False, "runtime_policy": "native_paper_sessions_v3",
                "internal_runtime_symlinks": sorted(internal_links, key=lambda item: item["path"]),
                "contains_exact_runtime": (root / "build-native-paper-v3/terrain_benchmark").is_file()}
    if input_archive:
        records.append(("source_input.zip", Path(input_archive)))
        metadata["source_input_sha256"] = sha256(input_archive)
    policy = LINUX_CASE_POLICY if sys.platform.startswith("linux") else PORTABLE_CASE_POLICY
    return archive(records, output, CHECKPOINT_KIND, metadata, path_case_policy=policy)


def restore_checkpoint(path, target, expected_sha256, *, source_sha256, root=ROOT, workspace=None):
    manifest = verify(path, expected_sha256)
    if manifest["kind"] != CHECKPOINT_KIND:
        raise ValueError("not a schema-12 native-paper checkpoint")
    metadata = manifest["metadata"]
    if metadata.get("protocol_id") != PROTOCOL_ID or metadata.get("runtime_policy") != "native_paper_sessions_v3":
        raise ValueError("checkpoint protocol/runtime policy mismatch")
    if metadata.get("source_input_sha256") != source_sha256:
        raise ValueError("checkpoint source identity mismatch")
    if str(Path(root).resolve()) != metadata["root_path"] or (
            workspace is not None and str(Path(workspace).resolve()) != metadata["workspace_path"]):
        raise ValueError("restore requires the same absolute repo/workspace paths")
    target, root = Path(target).resolve(), Path(root).resolve()
    if (target / "repo").resolve() != root or (target / "campaign").resolve() != Path(metadata["workspace_path"]).resolve():
        raise ValueError("restore target must preserve absolute repo/workspace paths")
    validate_extraction_case_policy(manifest, target)
    policy = manifest_path_policy(manifest)
    links = metadata.get("internal_runtime_symlinks", [])
    seen = set()
    for link in links:
        relative, destination = safe_name(link["path"]), safe_name(link["target"])
        if not relative.startswith("repo/") or not destination.startswith("repo/") or relative == destination:
            raise ValueError("invalid internal runtime link metadata")
        key = archive_path_key(relative, policy)
        if key in seen:
            raise ValueError("duplicate internal runtime link")
        seen.add(key)
        alias, real = target / relative, target / destination
        if root not in alias.parent.resolve().parents and alias.parent.resolve() != root:
            raise ValueError("runtime alias parent escapes repository")
        if root not in real.resolve().parents:
            raise ValueError("runtime link target escapes repository")
        if alias.is_symlink():
            if alias.resolve() != real.resolve():
                raise ValueError("refusing to replace a different existing runtime link")
        elif alias.exists():
            if link["target_is_directory"] or not alias.is_file() or sha256(alias) != link["target_sha256"]:
                raise ValueError("refusing to replace unverified runtime alias")
    # The generic ZIP extractor intentionally refuses symlinks. Remove only
    # known, checked aliases; a retry can reconstruct them from the same capsule.
    for link in links:
        alias = target / link["path"]
        if alias.is_symlink():
            alias.unlink()
    result = extract(path, target, expected_sha256, allow_identical=True)
    for link in links:
        alias, real = target / link["path"], target / link["target"]
        if link["target_is_directory"]:
            if not real.is_dir() or alias.exists():
                raise ValueError("missing target or conflicting directory alias")
        else:
            if not real.is_file() or sha256(real) != link["target_sha256"]:
                raise ValueError("restored runtime target hash mismatch")
            if not alias.is_file() or sha256(alias) != link["target_sha256"]:
                raise ValueError("restored runtime alias hash mismatch")
            alias.unlink()
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(os.path.relpath(real, alias.parent), target_is_directory=link["target_is_directory"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    source = subs.add_parser("source")
    source.add_argument("--output", type=Path, required=True)
    check = subs.add_parser("verify")
    check.add_argument("archive", type=Path)
    export = subs.add_parser("export")
    export.add_argument("--workspace", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--input-archive", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "source":
        result = build_source(args.output)
    elif args.command == "verify":
        result = verify(args.archive)
    else:
        result = export_checkpoint(args.workspace, args.output, input_archive=args.input_archive)
    print(json.dumps({"kind": result["kind"], "files": len(result["files"]),
                      "files_sha256": result["files_sha256"]}))


if __name__ == "__main__":
    main()
