"""Create a fresh, reviewable Cloud source checkout; never copy historic runs or binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ("include", "src", "apps", "tools", "tests", "cmake", "patches", "diagnostics", "configs", "docs", "prompts")
ROOT_FILES = ("CMakeLists.txt", "AGENTS.md", "requirements-cloud-benchmark.txt")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".git", ".venv"}
LICENSE_INPUTS = {
    "licenses/OMPL-LICENSE.txt": ".deps/ompl-1.7.0/LICENSE",
    "licenses/BOOST-LICENSE-1.0.txt": ".deps/boost_1_82_0/LICENSE_1_0.txt",
    **{f"licenses/Eigen-{name}": f".deps/eigen-3.4.0/{name}" for name in
       ("COPYING.README", "COPYING.MPL2", "COPYING.BSD", "COPYING.APACHE", "COPYING.GPL", "COPYING.LGPL", "COPYING.MINPACK")},
}
GITIGNORE = """/.deps/
/build*/
/.venv/
/data/*
!/data/a3_vs_all_v2/
/results/
/reports/
/logs/
/cloud-work/
__pycache__/
*.py[cod]
*.tmp
"""
README = """# BIT* 2.5D — Codex Cloud benchmark

Private research source for `a3_vs_all_v2`: five planners, 100 fixed maps,
three RNG seeds per planner/map, 30 seconds of solve budget per trial.
This is a new Linux collection, separate from historic Windows experiments.

Read `AGENTS.md`, `docs/A3_FIVE_PLANNER_CLOUD_V2_PROTOCOL.md`, and
`docs/CLOUD_BENCHMARK_RUNBOOK.md` before running the campaign. The SEM
comparator is **SEM-RRT*-adapted common-objective**, not the official
multi-objective implementation. No experimental ranking is asserted here.

Prepare a Python 3.12 virtual environment, install
`requirements-cloud-benchmark.txt`, then run:

```bash
.venv/bin/python tools/setup_cloud_benchmark.py --build-dir build-cloud --jobs 2
```

Setup builds and tests the pinned local OMPL library and performs only
correctness smoke. Collection requires the explicit benchmark flag and the
immutable global plan. Preserve every timeout, crash and invalid result.
"""


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def source_paths(root=ROOT):
    root = Path(root)
    paths = []
    for directory in SOURCE_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            raise ValueError("required source directory missing: " + directory)
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            if any(part in SKIP_PARTS for part in relative.parts) or path.suffix == ".pyc":
                continue
            if path.is_symlink():
                raise ValueError("source package forbids symlinks: " + relative.as_posix())
            if path.is_file():
                paths.append(path)
    for name in ROOT_FILES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("required root source missing or symlink: " + name)
        paths.append(path)
    return sorted(paths, key=lambda value: value.relative_to(root).as_posix())


def dataset_paths(root, dataset_root):
    root, dataset_root = Path(root).resolve(), Path(dataset_root).resolve()
    if dataset_root != root / "data/a3_vs_all_v2":
        raise ValueError("only the registered new data/a3_vs_all_v2 dataset may be packaged")
    sys.path.insert(0, str(root / "tools"))
    from experiment_common import load_config, read_json
    from cloud_campaign import validate_dataset
    config = load_config(root / "configs/a3_vs_all_v2.yaml", "combined")
    maps = read_json(dataset_root / "manifest.json")
    validate_dataset(config, maps, dataset_root, witnesses=False)
    if len(maps["maps"]) != 100:
        raise ValueError("Cloud dataset must contain exactly 100 registered maps")
    paths = [dataset_root / "manifest.json"]
    for entry in maps["maps"]:
        path = dataset_root / entry["path"]
        if path.is_symlink() or not path.is_file() or sha256(path) != entry["sha256"]:
            raise ValueError("registered map hash mismatch: " + entry["map_id"])
        paths += [path, path.with_suffix(".json")]
    expected = {path.resolve() for path in paths}
    actual = {path.resolve() for path in dataset_root.rglob("*") if path.is_file()}
    if expected != actual:
        raise ValueError("registered dataset contains unregistered files")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def package(output, root=ROOT, dataset_root=None):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if output == root or output in root.parents:
        raise ValueError("package output cannot contain the source checkout")
    if output.is_symlink() or (output.exists() and any(output.iterdir())):
        raise ValueError("package output must be new or empty; existing files are never overwritten")
    paths = source_paths(root)
    data_paths = dataset_paths(root, dataset_root) if dataset_root is not None else []
    paths += data_paths
    for relative in LICENSE_INPUTS.values():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("pinned dependency license missing: " + relative)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        relative = path.relative_to(root)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records.append({"path": relative.as_posix(), "size_bytes": target.stat().st_size, "sha256": sha256(target)})
    for relative, origin in LICENSE_INPUTS.items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / origin, target)
        records.append({"path": relative, "size_bytes": target.stat().st_size, "sha256": sha256(target)})
    for relative, content in {"README.md": README, ".gitignore": GITIGNORE,
                              ".gitattributes": "# Preserve source bytes and experiment identities across Windows/Linux.\n* -text\n"}.items():
        target = output / relative
        target.write_text(content, encoding="utf-8", newline="\n")
        records.append({"path": relative, "size_bytes": target.stat().st_size, "sha256": sha256(target)})
    records.sort(key=lambda record: record["path"])
    manifest = {"schema_version": 1, "classification": "SOURCE_AND_REGISTERED_NEW_DATASET" if data_paths else "CLOUD_BENCHMARK_SOURCE_ONLY",
                "files": records, "files_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "copied_planner_binaries": 0, "copied_historic_trials": 0,
                "dataset_inventory": [entry for entry in records if entry["path"].startswith("data/a3_vs_all_v2/")],
                "dataset_manifest_sha256": sha256(root / "data/a3_vs_all_v2/manifest.json") if data_paths else None,
                "excluded_roots": [".deps", "build*", "results", "reports", "data", "logs", "cloud_validation_repo"]}
    (output / "cloud_source_package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(output):
    output = Path(output).resolve()
    manifest = json.loads((output / "cloud_source_package.json").read_text(encoding="utf-8"))
    records = manifest["files"]
    expected = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if expected != manifest["files_sha256"]:
        raise ValueError("source package inventory hash mismatch")
    names = set()
    for entry in records:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in entry["path"] or entry["path"] in names:
            raise ValueError("invalid/duplicate source inventory path")
        names.add(entry["path"])
        path = output / relative
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
            raise ValueError("source package member changed: " + entry["path"])
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path, help="opt in to packaging exactly the registered new 100-map dataset")
    args = parser.parse_args()
    result = verify(args.output) if args.verify_only else package(args.output, dataset_root=args.dataset_root)
    print(json.dumps({"status": "PASS", "files": len(result["files"]), "source_inventory_sha256": result["files_sha256"],
                      "output": str(args.output.resolve()), "copied_planner_binaries": 0, "copied_historic_trials": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
