"""Hash-bound source/notebook and incremental Kaggle SEM-exposure pilot checkpoints."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

from kaggle_a3_bundle import archive, sha256
import kaggle_native_paper_v4_bundle as retained

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3t_sem_exposure_pilot_v1"
KIND = "A3T_SEM_EXPOSURE_PILOT_V1_SOURCE"
CHECKPOINT_KIND = "A3T_SEM_EXPOSURE_PILOT_V1_CHECKPOINT_PAYLOAD"
INDEX_KIND = "A3T_SEM_EXPOSURE_PILOT_V1_CHECKPOINT_INDEX"
RUNTIME_POLICY = "sem_exposure_pilot_sessions_v1"
BUILD_DIR = "build-sem-exposure-pilot"
VENV_DIR = ".venv-sem-exposure-pilot"
RUNTIME_DIRS = (BUILD_DIR, ".deps/install", VENV_DIR, ".managed-python")


def source_entries(root=ROOT):
    root = Path(root)
    entries = retained.source_entries(root)
    for name in ("docs/A3T_SEM_EXPOSURE.md", "docs/KAGGLE_SEM_EXPOSURE_PILOT.md",
                 "docs/A3T_SEM_EXPOSURE_PILOT.md"):
        if (root/name).is_file():
            entries.append((name, root/name))
    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise ValueError("duplicate pilot source entry")
    return entries


def build_source(output, root=ROOT):
    root, output = Path(root), Path(output)
    if output.suffix != ".zip":
        raise ValueError("source output must end in .zip")
    destinations = [output, output.with_suffix(".b64"), output.with_suffix(".ipynb"),
                    output.with_suffix(".README.md"), output.with_suffix(".zip.sha256")]
    if any(path.exists() for path in destinations):
        raise FileExistsError("pilot bundle output already exists")
    result = archive(source_entries(root), output, KIND, dict(protocol_id=PROTOCOL_ID,
        config_schema=1, pilot_requires_explicit_flag=True, full_enabled=False, contains_results=False))
    checksum = sha256(output)
    notebook = json.loads((root/"notebooks/sem_exposure_pilot_kaggle.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        cell["source"] = [line.replace("SOURCE_SHA256_PLACEHOLDER", checksum) for line in cell.get("source", [])]
    output.with_suffix(".ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    with output.open("rb") as source, output.with_suffix(".b64").open("xb") as encoded:
        for chunk in iter(lambda: source.read(3*1024*1024), b""):
            encoded.write(base64.b64encode(chunk))
    output.with_suffix(".zip.sha256").write_text(checksum+"  "+output.name+"\n", encoding="ascii")
    readme = (root/"docs/KAGGLE_SEM_EXPOSURE_PILOT.md").read_text(encoding="utf-8")
    output.with_suffix(".README.md").write_text(readme+f"\nGói nguồn: `{output.name}`. SHA-256: `{checksum}`.\n",
                                               encoding="utf-8")
    return result


def export_checkpoint(workspace, output, *, root=ROOT, input_archive=None, **limits):
    return retained.export_checkpoint(workspace, output, root=root, input_archive=input_archive,
        protocol_id=PROTOCOL_ID, checkpoint_kind=CHECKPOINT_KIND, index_kind=INDEX_KIND,
        schema_version=1, runtime_policy=RUNTIME_POLICY, runtime_dirs=RUNTIME_DIRS,
        runtime_binary=BUILD_DIR+"/terrain_benchmark", nested_shards=True,
        runtime_excluded_prefixes=(BUILD_DIR+"/Testing",), **limits)


def restore_checkpoint(index_path, target, expected_sha256, *, source_sha256, root=ROOT,
                       workspace=None, external_root=None, **limits):
    # A hard interruption may leave starts newer than the last exported index.
    # Never silently roll those physical journals/shards back to an older chain.
    index_path = Path(index_path).resolve()
    if sha256(index_path) != expected_sha256:
        raise ValueError("checkpoint index SHA-256 mismatch")
    index = retained._index(index_path, protocol_id=PROTOCOL_ID, index_kind=INDEX_KIND)
    physical_root = Path(external_root).resolve() if external_root else index_path.parents[2]
    physical_campaign = physical_root/"campaign"
    for path in physical_campaign.rglob("*"):
        if not path.is_file() or "exports" in path.relative_to(physical_campaign).parts:
            continue
        relative = path.relative_to(physical_root).as_posix()
        evidence = (path.name in ("start_journal.jsonl", "start.json", "execution.json", "launch.json", "worker.json") or
                    (path.suffix == ".zip" and "shards" in path.parts))
        if evidence:
            record = index["inventory"].get(relative)
            if path.is_symlink() or record is None or sha256(path) != record["sha256"]:
                raise ValueError("physical attempt evidence is newer than or differs from checkpoint: "+relative)
    return retained.restore_checkpoint(index_path, target, expected_sha256,
        source_sha256=source_sha256, root=root, workspace=workspace, external_root=external_root,
        protocol_id=PROTOCOL_ID, checkpoint_kind=CHECKPOINT_KIND, index_kind=INDEX_KIND, **limits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_source(args.output), indent=2))


if __name__ == "__main__":
    main()
