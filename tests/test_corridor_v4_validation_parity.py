"""Compare one/four Windows-spawn validators on identical retained raw trials.

Without --input this runs only the synthetic normalization unit tests.  With
--input it writes separate scratch spools/control receipts under --output and
never commits verdicts or other files into the supplied campaign.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))

# Deliberately retain source event times, solve/preprocessing timing, first-safe
# time, latency and all verdict fields.  Only newly measured oracle timings vary.
VOLATILE_VALIDATION_KEYS = {"validation_s", "validation_wall_s"}


def normalized_verdict(value):
    if isinstance(value, dict):
        return {key: normalized_verdict(item) for key, item in value.items()
                if key not in VOLATILE_VALIDATION_KEYS}
    if isinstance(value, list):
        return [normalized_verdict(item) for item in value]
    return value


def difference_paths(left, right, prefix=""):
    if type(left) is not type(right):
        return [prefix+":type"]
    if isinstance(left, dict):
        result = []
        for key in sorted(set(left)|set(right)):
            path = prefix+"."+key if prefix else key
            result.extend([path+":missing"] if key not in left or key not in right
                          else difference_paths(left[key], right[key], path))
        return result
    if isinstance(left, list):
        if len(left) != len(right):
            return [prefix+":length"]
        return [item for i, (a, b) in enumerate(zip(left, right))
                for item in difference_paths(a, b, prefix+"["+str(i)+"]")]
    return [] if left == right else [prefix]


def run_parity(input_directory, output_directory):
    from corridor_v4_execution import atomic_json, digest, file_hash, read, resource_preflight, run_spawn_lanes, utc_now
    from run_corridor_v4 import _validate_map, raw_inventory
    input_directory, output_directory = Path(input_directory).resolve(), Path(output_directory).resolve()
    if input_directory == output_directory or input_directory in output_directory.parents:
        raise ValueError("parity output must be outside the input campaign")
    if output_directory.exists():
        raise FileExistsError("refusing to overwrite parity evidence")
    manifest_path = input_directory/"manifest.json"
    manifest_hash = file_hash(manifest_path)
    manifest = read(manifest_path)
    if manifest.get("profile") != "smoke":
        raise ValueError("this bounded parity harness requires a smoke campaign")
    map_infos = manifest.get("maps", [])[:4]
    if len(map_infos) != 4:
        raise ValueError("four input maps are required for four spawn lanes")
    selected, frozen = [], {}
    for info in map_infos:
        entries = [entry for entry in manifest["entries"] if entry["map_id"] == info["map_id"]]
        match = None
        for entry in entries:
            directory = input_directory/"trials"/entry["trial_id"]
            request_path = directory/"request.json"
            if request_path.is_file() and read(request_path).get("safe_length", {}).get("policy_id") == "a3t_corridor_v4":
                match = (entry, directory)
                break
        if match is None:
            raise ValueError("no retained v4 trial for map "+info["map_id"])
        entry, directory = match
        inventory = read(directory/"raw_inventory.json")
        if raw_inventory(directory) != inventory:
            raise ValueError("input raw inventory mismatch: "+entry["trial_id"])
        frozen[entry["trial_id"]] = dict(directory=str(directory), raw_inventory=inventory,
            raw_inventory_sha256=file_hash(directory/"raw_inventory.json"),
            validation_sha256=file_hash(directory/"validation.json") if (directory/"validation.json").exists() else None)
        selected.append(dict(info=info, entry=entry, directory=str(directory), raw_inventory_sha256=digest(inventory)))
    config = read(input_directory/"config.json")
    reservation = config["execution"]["reserve_bytes_per_validation_worker"]
    admission = {count: resource_preflight(count, reservation) for count in (1, 4)}
    output_directory.mkdir(parents=True)
    evidence = dict(schema_version=1, status="RUNNING", created_utc=utc_now(),
        input_directory=str(input_directory), manifest_sha256=manifest_hash,
        selected_trial_ids=[row["entry"]["trial_id"] for row in selected],
        selection="first_v4_trial_in_schedule_for_each_of_first_four_manifest_maps",
        normalized_keys=sorted(VOLATILE_VALIDATION_KEYS), resource_admission=admission,
        no_planner_launch=True, no_input_mutation=True,
        source_sha256={str(path.relative_to(ROOT)):file_hash(path) for path in
            (Path(__file__),ROOT/"tools/corridor_v4_validation.py",ROOT/"tools/run_corridor_v4.py",ROOT/"tools/corridor_v4_execution.py")})
    atomic_json(output_directory/"receipt.json",evidence)
    if any(not value["admitted"] for value in admission.values()):
        evidence.update(status="BLOCKED_RESOURCES", finished_utc=utc_now())
        atomic_json(output_directory/"receipt.json",evidence)
        return evidence
    summaries = {}
    for count in (1, 4):
        prefix = output_directory/("workers_"+str(count))
        lanes = [[] for _ in range(count)]
        for index, row in enumerate(selected):
            lanes[index % count].append(dict(task_id=row["info"]["map_id"], info=row["info"],
                items=[{key:row[key] for key in ("entry","directory","raw_inventory_sha256")}],
                sources_sha256=digest(manifest["sources"]), spool=str(prefix/"spool")))
        summaries[count] = run_spawn_lanes(lanes,_validate_map,prefix/"control",cores=admission[count]["cores"])
        evidence["spawn_"+str(count)] = dict(status=summaries[count]["status"],
            max_concurrency=summaries[count]["max_concurrency"],
            pids=[result.get("child_pid") for result in summaries[count]["results"]])
        atomic_json(output_directory/"receipt.json",evidence)
        if summaries[count]["status"] != "COMPLETE":
            evidence.update(status="FAILED_INFRASTRUCTURE",finished_utc=utc_now())
            atomic_json(output_directory/"receipt.json",evidence)
            return evidence
    comparisons = []
    for row in selected:
        trial_id=row["entry"]["trial_id"]
        left=read(output_directory/"workers_1"/"spool"/(trial_id+".json"))
        right=read(output_directory/"workers_4"/"spool"/(trial_id+".json"))
        a,b=normalized_verdict(left["row"]),normalized_verdict(right["row"])
        differences=difference_paths(a,b)
        source=frozen[trial_id];directory=Path(source["directory"])
        preserved=(raw_inventory(directory)==source["raw_inventory"] and
            file_hash(directory/"raw_inventory.json")==source["raw_inventory_sha256"] and
            (file_hash(directory/"validation.json") if (directory/"validation.json").exists() else None)==source["validation_sha256"])
        comparisons.append(dict(trial_id=trial_id, identical=not differences, differences=differences,
            identity_equal=left["identity"]==right["identity"],input_preserved=preserved,
            correctness_errors=left["row"].get("correctness_errors",[]),
            normalized_sha256_1=digest(a),normalized_sha256_4=digest(b)))
    preserved_manifest=file_hash(manifest_path)==manifest_hash
    passed=(preserved_manifest and all(row["identical"] and row["identity_equal"] and row["input_preserved"] and
        not row["correctness_errors"] for row in comparisons) and
        evidence["spawn_1"]["max_concurrency"]==1 and evidence["spawn_4"]["max_concurrency"]==4 and
        len(set(evidence["spawn_4"]["pids"]))==4)
    evidence.update(status="PASS" if passed else "FAILED",comparisons=comparisons,
        manifest_preserved=preserved_manifest,finished_utc=utc_now())
    atomic_json(output_directory/"receipt.json",evidence)
    return evidence


class ParityNormalizationTests(unittest.TestCase):
    def test_only_new_validation_timing_is_removed(self):
        value=dict(validation_s=1.,first_safe_s=.1,preprocessing_s=2.,events=[dict(elapsed_s=.1)],
                   corridor=dict(L0_certificate=dict(validation_wall_s=3.,length=4.,cached=False)))
        normalized=normalized_verdict(value)
        self.assertEqual(normalized["first_safe_s"],.1)
        self.assertEqual(normalized["preprocessing_s"],2.)
        self.assertEqual(normalized["events"],[dict(elapsed_s=.1)])
        self.assertEqual(normalized["corridor"]["L0_certificate"],dict(length=4.,cached=False))
        self.assertEqual(difference_paths(normalized,normalized),[])
        other=normalized_verdict(dict(value,first_safe_s=.2))
        self.assertEqual(difference_paths(normalized,other),["first_safe_s"])


if __name__=="__main__":
    if "--input" not in sys.argv:
        unittest.main()
    else:
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--input",required=True,type=Path)
        parser.add_argument("--output",required=True,type=Path)
        args=parser.parse_args()
        result=run_parity(args.input,args.output)
        print(result["status"]+": "+str(args.output.resolve()/"receipt.json"))
        raise SystemExit(0 if result["status"]=="PASS" else 1)
