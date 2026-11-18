"""Summarize an existing schema-13 report; never run planners or rewrite raw data."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

METHODS = ("a3t_old", "a3t_safe", "sem_native")
LABELS = {"a3t_old": "A3-T cũ", "a3t_safe": "A3-T mới", "sem_native": "SEM-RRT*"}
NUMBERS = ("final_cost_terrain_m", "final_length_xy_m", "final_pri",
           "final_maximum_swept_footprint_slope_deg", "final_sharp_turn_count",
           "first_geometry_s", "first_safe_s", "first_geometry_latency_s",
           "first_safe_latency_s", "solve_time_s")
QUALITY = {"cost_terrain_m": "final_cost_terrain_m", "length_xy_m": "final_length_xy_m",
           "pri": "final_pri", "maximum_swept_footprint_slope_deg": "final_maximum_swept_footprint_slope_deg"}
TIMING = ("first_geometry_s", "first_safe_s", "first_geometry_latency_s", "first_safe_latency_s")


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values):
    values = [x for x in values if x is not None]
    return statistics.mean(values) if values else None


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_csv(path, records):
    keys = list(dict.fromkeys(key for row in records for key in row))
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def close(a, b):
    if a is None or b is None:
        assert a is b, (a, b)
    else:
        assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = read(source / "extraction_provenance.json")
    for item in provenance["files"]:
        path = source / Path(item["local"]).name
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"]
    summary = read(source / "summary.json")
    with (source / "trials.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key in NUMBERS:
            row[key] = float(row[key]) if row[key] else None
            assert row[key] is None or math.isfinite(row[key])
        for key in ("attained_safe", "success", "success_geometry"):
            assert row[key] in ("True", "False")
            row[key] = row[key] == "True"
        row["planner_seed"] = int(row["planner_seed"])
    original = {row["trial_id"]: row for row in summary["rows"]}
    assert len(original) == len(rows) == len({r["trial_id"] for r in rows}) == 1500
    assert summary["completion"] == "COMPLETE" and summary["correctness_failures"] == 0
    assert {r["backend"] for r in rows} == set(METHODS)
    for row in rows:
        other = original[row["trial_id"]]
        for key in ("backend", "map_id", "family", "query_id", "planner_seed", "execution_status", "success", "attained_safe"):
            assert row[key] == other[key], (row["trial_id"], key)
        for key in ("final_cost_terrain_m", "final_pri", "first_safe_s", "first_safe_latency_s"):
            close(row[key], other.get(key))
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["family"], row["map_id"], row["query_id"], row["planner_seed"])].append(row)
    assert len(groups) == 500 and all({r["backend"] for r in group} == set(METHODS) and len(group) == 3 for group in groups.values())
    assert len({r["map_id"] for r in rows}) == 50 and len({r["family"] for r in rows}) == 5
    assert collections.Counter(r["backend"] for r in rows) == dict.fromkeys(METHODS, 500)
    table_index = {(t["scope"], t["group"], t["backend"], t["selection"]): t for t in summary["tables"]}
    aggregate, families, maps = [], [], []
    for scope, field, destination in (("all", None, aggregate), ("family", "family", families), ("map", "map_id", maps)):
        for label in (["all"] if field is None else sorted({r[field] for r in rows})):
            for method in METHODS:
                group = [r for r in rows if r["backend"] == method and (field is None or r[field] == label)]
                valid = [r for r in group if r["success"]]
                attained = [r for r in group if r["attained_safe"]]
                table = table_index[(scope, label, method, "final_safe")]
                record = dict(scope=scope, group=label, backend=method, label=LABELS[method],
                    attempted=len(group), completed=sum(r["execution_status"] == "COMPLETED" for r in group),
                    final_safe=len(valid), geometry=sum(r["success_geometry"] for r in group),
                    attained_safe=len(attained), final_safe_rate=len(valid)/len(group),
                    first_safe_attainment_rate=len(attained)/len(group),
                    mean_first_safe_s=mean(r["first_safe_s"] for r in attained),
                    mean_first_safe_latency_s=mean(r["first_safe_latency_s"] for r in attained),
                    mean_first_geometry_s=mean(r["first_geometry_s"] for r in group),
                    sti=sum(r["final_sharp_turn_count"] for r in valid)/sum(r["final_cost_terrain_m"] for r in valid) if valid else None,
                    restricted_first_safe_mean_s=table["restricted_first_safe_mean_s"],
                    restricted_unavailable=table["restricted_unavailable"])
                assert table["success"] == len(valid) and table["attempted"] == len(group)
                close(record["final_safe_rate"], table["psr_paper"])
                close(record["sti"], table["sti"])
                close(record["mean_first_safe_s"], table["first_safe_all_attained"]["mean"])
                for metric, key in QUALITY.items():
                    record["mean_"+metric] = mean(r[key] for r in valid)
                    close(record["mean_"+metric], table[metric]["mean"])
                destination.append(record)
    pairs = []
    for competitor in ("a3t_old", "sem_native"):
        for group in groups.values():
            by_method = {r["backend"]: r for r in group}
            a, b = by_method["a3t_safe"], by_method[competitor]
            item = dict(competitor=competitor, family=a["family"], map_id=a["map_id"], query_id=a["query_id"], seed=a["planner_seed"],
                        both_final_safe=a["success"] and b["success"], both_attained_safe=a["attained_safe"] and b["attained_safe"],
                        same_timing_stratum=a["timing_stratum"] == b["timing_stratum"])
            for metric, key in QUALITY.items():
                item[metric+"_new"] = a[key] if item["both_final_safe"] else None
                item[metric+"_comparator"] = b[key] if item["both_final_safe"] else None
                item[metric+"_delta"] = a[key]-b[key] if item["both_final_safe"] else None
            for metric in TIMING:
                usable = item["same_timing_stratum"] and a[metric] is not None and b[metric] is not None
                item[metric+"_new"] = a[metric] if usable else None
                item[metric+"_comparator"] = b[metric] if usable else None
                item[metric+"_delta"] = a[metric]-b[metric] if usable else None
            pairs.append(item)
    contrasts = [p for p in summary["paired_contrasts"] if p["selection"] == "final_safe"]
    paired_details = []
    for contrast in contrasts:
        metric = contrast["metric"]
        if metric not in (*QUALITY, *TIMING):
            continue
        selected = [p for p in pairs if p["competitor"] == contrast["competitor"] and p.get(metric+"_delta") is not None]
        assert len(selected) == contrast["usable_pairs"]
        close(mean(p[metric+"_delta"] for p in selected), contrast["mean_difference"])
        a = mean(p[metric+"_new"] for p in selected)
        b = mean(p[metric+"_comparator"] for p in selected)
        paired_details.append(dict(**contrast, new_mean=a, comparator_mean=b,
            relative_change_pct=(a/b-1)*100 if b else None,
            new_lower=sum(p[metric+"_delta"] < 0 for p in selected),
            new_higher=sum(p[metric+"_delta"] > 0 for p in selected),
            equal=sum(p[metric+"_delta"] == 0 for p in selected)))
    safety = []
    for method in METHODS:
        method_rows = [r for r in original.values() if r["backend"] == method]
        events = [e for r in method_rows for e in r["events"]]
        selected_events = [next((e for e in reversed(r["events"]) if e["elapsed_s"] <= r["budget_s"]), None) for r in method_rows if r["success"]]
        assert all(e is not None and e["safe"] for e in selected_events)
        safety.append(dict(backend=method, snapshots=len(events), unsafe_snapshots=sum(not e["safe"] for e in events),
            safe_final_risk_max_mean=mean(e["sem_score"]["risk_max"] for e in selected_events),
            safe_final_risk_max_max=max(e["sem_score"]["risk_max"] for e in selected_events),
            safe_final_swept_slope_max=max(e["metrics"]["maximum_swept_footprint_slope_deg"] for e in selected_events),
            failed_status_count=sum(r["execution_status"] != "COMPLETED" for r in method_rows),
            formerly_safe_but_unsafe_final=sum(r["attained_safe"] and not r["success"] for r in method_rows)))
    attainment = []
    for method in METHODS:
        selected = [r for r in rows if r["backend"] == method]
        for clock in ("first_safe_s", "first_safe_latency_s"):
            for elapsed in (.001, .005, .01, .02, .05, .1, .25, .5, 1., 1.5, 2., 2.5, 3., 4., 5., 20.):
                attained = sum(r["attained_safe"] and r[clock] is not None and r[clock] <= elapsed for r in selected)
                attainment.append(dict(backend=method, clock=clock, elapsed_s=elapsed, attained=attained, attempted=len(selected), fraction=attained/len(selected)))
    result = dict(status="VERIFIED_EXISTING_REPORT", planned=1500, paired_blocks=500, maps=50, families=5, trials_per_method=500,
                  correctness_failures=summary["correctness_failures"], aggregate=aggregate, families_summary=families,
                  contrasts=contrasts, paired_details=paired_details, safety=safety,
                  source_provenance=provenance, python=sys.version, script_sha256=sha(__file__),
                  checks=dict(csv_matches_verdict_rows=True, all_final_aggregates_reconciled=True,
                              paired_means_and_coverage_reconciled=True, raw_modified=False, new_oracle_or_planner_runs=0),
                  inference=dict(bootstrap_unit=summary["bootstrap_unit"], replicates=summary["bootstrap_replicates"],
                                 confidence=summary["confidence"], seed=summary["bootstrap_seed"],
                                 ci_source="original Kaggle paired_contrasts; not newly fit"))
    write_json(output/"analysis.json", result)
    write_json(output/"plot_data.json", dict(aggregate=aggregate, pairs=pairs, times=[{k:r[k] for k in ("backend", "attained_safe", "first_safe_s", "first_safe_latency_s")} for r in rows]))
    for filename, records in (("aggregate.csv", aggregate), ("by_family.csv", families), ("by_map.csv", maps), ("paired_trials.csv", pairs), ("attainment.csv", attainment)):
        write_csv(output/filename, records)
    print(json.dumps(dict(status=result["status"], trials=len(rows), pairs=len(groups), report=str(output/"analysis.json"))))


if __name__ == "__main__":
    main()
