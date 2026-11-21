"""A5 verdict: the registered criteria of the A5 plan (section 3) and the time on both sides of the solve clock.

Reads the verdicts of one phase with the merge of tools/analyze_rp_bit.py (a strict verdict replaces the fast one) and
the report.json that tools/analyze_rp_bit.py wrote for the same phase; never runs or selects a planner. The worker-side
detail (time split, seed) is added when the phase still holds its trial directories.

  python -B tools/analyze_rp_bit_a5.py --input PHASE_DIR --report REPORT_DIR --output OUT_DIR

Registered criteria for the primary A5 arm (paired over the slots where both methods end on a final safe path):
  1  correctness: every planned trial has a verdict, no correctness error, the primary was strictly validated (its
     PSR is reported here and judged in 2a);
  2a the ten checks of plan 5.3 against SEM-RRT* (criteria of report.json);
  2b against the locked A4: one-sided upper 95% bound of the pooled relative J30 difference <= +0.1% (registered
     contrast of report.json);
  3  speed: mean SEM-RRT* solve time / mean A5 solve time >= 5, and A5 faster in >= 90% of the pairs.
Time is the solve time. The time before the solve clock (input loading + SEM field) is reported beside it only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean, median

from analyze_rp_bit import load
from corridor_v4_execution import atomic_json as dump, read
from rp_bit_protocol import A5, NATIVE

A4 = "a4_rpbit_l33_m20"
J30_MARGIN, SPEED_RATIO, FASTER_SHARE = 1e-3, 5., .9
SEED_ORIGIN = "a5_lattice_seed_refined"
CHECKS = ("psr_pooled", "psr_every_family", "sti_lower", "time_lower", "j_sem_mean_lower", "j_sem_upper95_below_zero",
          "apl_noninferior", "pri_noninferior", "apl_target_lower", "pri_target_lower")


def spread(values):
    return dict(mean=fmean(values), median=median(values), min=min(values), max=max(values)) if values else None


def before_solve(verdict):
    """Everything before the solve clock: input loading and the SEM field construction of either pipeline."""
    return (verdict.get("preprocessing_s") or 0.) + (verdict.get("field_build_s") or 0.)


def solve(verdict):
    return verdict["solve_time_s"]


def whole(verdict):
    return before_solve(verdict) + verdict["solve_time_s"]


def speed(rows, method, slots, family, clock):
    """SEM-RRT* time over the method's time on the slots where both final paths are safe, pooled and per family."""
    pairs = [s for s in slots if rows[method, *s]["success"] and rows[NATIVE, *s]["success"]]

    def summary(subset):
        if not subset:  # e.g. a family where SEM-RRT* or the method failed every slot
            return dict(pairs=0, sem_mean=None, method_mean=None, ratio_of_means=None, faster_pairs=0,
                        median_pair_ratio=None, min_pair_ratio=None)
        mine, theirs = [clock(rows[method, *s]) for s in subset], [clock(rows[NATIVE, *s]) for s in subset]
        return dict(pairs=len(subset), sem_mean=fmean(theirs), method_mean=fmean(mine),
                    ratio_of_means=fmean(theirs) / fmean(mine), faster_pairs=sum(a < b for a, b in zip(mine, theirs)),
                    median_pair_ratio=median(b / a for a, b in zip(mine, theirs)),
                    min_pair_ratio=min(b / a for a, b in zip(mine, theirs)))
    out = summary(pairs)
    out["by_family"] = {name: summary([s for s in pairs if family[s[0]] == name]) for name in dict.fromkeys(family.values())}
    return out


def j30_against(rows, method, baseline, slots):
    """Final J30 of the method against the baseline, slot by slot."""
    pairs = [s for s in slots if rows[method, *s]["success"] and rows[baseline, *s]["success"]]
    relative = [rows[method, *s]["final_j_sem"] / rows[baseline, *s]["final_j_sem"] - 1. for s in pairs]
    if not pairs:
        return dict(pairs=0)
    worst = max(zip(relative, pairs))
    return dict(pairs=len(pairs), relative_pooled=sum(rows[method, *s]["final_j_sem"] for s in pairs)
                / sum(rows[baseline, *s]["final_j_sem"] for s in pairs) - 1.,
                worse_by_more_than_0p1pct=sum(x > 1e-3 for x in relative),
                better_by_more_than_0p1pct=sum(x < -1e-3 for x in relative),
                worst_relative=worst[0], worst_slot=f"{worst[1][0]}__s{worst[1][1]}")


def worker_detail(trials, rows, method, slots, family):
    """Time split, first publication and seed of the final-safe runs (A4 and A5 publish the same events)."""
    first_time, first_ratio, publications = [], [], []
    timing, seed_status, seed_final, seed_gain, per_family_seed = {}, {}, 0, [], {}
    for s in slots:
        verdict = rows[method, *s]
        if not verdict["success"]:
            continue
        directory = trials / verdict["trial_id"]
        lines = (directory / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        trace = [e for e in map(json.loads, filter(None, lines)) if e.get("event_type") == "improvement"]
        first_time.append(trace[0]["elapsed_s"])
        first_ratio.append(trace[0]["objective_value"] / trace[-1]["objective_value"])
        publications.append(len(trace))
        worker = read(directory / "worker.json")
        if "a5_timing" not in worker:
            continue
        # seed_s includes the refinement of the seed, which refine_s counts too; rest_s is the solve outside
        # the setup and every refinement (BIT* search and the lattice A* of the seed).
        wall, t = worker["actual_solve_wall_s"], worker["a5_timing"]
        for key, value in (("wall_s", wall), ("setup_s", t["setup_s"]), ("seed_s", t["seed_s"]), ("refine_s", t["refine_s"]),
                           ("rest_s", wall - t["setup_s"] - t["refine_s"])):
            timing.setdefault(key, []).append(value)
        status = worker["seed_status"]
        seed_status[status] = seed_status.get(status, 0) + 1
        per_family_seed.setdefault(family[s[0]], {}).setdefault(status, 0)
        per_family_seed[family[s[0]]][status] += 1
        if trace[0].get("event_origin") == SEED_ORIGIN:
            seed_final += trace[-1].get("event_origin") == SEED_ORIGIN
            seed_gain.append(1. - trace[-1]["objective_value"] / trace[0]["objective_value"])
    out = dict(first_publication=dict(time_s=spread(first_time), j_over_final=spread(first_ratio),
                                      runs_first_more_than_1pct_above_final=sum(x > 1.01 for x in first_ratio),
                                      publications_per_run=fmean(publications) if publications else None))
    if timing:
        out["timing_mean_s"] = {key: fmean(values) for key, values in timing.items()}
        out["seed"] = dict(status=seed_status, status_by_family=per_family_seed, runs_with_seed=len(seed_gain),
                           final_path_is_the_seed=seed_final, bitstar_improved_the_seed=len(seed_gain) - seed_final,
                           final_gain_over_seed=spread(seed_gain))
    return out


def build(phase, report):
    items, methods, _, sources = load([phase])
    mine = [(s["config_sha256"], s["planned"], s["validated"], s["verdicts"]) for s in sources]
    if mine != [(s["config_sha256"], s["planned"], s["validated"], s["verdicts"]) for s in report["sources"]]:
        raise ValueError("report.json was not written for these verdicts")
    manifest = read(phase / "manifest.json")
    primary = manifest["config"]["phases"][manifest["phase"]]["primary_method"]
    if methods.get(primary, {}).get("kind") != A5 or NATIVE not in methods or A4 not in methods:
        raise ValueError("the phase needs SEM-RRT*, the locked A4 and an A5 primary arm")
    if any(item["verdict"] is None for item in items.values()):
        rows = None
    else:
        rows = {key: dict(item["verdict"], trial_id=item["trial_id"]) for key, item in items.items()}
    slots = sorted({key[1:] for key in items})
    family = {item["map_id"]: item["family"] for item in items.values()}
    out = dict(phase=str(phase), phase_name=manifest["phase"], primary=primary, methods={})
    # Criterion 1 is judged on every slot; the others need a verdict for every trial.
    strict = [item["verdict"] for key, item in items.items() if key[0] == primary]
    first = dict(complete=report["complete"], correctness_failures=report["correctness_failures"],
                 primary_strictly_validated=all(v is not None and v.get("full_validation") is True for v in strict),
                 primary_safe=sum(bool(v and v["success"]) for v in strict), slots=len(slots))
    first["pass"] = first["complete"] and first["correctness_failures"] == 0 and first["primary_strictly_validated"]
    checks = next(c for c in report["criteria"] if c["method"] == primary)
    contrast = next(c for c in report["contrasts"] if [c["method"], c["baseline"]] == [primary, A4])
    j_sem = contrast["comparisons"][0]["metrics"]["j_sem"]
    upper = j_sem["relative_bootstrap"]["upper95_one_sided"]
    criteria = {"1_correctness": first,
                "2a_ten_checks_vs_sem": dict({key: checks[key] for key in CHECKS}, passed=sum(checks[key] for key in CHECKS),
                                             **{"pass": all(checks[key] for key in CHECKS)}),
                "2b_j30_vs_a4": dict(pairs=contrast["comparisons"][0]["safe_pairs"], relative_pooled=j_sem["relative_pooled"],
                                     relative_upper95_one_sided=upper, margin=J30_MARGIN, **{"pass": upper <= J30_MARGIN})}
    for name in methods:
        subset = [item["verdict"] for key, item in items.items() if key[0] == name and item["verdict"]]
        safe = [v for v in subset if v["success"]]
        entry = out["methods"][name] = dict(kind=methods[name]["kind"], success=len(safe), slots=len(slots))
        entry["terminations"] = {}
        for verdict in subset:
            reason = verdict.get("termination_reason")
            entry["terminations"][reason] = entry["terminations"].get(reason, 0) + 1
        entry["time"] = spread([solve(v) for v in safe])
        entry["before_solve"] = dict(total=spread([before_solve(v) for v in safe]),
                                     worker_load=spread([v.get("preprocessing_s") or 0. for v in safe]),
                                     field_build=spread([v.get("field_build_s") or 0. for v in safe]))
        entry["time_with_before_solve"] = spread([whole(v) for v in safe])
        if rows is None:
            continue
        if name != NATIVE:
            entry["speed_vs_sem"] = speed(rows, name, slots, family, solve)
            entry["speed_vs_sem"]["with_before_solve"] = speed(rows, name, slots, family, whole)
        if methods[name]["kind"] == A5:
            entry["j30_vs_a4"] = j30_against(rows, name, A4, slots)
        if name != NATIVE and (phase / "trials").is_dir():
            entry.update(worker_detail(phase / "trials", rows, name, slots, family))
    if rows is not None and out["methods"][primary]["speed_vs_sem"]["pairs"] == 0:
        criteria["3_speed_vs_sem"] = {"pass": False, "reason": "no slot where both end on a final safe path"}
    elif rows is not None:
        ratio = out["methods"][primary]["speed_vs_sem"]
        criteria["3_speed_vs_sem"] = dict(pairs=ratio["pairs"], sem_mean_s=ratio["sem_mean"], a5_mean_s=ratio["method_mean"],
                                          ratio_of_means=ratio["ratio_of_means"], faster_pairs=ratio["faster_pairs"],
                                          faster_share=ratio["faster_pairs"] / ratio["pairs"], ratio_min=SPEED_RATIO,
                                          share_min=FASTER_SHARE, **{"pass": ratio["ratio_of_means"] >= SPEED_RATIO
                                                                     and ratio["faster_pairs"] >= FASTER_SHARE * ratio["pairs"]})
    else:
        criteria["3_speed_vs_sem"] = {"pass": False, "reason": "missing verdicts"}
    out["criteria"] = criteria
    out["all_pass"] = all(c["pass"] for c in criteria.values())
    return out


def markdown(out):
    mark = lambda flag: "đạt" if flag else "**không đạt**"
    c = out["criteria"]
    lines = [f"# Phán quyết A5 — `{out['phase_name']}`", "",
             f"Nhánh chính `{out['primary']}`. Time là thời gian solve; phần trước đồng hồ solve báo riêng, không dùng để chấm.", "",
             "| Tiêu chí | Số liệu | Kết luận |", "|---|---|---|"]
    one = c["1_correctness"]
    lines.append(f"| 1. Đúng đắn | đủ verdict: {one['complete']}; lỗi correctness {one['correctness_failures']}; "
                 f"validate nghiêm ngặt: {one['primary_strictly_validated']}; PSR {one['primary_safe']}/{one['slots']} "
                 f"(chỉ báo; PSR được chấm ở 2a) | {mark(one['pass'])} |")
    lines.append(f"| 2a. Mười tiêu chí so với SEM | {c['2a_ten_checks_vs_sem']['passed']}/10 | {mark(c['2a_ten_checks_vs_sem']['pass'])} |")
    two = c["2b_j30_vs_a4"]
    lines.append(f"| 2b. Không kém A4 | ΔJ₃₀ gộp {100 * two['relative_pooled']:+.3f}%, cận trên 95% "
                 f"{100 * two['relative_upper95_one_sided']:+.3f}% (ngưỡng +0,1%) | {mark(two['pass'])} |")
    three = c["3_speed_vs_sem"]
    if "ratio_of_means" in three:
        lines.append(f"| 3. Tốc độ | {three['sem_mean_s']:.3f} / {three['a5_mean_s']:.3f} = {three['ratio_of_means']:.2f} (ngưỡng 5); "
                     f"nhanh hơn ở {three['faster_pairs']}/{three['pairs']} cặp = {100 * three['faster_share']:.1f}% "
                     f"(ngưỡng 90%) | {mark(three['pass'])} |")
    else:
        lines.append(f"| 3. Tốc độ | {three['reason']} | {mark(False)} |")
    lines += ["", f"Cả bốn: {mark(out['all_pass'])}.", "", "## Thời gian (giây, trung bình trên lượt có đường an toàn)", "",
              "| Nhánh | Solve | Trước solve | Trong đó: trong worker | Trong đó: trường SEM dựng trước worker | Tổng | "
              "SEM/nhánh (solve) | SEM/nhánh (tổng) | Nhanh hơn SEM (solve) |",
              "|---|---|---|---|---|---|---|---|---|"]
    for name, m in out["methods"].items():
        if not m["time"]:
            continue
        b, s = m["before_solve"], m.get("speed_vs_sem")
        ratios = (f"{s['ratio_of_means']:.2f} | {s['with_before_solve']['ratio_of_means']:.2f} | {s['faster_pairs']}/{s['pairs']}"
                  if s and s["pairs"] else "— | — | —")
        lines.append(f"| `{name}` | {m['time']['mean']:.3f} | {b['total']['mean']:.3f} | {b['worker_load']['mean']:.3f} | "
                     f"{b['field_build']['mean']:.3f} | {m['time_with_before_solve']['mean']:.3f} | {ratios} |")
    speed_rows = out["methods"][out["primary"]].get("speed_vs_sem")
    if speed_rows:
        lines += ["", f"## Tỉ số SEM / `{out['primary']}` theo họ bản đồ", "",
                  "| Họ | Cặp | Solve | Tổng (cả trước solve) | Nhanh hơn (solve) |", "|---|---|---|---|---|"]
        for name, row in speed_rows["by_family"].items():
            if not row["pairs"]:
                lines.append(f"| {name} | 0 | — | — | — |")
                continue
            lines.append(f"| {name} | {row['pairs']} | {row['ratio_of_means']:.2f} | "
                         f"{speed_rows['with_before_solve']['by_family'][name]['ratio_of_means']:.2f} | {row['faster_pairs']}/{row['pairs']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="directory (or report.json) of tools/analyze_rp_bit.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.report / "report.json" if args.report.is_dir() else args.report
    out = build(args.input, read(report_path))
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "a5_verdict.json", out)
    (args.output / "A5_VERDICT.md").write_text(markdown(out), encoding="utf-8")
    print(f"all_pass={out['all_pass']} -> {args.output}")


if __name__ == "__main__":
    main()
