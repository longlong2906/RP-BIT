"""A5 verdict of a phase that runs only SEM-RRT* beside the primary A5 arm (heldout_a5_10s, 10 s cap).

Criteria 1, 2a and 3 of tools/analyze_rp_bit_a5.py with the same thresholds and helpers; criterion 2b (J30 against the
locked A4) is not judged because A4 is not run (user decision of 2026-11-22). Reads the verdicts of one phase and the
report.json that tools/analyze_rp_bit.py wrote for it; never runs or selects a planner.

  python -B tools/analyze_rp_bit_a5_sem.py --input PHASE_DIR --report REPORT_DIR --output OUT_DIR

  1  correctness: every planned trial has a verdict, no correctness error, the primary was strictly validated (its
     PSR is reported here and judged in 2a);
  2a the ten checks of plan 5.3 against SEM-RRT* (criteria of report.json);
  3  speed: mean SEM-RRT* solve time / mean A5 solve time >= 5, and A5 faster in >= 90% of the pairs.
Time is the solve time. The time before the solve clock (input loading + SEM field) is reported beside it only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from analyze_rp_bit import load
from analyze_rp_bit_a5 import CHECKS, FASTER_SHARE, SPEED_RATIO, before_solve, solve, speed, spread, whole, worker_detail
from corridor_v4_execution import atomic_json as dump, read
from rp_bit_protocol import A5, NATIVE


def build(phase, report):
    items, methods, _, sources = load([phase])
    mine = [(s["config_sha256"], s["planned"], s["validated"], s["verdicts"]) for s in sources]
    if mine != [(s["config_sha256"], s["planned"], s["validated"], s["verdicts"]) for s in report["sources"]]:
        raise ValueError("report.json was not written for these verdicts")
    manifest = read(phase / "manifest.json")
    primary = manifest["config"]["phases"][manifest["phase"]]["primary_method"]
    if methods.get(primary, {}).get("kind") != A5 or set(methods) != {NATIVE, primary}:
        raise ValueError("the phase needs exactly SEM-RRT* and an A5 primary arm")
    if any(item["verdict"] is None for item in items.values()):
        rows = None
    else:
        rows = {key: dict(item["verdict"], trial_id=item["trial_id"]) for key, item in items.items()}
    slots = sorted({key[1:] for key in items})
    family = {item["map_id"]: item["family"] for item in items.values()}
    out = dict(phase=str(phase), phase_name=manifest["phase"], primary=primary, methods={},
               not_judged={"2b_j30_vs_a4": "A4 is not run in this phase"})
    # Criterion 1 is judged on every slot; the others need a verdict for every trial.
    strict = [item["verdict"] for key, item in items.items() if key[0] == primary]
    first = dict(complete=report["complete"], correctness_failures=report["correctness_failures"],
                 primary_strictly_validated=all(v is not None and v.get("full_validation") is True for v in strict),
                 primary_safe=sum(bool(v and v["success"]) for v in strict), slots=len(slots))
    first["pass"] = first["complete"] and first["correctness_failures"] == 0 and first["primary_strictly_validated"]
    checks = next(c for c in report["criteria"] if c["method"] == primary)
    criteria = {"1_correctness": first,
                "2a_ten_checks_vs_sem": dict({key: checks[key] for key in CHECKS}, passed=sum(checks[key] for key in CHECKS),
                                             **{"pass": all(checks[key] for key in CHECKS)})}
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
        if rows is None or name == NATIVE:
            continue
        entry["speed_vs_sem"] = speed(rows, name, slots, family, solve)
        entry["speed_vs_sem"]["with_before_solve"] = speed(rows, name, slots, family, whole)
        if (phase / "trials").is_dir():
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
    lines = [f"# Phán quyết A5 so với SEM-RRT* — `{out['phase_name']}`", "",
             f"Nhánh chính `{out['primary']}`. Time là thời gian solve; phần trước đồng hồ solve báo riêng, không dùng để chấm. "
             "Tiêu chí 2b (so với A4) không chấm vì pha này không chạy A4.", "",
             "| Tiêu chí | Số liệu | Kết luận |", "|---|---|---|"]
    one = c["1_correctness"]
    lines.append(f"| 1. Đúng đắn | đủ verdict: {one['complete']}; lỗi correctness {one['correctness_failures']}; "
                 f"validate nghiêm ngặt: {one['primary_strictly_validated']}; PSR {one['primary_safe']}/{one['slots']} "
                 f"(chỉ báo; PSR được chấm ở 2a) | {mark(one['pass'])} |")
    lines.append(f"| 2a. Mười tiêu chí so với SEM | {c['2a_ten_checks_vs_sem']['passed']}/10 | {mark(c['2a_ten_checks_vs_sem']['pass'])} |")
    three = c["3_speed_vs_sem"]
    if "ratio_of_means" in three:
        lines.append(f"| 3. Tốc độ | {three['sem_mean_s']:.3f} / {three['a5_mean_s']:.3f} = {three['ratio_of_means']:.2f} (ngưỡng 5); "
                     f"nhanh hơn ở {three['faster_pairs']}/{three['pairs']} cặp = {100 * three['faster_share']:.1f}% "
                     f"(ngưỡng 90%) | {mark(three['pass'])} |")
    else:
        lines.append(f"| 3. Tốc độ | {three['reason']} | {mark(False)} |")
    lines += ["", f"Cả ba: {mark(out['all_pass'])}.", "", "## Thời gian (giây, trung bình trên lượt có đường an toàn)", "",
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
