"""A5 ablation: the 2^4 factorial of the four A5 mechanisms (16 cells) beside the self-stop SEM-RRT* (heldout_a5_abl).

Reads the verdicts of one phase and the report.json that tools/analyze_rp_bit.py wrote for it; never runs a planner.

  python -B tools/analyze_rp_bit_a5_ablation.py --input PHASE_DIR --report REPORT_DIR --output OUT_DIR

Cells are identified by their switches, every other parameter equal: C1 = batch_growth with batch_min 500 (off: the
A4 batches, batch_min 200 without growth), C2 = fast_validity, C3 = lazy_refine, C4 = seed. Cell 0000 is A4 (bit for
bit), cell 1111 the A5 primary.

Registered analysis (logs/a4_rpbit_a5_ablation_20261122/registration.json), paired by slot on the slots where all 16
cells end on a final safe path, with the hierarchical map -> seed bootstrap of tools/analyze_rp_bit.py:
  - main effect of Ci: mean over the 8 cell pairs that differ only in Ci of (on - off);
  - interaction Ci x Cj: mean over the 4 contexts of (effect of Ci with Cj on) - (effect of Ci with Cj off);
  - Shapley share of T(1111) - T(0000): mean marginal contribution of Ci over the 24 orders in which the four
    mechanisms can be added (subset weights 1/4, 1/12, 1/12, 1/4), so the four shares add up to the total;
  - read directly: remove Ci from 1111, add Ci to 0000;
for the solve time (s), its natural logarithm, and J30 relative to cell 0000 of the same slot.
Reading rule: Ci "reduces Time" if the one-sided upper 95% bound of its main effect on Time is < 0; Ci "costs no
quality" if the upper bound of its main effect on relative J30 is <= +0.1%. There is no overall pass/fail.
SEM-RRT* is a reference only: SEM/cell solve-time ratio and the ten checks of report.json for every cell.
Identity checks: equal final paths in every pair of cells that differ only in C2 (a decision-identical shortcut) and
between cell 0000 and the locked A4 where the phase runs it (gate); pairs where a run hit its time budget are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import fmean, pstdev

from analyze_rp_bit import _interval, bootstrap, load
from corridor_v4_execution import atomic_json as dump, read
from rp_bit_protocol import A5, NATIVE

A4 = "a4_rpbit_l33_m20"
FACTORS = ("C1", "C2", "C3", "C4")
SWITCHES = ("batch_growth", "fast_validity", "lazy_refine", "seed")
BITS = tuple(itertools.product((0, 1), repeat=4))
FULL, NONE = (1, 1, 1, 1), (0, 0, 0, 0)
SHAPLEY_WEIGHT = {0: 1 / 4, 1: 1 / 12, 2: 1 / 12, 3: 1 / 4}
J30_MARGIN = 1e-3
CHECKPOINTS = ("0.1", "0.25", "0.5", "1")
SEED_ORIGIN = "a5_lattice_seed_refined"
SIGNATURE_FIELDS = ("success", "final_j_sem", "final_cost_terrain_m", "final_length_xy_m", "final_pri",
                    "final_vertex_count", "final_sharp_turn_count")


def label(bits):
    return "".join(map(str, bits))


def flip(bits, index, value):
    return bits[:index] + (value,) + bits[index + 1:]


def cells(methods):
    """Cell bits -> method name; refuses anything that is not the registered 2^4 factorial."""
    out, common = {}, None
    for name, spec in methods.items():
        if spec["kind"] != A5:
            continue
        bits = tuple(int(spec[key]) for key in SWITCHES)
        rest = {key: value for key, value in spec.items() if key not in SWITCHES + ("batch_min",)}
        if spec["batch_min"] != (500 if bits[0] else 200) or bits in out or common not in (None, rest):
            raise ValueError("not the registered A5 factorial: " + name)
        common, out[bits] = rest, name
    if set(out) != set(BITS):
        raise ValueError("the phase needs all 16 cells of the A5 factorial")
    return out


def effects(y):
    """Every registered estimand of one slot, from its 16 cell values y[bits]."""
    out = {"total": y[FULL] - y[NONE]}
    for i, name in enumerate(FACTORS):
        steps = {bits: y[flip(bits, i, 1)] - y[bits] for bits in BITS if bits[i] == 0}
        out["main_" + name] = fmean(steps.values())
        out["shapley_" + name] = sum(SHAPLEY_WEIGHT[sum(bits)] * step for bits, step in steps.items())
        out["remove_" + name] = y[flip(FULL, i, 0)] - y[FULL]
        out["add_" + name] = y[flip(NONE, i, 1)] - y[NONE]
        for j in range(i + 1, 4):
            out[f"interaction_{name}x{FACTORS[j]}"] = fmean(steps[bits] - steps[flip(bits, j, 0)]
                                                           for bits in steps if bits[j] == 1)
    return out


def metric_values(rows, cell, slot):
    """Per-slot values of the three registered metrics, by cell."""
    time = {bits: rows[cell[bits], *slot]["solve_time_s"] for bits in BITS}
    reference = rows[cell[NONE], *slot]["final_j_sem"]
    return dict(time=time, log_time={bits: math.log(value) for bits, value in time.items()},
                j_rel={bits: rows[cell[bits], *slot]["final_j_sem"] / reference - 1. for bits in BITS})


def estimate(rows, cell, slots, analysis):
    """Point estimates and bootstrap intervals of every estimand, pooled over the slots."""
    per_slot = []
    for slot in slots:
        values = {f"{metric}.{key}": value for metric, y in metric_values(rows, cell, slot).items()
                  for key, value in effects(y).items()}
        per_slot.append(dict(map_id=slot[0], values=values))
    out = {}
    for metric in ("time", "log_time", "j_rel"):
        # One bootstrap per metric keeps the arrays small; the same seed draws the same maps and seeds each time.
        keys = [key for key in per_slot[0]["values"] if key.startswith(metric + ".")]
        pairs = [dict(map_id=s["map_id"], sides=(s["values"], dict.fromkeys(keys, 0.))) for s in per_slot]
        sums, sizes = bootstrap(pairs, keys, analysis["bootstrap_replicates"], analysis["bootstrap_seed"])
        means = sums[0] / sizes[:, None]
        for index, key in enumerate(keys):
            out[key] = dict(estimate=fmean(s["values"][key] for s in per_slot), **_interval(means[:, index]))
    return out


def by_family(rows, cell, slots, family):
    """Shapley shares of the solve time per family (point estimates)."""
    out = {}
    for name in dict.fromkeys(family[s[0]] for s in slots):
        subset = [s for s in slots if family[s[0]] == name]
        shares = [effects(metric_values(rows, cell, s)["time"]) for s in subset]
        out[name] = dict(slots=len(subset), **{key: fmean(x[key] for x in shares)
                                               for key in ["total"] + ["shapley_" + f for f in FACTORS]})
    return out


def signature(phase, verdict):
    """The final path file when the trial directory is kept, otherwise the verdict's final-path fields."""
    path = phase / "trials" / verdict["trial_id"] / "path_xy.csv"
    if (phase / "trials").is_dir():
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return tuple(verdict.get(key) for key in SIGNATURE_FIELDS)


def identity_checks(phase):
    """Final paths of the cell pairs that differ only in C2, and of cell 0000 against A4 where A4 runs."""
    items, methods, _, _ = load([phase])
    cell = cells(methods)
    slots = sorted({key[1:] for key in items})
    out = dict(basis="path_xy.csv" if (phase / "trials").is_dir() else "verdict final-path fields")
    pairs = {"c2_pairs": [(cell[bits], cell[flip(bits, 1, 1)]) for bits in BITS if bits[1] == 0]}
    if A4 in methods:
        pairs["cell_0000_vs_a4"] = [(cell[NONE], A4)]
    for name, couples in pairs.items():
        checked, skipped, differing = 0, 0, []
        for slot in slots:
            for left, right in couples:
                a, b = items[left, *slot]["verdict"], items[right, *slot]["verdict"]
                if a is None or b is None or "TIME_BUDGET" in (a.get("termination_reason"), b.get("termination_reason")):
                    skipped += 1
                    continue
                checked += 1
                if a["success"] != b["success"] or signature(phase, a) != signature(phase, b):
                    differing.append(f"{slot[0]}__s{slot[1]}: {left} vs {right}")
        out[name] = dict(checked=checked, skipped=skipped, differing=differing)
    return out


def worker_counters(phase, verdict):
    directory = phase / "trials" / verdict["trial_id"]
    worker = read(directory / "worker.json")
    counters, timing = worker.get("a5_counters", {}), worker.get("a5_timing", {})
    events = [json.loads(line) for line in (directory / "progress.jsonl").read_text(encoding="utf-8").splitlines() if line]
    trace = [e for e in events if e.get("event_type") == "improvement"]
    return dict(batches=worker.get("batches"), refinements=worker.get("refinement_count"), refine_s=timing.get("refine_s"),
                refine_deferred=counters.get("refine_deferred"), slope_exact=counters.get("slope_exact"),
                slope_certified=counters.get("slope_certified"), seed_published=worker.get("seed_status") == "published",
                final_is_seed=bool(trace) and trace[-1].get("event_origin") == SEED_ORIGIN)


def cell_summary(items, rows, name, slots, phase, reference, checks):
    verdicts = [items[name, *slot]["verdict"] for slot in slots]
    safe = [v for v in verdicts if v and v["success"]]
    times = [v["solve_time_s"] for v in safe]
    first = [v["first_safe_s"] for v in safe if v.get("first_safe_s") is not None]
    out = dict(method=name, success=len(safe), slots=len(slots), time_mean=fmean(times) if times else None,
               time_sd=pstdev(times) if times else None, first_safe_mean=fmean(first) if first else None,
               terminations={})
    for v in verdicts:
        if v:
            out["terminations"][v.get("termination_reason")] = out["terminations"].get(v.get("termination_reason"), 0) + 1
    out["anytime"] = {}
    for point in CHECKPOINTS:
        frozen = [(slot, (rows[name, *slot]["checkpoints"].get(point) or {})) for slot in reference if rows]
        hits = [(slot, p) for slot, p in frozen if p.get("safe")]
        out["anytime"][point] = dict(safe=len(hits), j_over_final_0000=fmean(
            p["j_sem"] / reference[slot] for slot, p in hits) if hits else None)
    if rows:
        both = [s for s in slots if rows[name, *s]["success"] and rows[NATIVE, *s]["success"]]
        mine, theirs = [rows[name, *s]["solve_time_s"] for s in both], [rows[NATIVE, *s]["solve_time_s"] for s in both]
        out["vs_sem"] = dict(pairs=len(both), ratio_of_means=fmean(theirs) / fmean(mine) if both else None,
                             faster_pairs=sum(a < b for a, b in zip(mine, theirs)))
    if checks:
        out["ten_checks_vs_sem"] = dict(passed=sum(bool(checks[k]) for k in checks if k != "method"),
                                        failed=[k for k in checks if k != "method" and not checks[k]])
    if (phase / "trials").is_dir() and safe:
        counters = [worker_counters(phase, v) for v in safe]
        out["mechanism"] = {key: fmean(float(c[key]) for c in counters) for key in counters[0]}
    return out


def build(phase, report):
    items, methods, analysis, sources = load([phase])
    mine = [(s["config_sha256"], s["planned"], s["validated"], s["verdicts"]) for s in sources]
    if mine != [(s["config_sha256"], s["planned"], s["validated"], s["verdicts"]) for s in report["sources"]]:
        raise ValueError("report.json was not written for these verdicts")
    cell = cells(methods)
    if NATIVE not in methods:
        raise ValueError("the phase needs the SEM-RRT* reference")
    manifest = read(phase / "manifest.json")
    complete = all(item["verdict"] is not None for item in items.values())
    rows = {key: item["verdict"] for key, item in items.items()} if complete else None
    slots = sorted({key[1:] for key in items})
    family = {item["map_id"]: item["family"] for item in items.values()}
    usable = [s for s in slots if rows and all(rows[cell[bits], *s]["success"] for bits in BITS)]
    reference = {s: rows[cell[NONE], *s]["final_j_sem"] for s in usable}
    checks = {c["method"]: c for c in report["criteria"]}
    out = dict(phase=str(phase), phase_name=manifest["phase"], complete=complete,
               correctness_failures=report["correctness_failures"], slots=len(slots), usable_slots=len(usable),
               cells={label(bits): cell_summary(items, rows, cell[bits], slots, phase, reference, checks.get(cell[bits]))
                      for bits in BITS},
               identity=identity_checks(phase))
    if len(usable) < 2:
        out["estimates"] = None
        return out
    est = out["estimates"] = estimate(rows, cell, usable, analysis)
    out["by_family"] = by_family(rows, cell, usable, family)
    out["reading"] = {name: dict(reduces_time=est["time.main_" + name]["upper95_one_sided"] < 0.,
                                 costs_no_quality=est["j_rel.main_" + name]["upper95_one_sided"] <= J30_MARGIN)
                      for name in FACTORS}
    return out


def markdown(out):
    fmt = lambda x, digits=3: "—" if x is None else f"{x:.{digits}f}"
    pct = lambda x: "—" if x is None else f"{100 * x:+.3f}%"
    lines = [f"# Ablation A5 (giai thừa 2⁴) — `{out['phase_name']}`", "",
             f"Đủ verdict: {out['complete']}; lỗi correctness: {out['correctness_failures']}; slot dùng cho hiệu ứng "
             f"(cả 16 ô đều có đường an toàn): {out['usable_slots']}/{out['slots']}. Ô ghi theo C1 C2 C3 C4; 0000 = A4, "
             "1111 = A5 chính. Time là thời gian solve (s).", "",
             "## Từng ô", "",
             "| Ô | PSR | Time | sd | SEM / ô | Nhanh hơn SEM | Đường đầu (s) | J₃₀ tại 0,25 s / J₃₀ cuối của 0000 | "
             "Có đường tại 0,25 s | Batch | Tinh chỉnh | Cạnh không có chứng chỉ độ dốc | Đường cuối là nghiệm mồi | "
             "10 tiêu chí so với SEM |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, c in out["cells"].items():
        sem, mech = c.get("vs_sem") or {}, c.get("mechanism") or {}
        early = c["anytime"].get("0.25", {})
        ten = c.get("ten_checks_vs_sem")
        # The slope counters exist only with C2 on; without it every check is exact.
        exact = fmt(mech.get('slope_exact'), 1) if name[1] == "1" else "—"
        lines.append(f"| {name} | {c['success']}/{c['slots']} | {fmt(c['time_mean'])} | {fmt(c['time_sd'])} | "
                     f"{fmt(sem.get('ratio_of_means'), 2)} | {sem.get('faster_pairs', '—')}/{sem.get('pairs', '—')} | "
                     f"{fmt(c['first_safe_mean'])} | {fmt(early.get('j_over_final_0000'), 4)} | {early.get('safe', '—')} | "
                     f"{fmt(mech.get('batches'), 1)} | {fmt(mech.get('refinements'), 2)} | {exact} | "
                     f"{fmt(mech.get('final_is_seed'), 2)} | {ten['passed'] if ten else '—'}/10 |")
    est = out.get("estimates")
    if est:
        total = est["time.total"]
        lines += ["", "## Đóng góp của từng cơ chế", "",
                  f"Tổng T(1111) − T(0000) = {total['estimate']:+.3f} s [95%: {total['ci95'][0]:+.3f}; {total['ci95'][1]:+.3f}].", "",
                  "| Cơ chế | Hiệu ứng chính Time (s) [cận trên] | Shapley (s) | Phần của tổng | Hiệu ứng chính log Time (×) | "
                  "Bỏ khỏi A5 (s) | Thêm vào A4 (s) | ΔJ₃₀ chính [cận trên] | Giảm Time | Không kém chất lượng |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for name in FACTORS:
            main, share, log = est["time.main_" + name], est["time.shapley_" + name], est["log_time.main_" + name]
            quality, reading = est["j_rel.main_" + name], out["reading"][name]
            lines.append(f"| {name} | {main['estimate']:+.3f} [{main['upper95_one_sided']:+.3f}] | {share['estimate']:+.3f} | "
                         f"{100 * share['estimate'] / total['estimate']:.1f}% | ×{math.exp(log['estimate']):.3f} | "
                         f"{-est['time.remove_' + name]['estimate']:+.3f} | {est['time.add_' + name]['estimate']:+.3f} | "
                         f"{pct(quality['estimate'])} [{pct(quality['upper95_one_sided'])}] | "
                         f"{'có' if reading['reduces_time'] else 'không'} | {'có' if reading['costs_no_quality'] else 'không'} |")
        lines += ["", "Cột \"Bỏ khỏi A5\" là T(1111) − T(1111 bỏ Ci): âm nghĩa là có Ci thì nhanh hơn.", "",
                  "## Tương tác (Time, s): hiệu ứng của Ci khi Cj bật trừ khi Cj tắt", "", "| Cặp | Ước lượng | 95% |", "|---|---:|---|"]
        for i, j in itertools.combinations(range(4), 2):
            row = est[f"time.interaction_{FACTORS[i]}x{FACTORS[j]}"]
            lines.append(f"| {FACTORS[i]} × {FACTORS[j]} | {row['estimate']:+.3f} | [{row['ci95'][0]:+.3f}; {row['ci95'][1]:+.3f}] |")
        lines += ["", "## Shapley theo họ bản đồ (Time, s)", "", "| Họ | Slot | Tổng | C1 | C2 | C3 | C4 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, row in out["by_family"].items():
            lines.append(f"| {name} | {row['slots']} | {row['total']:+.3f} | "
                         + " | ".join(f"{row['shapley_' + f]:+.3f}" for f in FACTORS) + " |")
    identity = out["identity"]
    lines += ["", f"## Kiểm tra trùng đường cuối (so bằng {identity['basis']})", ""]
    for name, row in identity.items():
        if name != "basis":
            lines.append(f"- `{name}`: so {row['checked']}, bỏ qua {row['skipped']} (chạm trần thời gian), khác {len(row['differing'])}.")
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
    dump(args.output / "a5_ablation.json", out)
    (args.output / "A5_ABLATION.md").write_text(markdown(out), encoding="utf-8")
    print(f"usable_slots={out['usable_slots']}/{out['slots']} -> {args.output}")


if __name__ == "__main__":
    main()
