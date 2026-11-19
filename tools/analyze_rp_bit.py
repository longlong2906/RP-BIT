"""A4 / RP-BIT* analysis: the five paper metrics, paired comparison with SEM-RRT*, anytime table.

Reads independent verdicts only and never runs or selects a planner. PSR keeps
every started attempt; quality is conditional on final-safe paths; paired
statistics use the (map, planner seed) slots where both methods are final-safe.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import fmean, median

import numpy as np

from corridor_v4_execution import atomic_json as dump, read
from rp_bit_protocol import NATIVE, PROTOCOL_ID

# report key -> verdict field (final path) / checkpoint field (anytime)
METRICS = (("apl", "final_cost_terrain_m", "cost_terrain_m"), ("pri", "final_pri", "pri"),
           ("j_sem", "final_j_sem", "j_sem"), ("turns", "final_sharp_turn_count", "sharp_turn_count"),
           ("time", "solve_time_s", None))
TOLERANCES = (0., .01)


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def load(directories):
    """Merge phase directories; a (method, map, planner seed) slot may occur once."""
    items, methods, analysis, sources = {}, {}, None, []
    for directory in map(Path, directories):
        manifest = read(directory / "manifest.json")
        if manifest.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("not an A4 RP-BIT* phase: " + str(directory))
        # A strict (full) verdict replaces the fast verdict of the same trial when both exist.
        names = [name for name in ("summary.json", "summary_full.json") if (directory / name).is_file()]
        verdicts = {row["trial_id"]: row for name in names for row in read(directory / name)["rows"]}
        name = "+".join(names)
        config = manifest["config"]
        if analysis not in (None, config["analysis"]):
            raise ValueError("phases disagree on the analysis settings")
        analysis = config["analysis"]
        for entry in manifest["entries"]:
            key, spec = (entry["method"], entry["map_id"], entry["planner_seed"]), config["methods"][entry["method"]]
            if key in items or methods.setdefault(entry["method"], spec) != spec:
                raise ValueError("duplicate slot or conflicting method definition: " + entry["trial_id"])
            items[key] = dict(entry, verdict=verdicts.get(entry["trial_id"]))
        phase = config["phases"][manifest["phase"]]
        sources.append(dict(directory=str(directory), phase=manifest["phase"], verdicts=name,
                            primary_method=phase.get("primary_method"),
                            sem_work_iterations=phase.get("sem_work_iterations", 0),
                            sem_budget_cap_s=phase.get("sem_budget_cap_s"),
                            config_sha256=manifest["config_sha256"], binary_sha256=manifest["binary_sha256"],
                            planned=len(manifest["entries"]), validated=len(verdicts)))
    return items, methods, analysis, sources


def sem_work_summary(items, sources):
    """Fixed-work SEM-RRT* phases: how many runs did the whole iteration budget before the wall-clock cap."""
    budgets = {s["phase"]: (s["sem_work_iterations"], s["sem_budget_cap_s"]) for s in sources if s["sem_work_iterations"]}
    runs = [(budgets[item["phase"]][0], item["verdict"]) for (name, _, _), item in items.items()
            if name == NATIVE and item["phase"] in budgets]
    if not runs:
        return None
    done = [(work, verdict) for work, verdict in runs if verdict]
    counts = [verdict["iterations"] for _, verdict in done if type(verdict.get("iterations")) is int]
    times = [verdict["solve_time_s"] for _, verdict in done if finite(verdict.get("solve_time_s"))]
    return dict(budgets=sorted(set(budgets.values())), planned=len(runs), verdicts=len(done),
                full_work=sum(verdict.get("iterations") == work for work, verdict in done),
                iterations_min=min(counts, default=None), iterations_max=max(counts, default=None),
                solve_time_mean_s=fmean(times) if times else None, solve_time_max_s=max(times, default=None))


def sem_iterations(items, methods):
    """Iterations of every SEM-RRT* run with a verdict; under a wall-clock budget they measure the engine speed."""
    counts = [item["verdict"]["iterations"] for (name, _, _), item in items.items()
              if name == NATIVE and item["verdict"] and type(item["verdict"].get("iterations")) is int]
    if not counts:
        return None
    return dict(engine=methods[NATIVE].get("engine", "python"), runs=len(counts), min=min(counts),
                median=median(counts), max=max(counts))


def terminations(items):
    """Why each run of a method ended (when its worker reports it) and its solve times."""
    out = {}
    for (name, _, _), item in items.items():
        verdict = item["verdict"]
        if verdict and verdict.get("termination_reason"):
            row = out.setdefault(name, dict(runs=0, reasons={}, times=[]))
            row["runs"] += 1
            row["reasons"][verdict["termination_reason"]] = row["reasons"].get(verdict["termination_reason"], 0) + 1
            if finite(verdict.get("solve_time_s")):
                row["times"].append(verdict["solve_time_s"])
    for row in out.values():
        times = row.pop("times")
        row.update(reasons=dict(sorted(row["reasons"].items())), solve_time_median_s=median(times) if times else None,
                   solve_time_mean_s=fmean(times) if times else None, solve_time_max_s=max(times, default=None))
    return out or None


def record(item, checkpoint=None):
    """One attempt at the end of its run, or frozen at an anytime checkpoint."""
    verdict = item["verdict"]
    out = dict(method=item["method"], map_id=item["map_id"], family=item["family"], planner_seed=item["planner_seed"],
               attempted=verdict is not None, safe=False, prep=None, **{key: None for key, _, _ in METRICS})
    if verdict is None:
        return out
    if checkpoint is None:
        out["safe"], source = bool(verdict["success"]), {key: verdict.get(field) for key, field, _ in METRICS}
        # Everything before the solve clock: input loading plus the SEM field construction of either pipeline.
        source["prep"] = sum(value for value in (verdict.get("preprocessing_s"), verdict.get("field_build_s")) if finite(value))
    else:
        point = verdict["checkpoints"].get(checkpoint)
        out["safe"] = bool(point and point["safe"] and verdict["execution_status"] == "COMPLETED"
                           and not verdict["correctness_errors"])
        source = {key: point.get(field) if field else None for key, _, field in METRICS} if point else {}
    if out["safe"]:
        out.update(source)
    return out


def aggregate(records):
    attempted = [r for r in records if r["attempted"]]
    safe = [r for r in attempted if r["safe"]]
    length = math.fsum(r["apl"] for r in safe)

    def mean(key):
        values = [r[key] for r in safe if finite(r[key])]
        return fmean(values) if values and len(values) == len(safe) else None
    return dict(planned=len(records), attempted=len(attempted), success=len(safe),
                psr=len(safe) / len(attempted) if attempted else None, apl=mean("apl"), pri=mean("pri"),
                j_sem=mean("j_sem"), time=mean("time"), prep=mean("prep"), turns_per_path=mean("turns"),
                sti=math.fsum(r["turns"] for r in safe) / length if length > 0 else None)


def bootstrap(pairs, keys, replicates, seed):
    """Map bootstrap, then paired-seed bootstrap inside each drawn map.

    Returns the per-replicate sums with shape (side, replicate, key) and the pair counts."""
    maps = sorted({pair["map_id"] for pair in pairs})
    groups = [[pair for pair in pairs if pair["map_id"] == name] for name in maps]
    counts = np.array([len(group) for group in groups])
    width = int(counts.max())
    values = np.zeros((2, len(maps), width, len(keys)))
    for i, group in enumerate(groups):
        for j, pair in enumerate(group):
            for side in (0, 1):
                values[side, i, j] = [pair["sides"][side][key] for key in keys]
    random, sums, sizes = np.random.default_rng(seed), [], []
    for start in range(0, replicates, 1024):
        size = min(1024, replicates - start)
        picked = random.integers(0, len(maps), (size, len(maps)))
        drawn = counts[picked]
        seeds = (random.random((size, len(maps), width)) * drawn[..., None]).astype(np.int64)
        present = (np.arange(width) < drawn[..., None])[None, ..., None]
        sums.append(np.where(present, values[:, picked[..., None], seeds], 0.).sum(axis=(2, 3)))
        sizes.append(drawn.sum(axis=1))
    return np.concatenate(sums, axis=1), np.concatenate(sizes)


def _interval(draws):
    low, high, upper = (float(np.quantile(draws, q)) for q in (.025, .975, .95))
    return dict(ci95=[low, high], upper95_one_sided=upper)


def compare(records, method, replicates=0, seed=0, baseline=NATIVE):
    """Method minus the baseline (SEM unless a registered contrast says otherwise) on the slots where both are final-safe."""
    left = {(r["map_id"], r["planner_seed"]): r for r in records if r["method"] == method}
    right = {(r["map_id"], r["planner_seed"]): r for r in records if r["method"] == baseline}
    slots = sorted(set(left) & set(right))
    good = [s for s in slots if left[s]["safe"] and right[s]["safe"]]
    other = "sem" if baseline == NATIVE else "baseline"
    out = dict(method=method, baseline=baseline, sign="method_minus_SEM" if baseline == NATIVE else "method_minus_baseline",
               planned_pairs=len(slots), safe_pairs=len(good), safe_maps=len({s[0] for s in good}), metrics={}, sti_difference=None)
    if not good:
        return out
    keys = [key for key, _, _ in METRICS if all(finite(left[s][key]) and finite(right[s][key]) for s in good)]
    sums = counts = None
    if replicates:
        sums, counts = bootstrap([dict(map_id=s[0], sides=(left[s], right[s])) for s in good], keys, replicates, seed)
    for index, key in enumerate(keys):
        a, b = [left[s][key] for s in good], [right[s][key] for s in good]
        total = math.fsum(b)
        item = dict(mean_method=fmean(a), **{"mean_" + other: fmean(b)}, mean_difference=fmean(x - y for x, y in zip(a, b)),
                    relative_pooled=(math.fsum(a) - total) / total if total > 0 else None,
                    wins=sum(x < y for x, y in zip(a, b)), ties=sum(x == y for x, y in zip(a, b)),
                    losses=sum(x > y for x, y in zip(a, b)))
        if sums is not None:
            delta = sums[0, :, index] - sums[1, :, index]
            item["difference_bootstrap"] = _interval(delta / counts)
            if (sums[1, :, index] > 0).all():
                item["relative_bootstrap"] = _interval(delta / sums[1, :, index])
        out["metrics"][key] = item
    if "apl" in keys and "turns" in keys:
        sti = lambda side: math.fsum(side[s]["turns"] for s in good) / math.fsum(side[s]["apl"] for s in good)
        out["sti_difference"] = dict(method=sti(left), **{other: sti(right)}, difference=sti(left) - sti(right))
        if sums is not None:
            i, j = keys.index("turns"), keys.index("apl")
            out["sti_difference"]["bootstrap"] = _interval(sums[0, :, i] / sums[0, :, j] - sums[1, :, i] / sums[1, :, j])
    return out


def time_to_quality(items, method, spec, weight, baseline=NATIVE):
    """First time the method's J (lambda = SEM-equivalent) reaches the baseline's final J of the same slot."""
    reached = {tolerance: [] for tolerance in TOLERANCES}
    pairs = 0
    for (name, map_id, seed), item in items.items():
        other = items.get((baseline, map_id, seed))
        verdict = item["verdict"]
        if name != method or not other or not other["verdict"] or not verdict:
            continue
        if not verdict["success"] or not other["verdict"]["success"]:
            continue
        # Independently valued trace when available; else the worker's canonical J, valid only at lambda = weight.
        trace = verdict.get("events_j_sem") or (verdict["reported"] if spec.get("lambda") == weight else None)
        if trace is None:
            return None
        pairs += 1
        target = other["verdict"]["final_j_sem"]
        for tolerance in TOLERANCES:
            hit = next((elapsed for elapsed, value in trace if finite(value) and value <= target * (1 + tolerance)), None)
            if hit is not None:
                reached[tolerance].append(hit)
    return [dict(tolerance=tolerance, pairs=pairs, reached=len(times), median_s=median(times) if times else None,
                 mean_s=fmean(times) if times else None, max_s=max(times) if times else None)
            for tolerance, times in reached.items()]


def criteria(tables, comparison, margin):
    """Registered checks (plan section 5.3) for one method against its baseline (SEM unless a contrast)."""
    find = lambda scope, method: next(t for t in tables if t["scope"] == scope and t["method"] == method)
    scopes = list(dict.fromkeys(t["scope"] for t in tables))
    method, metrics, baseline = comparison["method"], comparison["metrics"], comparison["baseline"]
    upper = lambda key, kind: metrics.get(key, {}).get(kind, {}).get("upper95_one_sided")
    below = lambda value, bound: finite(value) and value < bound
    psr = [find(scope, method)["psr"] is not None and find(scope, baseline)["psr"] is not None
           and find(scope, method)["psr"] >= find(scope, baseline)["psr"] for scope in scopes]
    a, b = find("all", method), find("all", baseline)
    return dict(method=method, psr_pooled=psr[0], psr_every_family=all(psr[1:]),
                sti_lower=below(a["sti"], b["sti"]) if finite(b["sti"]) else False,
                time_lower=below(metrics.get("time", {}).get("mean_difference"), 0.),
                j_sem_mean_lower=below(metrics.get("j_sem", {}).get("mean_difference"), 0.),
                j_sem_upper95_below_zero=below(upper("j_sem", "difference_bootstrap"), 0.),
                apl_noninferior=below(upper("apl", "relative_bootstrap"), margin + 1e-15),
                pri_noninferior=below(upper("pri", "relative_bootstrap"), margin + 1e-15),
                apl_target_lower=below(metrics.get("apl", {}).get("mean_difference"), 0.),
                pri_target_lower=below(metrics.get("pri", {}).get("mean_difference"), 0.))


def paired_at(frozen, name, analysis, baseline=NATIVE):
    """Equal-time paired differences at one anytime checkpoint, with the same hierarchical bootstrap bound."""
    paired = compare(frozen, name, analysis["bootstrap_replicates"], analysis["bootstrap_seed"], baseline)
    upper = lambda value, kind: (value.get(kind) or {}).get("upper95_one_sided")
    return dict(safe_pairs=paired["safe_pairs"], sti_difference=paired["sti_difference"],
                difference={key: value["mean_difference"] for key, value in paired["metrics"].items()},
                relative={key: value["relative_pooled"] for key, value in paired["metrics"].items()},
                difference_upper95={key: upper(value, "difference_bootstrap") for key, value in paired["metrics"].items()},
                relative_upper95={key: upper(value, "relative_bootstrap") for key, value in paired["metrics"].items()})


def decide(comparison, margin, tables):
    """Registered refine-attribution decision on the pooled contrast (rule refine_attribution_v1):
    A: J30 lower (mean < 0, one-sided upper 95% < 0); else B: Time lower (same test) with J30 non-inferior
    (relative upper 95% < margin); else C. A is tested first, so the chance of a false A or B is at most 2 alpha."""
    metrics = comparison["metrics"]
    upper = lambda key, kind: (metrics.get(key, {}).get(kind) or {}).get("upper95_one_sided")
    mean = lambda key: metrics.get(key, {}).get("mean_difference")
    below = lambda value, bound: finite(value) and value < bound
    quality = below(mean("j_sem"), 0.) and below(upper("j_sem", "difference_bootstrap"), 0.)
    speed = (below(mean("time"), 0.) and below(upper("time", "difference_bootstrap"), 0.)
             and below(upper("j_sem", "relative_bootstrap"), margin + 1e-15))
    psr = {name: next(t["psr"] for t in tables if t["scope"] == "all" and t["method"] == name)
           for name in (comparison["method"], comparison["baseline"])}
    return dict(rule="refine_attribution_v1", contrast=[comparison["method"], comparison["baseline"]],
                outcome="A" if quality else "B" if speed else "C", a_quality_lower=quality, b_faster_noninferior=speed,
                safe_pairs=comparison["safe_pairs"], planned_pairs=comparison["planned_pairs"],
                j_sem_mean_difference=mean("j_sem"), j_sem_upper95=upper("j_sem", "difference_bootstrap"),
                j_sem_relative_upper95=upper("j_sem", "relative_bootstrap"), time_mean_difference=mean("time"),
                time_upper95=upper("time", "difference_bootstrap"), psr=psr)


def replay_summary(items):
    """Per derived method: what the SEM + rpRefine replays did (from the verdicts)."""
    out = {}
    for (name, _, _), item in items.items():
        replay = (item["verdict"] or {}).get("replay")
        if replay:
            row = out.setdefault(name, dict(trials=0, overheads=[]))
            row["trials"] += 1
            row["overheads"].append(replay["overhead_s"])
            for key, value in replay.items():
                if key != "overhead_s":
                    row[key] = row.get(key, 0) + value
    for row in out.values():
        overheads = row.pop("overheads")
        row.update(overhead_mean_s=fmean(overheads), overhead_max_s=max(overheads))
    return out


def build_report(directories):
    items, methods, analysis, sources = load(directories)
    names = list(methods)
    final = [record(item) for item in items.values()]
    families = list(dict.fromkeys(r["family"] for r in final))
    scopes = [("all", final)] + [(family, [r for r in final if r["family"] == family]) for family in families]
    tables = [dict(scope=scope, method=name, **aggregate([r for r in subset if r["method"] == name]))
              for scope, subset in scopes for name in names]
    others = [name for name in names if name != NATIVE] if NATIVE in names else []
    comparisons = [dict(scope=scope, **compare(subset, name, analysis["bootstrap_replicates"] if scope == "all" else 0,
                                                analysis["bootstrap_seed"])) for scope, subset in scopes for name in others]
    anytime = []
    for limit in analysis["checkpoints_s"]:
        frozen = [record(item, format(limit, "g")) for item in items.values()]
        for name in names:
            row = dict(checkpoint_s=limit, method=name, **aggregate([r for r in frozen if r["method"] == name]))
            if name in others:
                # Same hierarchical bootstrap as the final comparison: equal-time differences carry an upper bound.
                row.update(paired_at(frozen, name, analysis))
            anytime.append(row)
    failures = [dict(trial_id=item["trial_id"], execution_status=item["verdict"]["execution_status"],
                     planner_status=item["verdict"]["planner_status"], correctness_errors=item["verdict"]["correctness_errors"],
                     final_unsafe_reasons=item["verdict"].get("final_unsafe_reasons"))
                for item in items.values() if item["verdict"] and not item["verdict"]["success"]]
    pooled = [c for c in comparisons if c["scope"] == "all"]
    report = dict(protocol_id=PROTOCOL_ID, sources=sources, methods=methods, analysis=analysis,
                complete=all(item["verdict"] is not None for item in items.values()),
                correctness_failures=sum(bool(item["verdict"] and item["verdict"]["correctness_errors"]) for item in items.values()),
                tables=tables, paired_comparisons=comparisons, anytime=anytime,
                time_to_quality={name: time_to_quality(items, name, methods[name], analysis["sem_equivalent_lambda"])
                                 for name in others},
                criteria=[criteria(tables, c, analysis["noninferiority_margin"]) for c in pooled if c["safe_pairs"]],
                sem_work=sem_work_summary(items, sources), sem_iterations=sem_iterations(items, methods) if NATIVE in methods else None,
                terminations=terminations(items), failures=failures, trials=final)
    if "contrasts" in analysis:
        # Registered extra comparisons (method minus baseline), present only when the config declares them,
        # so reports of earlier phases are unchanged.
        contrasts = []
        for name, baseline in analysis["contrasts"]:
            if name not in names or baseline not in names:
                continue
            rows = [dict(scope=scope, **compare(subset, name, analysis["bootstrap_replicates"] if scope == "all" else 0,
                                                 analysis["bootstrap_seed"], baseline)) for scope, subset in scopes]
            anytime_rows = [dict(checkpoint_s=limit, **paired_at([record(item, format(limit, "g")) for item in items.values()],
                                                                 name, analysis, baseline))
                            for limit in analysis["checkpoints_s"]]
            contrasts.append(dict(method=name, baseline=baseline, comparisons=rows, anytime=anytime_rows,
                                  criteria=criteria(tables, rows[0], analysis["noninferiority_margin"]) if rows[0]["safe_pairs"] else None,
                                  time_to_quality=time_to_quality(items, name, methods[name], analysis["sem_equivalent_lambda"], baseline)))
        report["contrasts"] = contrasts
        rule = analysis.get("decision")
        chosen = next((c for c in contrasts if rule and [c["method"], c["baseline"]] == rule["contrast"]), None)
        report["decision"] = (decide(chosen["comparisons"][0], analysis["noninferiority_margin"], tables)
                              if chosen and chosen["comparisons"][0]["safe_pairs"] else None)
        report["replay"] = replay_summary(items)
    return report


def markdown(report, title):
    fmt = lambda value, digits=4: "—" if value is None else f"{value:.{digits}g}"
    pct = lambda value: "—" if value is None else f"{100 * value:+.2f}%"
    bound = lambda item, kind: (item.get(kind) or {}).get("upper95_one_sided")
    cpp = report["methods"].get(NATIVE, {}).get("engine") == "cpp"
    lines = [f"# A4 / RP-BIT* — {title}", "",
             ("SEM-RRT* chạy bằng bản C++ (trùng từng bit với bản tái dựng Python đã khóa của dự án, không phải mã tác giả)"
              if cpp else "SEM-RRT* là bản tái dựng Python đã khóa của dự án (không phải mã tác giả)")
             + "; A4 và A3-T là C++ trên OMPL BIT* thật.",
             "PSR tính trên mọi lượt đã bắt đầu. APL = L_H, PRI = trung bình risk trên các ô duy nhất đường tâm chạm,",
             "STI = tổng số góc gấp > 30° / tổng L_H, Time = thời gian solve; J₃₀ = L_H + 30·E/40 là hàm mục tiêu tương đương",
             "trọng số của SEM-RRT*. Mọi số đều do oracle độc lập tính lại từ bản đồ và point cloud gốc.", "",
             "Nguồn: " + "; ".join(f"`{Path(s['directory']).name}` ({s['validated']}/{s['planned']} verdict, {s['verdicts']})"
                                   for s in report["sources"]) + ".",
             f"Đủ verdict: **{report['complete']}**; số lượt có lỗi correctness: **{report['correctness_failures']}**.", ""]
    work = report.get("sem_work")
    if work:
        budgets = ", ".join(f"{iterations} vòng lặp (trần {cap:g} s)" for iterations, cap in work["budgets"])
        lines += [f"Ngân sách SEM-RRT* cố định theo số vòng lặp: {budgets}. Chạy đủ số vòng lặp: **{work['full_work']}/"
                  f"{work['verdicts']}** lượt có verdict ({work['planned']} lượt đã đăng ký); số vòng lặp "
                  f"{work['iterations_min']}–{work['iterations_max']}; Time SEM trung bình {fmt(work['solve_time_mean_s'], 3)} s, "
                  f"lớn nhất {fmt(work['solve_time_max_s'], 3)} s.", ""]
    count = report.get("sem_iterations")
    if count:
        lines += [f"Số vòng lặp SEM-RRT* (bản {'C++' if count['engine'] == 'cpp' else 'Python'}) mỗi lượt: "
                  f"{count['min']}–{count['max']}, trung vị {count['median']:g} ({count['runs']} lượt có verdict).", ""]
    stop = report["methods"].get(NATIVE, {}).get("convergence_stop")
    if stop:
        lines += [f"SEM-RRT* tự dừng khi chi phí native đã công bố không giảm ít nhất {100 * stop['relative_eps']:g}% (tương đối) "
                  f"qua {stop['window_iterations']} vòng lặp liên tiếp; trần thời gian vẫn giữ.", ""]
    for name, row in (report.get("terminations") or {}).items():
        reasons = ", ".join(f"{reason} {n}" for reason, n in row["reasons"].items())
        lines.append(f"- Lý do dừng của `{name}` ({row['runs']} lượt có verdict): {reasons}; Time trung vị "
                     f"{fmt(row['solve_time_median_s'], 3)} s, trung bình {fmt(row['solve_time_mean_s'], 3)} s, "
                     f"lớn nhất {fmt(row['solve_time_max_s'], 3)} s.")
    if report.get("terminations"):
        lines.append("")
    lines += ["## 1. Năm metric (đường cuối)", "",
             "Time là thời gian solve (với A4 gồm dựng potential, BIT* và tinh chỉnh); cột tiền xử lý là phần trước đồng hồ solve",
             "(nạp dữ liệu và dựng trường SEM của từng pipeline), báo riêng để so sánh minh bạch.", "",
             "| Nhóm map | Thuật toán | Success/attempt | PSR | APL (m) | PRI | STI (1/m) | Góc gấp/đường | Time (s) | Tiền xử lý (s) | J₃₀ |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["tables"]:
        if row["attempted"]:
            lines.append(f"| {row['scope']} | {row['method']} | {row['success']}/{row['attempted']} | {fmt(100 * row['psr'])}% | "
                         f"{fmt(row['apl'], 5)} | {fmt(row['pri'])} | {fmt(row['sti'], 3)} | {fmt(row['turns_per_path'], 3)} | "
                         f"{fmt(row['time'], 3)} | {fmt(row['prep'], 3)} | {fmt(row['j_sem'], 5)} |")
    lines += ["", "## 2. So ghép cặp với SEM-RRT* (thuật toán − SEM; chỉ các cặp cùng map/seed đều final-safe)", "",
              "Δ% gộp = Σ(thuật toán − SEM)/Σ SEM. Cận trên: bootstrap phân cấp map → seed, một phía 95% (chỉ tính cho nhóm `all`).", "",
              "| Nhóm map | Thuật toán | Cặp | ΔAPL (m) | ΔAPL gộp [cận trên] | ΔPRI | ΔPRI gộp [cận trên] | ΔJ₃₀ [cận trên] | ΔTime (s) | ΔSTI |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["paired_comparisons"]:
        m = row["metrics"]
        if not m:
            continue
        cell = lambda key: fmt(m[key]["mean_difference"]) if key in m else "—"
        rel = lambda key: (pct(m[key]["relative_pooled"]) + (f" [{pct(bound(m[key], 'relative_bootstrap'))}]"
                           if "relative_bootstrap" in m[key] else "")) if key in m else "—"
        j = (fmt(m["j_sem"]["mean_difference"]) + (f" [{fmt(bound(m['j_sem'], 'difference_bootstrap'))}]"
             if "difference_bootstrap" in m["j_sem"] else "")) if "j_sem" in m else "—"
        sti = row["sti_difference"]
        lines.append(f"| {row['scope']} | {row['method']} | {row['safe_pairs']}/{row['planned_pairs']} | {cell('apl')} | {rel('apl')} | "
                     f"{cell('pri')} | {rel('pri')} | {j} | {cell('time')} | {fmt(sti['difference'], 3) if sti else '—'} |")
    lines += ["", "## 3. Anytime (đường tốt nhất đã công bố tới thời điểm T; an toàn tại T do oracle chấm)", "",
              "Δ là hiệu trung bình ghép cặp tại cùng T; trong ngoặc là cận trên một phía 95% của cùng bootstrap phân cấp.", "",
              "| T (s) | Thuật toán | Safe@T/attempt | APL (m) | PRI | STI (1/m) | J₃₀ | Cặp với SEM@T | ΔAPL | ΔPRI | ΔJ₃₀ [cận trên] |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["anytime"]:
        if not row["attempted"]:
            continue
        d, u = row.get("difference", {}), row.get("difference_upper95", {})
        j = fmt(d.get("j_sem")) + (f" [{fmt(u['j_sem'])}]" if u.get("j_sem") is not None else "")
        lines.append(f"| {row['checkpoint_s']:g} | {row['method']} | {row['success']}/{row['attempted']} | {fmt(row['apl'], 5)} | "
                     f"{fmt(row['pri'])} | {fmt(row['sti'], 3)} | {fmt(row['j_sem'], 5)} | {row.get('safe_pairs', '—')} | "
                     f"{fmt(d.get('apl'))} | {fmt(d.get('pri'))} | {j} |")
    lines += ["", "## 4. Thời gian đạt chất lượng cuối của SEM-RRT* trên cùng map/seed", "",
              "| Thuật toán | Ngưỡng J₃₀ | Đạt/cặp | Trung vị (s) | Trung bình (s) | Lớn nhất (s) |", "|---|---|---:|---:|---:|---:|"]
    for name, rows in report["time_to_quality"].items():
        for row in rows or []:
            lines.append(f"| {name} | ≤ SEM cuối × {1 + row['tolerance']:g} | {row['reached']}/{row['pairs']} | "
                         f"{fmt(row['median_s'], 3)} | {fmt(row['mean_s'], 3)} | {fmt(row['max_s'], 3)} |")
    lines += ["", "## 5. Tiêu chí đã đăng ký (mục 5.3 của kế hoạch)", "",
              "| Thuật toán | PSR gộp ≥ | PSR từng họ ≥ | STI < | Time < | ΔJ₃₀ < 0 | cận trên ΔJ₃₀ < 0 | APL không kém 1% | PRI không kém 1% | ΔAPL < 0 (mục tiêu) | ΔPRI < 0 (mục tiêu) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    mark = lambda ok: "đạt" if ok else "không"
    primary = {source.get("primary_method") for source in report["sources"]} - {None}
    for row in report["criteria"]:
        lines.append(f"| {row['method']}{' **(cấu hình chính đã đăng ký)**' if row['method'] in primary else ''} | "
                     + " | ".join(mark(row[key]) for key in (
            "psr_pooled", "psr_every_family", "sti_lower", "time_lower", "j_sem_mean_lower", "j_sem_upper95_below_zero",
            "apl_noninferior", "pri_noninferior", "apl_target_lower", "pri_target_lower")) + " |")
    lines += ["", "## 6. Lượt không thành công", ""]
    lines += [f"- `{row['trial_id']}`: {row['execution_status']}/{row['planner_status']}; "
              f"correctness={row['correctness_errors']}; unsafe={row['final_unsafe_reasons']}" for row in report["failures"]] or ["Không có."]
    if "contrasts" in report:
        lines += markdown_contrasts(report, fmt, pct, bound, mark)
    return "\n".join(lines) + "\n"


def markdown_contrasts(report, fmt, pct, bound, mark):
    """Sections 7-9: registered contrasts, the refine-attribution decision and the replay summary."""
    lines = ["", "## 7. Tương phản đã đăng ký (thuật toán − đối chứng; chỉ các cặp cùng map/seed đều final-safe)", "",
             "Cận trên: bootstrap phân cấp map → seed, một phía 95% (chỉ nhóm `all`).", "",
             "| Nhóm map | Thuật toán | Đối chứng | Cặp | ΔAPL (m) | ΔAPL gộp [cận trên] | ΔPRI | ΔPRI gộp [cận trên] | ΔJ₃₀ [cận trên] | ΔTime (s) [cận trên] | ΔSTI |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for contrast in report["contrasts"]:
        for row in contrast["comparisons"]:
            m = row["metrics"]
            if not m:
                continue
            cell = lambda key: fmt(m[key]["mean_difference"]) if key in m else "—"
            rel = lambda key: (pct(m[key]["relative_pooled"]) + (f" [{pct(bound(m[key], 'relative_bootstrap'))}]"
                               if "relative_bootstrap" in m[key] else "")) if key in m else "—"
            diff = lambda key: (fmt(m[key]["mean_difference"]) + (f" [{fmt(bound(m[key], 'difference_bootstrap'))}]"
                                if "difference_bootstrap" in m[key] else "")) if key in m else "—"
            sti = row["sti_difference"]
            lines.append(f"| {row['scope']} | {row['method']} | {row['baseline']} | {row['safe_pairs']}/{row['planned_pairs']} | "
                         f"{cell('apl')} | {rel('apl')} | {cell('pri')} | {rel('pri')} | {diff('j_sem')} | {diff('time')} | "
                         f"{fmt(sti['difference'], 3) if sti else '—'} |")
    lines += ["", "Anytime (cùng thời điểm T): ΔJ₃₀ [cận trên]; cặp là các slot mà cả hai đều có đường an toàn tại T.", "",
              "| T (s) | Thuật toán | Đối chứng | Cặp | ΔJ₃₀ [cận trên] | ΔAPL | ΔPRI |", "|---:|---|---|---:|---:|---:|---:|"]
    for contrast in report["contrasts"]:
        for row in contrast["anytime"]:
            d, u = row["difference"], row["difference_upper95"]
            j = fmt(d.get("j_sem")) + (f" [{fmt(u['j_sem'])}]" if u.get("j_sem") is not None else "")
            lines.append(f"| {row['checkpoint_s']:g} | {contrast['method']} | {contrast['baseline']} | {row['safe_pairs']} | {j} | "
                         f"{fmt(d.get('apl'))} | {fmt(d.get('pri'))} |")
    lines += ["", "Tiêu chí của mục 5 áp cho từng tương phản (đối chứng thay cho SEM):", "",
              "| Thuật toán | Đối chứng | PSR gộp ≥ | PSR từng họ ≥ | STI < | Time < | ΔJ₃₀ < 0 | cận trên ΔJ₃₀ < 0 | APL không kém 1% | PRI không kém 1% |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for contrast in report["contrasts"]:
        row = contrast["criteria"]
        if row:
            lines.append(f"| {contrast['method']} | {contrast['baseline']} | " + " | ".join(mark(row[key]) for key in (
                "psr_pooled", "psr_every_family", "sti_lower", "time_lower", "j_sem_mean_lower", "j_sem_upper95_below_zero",
                "apl_noninferior", "pri_noninferior")) + " |")
    # Sections 8-9 belong to configs that register a decision rule / derived replay trials (the factorial);
    # a contrast-only config (e.g. the PUTN dev preview) has neither, so they are left out.
    if report["analysis"].get("decision") or report.get("decision"):
        lines += _decision_lines(report.get("decision"), fmt, pct, mark)
    if any(spec.get("kind") == "sem_refine" for spec in report["methods"].values()):
        lines += _replay_lines(report, fmt)
    return lines


def _decision_lines(decision, fmt, pct, mark):
    lines = ["", "## 8. Quyết định đã đăng ký (luật `refine_attribution_v1`)", ""]
    if decision:
        meaning = {"A": "A — planner của A4 đóng góp chất lượng: J₃₀ thấp hơn có ý nghĩa ngay cả khi đối chứng cũng có rpRefine",
                   "B": "B — không thấp hơn về J₃₀ nhưng nhanh hơn có ý nghĩa và J₃₀ không kém quá 1%",
                   "C": "C — không đạt A hay B: lợi thế chủ yếu đến từ rpRefine"}
        method, baseline = decision["contrast"]
        lines += [f"Tương phản `{method}` − `{baseline}` trên {decision['safe_pairs']}/{decision['planned_pairs']} cặp: "
                  f"**{meaning[decision['outcome']]}**.", "",
                  f"- ΔJ₃₀ = {fmt(decision['j_sem_mean_difference'])} (cận trên {fmt(decision['j_sem_upper95'])}; "
                  f"cận trên tương đối {pct(decision['j_sem_relative_upper95'])}); A: {mark(decision['a_quality_lower'])}.",
                  f"- ΔTime = {fmt(decision['time_mean_difference'], 3)} s (cận trên {fmt(decision['time_upper95'], 3)} s); "
                  f"B: {mark(decision['b_faster_noninferior'])}.",
                  f"- PSR (không dùng để quyết định): " + ", ".join(f"`{name}` {fmt(100 * value)}%"
                                                                    for name, value in decision["psr"].items()) + ".",
                  "- A được xét trước B, nên xác suất kết luận sai A hoặc B tối đa là 2α; đây là quyết định hướng bài, không phải claim."]
    else:
        lines.append("Không tính được (thiếu tương phản hoặc không có cặp an toàn).")
    return lines


def _replay_lines(report, fmt):
    lines = ["", "## 9. Replay SEM-RRT* + rpRefine", ""]
    for name, row in (report.get("replay") or {}).items():
        lines.append(f"- `{name}` ({row['trials']} lượt): {row['refinements']} lần refine, công bố {row['refine_published']}, "
                     f"không hạ được J đã công bố {row['refine_not_improving']}, không hợp lệ/không an toàn {row['refine_rejected']}; "
                     f"đường thô SEM bị loại vì không an toàn {row['raw_rejected']}, đường thô được công bố {row['raw_published']}; "
                     f"ứng viên quá ngân sách {row['late']}; overhead trung bình {fmt(row['overhead_mean_s'], 3)} s, "
                     f"lớn nhất {fmt(row['overhead_max_s'], 3)} s.")
    if not report.get("replay"):
        lines.append("Không có.")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, nargs="+", type=Path, help="phase directories (with manifest.json)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="báo cáo phân tích")
    args = parser.parse_args(argv)
    report = build_report([path.resolve() for path in args.input])
    args.output.mkdir(parents=True, exist_ok=True)
    dump(args.output / "report.json", report)
    (args.output / "REPORT.md").write_text(markdown(report, args.title), encoding="utf-8")
    print(f"complete={report['complete']} correctness_failures={report['correctness_failures']} "
          f"trials={len(report['trials'])} failures={len(report['failures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
