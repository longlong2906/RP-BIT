"""Independent scoring of A4 / RP-BIT* campaign trials from raw map/cloud inputs.

Fast mode scores the final in-budget path and the registered anytime
checkpoints. Full mode additionally runs the project's strict per-snapshot
validators and requires them to pass. The planner's own field, costs and
validity verdicts are never used as an oracle here.
"""
from __future__ import annotations

import math
from pathlib import Path

from corridor_v3_validation import risk_metrics
from corridor_v4_execution import file_hash, read
from pri_portfolio_validation import sharp_turns
from rp_bit_protocol import NATIVE_KINDS, PUTN
from run_safe_length import _csv_path, _events
from sem_exposure_validation import objective_metrics, validate_geometry

VERSION = "a4_rpbit_independent_scoring_v1"
FINAL = ("cost_terrain_m", "pri", "length_xy_m", "sharp_turn_count", "swept_slope_deg", "exposure",
         "j_sem", "risk_sum", "risk_count", "risk_max", "vertex_count")


def finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def score_path(oracle, field, path, start, goal, weight):
    """Hard safety contract of the registered validators plus the reported metrics."""
    geometry = validate_geometry(oracle, path, start, goal, None)
    sem = field.score_path(path)
    slope = geometry.get("maximum_swept_footprint_slope_deg")
    safe = bool(geometry["valid"] and sem["valid"] and sem["risk_max"] <= 40. and finite(slope) and slope <= 35.)
    out = dict(safe=safe, geometry_errors=geometry["errors"], sem_reasons=sem["reasons"],
               cost_terrain_m=geometry.get("cost_terrain_m"), swept_slope_deg=slope, pri=sem["pri"],
               risk_max=sem["risk_max"], length_xy_m=math.fsum(math.dist(a, b) for a, b in zip(path, path[1:])),
               sharp_turn_count=sharp_turns(path), vertex_count=len(path))
    if safe:
        values, risk = objective_metrics(oracle, field, path, weight), risk_metrics(field, path)
        out.update(exposure=values["sem_exposure_risk_m"], j_sem=values["objective_value"],
                   risk_sum=risk["risk_sum"], risk_count=risk["risk_count"])
    return out


def _convergence_consistent(rule, outcome, worker):
    """A converged SEM run stopped exactly one window after its last reference, and its final published
    cost lies in [reference * (1 - eps), reference]: no qualifying improvement inside the window."""
    snapshots, reference = worker.get("snapshots") or [], outcome.get("reference_cost")
    if not snapshots or not finite(reference):
        return False
    last = snapshots[-1]["native_objective_value"]
    return (worker.get("counters", {}).get("iterations") == outcome["reference_iteration"] + rule["window_iterations"]
            and worker.get("termination_reason") == "CONVERGED_NO_IMPROVEMENT"
            and reference * (1 - rule["relative_eps"]) <= last <= reference)


def _same(a, b):
    """Structural equality with floats equal to 1e-12 relative."""
    if type(a) is float or type(b) is float:
        return finite(a) and finite(b) and abs(a - b) <= 1e-12 * max(1., abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _replay_checks(directory, entry, request, worker, method):
    """A derived SEM + rpRefine trial: bound to its finished source; schedule, timing and publications re-derived."""
    from rp_bit_protocol import coordinate_from_bits
    errors, spec = [], request["rp_refine_replay"]
    source = directory.parent / spec["source_trial_id"]
    if spec["source_trial_id"] != f"{entry['map_id']}__s{entry['planner_seed']}__{method['source']}":
        errors.append("replay_source_not_the_same_slot")
    if not (source / "worker.json").is_file() or file_hash(source / "worker.json") != spec["source_worker_sha256"]:
        return errors + ["replay_source_missing_or_changed"]
    origin = read(source / "worker.json")
    timings, incumbents = origin["phase_timings"], origin.get("snapshots") or []
    embedded = [dict(snapshot_id=s["snapshot_id"], elapsed_s=s["elapsed_s"],
                     path_xy=[[coordinate_from_bits(x), coordinate_from_bits(y)] for x, y in s["path_bits"]])
                for s in spec["incumbents"]]
    if (embedded != [dict(snapshot_id=s["snapshot_id"], elapsed_s=s["elapsed_s"], path_xy=s["path_xy"]) for s in incumbents]
            or spec["source_solve_s"] != timings["solve_s"] or spec["source_preprocessing_s"] != timings["total_before_solve_s"]
            or spec["source_planner_status"] != origin["planner_status"]
            or spec["source_termination_reason"] != origin["termination_reason"]):
        errors.append("replay_source_mismatch")
    if spec["refine_at_s"] != method["refine_at_s"] or request["sem_exposure"]["lambda"] != method["lambda"]:
        errors.append("replay_spec_not_registered")
    # The echo passed through Boost.JSON 1.82, whose decimal parsing may be off by one ULP: numbers compare to 1e-12.
    if not _same(worker.get("requested_rp_refine_replay"), {key: value for key, value in spec.items() if key != "incumbents"}):
        errors.append("requested_replay_mismatch")
    if worker.get("planner_status") in ("INVALID_START", "INVALID_GOAL"):
        return errors
    # Raw incumbents in time order; refine points before the stop, then the stop; raw first on equal times.
    # The incumbent standing at a refine point is refined once, and only if its own raw check passed.
    stop, times = spec["source_solve_s"], [s["elapsed_s"] for s in spec["incumbents"]]
    points = [r for r in spec["refine_at_s"] if r < stop] + [stop]
    plan, i, k = [], 0, 0
    while i < len(times) or k < len(points):
        if i < len(times) and (k == len(points) or times[i] <= points[k]):
            plan.append(("raw", times[i], i))
            i += 1
        else:
            plan.append(("refine", points[k], i))
            k += 1
    steps = worker.get("replay_steps") or []
    if len(steps) != len(plan):
        return errors + ["replay_schedule_mismatch"]
    close = lambda a, b, tol: finite(a) and finite(b) and abs(a - b) <= tol * max(1., abs(b))
    ids, raw_ok, refined, published = [s["snapshot_id"] for s in spec["incumbents"]], {}, set(), []
    offset, best, budget = worker.get("setup_s"), math.inf, request["planning_budget_s"]
    if not finite(offset):
        return errors + ["replay_timing_mismatch"]
    for (kind, at, n), step in zip(plan, steps):
        skipped, index = None, n if kind == "raw" else n - 1
        if kind == "refine":
            if not n:
                skipped = "no_incumbent"
            elif not raw_ok[index]:
                skipped = "unsafe_source"
            elif index in refined:
                skipped = "already_refined"
            else:
                refined.add(index)
        source_id = ids[index] if index >= 0 else None
        if (step["kind"], step["source_snapshot_id"], step["skipped"]) != (kind, source_id, skipped) or not close(step["at_s"], at, 1e-12):
            return errors + ["replay_schedule_mismatch"]
        if skipped:
            if step["duration_s"] != 0 or step["published"]:
                errors.append("replay_skipped_step_acted")
            continue
        offset += step["duration_s"]
        candidate = bool(step["valid"] and step["safe"])
        if kind == "raw":
            raw_ok[index] = candidate
        if not candidate:
            if step["published"] or step["stamp_s"] is not None:
                errors.append("replay_rejected_step_published")
            continue
        if not close(step["stamp_s"], at + offset, 1e-9):
            errors.append("replay_timing_mismatch")
        improves = not math.isfinite(best) or step["objective"] < best - 1e-6 * max(1., best)
        if step["published"] != (step["stamp_s"] <= budget and improves):
            errors.append("replay_publication_rule_violated")
        if step["published"]:
            best = step["objective"]
            published.append((step, kind, index))
    if not close(worker.get("actual_solve_wall_s"), stop + offset, 1e-9):
        errors.append("replay_timing_mismatch")
    events = _events(directory, worker)
    if len(events) != len(published):
        return errors + ["replay_events_mismatch"]
    for event, (step, kind, index) in zip(events, published):
        if (event["snapshot_id"] != step["snapshot_id"] or event["elapsed_s"] != step["stamp_s"]
                or event.get("event_origin") != ("sem_incumbent" if kind == "raw" else "rp_refinement")
                or event.get("source_snapshot_id") != ids[index] or event.get("search_objective_value") != step["objective"]):
            return errors + ["replay_events_mismatch"]
        if kind == "raw" and _csv_path(directory / "snapshots" / event["snapshot_id"] / "path_xy.csv") != incumbents[index]["path_xy"]:
            errors.append("replay_raw_path_differs_from_source")
    return errors


def _replay_summary(worker):
    """What the replay did, for the report: refinements that were published, lost to the published J, rejected or late."""
    steps = [s for s in worker.get("replay_steps") or [] if not s["skipped"]]
    refines = [s for s in steps if s["kind"] == "refine"]
    candidate = lambda s: s["valid"] and s["safe"]
    return dict(overhead_s=worker.get("replay_overhead_s"), refinements=len(refines),
                refine_published=sum(s["published"] for s in refines),
                refine_not_improving=sum(candidate(s) and not s["published"] and s["stamp_s"] <= worker["time_budget_s"]
                                         for s in refines),
                refine_rejected=sum(not candidate(s) for s in refines),
                late=sum(candidate(s) and s["stamp_s"] > worker["time_budget_s"] for s in steps),
                raw_rejected=sum(s["kind"] == "raw" and not candidate(s) for s in steps),
                raw_published=sum(s["kind"] == "raw" and s["published"] for s in steps))


def _strict(kind, directory, request, info, execution, oracle, field):
    if kind in NATIVE_KINDS:
        from pri_portfolio_validation import validate_trial as strict
    else:
        from sem_exposure_validation import validate_trial as strict
    return strict(directory, request, info, execution, oracle=oracle, field=field)


def validate_trial(directory, entry, info, config, *, oracle, field, full=False):
    directory = Path(directory)
    execution = read(directory / "execution.json")
    row = dict(entry, success=False, attained_safe=False, execution_status=execution["status"],
               execution_launched=execution.get("launched", False), planner_status=None, correctness_errors=[],
               field_build_s=execution.get("field_build_s"), peak_rss_bytes=execution.get("peak_process_tree_rss_bytes"),
               queue_wait_s=execution.get("queue_wait_s"), oracle_version=VERSION, full_validation=full,
               solve_time_s=None, first_solution_s=None, first_safe_s=None, first_safe_latency_s=None,
               preprocessing_s=None, checkpoints={}, reported=[], **{"final_" + key: None for key in FINAL})
    errors = row["correctness_errors"]
    if not (directory / "worker.json").is_file() or not (directory / "request.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        return row
    request, worker = read(directory / "request.json"), read(directory / "worker.json")
    kind = config["methods"][entry["method"]]["kind"]
    native = kind in NATIVE_KINDS
    if kind == PUTN:
        # The registered PUTN convergence stop and tau slope cap must be the ones that ran.
        method = config["methods"][entry["method"]]
        rule, outcome = method.get("convergence_stop"), worker.get("putn_convergence")
        if rule != (outcome or {}).get("rule") or rule != request.get("putn_convergence_stop"):
            errors.append("putn_convergence_stop_mismatch")
        elif rule and outcome["converged"] and not _convergence_consistent(rule, outcome, worker):
            errors.append("putn_convergence_inconsistent")
        cap = method.get("tau_slope_cap_deg")
        if cap != request.get("putn_tau_slope_cap_deg") or cap != (worker.get("native_model") or {}).get("tau_slope_cap_deg"):
            errors.append("putn_tau_slope_cap_mismatch")
    if kind == "sem_native":
        # The registered SEM search engine must be the one that ran (C++ port: also the exact library).
        engine, ran = config["methods"][entry["method"]].get("engine", "python"), worker.get("provenance", {})
        if ran.get("sem_engine", "python") != engine or (
                engine == "cpp" and ran.get("sem_library_sha256") != request.get("sem_library_sha256")):
            errors.append("sem_engine_mismatch")
        rule, outcome = config["methods"][entry["method"]].get("convergence_stop"), worker.get("sem_convergence")
        if rule != (outcome or {}).get("rule") or rule != request.get("sem_convergence_stop"):
            errors.append("sem_convergence_stop_mismatch")
        elif rule and outcome["converged"] and not _convergence_consistent(rule, outcome, worker):
            errors.append("sem_convergence_inconsistent")
    budget, weight = request["planning_budget_s"], config["analysis"]["sem_equivalent_lambda"]
    timings = worker.get("phase_timings", {})
    row.update(planner_status=worker.get("planner_status"), termination_reason=worker.get("termination_reason"),
               solve_time_s=timings.get("solve_s") if native else worker.get("actual_solve_wall_s"),
               preprocessing_s=timings.get("total_before_solve_s") if native else worker.get("preprocessing_s"),
               iterations=worker.get("counters", {}).get("iterations") if native else worker.get("iterations"),
               batches=worker.get("batches"))
    usable = [e for e in _events(directory, worker) if finite(e.get("elapsed_s")) and 0 <= e["elapsed_s"] <= budget]
    cache = {}

    def scored(event):
        sid = event["snapshot_id"]
        if sid not in cache:
            cache[sid] = score_path(oracle, field, _csv_path(directory / "snapshots" / sid / "path_xy.csv"),
                                    request["start"], request["goal"], weight)
        return cache[sid]

    objective = "native_objective_value" if native else "objective_value"
    row["reported"] = [[e["elapsed_s"], e.get(objective)] for e in usable]
    for limit in config["analysis"]["checkpoints_s"]:
        if limit <= budget + 1e-9:
            prior = [e for e in usable if e["elapsed_s"] <= limit]
            row["checkpoints"][format(limit, "g")] = (dict(snapshot_id=prior[-1]["snapshot_id"],
                elapsed_s=prior[-1]["elapsed_s"], **scored(prior[-1])) if prior else None)
    final = scored(usable[-1]) if usable else None
    if final:
        row.update(selected_snapshot_id=usable[-1]["snapshot_id"], first_solution_s=usable[0]["elapsed_s"],
                   final_safe=final["safe"], **{"final_" + key: final.get(key) for key in FINAL})
        if not final["safe"]:
            row["final_unsafe_reasons"] = final["geometry_errors"] + final["sem_reasons"]
    if kind == "rp_bit":
        if worker.get("requested_rp_bit") != request["rp_bit"]:
            errors.append("requested_rp_bit_mismatch")
        if worker.get("rp_counters", {}).get("heuristic_violations") != 0:
            errors.append("heuristic_bound_violation_on_incumbent")
        if not request["rp_bit"].get("refine", True) and any(e.get("event_origin") == "rp_refinement" for e in usable):
            errors.append("refinement_event_with_refine_off")
    if kind == "sem_refine":
        errors.extend(_replay_checks(directory, entry, request, worker, config["methods"][entry["method"]]))
        row["replay"] = _replay_summary(worker)
    safe = bool(final and final["safe"] and execution["status"] == "COMPLETED"
                and row["planner_status"] == "EXACT_SOLUTION")
    if full:
        strict = _strict(kind, directory, request, info, execution, oracle, field)
        row.update(strict_success=bool(strict["success"]), strict_errors=strict["correctness_errors"],
                   first_safe_s=strict.get("first_safe_s"), first_safe_latency_s=strict.get("first_safe_latency_s"))
        errors.extend(strict["correctness_errors"])
        # An unsafe final path is a planner failure, not a correctness error;
        # only a disagreement between the two independent verdicts is.
        if not strict["correctness_errors"] and bool(strict["success"]) != safe:
            errors.append("strict_and_fast_verdicts_disagree")
        if kind in ("rp_bit", "sem_refine"):
            row["events_j_sem"] = [[e["elapsed_s"], scored(e).get("j_sem")] for e in usable]
    row["success"] = safe and not errors
    row["attained_safe"] = row["success"]
    return row
