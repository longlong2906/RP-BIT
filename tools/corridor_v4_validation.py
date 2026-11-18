"""Independent raw-map/cloud validation for bounded corridor-v4 guidance.

The finite graph certificate and geometry come from independent Python
primitives.  This does not reproduce the approximate C++ label search or
claim that its candidate set contains a globally minimum-PRI path.
"""
from __future__ import annotations

from datetime import datetime
from fractions import Fraction
import math
from pathlib import Path

import numpy as np

from corridor_v3_validation import (FAILURES, READY, ROOT, _hard_edge, _metric_close,
    _risk_close, canonical_sha256, path_in_mask, reconstruct_mask, risk_exposure,
    risk_metrics, shortest_graph_length, witness_on_registered_graph)
from run_safe_length import _csv_path, _events, dump, read
from safe_length_field import file_hash
from safe_length_v2_validation import finite

VERSION = "independent_corridor_v4_v1"
POLICY_ID = "a3t_corridor_v4"


def _compact(path):
    return [p for i, p in enumerate(path) if i == 0 or p != path[i-1]]


def _orientation(a, b, c):
    # Exact arithmetic on the submitted binary64 coordinates, independent of
    # the C++ segment predicates.  No physical-width tolerance is introduced.
    ax, ay, bx, by, cx, cy = map(Fraction, (*a, *b, *c))
    value = (bx-ax)*(cy-ay)-(by-ay)*(cx-ax)
    return (value > 0)-(value < 0)


def _on_segment(a, b, p):
    return (_orientation(a, b, p) == 0 and
            all(min(x, y) <= z <= max(x, y) for x, y, z in zip(a, b, p)))


def _segments_touch(a, b, c, d):
    ab_c, ab_d = _orientation(a, b, c), _orientation(a, b, d)
    cd_a, cd_b = _orientation(c, d, a), _orientation(c, d, b)
    if ab_c*ab_d < 0 and cd_a*cd_b < 0:
        return True
    return ((ab_c == 0 and _on_segment(a, b, c)) or
            (ab_d == 0 and _on_segment(a, b, d)) or
            (cd_a == 0 and _on_segment(c, d, a)) or
            (cd_b == 0 and _on_segment(c, d, b)))


def path_is_simple(path):
    """Reject repeated vertices, crossings, nonadjacent touches and overlaps."""
    if (not path or any(not isinstance(p, (list, tuple)) or len(p) != 2 or
                       any(not finite(v) for v in p) for p in path)):
        return False
    points = _compact(path)
    if len({tuple(p) for p in points}) != len(points):
        return False
    for i in range(len(points)-2):
        a, b, c = points[i:i+3]
        if _orientation(a, b, c) == 0 and (_on_segment(a, b, c) or _on_segment(b, c, a)):
            return False
    for i in range(len(points)-1):
        a, b = points[i:i+2]
        for j in range(i+2, len(points)-1):
            c, d = points[j:j+2]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]) or
                    max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            if _segments_touch(a, b, c, d):
                return False
    return True


def ordered_subsequence_indices(raw, preview):
    raw = _compact(raw)
    if not raw or not preview or preview[0] != raw[0] or preview[-1] != raw[-1]:
        return None
    indices, start = [], 0
    for point in preview:
        found = next((i for i in range(start, len(raw)) if raw[i] == point), None)
        if found is None:
            return None
        indices.append(found)
        start = found+1
    return indices


def reconstruct_preview(oracle, field, raw, max_edge_checks=2000000):
    """Independent deterministic farthest-visible, strict-length shortcut pass."""
    raw = _compact(raw)
    cache = getattr(oracle, "_corridor_v4_preview_cache", {})
    cache_key = (id(field), tuple(map(tuple, raw)), max_edge_checks)
    if cache_key in cache:
        return cache[cache_key]
    if not path_is_simple(raw):
        raise ValueError("raw_guide_not_simple")
    # A simple exactly collinear path is monotone along its segment. Every
    # candidate shortcut therefore follows the identical lifted curve on H;
    # it cannot strictly reduce surface length, even across terrain folds.
    # Certify this structurally instead of comparing separately rounded sums.
    if all(_orientation(raw[0],raw[-1],point)==0 for point in raw):
        checks=(len(raw)-1)*(len(raw)-2)//2
        if checks>max_edge_checks:
            raise ValueError("preview_edge_check_limit")
        if len(cache)<1024:
            cache[cache_key]=(raw,checks)
            oracle._corridor_v4_preview_cache=cache
        return raw,checks
    mask = reconstruct_mask(field, raw, 1)
    lengths = [oracle.surface(a, b)[0] for a, b in zip(raw, raw[1:])]
    out, i, checks = [raw[0]], 0, 0
    while i < len(raw)-1:
        chosen = i+1
        for j in range(len(raw)-1, i+1, -1):
            checks += 1
            if checks > max_edge_checks:
                raise ValueError("preview_edge_check_limit")
            if not path_in_mask(field, [raw[i], raw[j]], mask):
                continue
            length = _hard_edge(oracle, field, raw[i], raw[j])
            old_length = math.fsum(lengths[i:j])
            if length is None or not old_length-length > 64*math.ulp(max(1.,abs(old_length),abs(length))):
                continue
            if not path_is_simple(out+[raw[j]]+raw[j+1:]):
                continue
            chosen = j
            break
        out.append(raw[chosen])
        i = chosen
    if len(cache) < 1024:
        cache[cache_key] = (out, checks)
        oracle._corridor_v4_preview_cache = cache
    return out, checks


def cell_delta_diagnostics(field, before, after):
    left, right = set(field.path_cells(before)), set(field.path_cells(after))
    def describe(cells):
        ordered = sorted(cells, key=lambda cell: (cell[1], cell[0]))
        values = [float(field.risk[j, i]) for i, j in ordered]
        return dict(count=len(ordered), risk_sum=math.fsum(values), cells=[list(p) for p in ordered])
    return dict(before=risk_metrics(field, before), after=risk_metrics(field, after),
                added=describe(right-left), removed=describe(left-right), retained_count=len(left & right))


def _exact_original_intervals(oracle, a, b):
    """Exact closed triangle clipping of the original binary64 edge."""
    aa, bb = [Fraction(float(v)) for v in a], [Fraction(float(v)) for v in b]
    lower = (np.minimum(a,b)-[oracle.x0,oracle.y0])/[oracle.dx,oracle.dy]
    upper = (np.maximum(a,b)-[oracle.x0,oracle.y0])/[oracle.dx,oracle.dy]
    i0,j0 = max(0,math.floor(lower[0])-1),max(0,math.floor(lower[1])-1)
    i1,j1 = min(oracle.nx-2,math.floor(upper[0])+1),min(oracle.ny-2,math.floor(upper[1])+1)
    intervals=[]
    for index in oracle.cell_triangle_ids[j0:j1+1,i0:i1+1].reshape(-1):
        if index < 0:
            continue
        vertices=[[Fraction(float(v)) for v in point] for point in oracle.vertices[index]]
        low,high,possible=Fraction(0),Fraction(1),True
        for k in range(3):
            p,q=vertices[k],vertices[(k+1)%3]
            ex,ey=q[0]-p[0],q[1]-p[1]
            start=ex*(aa[1]-p[1])-ey*(aa[0]-p[0])
            end=ex*(bb[1]-p[1])-ey*(bb[0]-p[0])
            if start<0 and end<0:
                possible=False
                break
            if start<0:
                low=max(low,start/(start-end))
            elif end<0:
                high=min(high,start/(start-end))
            if high<low:
                possible=False
                break
        if possible:
            intervals.append((low,high,oracle.coefficients[index]))
    return intervals


def _original_parameter_exposure(oracle, field, path):
    parts=[]
    for aa,bb in zip(path,path[1:]):
        a,b=np.asarray(aa,dtype=float),np.asarray(bb,dtype=float)
        delta=b-a
        cells=field.path_cells([aa,bb])
        if not cells or any(not field.valid[j,i] or not math.isfinite(float(field.risk[j,i])) for i,j in cells):
            raise ValueError("missing_sem_support")
        raster_cuts={0.,1.}
        for axis in (0,1):
            if delta[axis]==0:
                continue
            origin=field.bounds[axis]
            low,high=sorted(((a[axis]-origin)/field.resolution,(b[axis]-origin)/field.resolution))
            for k in range(math.ceil(low),math.floor(high)+1):
                value=(origin+k*field.resolution-a[axis])/delta[axis]
                if 0.<value<1.:
                    raster_cuts.add(float(value))
        def integrate(intervals,cuts):
            cuts=sorted(set(cuts)|{value for low,high,_ in intervals for value in (low,high)})
            output=[]
            for low,high in zip(cuts,cuts[1:]):
                if high<=low:
                    continue
                midpoint=(low+high)/2
                coefficient=next((c for left,right,c in intervals if left<=midpoint<=right),None)
                if coefficient is None:
                    raise ValueError("NoData gap")
                touched=field.path_cells([(a+float(midpoint)*delta).tolist()])
                if not touched or any(not field.valid[j,i] for i,j in touched):
                    raise ValueError("missing_sem_support")
                risk=max(float(field.risk[j,i]) for i,j in touched)
                speed=math.hypot(float(np.linalg.norm(delta)),float(coefficient[:2]@delta))
                output.append(risk*float(high-low)*speed)
            return output
        try:
            pieces=integrate(oracle.intervals(a,b),raster_cuts)
        except ValueError as error:
            if str(error)!="NoData gap":
                raise
            pieces=integrate(_exact_original_intervals(oracle,aa,bb),map(Fraction,raster_cuts))
        parts.extend(pieces)
    return math.fsum(parts)


def risk_exposure_v4(oracle, field, path):
    """Preserve inherited results; repair only its rounded-subchord gap error.

    The fallback never rounds new geometry endpoints, expands a triangle,
    drops a positive interval or treats absent terrain/support as zero.
    """
    try:
        return risk_exposure(oracle,field,path)
    except ValueError as error:
        if str(error)!="NoData gap":
            raise
        return _original_parameter_exposure(oracle,field,path)


def _candidate_surface_length(oracle,a,b):
    """Repair only a candidate metric's false float-clipping NoData gap.

    The exact intervals cover the original binary64 segment. Every positive
    interval requires actual terrain support; a genuine hole remains an error.
    This does not replace the independent final-path or footprint validator.
    """
    try:
        return oracle.surface(a,b)[0]
    except ValueError as error:
        if str(error)!="NoData gap":
            raise
        intervals=_exact_original_intervals(oracle,a,b)
        cuts=sorted({Fraction(0),Fraction(1),*(t for low,high,_ in intervals for t in (low,high))})
        delta=np.asarray(b,dtype=float)-np.asarray(a,dtype=float)
        parts=[]
        for low,high in zip(cuts,cuts[1:]):
            if high<=low:
                continue
            midpoint=(low+high)/2
            coefficient=next((c for left,right,c in intervals if left<=midpoint<=right),None)
            if coefficient is None:
                raise ValueError("NoData gap")
            speed=math.hypot(float(np.linalg.norm(delta)),float(coefficient[:2]@delta))
            parts.append(float(high-low)*speed)
        return math.fsum(parts)


def _stats(oracle, field, path):
    cache = getattr(oracle, "_corridor_v4_stats_cache", {})
    key = (id(field), tuple(map(tuple, path)))
    if key not in cache:
        measured = dict(length=math.fsum(_candidate_surface_length(oracle,a,b) for a,b in zip(path,path[1:])),
                        exposure=risk_exposure_v4(oracle, field, path), **risk_metrics(field, path))
        if len(cache) >= 1024:
            return measured
        cache[key] = measured
        oracle._corridor_v4_stats_cache = cache
    return cache[key]


def _node_ids_match(points, ids, start, goal, bounds):
    if not isinstance(ids, list) or len(points) != len(ids):
        return False
    nx = math.floor((bounds[2]-bounds[0])/.1)+1
    ny = math.floor((bounds[3]-bounds[1])/.1)+1
    count = nx*ny
    for point, node in zip(points, ids):
        if type(node) is not int or not 0 <= node <= count+1:
            return False
        expected = start if node == count else goal if node == count+1 else [bounds[0]+(node % nx)*.1, bounds[1]+(node//nx)*.1]
        if any(abs(actual-wanted) > 16*math.ulp(max(1.,abs(actual),abs(wanted))) for actual,wanted in zip(point,expected)):
            return False
    return True


def _check_stats(actual, expected, prefix, errors):
    if not isinstance(actual, dict):
        errors.append(prefix+":missing_stats")
        return
    for key in ("length", "exposure", "pri", "risk_sum"):
        close = _risk_close if key in ("pri", "risk_sum") else _metric_close
        if not close(actual.get(key), expected[key]):
            errors.append(prefix+":metric_mismatch:"+key)
    if actual.get("risk_count") != expected["risk_count"]:
        errors.append(prefix+":metric_mismatch:risk_count")
    quantiles = actual.get("risk_quantiles")
    if (not isinstance(quantiles, list) or len(quantiles) != 5 or
            not all(_risk_close(a, b) for a, b in zip(quantiles, expected["risk_quantiles"]))):
        errors.append(prefix+":metric_mismatch:risk_quantiles")


def record_corridor_receipt(directory):
    directory = Path(directory)
    source = directory/"worker.json"
    if not source.is_file():
        return None
    worker = read(source)
    corridor = worker.get("corridor")
    if corridor is None:
        return None
    if worker.get("requested_safe_length", {}).get("policy_id") != POLICY_ID:
        from corridor_v3_validation import record_corridor_receipt as previous
        return previous(directory)
    receipt = dict(version=VERSION, worker_sha256=file_hash(source), status=corridor.get("status"),
                   mask_sha256=canonical_sha256(corridor.get("mask", [])),
                   candidate_diagnostics_sha256=canonical_sha256(corridor.get("candidate_diagnostics", [])))
    for prefix, key, filename in (("guide", "guide_xy", "corridor_guide_xy.csv"),
                                  ("raw_guide", "raw_guide_xy", "corridor_raw_guide_xy.csv"),
                                  ("preview", "preview_xy", "corridor_preview_xy.csv")):
        path = directory/filename
        receipt[prefix+"_sha256"] = file_hash(path) if path.is_file() else None
        receipt[prefix+"_inline_sha256"] = canonical_sha256(corridor.get(key, []))
    target = directory/"corridor_receipt.json"
    if target.exists():
        if read(target) != receipt:
            raise ValueError("immutable corridor receipt mismatch")
    else:
        dump(target, receipt)
    return receipt


def validate_corridor(directory, worker, request, oracle, field):
    directory = Path(directory)
    corridor = worker.get("corridor", {})
    errors = []
    result = dict(status=corridor.get("status"), independent_valid=False, errors=errors,
                  no_continuous_infeasibility_claim=True, no_global_PRI_optimality_claim=True)
    policy = request.get("corridor_policy", request["safe_length"].get("corridor_policy", {}))
    receipt_path = directory/"corridor_receipt.json"
    receipt = read(receipt_path) if receipt_path.is_file() else {}
    if not receipt:
        errors.append("missing_corridor_receipt")
    for key, expected in (("version", VERSION), ("worker_sha256", file_hash(directory/"worker.json")),
                          ("status", corridor.get("status")),
                          ("mask_sha256", canonical_sha256(corridor.get("mask", []))),
                          ("candidate_diagnostics_sha256", canonical_sha256(corridor.get("candidate_diagnostics", [])))):
        if receipt.get(key) != expected:
            errors.append("corridor_receipt_mismatch:"+key)
    for key in ("graph_s", "guidance_s", "preview_s", "mask_s", "total_s"):
        if not finite(corridor.get(key)) or corridor[key] < 0:
            errors.append("invalid_corridor_time:"+key)
    if "beam_scheduler" in policy:
        scheduler = policy["beam_scheduler"]
        counters = corridor.get("counters", {})
        if corridor.get("beam_scheduler") != scheduler:
            errors.append("corridor_metadata_mismatch:beam_scheduler")
        counter_keys = ("labels_allocated", "beam_labels_allocated", "beam_replacement_labels",
                        "beam_role_replacements", "beam_equal_score_retains", "beam_retired",
                        "beam_expansions", "beam_stale_pops")
        if any(type(counters.get(key)) is not int or counters[key] < 0 for key in counter_keys):
            errors.append("invalid_beam_counters")
        else:
            labels = counters["labels_allocated"]
            beam = counters["beam_labels_allocated"]
            replacements = counters["beam_replacement_labels"]
            if not (0 <= replacements <= beam <= labels <= policy["max_labels"]
                    and replacements <= counters["beam_role_replacements"] <= 4 * replacements
                    and counters["beam_retired"] <= beam
                    and counters["beam_expansions"] + counters["beam_stale_pops"] <= beam):
                errors.append("inconsistent_beam_counters")
            result.update(beam_scheduler=scheduler,
                          beam_counters={key: counters[key] for key in counter_keys})
    status = corridor.get("status")
    if status != READY:
        if status not in FAILURES:
            errors.append("unknown_corridor_status")
        if (any(corridor.get(key) for key in ("guide_xy", "raw_guide_xy", "preview_xy", "mask")) or
                _events(directory, worker) or worker.get("planner_solve_called") is not False or
                worker.get("planner_status") == "EXACT_SOLUTION"):
            errors.append("corridor_failure_published_guide_mask_or_solution")
        if status in ("INVALID_START", "INVALID_GOAL"):
            point = request["start" if status == "INVALID_START" else "goal"]
            check = oracle.validate_path([point], point, point, geometric_safety_metrics=True, clearance_metrics=False)
            if (check["valid"] and field.score_path([point])["valid"] and
                    check["maximum_swept_footprint_slope_deg"] <= 35.):
                errors.append("corridor_claimed_invalid_endpoint_is_valid")
        result["independent_valid"] = not errors
        return result
    selector = policy.get("selector")
    dilation = policy.get("dilation_cells")
    if selector not in ("raw_pri", "preview_pri") or dilation not in (0, 1) or (selector == "raw_pri" and dilation != 1):
        errors.append("unregistered_selector_dilation")
    defaults = dict(label_bins=16, representatives_per_bin=4, max_preview_candidates=64,
                    max_preview_edge_checks=2000000, max_labels=2000000, max_memory_bytes=536870912)
    for key, expected in (("alpha", policy.get("alpha")), ("selector", selector), ("dilation_cells", dilation),
                          ("grid_spacing_m", .1), ("endpoint_radius_m", .2), ("guide_used_as_incumbent", False),
                          ("preview_used_as_incumbent", False),
                          ("preview_length_decrease_ulps", 64),
                          ("preview_length_decrease_scale", "max(1,abs(old),abs(new))"),
                          *((key, policy.get(key, value)) for key, value in defaults.items())):
        if corridor.get(key) != expected:
            errors.append("corridor_metadata_mismatch:"+key)
    raw, preview, guide = (corridor.get(key, []) for key in ("raw_guide_xy", "preview_xy", "guide_xy"))
    for prefix, key, filename, required in (("guide", "guide_xy", "corridor_guide_xy.csv", True),
                                          ("raw_guide", "raw_guide_xy", "corridor_raw_guide_xy.csv", True),
                                          ("preview", "preview_xy", "corridor_preview_xy.csv", selector == "preview_pri")):
        path, points = directory/filename, corridor.get(key, [])
        if receipt.get(prefix+"_inline_sha256") != canonical_sha256(points):
            errors.append("corridor_receipt_mismatch:"+prefix+"_inline_sha256")
        if required:
            if corridor.get(prefix+"_file") != filename or not path.is_file():
                errors.append("missing_or_unregistered_"+prefix+"_csv")
            else:
                if _csv_path(path) != points:
                    errors.append(prefix+"_csv_inline_mismatch")
                if receipt.get(prefix+"_sha256") != file_hash(path):
                    errors.append(prefix+"_csv_hash_mismatch")
        elif points or corridor.get(prefix+"_file") is not None or path.exists() or receipt.get(prefix+"_sha256") is not None:
            errors.append("inactive_preview_published")
    if not raw or not guide:
        errors.append("missing_corridor_guide")
        return result
    if not witness_on_registered_graph(raw, request["start"], request["goal"], field.bounds):
        errors.append("raw_guide_not_on_registered_graph")
    for name, path in (("raw_guide", raw), ("guide", guide), *(([("preview", preview)]) if preview else [])):
        if not path_is_simple(path):
            errors.append(name+"_not_simple")
    expected_guide = preview if selector == "preview_pri" else raw
    if guide != expected_guide:
        errors.append("guide_not_selected_representation")
    if selector == "preview_pri":
        if ordered_subsequence_indices(raw, preview) is None:
            errors.append("preview_not_ordered_raw_subsequence")
        try:
            rebuilt_preview, checks = reconstruct_preview(oracle, field, raw, policy.get("max_preview_edge_checks", 2000000))
            result["selected_preview_independent_edge_checks"] = checks
            if preview != rebuilt_preview:
                errors.append("preview_not_registered_farthest_visible_pass")
        except (ValueError, TypeError) as error:
            errors.append("preview_reconstruction_failed:"+str(error))
    alpha, L0 = policy.get("alpha"), corridor.get("L0")
    for name, path in (("raw_guide", raw), ("guide", guide)):
        check = oracle.validate_path(path, request["start"], request["goal"], geometric_safety_metrics=True, clearance_metrics=False)
        if not check["valid"] or not field.score_path(path)["valid"] or check["maximum_swept_footprint_slope_deg"] > 35.:
            errors.append(name+"_independent_hard_safety_failed")
        result[name+"_geometry"] = check
    if not finite(L0) or L0 < 0 or not finite(alpha) or alpha < 1:
        errors.append("invalid_guide_length_budget")
    else:
        certificate = shortest_graph_length(oracle, field, request["start"], request["goal"], L0)
        result["L0_certificate"] = certificate
        if not _metric_close(certificate["length"], L0):
            errors.append("L0_not_independent_graph_shortest_length")
    try:
        raw_stats, guide_stats = _stats(oracle, field, raw), _stats(oracle, field, guide)
        _check_stats(corridor.get("raw_guide_stats"), raw_stats, "raw_guide", errors)
        if preview:
            preview_stats = _stats(oracle, field, preview)
            _check_stats(corridor.get("preview_stats"), preview_stats, "preview", errors)
            result["guide_to_preview_cells"] = cell_delta_diagnostics(field, raw, preview)
        for key, stat in (("guide_length", "length"), ("guide_pri", "pri"), ("guide_exposure", "exposure"), ("risk_sum", "risk_sum")):
            close = _risk_close if stat in ("pri", "risk_sum") else _metric_close
            if not close(corridor.get(key), guide_stats[stat]):
                errors.append("guide_metric_mismatch:"+key)
        if corridor.get("risk_count") != guide_stats["risk_count"]:
            errors.append("guide_metric_mismatch:risk_count")
        quantiles = corridor.get("risk_quantiles")
        if not isinstance(quantiles, list) or len(quantiles) != 5 or not all(_risk_close(a, b) for a, b in zip(quantiles, guide_stats["risk_quantiles"])):
            errors.append("guide_metric_mismatch:risk_quantiles")
        if finite(L0) and finite(alpha) and raw_stats["length"] > alpha*L0+1e-8+1e-8*abs(alpha*L0):
            errors.append("raw_guide_exceeds_length_budget")
        rebuilt_mask = reconstruct_mask(field, guide, dilation)
        if corridor.get("mask") != rebuilt_mask:
            errors.append("corridor_mask_not_exact_selected_guide_dilation")
        if not path_in_mask(field, guide, corridor.get("mask")):
            errors.append("guide_outside_corridor_mask")
        result.update(raw_guide_stats=raw_stats, guide_stats=guide_stats,
                      reconstructed_mask_sha256=canonical_sha256(rebuilt_mask))
    except (ValueError, TypeError) as error:
        errors.append("guide_metrics_unavailable:"+str(error))
    candidates = corridor.get("candidate_diagnostics")
    if not isinstance(candidates, list) or not candidates or len(candidates) > policy.get("max_preview_candidates", 64)+2:
        errors.append("invalid_candidate_diagnostics_count")
    else:
        selected = [candidate for candidate in candidates if candidate.get("selected") is True]
        if len(selected) != 1 or selected[0].get("raw_xy") != raw or selected[0].get("preview_xy", []) != preview:
            errors.append("selected_candidate_diagnostics_mismatch")
        for i, candidate in enumerate(candidates):
            candidate_raw = candidate.get("raw_xy", [])
            if candidate.get("source") not in ("shortest", "v3", "beam"):
                errors.append("unknown_candidate_source:"+str(i))
            if not witness_on_registered_graph(candidate_raw, request["start"], request["goal"], field.bounds):
                errors.append("candidate_raw_not_on_registered_graph:"+str(i))
            if not _node_ids_match(candidate_raw, candidate.get("raw_node_ids"), request["start"], request["goal"], field.bounds):
                errors.append("candidate_node_ids_mismatch:"+str(i))
            if selector == "preview_pri":
                candidate_preview = candidate.get("preview_xy", [])
                if ordered_subsequence_indices(candidate_raw, candidate_preview) is None:
                    errors.append("candidate_preview_not_ordered_subsequence:"+str(i))
                elif not path_in_mask(field, candidate_preview, reconstruct_mask(field, candidate_raw, 1)):
                    errors.append("candidate_preview_outside_provisional_mask:"+str(i))
            for name, key in (("raw", "raw_xy"), *(([("preview", "preview_xy")]) if selector == "preview_pri" else [])):
                points = candidate.get(key, [])
                if not path_is_simple(points):
                    errors.append("candidate_not_simple:"+str(i)+":"+name)
                    continue
                try:
                    measured = _stats(oracle, field, points)
                    _check_stats(candidate.get(name+"_stats"), measured, "candidate:"+str(i)+":"+name, errors)
                except (ValueError, TypeError) as error:
                    errors.append("candidate_metrics_unavailable:"+str(i)+":"+str(error))
        metric_name = "preview_stats" if selector == "preview_pri" else "raw_stats"
        if selected:
            chosen = selected[0].get(metric_name, {})
            chosen_key = (chosen.get("pri", math.inf), chosen.get("length", math.inf), selected[0].get("raw_node_ids", []))
            for candidate in candidates:
                stats = candidate.get(metric_name) or {}
                if all(finite(stats.get(key)) for key in ("pri", "length")) and (stats["pri"], stats["length"], candidate.get("raw_node_ids", [])) < chosen_key:
                    errors.append("selected_candidate_not_minimum_published_PRI_length")
                    break
    for key, maximum in (("labels_allocated", policy.get("max_labels", 2000000)),
                         ("peak_memory_bytes", policy.get("max_memory_bytes", 536870912)),
                         ("preview_edge_checks", policy.get("max_preview_edge_checks", 2000000))):
        value = corridor.get("counters", {}).get(key)
        if type(value) is not int or not 0 <= value <= maximum:
            errors.append("invalid_resource_counter:"+key)
    result["independent_valid"] = not errors
    return result


def validate_trial(directory, request, info, execution, *, oracle=None, field=None):
    if request.get("safe_length", {}).get("policy_id") != POLICY_ID:
        from corridor_v3_validation import validate_trial as previous
        row = previous(directory, request, info, execution, oracle=oracle, field=field)
        if row.get("success") and field is not None and oracle is not None:
            worker = read(Path(directory)/"worker.json")
            selected = row.get("selected_snapshot_id")
            if selected:
                points = _csv_path(Path(directory)/"snapshots"/selected/"path_xy.csv")
                metrics = risk_metrics(field, points)
                row.update(final_risk_sum=metrics["risk_sum"], final_risk_count=metrics["risk_count"],
                           final_risk_quantiles=metrics["risk_quantiles"], final_exposure=risk_exposure(oracle, field, points))
        return row
    from native_paper_oracle import NativePaperOracle
    from native_paper_v3_sem_evaluator import SharedSEMField
    directory = Path(directory)
    for key, hashkey in (("map_path", "map_sha256"), ("cloud_path", "cloud_sha256")):
        if file_hash(info[key]) != info[hashkey]:
            raise ValueError("raw input hash mismatch during corridor validation")
    row = dict(success=False, success_geometry=False, attained_safe=False, first_safe_s=None,
               first_safe_latency_s=None, final_cost_terrain_m=None, final_pri=None,
               final_swept_slope_deg=None, events=[], correctness_errors=[],
               execution_status=execution["status"], planner_status=None,
               field_build_s=execution.get("field_build_s"), solve_time_s=None,
               selection="last_published_incumbent_in_budget_before_validation")
    if not (directory/"worker.json").is_file():
        row["unavailable_reason"] = "no_worker_output"
        if execution["status"] == "COMPLETED":
            row["correctness_errors"].append("completed_process_missing_worker_output")
        return row
    worker = read(directory/"worker.json")
    oracle = oracle or NativePaperOracle(info["map_path"])
    field = field or SharedSEMField(np.load(info["cloud_path"], allow_pickle=False), info["bounds"])
    errors = row["correctness_errors"]
    policy = request["safe_length"]
    effective = worker.get("effective_safe_length", {})
    row.update(planner_status=worker.get("planner_status", worker.get("status")),
               worker_execution_status=worker.get("execution_status"), termination_reason=worker.get("termination_reason"),
               preprocessing_s=worker.get("preprocessing_s"),
               solve_time_s=worker.get("actual_solve_wall_s"),
               ompl_version=worker.get("ompl_version"), compiler=worker.get("compiler"),
               effective_risk_cap=40., corridor_s=worker.get("corridor", {}).get("total_s"),
               corridor_status=worker.get("corridor", {}).get("status"))
    row.update({key: worker.get("corridor", {}).get(key) for key in
                ("graph_s", "guidance_s", "preview_s", "mask_s", "guide_length", "guide_pri", "guide_exposure", "L0")})
    if worker.get("requested_safe_length") != policy:
        errors.append("requested_safety_policy_mismatch")
    for key, expected in (("policy_id", POLICY_ID), ("slope_cap_deg", 35.),
                          ("atlas_sampling", policy.get("atlas_sampling")),
                          ("thresholds_immutable_during_solve", True),
                          ("domain", "immutable_centerline_corridor")):
        if effective.get(key) != expected:
            errors.append("effective_safety_policy_mismatch:"+key)
    checked = validate_corridor(directory, worker, request, oracle, field)
    row["corridor"] = checked
    errors.extend(checked["errors"])
    if checked["status"] == READY:
        if effective.get("risk_cap") != 40.:
            errors.append("effective_safety_policy_mismatch:risk_cap")
        if worker.get("planner_solve_called") is not True:
            errors.append("planner_not_called_after_guidance_ready")
    if effective.get("guide_used_as_incumbent", worker.get("corridor", {}).get("guide_used_as_incumbent")) is not False:
        errors.append("guide_incumbent_contract_missing")
    effective_policy = worker.get("effective_policy", {})
    if ("safety_atlas_sampling" in effective_policy and
            effective_policy["safety_atlas_sampling"] != policy.get("atlas_sampling")):
        errors.append("effective_atlas_policy_mismatch")
    if "batch_policy" in effective_policy:
        for key in ("batch_after_first_solution", "samples_per_batch"):
            if key == "samples_per_batch" and request["planner_id"] != "bitstar_a3":
                continue
            if effective_policy.get(key) != request[key]:
                errors.append("effective_batch_policy_mismatch:"+key)
    before, total_before = None, None
    if worker.get("solve_started_utc") and execution.get("launch_utc"):
        parse = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        before = (parse(worker["solve_started_utc"])-parse(execution["launch_utc"])).total_seconds()
        if before < 0:
            raise ValueError("negative startup time")
        if execution.get("started_utc"):
            total_before = (parse(worker["solve_started_utc"])-parse(execution["started_utc"])).total_seconds()
            if total_before < before:
                raise ValueError("attempt started after launch")
    row.update(launch_to_solve_s=before, attempt_to_solve_s=total_before,
               latency_origin="attempt_start_before_input_hash_field_build_and_process_launch")
    events = _events(directory, worker)
    seen, previous_time, previous_cost, selected = set(), -1., math.inf, None
    mask = worker.get("corridor", {}).get("mask", [])
    for event in events:
        sid = event["snapshot_id"]
        if not isinstance(sid, str) or Path(sid).name != sid or sid in (".", "..") or sid in seen:
            raise ValueError("unsafe or repeated snapshot id")
        seen.add(sid)
        elapsed = event.get("elapsed_s", event.get("timestamp_s"))
        cost = event.get("native_objective_value", event.get("objective_value", event.get("best_cost_terrain_m")))
        if not finite(elapsed) or elapsed < max(0., previous_time) or not finite(cost) or cost > previous_cost+1e-8:
            raise ValueError("invalid incumbent time or cost order")
        path = _csv_path(directory/"snapshots"/sid/"path_xy.csv")
        if event.get("path_xy") is not None and event["path_xy"] != path:
            raise ValueError("snapshot CSV differs from callback coordinates")
        geometry = oracle.validate_path(path, request["start"], request["goal"],
                                        claimed_cost=event.get("cost_terrain_m", event.get("best_cost_terrain_m")),
                                        geometric_safety_metrics=True, clearance_metrics=False)
        sem = field.score_path(path)
        common = bool(geometry["valid"] and sem["valid"] and geometry["maximum_swept_footprint_slope_deg"] <= 35.)
        inside = path_in_mask(field, path, mask)
        safe = common and inside
        if not geometry["valid"]:
            errors.append(sid+":independent_geometry_invalid")
        if not safe:
            errors.append(sid+":corridor_hard_safety_contract_failed")
        if not _metric_close(geometry["cost_terrain_m"], cost):
            errors.append(sid+":surface_length_objective_mismatch")
        measured = dict(snapshot_id=sid, elapsed_s=elapsed, in_budget=elapsed <= request["planning_budget_s"],
                        safe=safe, common_safe=common, geometry_valid=geometry["valid"],
                        geometry=geometry, sem=sem, inside_corridor=inside)
        if common:
            measured.update(risk_metrics=risk_metrics(field, path), exposure=risk_exposure_v4(oracle, field, path))
        row["events"].append(measured)
        if measured["in_budget"]:
            selected = measured
            if safe and row["first_safe_s"] is None:
                row["first_safe_s"] = elapsed
                if total_before is not None:
                    row["first_safe_latency_s"] = total_before+elapsed
                if before is not None:
                    row["first_safe_launch_latency_s"] = before+elapsed
        previous_time, previous_cost = elapsed, cost
    if selected is not None:
        final = directory/"path_xy.csv"
        if not final.is_file() or _csv_path(final) != _csv_path(directory/"snapshots"/events[-1]["snapshot_id"]/"path_xy.csv"):
            errors.append("final_path_not_last_published_incumbent")
        metrics = selected.get("risk_metrics", {})
        row.update(selected_snapshot_id=selected["snapshot_id"], success_geometry=selected["geometry_valid"],
                   final_cost_terrain_m=selected["geometry"]["cost_terrain_m"], final_pri=selected["sem"]["pri"],
                   final_swept_slope_deg=selected["geometry"]["maximum_swept_footprint_slope_deg"],
                   final_risk_sum=metrics.get("risk_sum"), final_risk_count=metrics.get("risk_count"),
                   final_risk_quantiles=metrics.get("risk_quantiles"), final_exposure=selected.get("exposure"))
    elif row["planner_status"] == "EXACT_SOLUTION":
        errors.append("exact_solution_without_in_budget_snapshot")
    if events and request.get("batch_after_first_solution") and worker.get("batch_transition_count") != 1:
        errors.append("adaptive_batch_transition_not_observed")
    row["success"] = bool(selected is not None and selected["safe"] and execution["status"] == "COMPLETED" and
                          row["planner_status"] == "EXACT_SOLUTION" and not errors)
    row["attained_safe"] = bool(row["first_safe_s"] is not None and execution["status"] == "COMPLETED" and not errors)
    row["late_snapshot_count"] = sum(not e["in_budget"] for e in row["events"])
    if execution["status"] != "COMPLETED" or errors:
        row["safe_snapshot_observed_in_failed_attempt"] = row["first_safe_s"] is not None
        row["first_safe_s"], row["first_safe_latency_s"] = None, None
    if selected is not None and selected["safe"]:
        final_points = _csv_path(directory/"snapshots"/selected["snapshot_id"]/"path_xy.csv")
        raw = worker["corridor"].get("raw_guide_xy", [])
        preview = worker["corridor"].get("preview_xy", [])
        guide = worker["corridor"].get("guide_xy", [])
        row["cell_delta_diagnostics"] = dict(
            raw_guide_to_final=cell_delta_diagnostics(field, raw, final_points),
            selected_guide_to_final=cell_delta_diagnostics(field, guide, final_points))
        if preview:
            row["cell_delta_diagnostics"].update(
                raw_guide_to_preview=cell_delta_diagnostics(field, raw, preview),
                preview_to_final=cell_delta_diagnostics(field, preview, final_points))
    row["oracle_sources"] = {name: file_hash(ROOT/"tools"/name) for name in
                             ("independent_verifier.py", "native_paper_oracle.py", "native_paper_v3_sem_evaluator.py",
                              "safe_length_v2_validation.py", "corridor_v3_validation.py", "corridor_v4_validation.py")}
    return row
