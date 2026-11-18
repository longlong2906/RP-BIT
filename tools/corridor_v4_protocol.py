"""Registered methods, paired schedules and provenance for corridor v4."""
from __future__ import annotations

from pathlib import Path

import corridor_v3_protocol as v3
from run_safe_length import digest, read
from safe_length_field import file_hash

ROOT = v3.ROOT
CONFIG = ROOT / "configs/a3t_corridor_v4.json"
PROTOCOL_ID = "a3t_corridor_v4"
ALPHAS = (1.20, 1.30, 1.40)
PRIMARY = "a3t_corridor_v4"
COMMON = "sem_common_corridor_v4"
LEGACY = "a3t_corridor_v3"
RAW = "a3t_corridor_v4_raw_d1"
WIDE = "a3t_corridor_v4_preview_d1"
NATIVE = "sem_native"
METHOD_POLICIES = {LEGACY: "legacy", RAW: "raw", WIDE: "preview_d1", NATIVE: "native"}


def default_config(workers=4, validation_workers=4, *, beam_scheduler=None):
    if any(type(n) is not int or n not in (1, 4) for n in (workers, validation_workers)):
        raise ValueError("registered worker counts are 1 and 4")
    config = v3.default_config()
    config.update(schema_version=4, protocol_id=PROTOCOL_ID)
    config["corridor_policy"].update(alpha=1.2, dilation_cells=0, selector="preview_pri",
        representatives_per_bin=4, max_preview_candidates=64, max_preview_edge_checks=2_000_000)
    config["guidance"].update(
        approximation="four_representatives_per_slack_bin; bounded_guide_search",
        guide_selection="unique_cell_PRI_after_one_shortcut_pass_then_surface_length_then_node_sequence",
        preview_dilation_cells=1, label_representatives=["pri", "exposure", "length", "maximum_risk"])
    if beam_scheduler is not None:
        if beam_scheduler != "length_first_stable":
            raise ValueError("unregistered v4 beam scheduler")
        config["corridor_policy"]["beam_scheduler"] = beam_scheduler
        config["guidance"]["beam_order"] = "prefix_length_then_pri_then_insertion"
        config["guidance"]["equal_score_policy"] = "keep_first"
    config["execution"].update(max_workers=workers, validation_workers=validation_workers,
        start_method="spawn", phase_order=["collection", "independent_validation"],
        reserve_bytes_per_collection_worker=1024**3,
        reserve_bytes_per_validation_worker=512*1024**2)
    config["profiles"]["smoke"]["alpha"] = 1.2
    config["profiles"]["pilot"]["alphas"] = list(ALPHAS)
    config["profiles"]["full"].update(map_seeds=list(range(35001, 35011)),
        planner_seeds=list(range(356001, 356011)))
    return config


def validate_config(config):
    execution = config["execution"]
    scheduler = config["corridor_policy"].get("beam_scheduler")
    if config != default_config(execution["max_workers"], execution["validation_workers"],
                                beam_scheduler=scheduler):
        raise ValueError("unregistered corridor v4 configuration")
    return config


def load_config(path=CONFIG):
    return validate_config(read(path))


def method_specs(profile, selected_alpha=None):
    if profile == "smoke":
        return [(name, None if name == NATIVE else 1.2)
                for name in (LEGACY, RAW, WIDE, PRIMARY, COMMON, NATIVE)]
    if profile == "pilot":
        return [(name + "_a" + v3.alpha_suffix(alpha), alpha)
                for alpha in ALPHAS for name in (PRIMARY, COMMON)] + [
                    (LEGACY, 1.2), (RAW, 1.2), (WIDE, 1.2), (NATIVE, None)]
    if profile == "full":
        if selected_alpha is not None and selected_alpha not in ALPHAS:
            raise ValueError("full alpha must be selected from the registered pilot")
        return [(PRIMARY, selected_alpha), (COMMON, selected_alpha), (LEGACY, 1.2), (NATIVE, None)]
    raise ValueError("unknown corridor v4 profile")


def inventory(config, profile="smoke", selected_alpha=None):
    validate_config(config)
    settings = config["profiles"][profile]
    methods = method_specs(profile, selected_alpha)
    entries = []
    block = 0
    for family in settings["families"]:
        for seed in settings["map_seeds"]:
            map_id = f"{family}_{seed}"
            for planner_seed in settings["planner_seeds"]:
                offset = block % len(methods)
                for method, alpha in methods[offset:] + methods[:offset]:
                    entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}",
                        map_id=map_id, query_id="q0", family=family, map_seed=seed,
                        planner_seed=planner_seed, method=method, alpha=alpha,
                        policy=METHOD_POLICIES.get(method, "preview_d0"),
                        block_index=block, status="NOT_RUN"))
                block += 1
    return dict(protocol_id=PROTOCOL_ID, schema_version=4, profile=profile,
        status="PREPARED_INVENTORY", config_sha256=digest(config), entries=entries,
        planned_trials=len(entries), map_count=len(settings["families"])*len(settings["map_seeds"]),
        planning_budget_s=settings["planning_budget_s"], requires_selected_pilot=profile == "full",
        planner_execution_count=0, seed_replacements=[])


def source_hashes():
    hashes = v3.source_hashes()
    files = ("tools/corridor_v3_protocol.py", "tools/corridor_v3_validation.py",
             "tools/analyze_corridor_v3.py", "tools/run_corridor_v4.py", "tools/corridor_v4_execution.py",
             "tools/corridor_v4_protocol.py", "tools/corridor_v4_validation.py",
             "tools/corridor_v4_l0_cache.py", "tools/analyze_corridor_v4.py",
             "include/terrain25d/Corridor.hpp", "src/Corridor.cpp")
    hashes.update({name: file_hash(ROOT / name) for name in files})
    return hashes


def request_for(entry, info, config, profile, field_path=None, field_metadata=None, bridge=None):
    method, alpha = entry["method"], entry["alpha"]
    specs = dict(method_specs(profile, alpha if profile == "full" and method in (PRIMARY, COMMON) else None))
    if method not in specs or specs[method] != alpha:
        raise ValueError("unregistered v4 method/alpha")
    policy = METHOD_POLICIES.get(method, "preview_d0")
    if entry["policy"] != policy:
        raise ValueError("unregistered v4 method policy")
    # Reuse the registered v3 worker contracts for unchanged planner parameters
    # and SEM native reconstruction. Only the corridor policy varies.
    common = method.startswith(COMMON)
    registered = (NATIVE if method == NATIVE else
                  "sem_common_corridor_a1p2" if common else "a3t_corridor_a1p2")
    base = v3.default_config()
    base["profiles"]["pilot"]["planning_budget_s"] = config["profiles"][profile]["planning_budget_s"]
    request = v3.request_for(dict(entry, method=registered,
                                 alpha=None if method == NATIVE else 1.2),
                             info, base, "pilot", field_path, field_metadata, bridge)
    request["variant_id"] = method
    if policy not in ("native", "legacy"):
        safety = request["safe_length"]
        safety["policy_id"] = PROTOCOL_ID
        safety["corridor_policy"] = dict(config["corridor_policy"], alpha=alpha,
            dilation_cells=int(policy != "preview_d0"),
            selector="raw_pri" if policy == "raw" else "preview_pri")
    return request


def check_selection(path, config, *, binary_sha256=None):
    from analyze_corridor_v4 import build_report

    selection = read(path)
    sources = source_hashes()
    if (selection.get("status") != "SELECTED" or selection.get("protocol_id") != PROTOCOL_ID
            or selection.get("config_sha256") != digest(config)
            or selection.get("sources_sha256") != digest(sources)):
        raise ValueError("full requires a hash-bound selected v4 pilot with the same worker policy")
    records = {}
    for key in ("manifest", "summary"):
        artifact = Path(selection["pilot_" + key + "_path"])
        if file_hash(artifact) != selection["pilot_" + key + "_sha256"]:
            raise ValueError("pilot evidence hash mismatch")
        records[key] = read(artifact)
    pilot = records["manifest"]
    if (pilot.get("profile") != "pilot" or pilot.get("sources") != sources
            or pilot.get("binary_sha256") != selection.get("binary_sha256")
            or binary_sha256 is not None and pilot["binary_sha256"] != binary_sha256):
        raise ValueError("pilot source/binary/profile mismatch")
    actual = build_report(records["summary"], pilot, config)["pilot_selection"]
    if any(actual[key] != selection[key] for key in ("status", "selected_alpha", "selected_method")):
        raise ValueError("pilot selection does not satisfy the registered gates")
    return selection
