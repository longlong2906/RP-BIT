"""Smoke-only registration for contextual PRI segment search and v1 full.

Each backend run produces four correlated selectors. No pilot, confirmation,
full campaign or automatic alpha selection is registered by this protocol.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import random

from corridor_v4_execution import digest, file_hash, read
import pri_pathsearch_protocol as previous

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_pri_segment_v2.json"
PROTOCOL_ID = "a3t_pri_segment_v2"
SEGMENT = PROTOCOL_ID
BASELINE = previous.FULL
METHODS = (SEGMENT, BASELINE)
SELECTOR_BACKENDS = METHODS
ALPHAS = previous.ALPHAS
PHASES = ("smoke_serial", "smoke_parallel")
MODES = {SEGMENT: "segment", BASELINE: "full"}
SEGMENT_POLICY = dict(operator="contextual_bounded_beam",
    schedule=["shortcut", "deform", "segment_reduce", "segment_contract"],
    max_nodes=24, beam_width=8, max_expansions=128, max_depth=8,
    risk_target_ratio=.99, target_cap_indices=[0, 1, 2, 2, 3],
    objective="min_surface_length_subject_to_whole_path_PRI_cap",
    prefix_PRI_pruning=False, shared_outside_cells=True)


def default_config():
    old = previous.default_config()
    profiles = {phase: dict(deepcopy(old["profiles"][phase]), planned_trials=14) for phase in PHASES}
    return dict(protocol_id=PROTOCOL_ID, schema_version=1,
        model=deepcopy(old["model"]), safety=deepcopy(old["safety"]),
        algorithm=deepcopy(old["algorithm"]), backends=list(METHODS),
        pathsearch=dict(deepcopy(old["pathsearch"]), modes=MODES,
                        segment_search=deepcopy(SEGMENT_POLICY)),
        profiles=profiles, execution=deepcopy(old["execution"]),
        analysis=dict(final_selection="in_budget_before_independent_validation_no_rescue",
            comparator=BASELINE, selectors_are_correlated_outputs=True,
            report_metrics=["cost_terrain_m", "pri", "risk_sum", "risk_count", "exposure"],
            alpha_selection=False, stop_after_smoke=True,
            pilot_registered=False, confirmation_registered=False, full_registered=False,
            evidence_scope="correctness_and_fixture_smoke_only"))


def validate_config(config):
    if digest(config) != digest(default_config()):
        raise ValueError("unregistered segment smoke configuration")
    return config


def load_config(path=CONFIG):
    return validate_config(read(path))


def entry_matches(actual, registered):
    return previous.entry_matches(actual, registered)


def inventory(config, phase="smoke_serial", selected_alpha=None):
    validate_config(config)
    if phase not in PHASES or selected_alpha is not None:
        raise ValueError("segment protocol permits only smoke; no alpha selection or later phase")
    settings = config["profiles"][phase]
    maps = [(seed, family) for seed in settings["map_seeds"] for family in settings["families"]]
    random.Random(config["execution"]["shuffle_seed"]).shuffle(maps)
    entries = []
    for block, (seed, family) in enumerate(maps):
        for planner_seed in settings["planner_seeds"]:
            order = METHODS[block % 2:] + METHODS[:block % 2]
            for method in order:
                map_id = f"{family}_{seed}"
                entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}", phase=phase,
                    selected_alpha=None, map_id=map_id, family=family, map_seed=seed,
                    planner_seed=planner_seed, query_id="q0", method=method, alpha=None,
                    sem_lambda=None, policy="pri_pathsearch", mode=MODES[method],
                    block_index=block, status="NOT_RUN"))
    if len(entries) != settings["planned_trials"]:
        raise AssertionError("registered segment physical trial count differs")
    return dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
        status="PREPARED_INVENTORY", selected_alpha=None, config_sha256=digest(config),
        entries=entries, planned_trials=len(entries), map_count=len(maps),
        planning_budget_s=settings["planning_budget_s"], planner_execution_count=0,
        seed_replacements=[], stop_after_smoke=True)


def source_hashes():
    sources = previous.source_hashes()
    sources[CONFIG.relative_to(ROOT).as_posix()] = file_hash(CONFIG)
    for pattern in ("test_pri_segment_*.py",):
        for path in sorted((ROOT / "tests").glob(pattern)):
            sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    for path in sorted((ROOT / "tests/integration").glob("*pri_segment*")):
        if path.is_file():
            sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return sources


def request_for(entry, info, config, phase=None, field_path=None, field_metadata=None, bridge=None):
    phase = phase or entry.get("phase")
    expected = next((row for row in inventory(config, phase)["entries"]
                     if row["trial_id"] == entry.get("trial_id")), None)
    if expected is None or phase != entry.get("phase") or not entry_matches(entry, expected):
        raise ValueError("unregistered segment smoke slot")
    if info.get("map_id") != entry["map_id"] or info.get("query_id", "q0") != "q0":
        raise ValueError("segment request map/query differs")
    from run_safe_length import request_for as safe_request
    request = safe_request("atlas_adaptive100_200", info,
        dict(planning_budget_s=config["profiles"][phase]["planning_budget_s"], risk_cap=40., slope_cap_deg=35.),
        field_path, field_metadata, bridge)
    request.update(variant_id=entry["method"], planner_seed=entry["planner_seed"],
        objective_type="surface_length", pri_pathsearch={"mode":MODES[entry["method"]]})
    return request
