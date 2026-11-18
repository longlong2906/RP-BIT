"""Frozen development protocol: five real backends, four correlated selectors.

This protocol stops after pilot. Its reused development maps cannot establish
held-out superiority, and no confirmation/full execution is registered.
"""
from __future__ import annotations

import math
from pathlib import Path
import random

from corridor_v4_execution import digest, read
from pri_portfolio_protocol import default_config as portfolio_defaults

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_pri_pathsearch_v1.json"
PROTOCOL_ID = "a3t_pri_pathsearch_v1"
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
ALPHAS = (1., 1.025, 1.05, 1.1)
NATIVE = "sem_native"
LENGTH = "a3t_bitstar_length"
LEGACY = "a3t_pri_portfolio_a1100"
SHORTCUT = "a3t_pri_pathsearch_shortcut"
FULL = "a3t_pri_pathsearch_v1"
METHODS = (LENGTH, LEGACY, SHORTCUT, FULL, NATIVE)
SELECTOR_BACKENDS = (SHORTCUT, FULL)
PHASES = ("smoke_serial", "smoke_parallel", "pilot")
MODES = {LENGTH: "length", SHORTCUT: "shortcut", FULL: "full"}


def selector_method(backend, alpha):
    if backend not in SELECTOR_BACKENDS or type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha not in ALPHAS:
        raise ValueError("unregistered backend/selector")
    return backend + "__a" + str(round(alpha * 1000))


def default_config():
    previous = portfolio_defaults()
    return dict(protocol_id=PROTOCOL_ID, schema_version=1,
        model=previous["model"], safety=previous["safety"], algorithm=previous["algorithm"],
        native_sem=previous["native_sem"],
        backends=list(METHODS),
        pathsearch=dict(modes=MODES, selector_alphas=list(ALPHAS), anchor_fraction=.25,
            bitstar_slice_fraction=.005, optimizer_slice_fraction=.015,
            archive_length_bins=16, archive_paths_per_bin=4, internal_length_cap_ratio=1.1,
            replacement_windows_m=[.5, 1., 2., 4., 8., 16., "whole"],
            displacement_resolution_multipliers=[1, 2, 4, 8, 16, 32, 64], splice_distance_m=1.,
            objective="surface_length", selection_metric="exact_unique_closed_cell_PRI",
            selectors_are_correlated_outputs=True, selector_count_is_not_trial_count=True),
        legacy_portfolio=dict(length_cap_ratio=1.1, implementation="unchanged_pri_portfolio_v1"),
        profiles={name: previous["profiles"][name] for name in PHASES},
        execution=previous["execution"],
        analysis=dict(final_selection="in_budget_before_independent_validation_no_rescue",
            smoke_selects_alpha=False, pilot_is_development=True,
            tuning_common_cohort="SEM_and_all_four_full_selectors_final_safe",
            tuning_requires_every_map=True,
            tuning_gate="each_family:mean_delta_PRI<0;mean_delta_L_H<0;observed_delta_PSR>=0",
            ranking_order=["maximum_family_delta_PRI", "mean_family_delta_L_H", "alpha"],
            selector_unit="same_physical_backend_attempt; paired_not_independent",
            psr_gate="observed_no_decrease_not_population_noninferiority",
            stop_after_pilot=True, confirmation_registered=False))


def validate_config(config):
    if digest(config) != digest(default_config()):
        raise ValueError("unregistered pathsearch configuration")
    return config


def load_config(path=CONFIG):
    return validate_config(read(path))


def entry_matches(actual, registered):
    keys = [key for key in registered if key != "status"]
    return digest({key: actual.get(key) for key in keys}) == digest({key: registered[key] for key in keys})


def inventory(config, phase="pilot", selected_alpha=None):
    validate_config(config)
    if phase not in PHASES or selected_alpha is not None:
        raise ValueError("only smoke/pilot are registered; selection never authorizes more execution")
    settings = config["profiles"][phase]
    maps = [(seed, family) for seed in settings["map_seeds"] for family in settings["families"]]
    random.Random(config["execution"]["shuffle_seed"]).shuffle(maps)
    entries, block = [], 0
    for map_seed, family in maps:
        for planner_seed in settings["planner_seeds"]:
            offset = block % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                map_id = f"{family}_{map_seed}"
                entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}", phase=phase,
                    selected_alpha=None, map_id=map_id, family=family, map_seed=map_seed,
                    planner_seed=planner_seed, query_id="q0", method=method,
                    alpha=1.1 if method == LEGACY else None, sem_lambda=None,
                    policy="native" if method == NATIVE else "pri_portfolio" if method == LEGACY else "pri_pathsearch",
                    mode=MODES.get(method), block_index=block, status="NOT_RUN"))
            block += 1
    if len(entries) != settings["planned_trials"]:
        raise AssertionError("registered physical trial count mismatch")
    return dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
        status="PREPARED_INVENTORY", selected_alpha=None, config_sha256=digest(config), entries=entries,
        planned_trials=len(entries), map_count=len(maps), planning_budget_s=settings["planning_budget_s"],
        planner_execution_count=0, seed_replacements=[], stop_after_pilot=True)


def source_hashes():
    from safe_length_v2_protocol import source_hashes as scientific_sources
    from corridor_v4_execution import file_hash
    sources = scientific_sources()
    sources[CONFIG.relative_to(ROOT).as_posix()] = file_hash(CONFIG)
    for path in sorted((ROOT / "tests").glob("test_pri_pathsearch_*.py")):
        sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    for path in sorted((ROOT / "tests/integration").glob("*pri_pathsearch*")):
        if path.is_file():
            sources[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return sources


def request_for(entry, info, config, phase=None, field_path=None, field_metadata=None, bridge=None):
    phase = phase or entry.get("phase")
    expected = next((row for row in inventory(config, phase)["entries"]
                     if row["trial_id"] == entry.get("trial_id")), None)
    if phase != entry.get("phase") or expected is None or not entry_matches(entry, expected):
        raise ValueError("unregistered pathsearch map/query/seed/backend slot")
    if info.get("map_id") != entry["map_id"] or info.get("query_id", "q0") != "q0":
        raise ValueError("request input differs from registered map/query")
    budget = config["profiles"][phase]["planning_budget_s"]
    if entry["method"] == NATIVE:
        from native_paper_v3_protocol import request_for as native_request, default_config as native_config
        request = native_request("sem", dict(start=info["start"], goal=info["goal"]),
            info["map_path"], info["cloud_path"], info["cloud_sha256"], entry["planner_seed"], budget, native_config())
        request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(bridge).resolve()) if bridge else None,
            library_search_paths=[str(Path(bridge).resolve().parent), "C:/mingw64/bin"] if bridge else [],
            variant_id=NATIVE, algorithm_budget_policy="pri_portfolio_v1")
        return request
    from run_safe_length import request_for as safe_request
    request = safe_request("atlas_adaptive100_200", info,
        dict(planning_budget_s=budget, risk_cap=40., slope_cap_deg=35.), field_path, field_metadata, bridge)
    request.update(variant_id=entry["method"], planner_seed=entry["planner_seed"], objective_type="surface_length")
    if entry["method"] == LEGACY:
        request["pri_portfolio"] = {"length_cap_ratio": 1.1}
    else:
        request["pri_pathsearch"] = {"mode": MODES[entry["method"]]}
    return request
