"""Shared configuration/provenance helpers, independent of geometry algorithms."""
import hashlib
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import shutil
import sys
import tempfile
import time
import yaml

ROOT=Path(__file__).resolve().parents[1]
KEYS={
    "root":"schema_version project language model algorithm execution logging analysis profiles optional_ablations optional_extensions",
    "model":"state local_connector objective surface_interpolation cell_diagonal height_units coordinate_units nodata_policy obstacle_contact_policy include_outside_map_as_forbidden robot_radius_m safety_margin_m max_slope_deg roughness_limit_m roughness_window_m goal goal_roundoff_tolerance_m numerical_abs_tolerance_m numerical_rel_tolerance",
    "algorithm":"nearest_neighbor_metric bitstar_connection bitstar_samples_per_batch bitstar_rewire_factor bitstar_just_in_time_sampling informed_sample_attempts_per_call informed_measure_policy edge_cache graph_pruning_requested fail_if_requested_variant_not_effective simplification approximate_solution_counts_as_success",
    "execution":"jobs one_process_per_trial seed_before_creating_planner shuffle_trial_order_seed save_trial_manifest resume_by_config_hash watchdog_grace_s memory_limit_mb default_profile full_requires_explicit_flag timing_origin record_preprocessing_separately record_setup_separately offline_validation_separately",
    "logging":"progress_interval_s save_improvement_events save_improvement_paths save_ompl_benchmark_log save_path_xy save_path_surface_breakpoints save_full_tree keep_failures strict_json_no_nan_or_infinity",
    "analysis":"target_relative_tolerances require_reference_provenance confidence_level bootstrap_unit_order ignore_unexecuted_trials_in_success_denominator separately_report_unexecuted_trials include_executed_timeouts_in_success_denominator",
    "profile":"map_extent_m grid_spacing_m families maps_per_family queries_per_map map_seeds planner_seeds planners planning_budget_s purpose",
    "optional_ablations":"enabled require_independent_policy_tests planners",
    "optional_extensions":"real_maps cgal_reference aitstar abitstar physics_simulation constrained_3d_comparison",
}

ADVANCED_BENCHMARK_KEYS=set("""protocol_id planner_ids primary_planner_id comparator_planner_ids
    informed_planner_ids required_query_ids order_policy expected_trials planner_block_size
    required_allow_flag seed_replacement_policy timer_scope collection_soft_stop_s
    validation_checkpoint_s hard_finalize_s bootstrap_replicates bootstrap_seed
    report_metrics""".split())

SEM_COMPARISON_KEYS=ADVANCED_BENCHMARK_KEYS|set("""sem_range_m sem_neighbor_radius_m
    search_direction lower_bound_policy informed_domain_policy rolling_optimization
    terrain_risk_cost turning_angle_cost paper_sample_checkpoint""".split())

COMBINED_BENCHMARK_KEYS=ADVANCED_BENCHMARK_KEYS|set("""sem_range_m
    sem_neighbor_radius_m search_direction lower_bound_policy
    informed_domain_policy rolling_optimization terrain_risk_cost
    turning_angle_cost official_sem_rrtstar_implementation
    sem_comparator_label""".split())

# Version 6 is an opt-in smoke protocol. Existing versioned campaigns keep their
# exact schema and default worker request, including the original A3 control.
VARIANT_BACKENDS={"bitstar_a0","bitstar_a3","rrtstar","informed_rrtstar",
                  "abitstar","aitstar","lazyprmstar_hauser2015"}
VARIANT_FIELDS={"planner_id","state_cache","scalar_cost","collision_index",
                "samples_per_batch","batch_after_first_solution"}


def resolved_planner_options(config,planner_id):
    """Resolve opt-in options without changing a legacy config or trial label."""
    definition=config.get("planner_variants",{}).get(planner_id,{})
    result = {"planner_id":definition.get("planner_id",planner_id),
            "state_cache":definition.get("state_cache",False),
            "scalar_cost":definition.get("scalar_cost",False),
            "collision_index":definition.get("collision_index",False),
            "samples_per_batch":definition.get("samples_per_batch",config["algorithm"]["bitstar_samples_per_batch"]),
            "batch_after_first_solution":definition.get("batch_after_first_solution",0)}
    if config.get("schema_version") == 9:
        for key, default in (("sampler_query_cache", False),
                             ("traversability_build_policy", "legacy"),
                             ("exact_flat_certificate", False)):
            result[key] = definition.get(key, config["algorithm"].get(key, default))
    return result


def _validate_planner_variants(cfg, *, traversability=False):
    variants=cfg["planner_variants"]
    if not isinstance(variants,dict) or not variants:
        raise ValueError("planner_variants must be a nonempty mapping")
    for label,definition in variants.items():
        if (not isinstance(label,str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,79}",label)
                or label in VARIANT_BACKENDS or label=="sem_rrtstar_common_v1"):
            raise ValueError("variant label must be distinct from registered planners and path-safe")
        if (not isinstance(definition,dict) or "planner_id" not in definition
                or set(definition)-(VARIANT_FIELDS|({"lambda"} if traversability else set()))):
            raise ValueError(f"planner_variants.{label}: missing planner_id or unknown fields")
        options=resolved_planner_options(cfg,label)
        backend=options["planner_id"]
        if backend not in ({"bitstar_a0","bitstar_a3"} if traversability else VARIANT_BACKENDS):
            raise ValueError(f"planner_variants.{label}: unsupported backend")
        for key in ("state_cache","scalar_cost","collision_index"):
            if type(options[key]) is not bool:
                raise ValueError(f"planner_variants.{label}.{key} must be a boolean")
        for key,minimum in (("samples_per_batch",1),("batch_after_first_solution",0)):
            if type(options[key]) is not int or not minimum<=options[key]<=2**32-1:
                raise ValueError(f"planner_variants.{label}.{key} must be a bounded integer")
        if "samples_per_batch" in definition and backend not in {"bitstar_a0","bitstar_a3","abitstar","aitstar"}:
            raise ValueError(f"planner_variants.{label}: batch size is not used by this backend")
        if options["batch_after_first_solution"] and backend!="bitstar_a3":
            raise ValueError("two-phase batch policy is supported only for bitstar_a3")
        if options["batch_after_first_solution"]==options["samples_per_batch"]:
            raise ValueError("two-phase batch policy must change the batch size")
        defaults=resolved_planner_options(cfg,backend)
        if options==defaults and not traversability:
            raise ValueError(f"planner_variants.{label}: variant has no effective option changes")
    planners=cfg["profiles"]["smoke"]["planners"]
    if (not isinstance(planners,list) or not planners or len(set(planners))!=len(planners)
            or any(p not in VARIANT_BACKENDS and p not in variants for p in planners)):
        raise ValueError("smoke planners must contain distinct registered backends or variant labels")
    if not set(variants)<=set(planners):
        raise ValueError("unused planner_variants are not permitted in the smoke config")
    if "bitstar_a3" not in planners and not traversability:
        raise ValueError("A3 improvement smoke must retain the original bitstar_a3 control")


def _exact_keys(value, expected, section):
    expected=set(expected)
    if not isinstance(value,dict) or set(value)!=expected:
        actual=set(value) if isinstance(value,dict) else set()
        raise ValueError(f"{section}: missing/unknown fields: {actual^expected}")


def _validate_advanced_config(cfg,profile):
    """Validate the separately versioned, locked A3-vs-advanced campaign.

    Schema v1 intentionally continues through the original validation path below;
    G5 config loading and its exact field allowlists therefore do not change.
    """
    _exact_keys(cfg,set(KEYS["root"].split())|{"advanced_benchmark"},"root")
    for section in ("model","algorithm","execution","logging","analysis","optional_ablations","optional_extensions"):
        _exact_keys(cfg[section],KEYS[section].split(),section)
    _exact_keys(cfg["advanced_benchmark"],ADVANCED_BENCHMARK_KEYS,"advanced_benchmark")
    if set(cfg["profiles"])!={"advanced"}:
        raise ValueError("schema v2 requires exactly the advanced profile")
    _exact_keys(cfg["profiles"]["advanced"],KEYS["profile"].split(),"profile")
    if cfg["project"]!="bitstar_25d" or profile!="advanced":
        raise ValueError("schema v2 only supports project bitstar_25d/profile advanced")

    fixed={
        "model":{"state":"xy","objective":"surface_length","local_connector":"straight_xy_lifted_to_piecewise_affine_heightfield",
            "surface_interpolation":"fixed_diagonal_triangles","cell_diagonal":"southwest_to_northeast","height_units":"m","coordinate_units":"m",
            "nodata_policy":"forbidden","obstacle_contact_policy":"forbidden","include_outside_map_as_forbidden":True,"roughness_limit_m":None,
            "goal":"singleton","goal_roundoff_tolerance_m":1e-9},
        "algorithm":{"nearest_neighbor_metric":"euclidean_xy","bitstar_connection":"k_nearest","bitstar_just_in_time_sampling":False,
            "informed_measure_policy":"audit_and_use_documented_conservative_fallback","edge_cache":False,"graph_pruning_requested":True,
            "fail_if_requested_variant_not_effective":True,"simplification":False,"approximate_solution_counts_as_success":False},
        "execution":{"jobs":1,"one_process_per_trial":True,"seed_before_creating_planner":True,"save_trial_manifest":True,
            "resume_by_config_hash":True,"full_requires_explicit_flag":True,"timing_origin":"immediately_before_solve",
            "record_preprocessing_separately":True,"record_setup_separately":True,"offline_validation_separately":True},
        "logging":{"save_improvement_events":True,"save_improvement_paths":True,"save_ompl_benchmark_log":True,
            "save_path_xy":True,"save_path_surface_breakpoints":True,"save_full_tree":False,"keep_failures":True,"strict_json_no_nan_or_infinity":True},
        "analysis":{"require_reference_provenance":True,"ignore_unexecuted_trials_in_success_denominator":True,
            "separately_report_unexecuted_trials":True,"include_executed_timeouts_in_success_denominator":True},
    }
    for section,values in fixed.items():
        for key,value in values.items():
            if cfg[section][key]!=value:
                raise ValueError(f"unsupported {section}.{key}={cfg[section][key]!r}")
    # A versioned protocol ID denotes one exact experimental design.  Keep all
    # values that affect maps, planner work, watchdogs, logging or statistics
    # fixed rather than merely range-checking them under the same label.
    locked_remaining={
        "root":{"language":"vi"},
        "model":{"robot_radius_m":0.30,"safety_margin_m":0.10,"max_slope_deg":35.0,
            "roughness_window_m":1.0,"numerical_abs_tolerance_m":1.0e-8,
            "numerical_rel_tolerance":1.0e-8},
        "algorithm":{"bitstar_samples_per_batch":200,"bitstar_rewire_factor":1.1,
            "informed_sample_attempts_per_call":100},
        "execution":{"shuffle_trial_order_seed":73131,"watchdog_grace_s":5.0,
            "memory_limit_mb":4096,"default_profile":"advanced"},
        "logging":{"progress_interval_s":0.05},
        "analysis":{"target_relative_tolerances":[0.10,0.05,0.02],
            "confidence_level":0.95,"bootstrap_unit_order":["map","query","seed"]},
        "optional_ablations":{"enabled":False,"require_independent_policy_tests":True,
            "planners":["bitstar_a1","bitstar_a2"]},
        "optional_extensions":{"real_maps":False,"cgal_reference":False,"aitstar":False,
            "abitstar":False,"physics_simulation":False,"constrained_3d_comparison":False},
    }
    for section,values in locked_remaining.items():
        container=cfg if section=="root" else cfg[section]
        for key,value in values.items():
            if container[key]!=value:
                raise ValueError(f"locked protocol value differs: {section}.{key}")
    if cfg["optional_ablations"]["enabled"] or any(cfg["optional_extensions"].values()):
        raise ValueError("extensions/ablations UNSUPPORTED")

    planners=["bitstar_a3","abitstar","aitstar","lazyprmstar_hauser2015"]
    advanced=cfg["advanced_benchmark"]
    expected_advanced={
        "protocol_id":"a3_vs_advanced_v1",
        "planner_ids":planners,
        "primary_planner_id":"bitstar_a3",
        "comparator_planner_ids":planners[1:],
        "informed_planner_ids":planners[:3],
        "required_query_ids":["q0"],
        "order_policy":"latin_square_problem_slots_v1",
        "expected_trials":400,
        "planner_block_size":4,
        "required_allow_flag":"--allow-advanced-benchmark",
        "seed_replacement_policy":"forbidden_predeclared_held_out_seeds",
        "timer_scope":"planner_collection_offline_validation_analysis_report",
        "collection_soft_stop_s":5400.0,
        "validation_checkpoint_s":6600.0,
        "hard_finalize_s":7200.0,
        "bootstrap_replicates":2000,
        "bootstrap_seed":73132,
        "report_metrics":["validated_success","first_solution_time_s","cost_terrain_m_at_budget",
            "length_xy_m_final","actual_solve_wall_s","worker_wall_s","validation_wall_s",
            "peak_rss_mb","motion_validity_calls","state_validity_calls","coverage"],
    }
    for key,value in expected_advanced.items():
        if advanced[key]!=value:
            raise ValueError(f"advanced_benchmark.{key} differs from locked protocol")
    p=cfg["profiles"]["advanced"]
    expected_profile={
        "map_extent_m":[40.,40.],"grid_spacing_m":.1,
        "families":["traversable_hill","ridge_pass","rolling_terrain","narrow_passage","mixed_obstacles"],
        "maps_per_family":10,"queries_per_map":1,"map_seeds":list(range(3001,3011)),
        "planner_seeds":[91001,91002],"planners":planners,"planning_budget_s":5.,
        "purpose":"held_out_a3_vs_advanced_algorithms",
    }
    if any(p[k]!=value for k,value in expected_profile.items()):
        raise ValueError("advanced profile differs from locked 400-trial protocol")
    if p["maps_per_family"]!=len(p["map_seeds"]):
        raise ValueError("inconsistent advanced map seeds")
    if (len(p["families"])*p["maps_per_family"]*p["queries_per_map"]*
            len(p["planner_seeds"])*len(p["planners"])!=advanced["expected_trials"]):
        raise ValueError("advanced trial cardinality mismatch")
    if cfg["execution"]["shuffle_trial_order_seed"]!=73131:
        raise ValueError("advanced shuffle seed differs from locked protocol")
    for section,key in [("model","robot_radius_m"),("model","safety_margin_m"),("model","max_slope_deg"),
        ("algorithm","bitstar_samples_per_batch"),("algorithm","informed_sample_attempts_per_call"),
        ("logging","progress_interval_s"),("execution","watchdog_grace_s"),("execution","memory_limit_mb")]:
        value=cfg[section][key]
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError(f"invalid {section}.{key}")
    if not 0<cfg["model"]["max_slope_deg"]<90 or cfg["algorithm"]["bitstar_rewire_factor"]<=1:
        raise ValueError("invalid limits")
    return cfg


def _validate_sem_comparison_config(cfg,profile):
    """Validate the immutable A3-vs-SEM-RRT*-adapted campaign design."""
    _exact_keys(cfg,set(KEYS["root"].split())|{"sem_comparison"},"root")
    for section in ("model","algorithm","execution","logging","analysis","optional_ablations","optional_extensions"):
        _exact_keys(cfg[section],KEYS[section].split(),section)
    _exact_keys(cfg["sem_comparison"],SEM_COMPARISON_KEYS,"sem_comparison")
    if set(cfg["profiles"])!={"sem_comparison"}:
        raise ValueError("schema v3 requires exactly the sem_comparison profile")
    _exact_keys(cfg["profiles"]["sem_comparison"],KEYS["profile"].split(),"profile")
    if cfg["project"]!="bitstar_25d" or profile!="sem_comparison":
        raise ValueError("schema v3 only supports project bitstar_25d/profile sem_comparison")

    fixed={
        "model":{"state":"xy","objective":"surface_length","local_connector":"straight_xy_lifted_to_piecewise_affine_heightfield",
            "surface_interpolation":"fixed_diagonal_triangles","cell_diagonal":"southwest_to_northeast","height_units":"m","coordinate_units":"m",
            "nodata_policy":"forbidden","obstacle_contact_policy":"forbidden","include_outside_map_as_forbidden":True,"roughness_limit_m":None,
            "goal":"singleton","goal_roundoff_tolerance_m":1e-9},
        "algorithm":{"nearest_neighbor_metric":"euclidean_xy","bitstar_connection":"k_nearest","bitstar_just_in_time_sampling":False,
            "informed_measure_policy":"audit_and_use_documented_conservative_fallback","edge_cache":False,"graph_pruning_requested":True,
            "fail_if_requested_variant_not_effective":True,"simplification":False,"approximate_solution_counts_as_success":False},
        "execution":{"jobs":1,"one_process_per_trial":True,"seed_before_creating_planner":True,"save_trial_manifest":True,
            "resume_by_config_hash":True,"full_requires_explicit_flag":True,"timing_origin":"immediately_before_solve",
            "record_preprocessing_separately":True,"record_setup_separately":True,"offline_validation_separately":True},
        "logging":{"save_improvement_events":True,"save_improvement_paths":True,"save_ompl_benchmark_log":True,
            "save_path_xy":True,"save_path_surface_breakpoints":True,"save_full_tree":False,"keep_failures":True,"strict_json_no_nan_or_infinity":True},
        "analysis":{"require_reference_provenance":True,"ignore_unexecuted_trials_in_success_denominator":True,
            "separately_report_unexecuted_trials":True,"include_executed_timeouts_in_success_denominator":True},
    }
    for section,values in fixed.items():
        for key,value in values.items():
            if cfg[section][key]!=value:
                raise ValueError(f"unsupported {section}.{key}={cfg[section][key]!r}")
    locked_remaining={
        "root":{"language":"vi"},
        "model":{"robot_radius_m":0.30,"safety_margin_m":0.10,"max_slope_deg":35.0,
            "roughness_window_m":1.0,"numerical_abs_tolerance_m":1.0e-8,"numerical_rel_tolerance":1.0e-8},
        "algorithm":{"bitstar_samples_per_batch":200,"bitstar_rewire_factor":1.1,
            "informed_sample_attempts_per_call":100},
        "execution":{"shuffle_trial_order_seed":73134,"watchdog_grace_s":5.0,
            "memory_limit_mb":4096,"default_profile":"sem_comparison"},
        "logging":{"progress_interval_s":0.05},
        "analysis":{"target_relative_tolerances":[0.10,0.05,0.02],"confidence_level":0.95,
            "bootstrap_unit_order":["family","map","seed"]},
        "optional_ablations":{"enabled":False,"require_independent_policy_tests":True,
            "planners":["bitstar_a1","bitstar_a2"]},
        "optional_extensions":{"real_maps":False,"cgal_reference":False,"aitstar":False,
            "abitstar":False,"physics_simulation":False,"constrained_3d_comparison":False},
    }
    for section,values in locked_remaining.items():
        container=cfg if section=="root" else cfg[section]
        for key,value in values.items():
            if container[key]!=value:
                raise ValueError(f"locked protocol value differs: {section}.{key}")
    if cfg["optional_ablations"]["enabled"] or any(cfg["optional_extensions"].values()):
        raise ValueError("extensions/ablations UNSUPPORTED")

    planners=["bitstar_a3","sem_rrtstar_common_v1"]
    protocol=cfg["sem_comparison"]
    expected_protocol={
        "protocol_id":"a3_vs_sem_rrtstar_common_v1","planner_ids":planners,
        "primary_planner_id":"bitstar_a3","comparator_planner_ids":["sem_rrtstar_common_v1"],
        "informed_planner_ids":planners,"required_query_ids":["q0"],
        "order_policy":"paired_map_parity_seed_v1","expected_trials":300,"planner_block_size":2,
        "required_allow_flag":"--allow-sem-comparison",
        "seed_replacement_policy":"forbidden_predeclared_held_out_seeds",
        "timer_scope":"planner_collection_offline_validation_analysis_report",
        "collection_soft_stop_s":5400.0,"validation_checkpoint_s":6600.0,"hard_finalize_s":7200.0,
        "bootstrap_replicates":2000,"bootstrap_seed":73135,"sem_range_m":1.0,
        "sem_neighbor_radius_m":3.0,"search_direction":"backward_goal_to_start",
        "lower_bound_policy":"d_xy","informed_domain_policy":"E_xy","rolling_optimization":False,
        "terrain_risk_cost":False,"turning_angle_cost":False,"paper_sample_checkpoint":5000,
        "report_metrics":["validated_success","first_solution_time_s","cost_terrain_m_at_budget",
            "length_xy_m_final","actual_solve_wall_s","worker_wall_s","validation_wall_s",
            "peak_rss_mb","motion_validity_calls","state_validity_calls","full_surface_cost_calls",
            "triangle_visits","height_queries","sample_proposals","coverage"],
    }
    for key,value in expected_protocol.items():
        if protocol[key]!=value:
            raise ValueError(f"sem_comparison.{key} differs from locked protocol")
    settings=cfg["profiles"]["sem_comparison"]
    expected_profile={
        "map_extent_m":[40.,40.],"grid_spacing_m":.1,
        "families":["traversable_hill","ridge_pass","rolling_terrain","narrow_passage","mixed_obstacles"],
        "maps_per_family":10,"queries_per_map":1,"map_seeds":list(range(4001,4011)),
        "planner_seeds":[92001,92002,92003],"planners":planners,"planning_budget_s":10.,
        "purpose":"held_out_a3_vs_sem_rrtstar_common_objective",
    }
    if any(settings[key]!=value for key,value in expected_profile.items()):
        raise ValueError("sem_comparison profile differs from locked 300-trial protocol")
    if settings["maps_per_family"]!=len(settings["map_seeds"]):
        raise ValueError("inconsistent sem_comparison map seeds")
    cardinality=(len(settings["families"])*settings["maps_per_family"]*settings["queries_per_map"]*
                 len(settings["planner_seeds"])*len(settings["planners"]))
    if cardinality!=protocol["expected_trials"]:
        raise ValueError("sem_comparison trial cardinality mismatch")
    for section,key in [("model","robot_radius_m"),("model","safety_margin_m"),("model","max_slope_deg"),
        ("algorithm","bitstar_samples_per_batch"),("algorithm","informed_sample_attempts_per_call"),
        ("logging","progress_interval_s"),("execution","watchdog_grace_s"),("execution","memory_limit_mb")]:
        value=cfg[section][key]
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
            raise ValueError(f"invalid {section}.{key}")
    if not 0<cfg["model"]["max_slope_deg"]<90 or cfg["algorithm"]["bitstar_rewire_factor"]<=1:
        raise ValueError("invalid limits")
    return cfg


def _validate_combined_config(cfg,profile):
    """Validate the immutable 250-trial, five-planner comparison campaign."""
    _exact_keys(cfg,set(KEYS["root"].split())|{"combined_benchmark"},"root")
    for section in ("model","algorithm","execution","logging","analysis","optional_ablations","optional_extensions"):
        _exact_keys(cfg[section],KEYS[section].split(),section)
    _exact_keys(cfg["combined_benchmark"],COMBINED_BENCHMARK_KEYS,"combined_benchmark")
    if set(cfg["profiles"])!={"combined"}:
        raise ValueError("schema v4 requires exactly the combined profile")
    _exact_keys(cfg["profiles"]["combined"],KEYS["profile"].split(),"profile")
    if cfg["project"]!="bitstar_25d" or profile!="combined":
        raise ValueError("schema v4 only supports project bitstar_25d/profile combined")

    fixed={
        "model":{"state":"xy","objective":"surface_length","local_connector":"straight_xy_lifted_to_piecewise_affine_heightfield",
            "surface_interpolation":"fixed_diagonal_triangles","cell_diagonal":"southwest_to_northeast","height_units":"m","coordinate_units":"m",
            "nodata_policy":"forbidden","obstacle_contact_policy":"forbidden","include_outside_map_as_forbidden":True,"roughness_limit_m":None,
            "goal":"singleton","goal_roundoff_tolerance_m":1e-9},
        "algorithm":{"nearest_neighbor_metric":"euclidean_xy","bitstar_connection":"k_nearest","bitstar_just_in_time_sampling":False,
            "informed_measure_policy":"audit_and_use_documented_conservative_fallback","edge_cache":False,"graph_pruning_requested":True,
            "fail_if_requested_variant_not_effective":True,"simplification":False,"approximate_solution_counts_as_success":False},
        "execution":{"jobs":1,"one_process_per_trial":True,"seed_before_creating_planner":True,"save_trial_manifest":True,
            "resume_by_config_hash":True,"full_requires_explicit_flag":True,"timing_origin":"immediately_before_solve",
            "record_preprocessing_separately":True,"record_setup_separately":True,"offline_validation_separately":True},
        "logging":{"save_improvement_events":True,"save_improvement_paths":True,"save_ompl_benchmark_log":True,
            "save_path_xy":True,"save_path_surface_breakpoints":True,"save_full_tree":False,"keep_failures":True,"strict_json_no_nan_or_infinity":True},
        "analysis":{"require_reference_provenance":True,"ignore_unexecuted_trials_in_success_denominator":True,
            "separately_report_unexecuted_trials":True,"include_executed_timeouts_in_success_denominator":True},
    }
    for section,values in fixed.items():
        for key,value in values.items():
            if cfg[section][key]!=value:
                raise ValueError(f"unsupported {section}.{key}={cfg[section][key]!r}")
    locked_remaining={
        "root":{"language":"vi"},
        "model":{"robot_radius_m":0.30,"safety_margin_m":0.10,"max_slope_deg":35.0,
            "roughness_window_m":1.0,"numerical_abs_tolerance_m":1.0e-8,"numerical_rel_tolerance":1.0e-8},
        "algorithm":{"bitstar_samples_per_batch":200,"bitstar_rewire_factor":1.1,
            "informed_sample_attempts_per_call":100},
        "execution":{"shuffle_trial_order_seed":73136,"watchdog_grace_s":5.0,
            "memory_limit_mb":4096,"default_profile":"combined"},
        "logging":{"progress_interval_s":0.05},
        "analysis":{"target_relative_tolerances":[0.10,0.05,0.02],"confidence_level":0.95,
            "bootstrap_unit_order":["map","query","seed"]},
        "optional_ablations":{"enabled":False,"require_independent_policy_tests":True,
            "planners":["bitstar_a1","bitstar_a2"]},
        "optional_extensions":{"real_maps":False,"cgal_reference":False,"aitstar":False,
            "abitstar":False,"physics_simulation":False,"constrained_3d_comparison":False},
    }
    for section,values in locked_remaining.items():
        container=cfg if section=="root" else cfg[section]
        for key,value in values.items():
            if container[key]!=value:
                raise ValueError(f"locked protocol value differs: {section}.{key}")
    if cfg["optional_ablations"]["enabled"] or any(cfg["optional_extensions"].values()):
        raise ValueError("extensions/ablations UNSUPPORTED")

    planners=["bitstar_a3","abitstar","aitstar","lazyprmstar_hauser2015",
              "sem_rrtstar_common_v1"]
    protocol=cfg["combined_benchmark"]
    expected_protocol={
        "protocol_id":"a3_vs_all_v1","planner_ids":planners,
        "primary_planner_id":"bitstar_a3","comparator_planner_ids":planners[1:],
        "informed_planner_ids":["bitstar_a3","abitstar","aitstar","sem_rrtstar_common_v1"],
        "required_query_ids":["q0"],"order_policy":"latin_square_five_planners_v1",
        "expected_trials":250,"planner_block_size":5,
        "required_allow_flag":"--allow-combined-benchmark",
        "seed_replacement_policy":"forbidden_predeclared_held_out_seeds",
        "timer_scope":"planner_collection_offline_validation_analysis_report",
        "collection_soft_stop_s":6300.0,"validation_checkpoint_s":9600.0,
        "hard_finalize_s":10800.0,"bootstrap_replicates":2000,"bootstrap_seed":73137,
        "sem_range_m":1.0,"sem_neighbor_radius_m":3.0,
        "search_direction":"backward_goal_to_start","lower_bound_policy":"d_xy",
        "informed_domain_policy":"E_xy","rolling_optimization":False,
        "terrain_risk_cost":False,"turning_angle_cost":False,
        "official_sem_rrtstar_implementation":False,
        "sem_comparator_label":"SEM-RRT*-adapted common-objective",
        "report_metrics":["validated_success","first_solution_time_s","cost_terrain_m_at_budget",
            "length_xy_m_final","actual_solve_wall_s","worker_wall_s","validation_wall_s",
            "peak_rss_mb","motion_validity_calls","state_validity_calls","full_surface_cost_calls",
            "triangle_visits","height_queries","sample_proposals","coverage"],
    }
    for key,value in expected_protocol.items():
        if protocol[key]!=value:
            raise ValueError(f"combined_benchmark.{key} differs from locked protocol")
    settings=cfg["profiles"]["combined"]
    expected_profile={
        "map_extent_m":[40.,40.],"grid_spacing_m":.1,
        "families":["traversable_hill","ridge_pass","rolling_terrain","narrow_passage","mixed_obstacles"],
        "maps_per_family":10,"queries_per_map":1,"map_seeds":list(range(5001,5011)),
        "planner_seeds":[93001],"planners":planners,"planning_budget_s":20.,
        "purpose":"held_out_a3_vs_all_common_objective",
    }
    if any(settings[key]!=value for key,value in expected_profile.items()):
        raise ValueError("combined profile differs from locked 250-trial protocol")
    if settings["maps_per_family"]!=len(settings["map_seeds"]):
        raise ValueError("inconsistent combined map seeds")
    cardinality=(len(settings["families"])*settings["maps_per_family"]*settings["queries_per_map"]*
                 len(settings["planner_seeds"])*len(settings["planners"]))
    if cardinality!=protocol["expected_trials"]:
        raise ValueError("combined trial cardinality mismatch")
    for section,key in [("model","robot_radius_m"),("model","safety_margin_m"),("model","max_slope_deg"),
        ("algorithm","bitstar_samples_per_batch"),("algorithm","informed_sample_attempts_per_call"),
        ("logging","progress_interval_s"),("execution","watchdog_grace_s"),("execution","memory_limit_mb")]:
        value=cfg[section][key]
        if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:
            raise ValueError(f"invalid {section}.{key}")
    if not 0<cfg["model"]["max_slope_deg"]<90 or cfg["algorithm"]["bitstar_rewire_factor"]<=1:
        raise ValueError("invalid limits")
    return cfg

def _validate_cloud_config(cfg, profile):
    """Versioned 1500-trial design; the v1 contract remains immutable."""
    extra = {"shard_count", "max_cloud_tasks", "setup_watchdog_s", "serialization_watchdog_s"}
    _exact_keys(cfg, set(KEYS["root"].split()) | {"combined_benchmark"}, "root")
    _exact_keys(cfg["combined_benchmark"], COMBINED_BENCHMARK_KEYS | extra, "combined_benchmark")
    if profile != "combined" or set(cfg["profiles"]) != {"combined"}:
        raise ValueError("schema v5 requires the combined profile")
    p = cfg["profiles"]["combined"]
    expected_profile = {"maps_per_family": 20, "map_seeds": list(range(6001, 6021)),
                        "planner_seeds": [94001, 94002, 94003], "planning_budget_s": 30.0}
    protocol = cfg["combined_benchmark"]
    expected_protocol = {"protocol_id": "a3_vs_all_v2", "expected_trials": 1500,
        "order_policy": "latin_square_shards_five_planners_v2", "bootstrap_seed": 73139,
        "timer_scope": "resumable_phase_worklist_no_campaign_deadline",
        "collection_soft_stop_s": None, "validation_checkpoint_s": None, "hard_finalize_s": None,
        "shard_count": 20, "max_cloud_tasks": 4, "setup_watchdog_s": 60.0,
        "serialization_watchdog_s": 60.0}
    for key, value in expected_profile.items():
        if p.get(key) != value:
            raise ValueError("cloud profile differs from locked protocol: " + key)
    for key, value in expected_protocol.items():
        if protocol.get(key) != value:
            raise ValueError("cloud protocol differs from locked protocol: " + key)
    if cfg["execution"].get("shuffle_trial_order_seed") != 73138:
        raise ValueError("cloud shuffle seed differs from locked protocol")
    if cfg["analysis"].get("bootstrap_unit_order") != ["family", "map", "seed"]:
        raise ValueError("cloud bootstrap hierarchy differs from locked protocol")
    # Validate every unchanged setting through the existing strict contract.
    projected = copy.deepcopy(cfg)
    projected["schema_version"] = 4
    projected["execution"]["shuffle_trial_order_seed"] = 73136
    projected["analysis"]["bootstrap_unit_order"] = ["map", "query", "seed"]
    projected["profiles"]["combined"].update(maps_per_family=10, map_seeds=list(range(5001, 5011)),
        planner_seeds=[93001], planning_budget_s=20.0)
    for key in extra:
        del projected["combined_benchmark"][key]
    projected["combined_benchmark"].update(protocol_id="a3_vs_all_v1", expected_trials=250,
        order_policy="latin_square_five_planners_v1", bootstrap_seed=73137,
        timer_scope="planner_collection_offline_validation_analysis_report",
        collection_soft_stop_s=6300.0, validation_checkpoint_s=9600.0, hard_finalize_s=10800.0)
    _validate_combined_config(projected, profile)
    return cfg


def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    serialized=json.dumps(data,indent=2,allow_nan=False,ensure_ascii=False)+"\n"
    # A stopped process must leave either the previous complete manifest or the
    # new complete manifest. Temporary files remain visible evidence on failure.
    with tempfile.NamedTemporaryFile(mode="w",encoding="utf-8",dir=path.parent,prefix=path.name+".",suffix=".tmp",delete=False) as handle:
        handle.write(serialized);handle.flush();os.fsync(handle.fileno());temporary=handle.name
    # Windows can transiently deny an otherwise valid atomic replacement when
    # a reader or filesystem scanner briefly holds the destination. Retry only
    # this metadata commit; planner attempts are never retried here.
    for attempt in range(10):
        try:
            os.replace(temporary,path)
            break
        except PermissionError:
            if attempt==9:
                raise
            time.sleep(.05*(attempt+1))

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite JSON: "+x)))

def digest(data):
    if not isinstance(data,bytes):data=json.dumps(data,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()

def file_hash(path):
    with Path(path).open("rb") as handle:return hashlib.file_digest(handle,"sha256").hexdigest()

def load_config(path,profile="smoke"):
    cfg=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(cfg,dict) and cfg.get("schema_version")==9:
        from a3_traversability_perf_protocol import validate_config
        return validate_config(cfg,profile)
    if isinstance(cfg,dict) and cfg.get("schema_version")==8:
        from a3_traversability_protocol import validate_config
        return validate_config(cfg,profile)
    if isinstance(cfg,dict) and cfg.get("schema_version")==7:
        from a3_pilot_protocol import validate_config
        return validate_config(cfg,profile)
    if isinstance(cfg,dict) and cfg.get("schema_version")==5:
        return _validate_cloud_config(cfg,profile)
    if isinstance(cfg,dict) and cfg.get("schema_version")==2:
        return _validate_advanced_config(cfg,profile)
    if isinstance(cfg,dict) and cfg.get("schema_version")==3:
        return _validate_sem_comparison_config(cfg,profile)
    if isinstance(cfg,dict) and cfg.get("schema_version")==4:
        return _validate_combined_config(cfg,profile)
    return _validate_base_config(cfg,profile)


def _validate_base_config(cfg,profile="smoke", *, traversability=False):
    """Shared schema validation; new objectives must opt in explicitly."""
    variants_enabled=isinstance(cfg,dict) and cfg.get("schema_version") in (6,8)
    def keys(value,section):
        expected=set(KEYS[section].split())
        if section=="root" and variants_enabled:expected.add("planner_variants")
        if section=="root" and traversability:expected.add("traversability")
        if not isinstance(value,dict) or set(value)!=expected:
            actual=set(value) if isinstance(value,dict) else set()
            raise ValueError(f"{section}: missing/unknown fields: {actual^expected}")
    keys(cfg,"root")
    for section in ("model","algorithm","execution","logging","analysis","optional_ablations","optional_extensions"):keys(cfg[section],section)
    if set(cfg["profiles"])!=({"smoke"} if variants_enabled else {"smoke","pilot","full"}):
        raise ValueError("unknown/missing profile")
    for p in cfg["profiles"].values():keys(p,"profile")
    if cfg["schema_version"] not in ((8,) if traversability else (1,6)) or cfg["project"]!="bitstar_25d":raise ValueError("unsupported schema/project")
    fixed={
        "model":{"state":"xy","objective":"surface_length","local_connector":"straight_xy_lifted_to_piecewise_affine_heightfield",
            "surface_interpolation":"fixed_diagonal_triangles","cell_diagonal":"southwest_to_northeast","height_units":"m","coordinate_units":"m",
            "nodata_policy":"forbidden","obstacle_contact_policy":"forbidden","include_outside_map_as_forbidden":True,"roughness_limit_m":None,"goal":"singleton","goal_roundoff_tolerance_m":1e-9},
        "algorithm":{"nearest_neighbor_metric":"euclidean_xy","bitstar_connection":"k_nearest","bitstar_just_in_time_sampling":False,
            "informed_measure_policy":"audit_and_use_documented_conservative_fallback","edge_cache":False,"graph_pruning_requested":True,
            "fail_if_requested_variant_not_effective":True,"simplification":False,"approximate_solution_counts_as_success":False},
        "execution":{"jobs":1,"one_process_per_trial":True,"seed_before_creating_planner":True,"save_trial_manifest":True,"resume_by_config_hash":True,
            "full_requires_explicit_flag":True,"timing_origin":"immediately_before_solve","record_preprocessing_separately":True,
            "record_setup_separately":True,"offline_validation_separately":True},
        "logging":{"save_improvement_events":True,"save_improvement_paths":True,"save_ompl_benchmark_log":True,
            "save_path_xy":True,"save_path_surface_breakpoints":True,"save_full_tree":False,"keep_failures":True,"strict_json_no_nan_or_infinity":True},
        "analysis":{"require_reference_provenance":True,"ignore_unexecuted_trials_in_success_denominator":True,
            "separately_report_unexecuted_trials":True,"include_executed_timeouts_in_success_denominator":True},
    }
    if traversability:fixed["model"]["objective"]="surface_length_traversability"
    for section,values in fixed.items():
        for key,value in values.items():
            if cfg[section][key]!=value:raise ValueError(f"unsupported {section}.{key}={cfg[section][key]!r}")
    if cfg["optional_ablations"]["enabled"] or any(cfg["optional_extensions"].values()):raise ValueError("extensions/ablations UNSUPPORTED")
    if variants_enabled:
        if profile!="smoke" or cfg["execution"]["default_profile"]!="smoke":
            raise ValueError("schema v6 authorizes smoke only; pilot/full need a separate protocol")
        _validate_planner_variants(cfg,traversability=traversability)
    if profile not in ("smoke","pilot"):raise ValueError("only smoke/pilot supported; full is prohibited for G3-G4")
    p=cfg["profiles"][profile]
    if p["maps_per_family"]!=len(p["map_seeds"]) or p["queries_per_map"]!=(1 if profile=="smoke" else 2):raise ValueError("inconsistent map seeds or unsupported query count")
    if not traversability and set(p["families"])-{"flat","tilted_plane","traversable_hill","narrow_passage","ridge_pass","rolling_terrain","mixed_obstacles"}:raise ValueError("unsupported family")
    if not variants_enabled and set(p["planners"])-{"bitstar_a0","bitstar_a3","rrtstar","informed_rrtstar"}:raise ValueError("unsupported planner")
    if profile=="pilot":
        expected={"map_extent_m":[40.,40.],"grid_spacing_m":.1,"families":["traversable_hill","ridge_pass","rolling_terrain","narrow_passage","mixed_obstacles"],
            "maps_per_family":2,"queries_per_map":2,"map_seeds":[201,202],"planner_seeds":[11,22,33,44,55],
            "planners":["bitstar_a0","bitstar_a3","rrtstar","informed_rrtstar"],"planning_budget_s":5.}
        if any(p[k]!=v for k,v in expected.items()):raise ValueError("pilot differs from locked 400-trial protocol")
    for section,key in [("model","robot_radius_m"),("model","safety_margin_m"),("model","max_slope_deg"),
        ("algorithm","bitstar_samples_per_batch"),("algorithm","informed_sample_attempts_per_call"),
        ("logging","progress_interval_s"),("execution","watchdog_grace_s"),("execution","memory_limit_mb")]:
        v=cfg[section][key]
        if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError(f"invalid {section}.{key}")
    if not 0<cfg["model"]["max_slope_deg"]<90 or cfg["algorithm"]["bitstar_rewire_factor"]<=1:raise ValueError("invalid limits")
    return cfg

def source_files():
    files=[]
    for directory in ("include","src","apps","tools","tests","cmake","patches","diagnostics"):
        files += [p for p in (ROOT/directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    files += [ROOT/"CMakeLists.txt"]
    return sorted(files)

def code_hash():
    return digest([(str(p.relative_to(ROOT)),file_hash(p)) for p in source_files()])

def environment():
    import psutil
    return {"os":platform.platform(),"cpu":platform.processor(),"logical_cpus":os.cpu_count(),
        "ram_mb":psutil.virtual_memory().total/2**20,"python":sys.version,"python_executable":sys.executable,
        "packages":{name:importlib.metadata.version(name) for name in ("numpy","matplotlib","PyYAML","psutil")},
        "code_commit":None,"git_note":"initial directory has no .git","code_sha256":code_hash(),
        "compiler":"GCC 13.2.0 MinGW-W64 x86_64-ucrt-posix-seh","cmake":"3.28.2","build_flags":"Release -O3 -DNDEBUG C++17",
        "dependencies":read_json(ROOT/".deps/downloads.json") if (ROOT/".deps/downloads.json").exists() else {},
        "ompl_version":"1.7.0","boost_version":"1.82.0","eigen_version":"3.4.0"}

def worker_env():
    env=dict(os.environ)
    if os.name=="nt":env["PATH"]=";".join([str(ROOT/".deps/install/lib"),"C:/mingw64/bin",env["PATH"]])
    else:env["LD_LIBRARY_PATH"]=str(ROOT/".deps/install/lib")+os.pathsep+env.get("LD_LIBRARY_PATH", "")
    return env

def default_worker():return ROOT/"build"/("terrain_benchmark.exe" if os.name=="nt" else "terrain_benchmark")


def cloud_code_hash():
    """Canonical names, so identical source has identical identity on both OSes."""
    records = [(p.relative_to(ROOT).as_posix(), file_hash(p)) for p in source_files()]
    # Path sorting has OS-dependent case rules. Sort the canonical names instead.
    return digest(sorted(records, key=lambda record: record[0]))


def _cloud_probe(argv, *, env=None):
    try:
        result = subprocess.run([str(x) for x in argv], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=30, env=env)
        return {"argv": [str(x) for x in argv], "exit_code": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": [str(x) for x in argv], "exit_code": None, "error": str(error)}


def cloud_ompl_library(worker):
    worker = Path(worker).resolve()
    if os.name == "nt":
        # The launch environment places this dependency directory first.
        candidate = ROOT / ".deps/install/lib/libompl.dll"
        if not candidate.is_file():
            raise ValueError("Cloud gate cannot identify the OMPL DLL")
        return candidate.resolve()
    import re
    evidence = _cloud_probe(["ldd", worker], env=worker_env())
    if evidence.get("exit_code") != 0:
        raise ValueError("cannot inspect worker linked libraries: " + str(evidence))
    matches = re.findall(r"^\s*libompl[^\s]*\s+=>\s+(\S+)\s", evidence["stdout"], re.MULTILINE)
    if len(matches) != 1 or not Path(matches[0]).is_file():
        raise ValueError("worker must link exactly one identifiable OMPL shared library")
    return Path(matches[0]).resolve()


def cloud_environment(worker):
    """Probe observed build/runtime facts; historical Windows metadata is untouched."""
    import psutil
    worker = Path(worker).resolve()
    library = cloud_ompl_library(worker)
    cache_path = worker.parent / "CMakeCache.txt"
    cache = {}
    if cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line and not line.startswith(("#", "//")) and "=" in line and ":" in line.split("=", 1)[0]:
                left, value = line.split("=", 1); cache[left.split(":", 1)[0]] = value
    compiler = cache.get("CMAKE_CXX_COMPILER") or shutil.which("g++")
    compiler_probe = _cloud_probe([compiler, "--version"]) if compiler else {"error": "compiler_unavailable"}
    cmake_probe = _cloud_probe(["cmake", "--version"])
    git_probe = _cloud_probe(["git", "-C", ROOT, "rev-parse", "HEAD"])
    import re
    dependency_evidence = {}
    header_paths = {"ompl": ROOT / ".deps/install/include/ompl-1.7/ompl/config.h",
        "boost": ROOT / ".deps/boost_1_82_0/boost/version.hpp",
        "eigen": ROOT / ".deps/install/include/eigen3/Eigen/src/Core/util/Macros.h"}
    for name, path in header_paths.items():
        version = None
        if path.is_file():
            content = path.read_text(encoding="utf-8", errors="replace")
            if name == "ompl":
                match = re.search(r'^#define OMPL_VERSION "([^"]+)"', content, re.MULTILINE)
                if match: version = match.group(1)
            elif name == "boost":
                match = re.search(r"^#define BOOST_VERSION (\d+)", content, re.MULTILINE)
                if match:
                    value = int(match.group(1)); version = f"{value//100000}.{value//100%1000}.{value%100}"
            else:
                parts = [re.search(r"^#define EIGEN_" + field + r"_VERSION (\d+)", content, re.MULTILINE)
                         for field in ("WORLD", "MAJOR", "MINOR")]
                if all(parts): version = ".".join(part.group(1) for part in parts)
        dependency_evidence[name] = {"header": str(path), "version": version,
            "sha256": file_hash(path) if path.is_file() else None}
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if not cpu and os.name != "nt" and cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip(); break
    packages = {}
    for name in ("numpy", "matplotlib", "PyYAML", "psutil"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = None
    return {"os": platform.platform(), "cpu": cpu, "logical_cpus": os.cpu_count(),
        "ram_mb": psutil.virtual_memory().total / 2**20, "python": sys.version,
        "python_executable": sys.executable, "packages": packages,
        "code_commit": git_probe.get("stdout") if git_probe.get("exit_code") == 0 else None,
        "git_note": None if git_probe.get("exit_code") == 0 else "no_git_checkout_source_inventory_is_authoritative",
        "code_sha256": cloud_code_hash(), "compiler": compiler_probe.get("stdout"),
        "cmake": cmake_probe.get("stdout"), "build_flags": {key: cache.get(key) for key in
            ("CMAKE_BUILD_TYPE", "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE", "CMAKE_CXX_STANDARD")},
        "build_cache_sha256": file_hash(cache_path) if cache_path.is_file() else None,
        "worker_sha256": file_hash(worker), "ompl_library_sha256": file_hash(library),
        "ompl_library_path": str(library), "version_probe_evidence": {
            "compiler": compiler_probe, "cmake": cmake_probe, "git": git_probe,
            "linked_libraries": _cloud_probe(["ldd", worker], env=worker_env()) if os.name != "nt" else None},
        "dependency_version_evidence": dependency_evidence,
        "ompl_version": dependency_evidence["ompl"]["version"],
        "boost_version": dependency_evidence["boost"]["version"],
        "eigen_version": dependency_evidence["eigen"]["version"],
        "dependency_lock": read_json(ROOT / "configs/dependencies.lock.json")}


def cloud_identity(config, maps, worker):
    dependencies = read_json(ROOT / "configs/dependencies.lock.json")
    archives = {}
    for name, record in dependencies["archives"].items():
        archive = ROOT / ".deps" / (name + ".tar.gz")
        if not archive.is_file() or file_hash(archive) != record["sha256"]:
            raise ValueError("pinned dependency archive missing or changed: " + name)
        archives[name] = record["sha256"]
    return {"config_hash": digest(config), "code_sha256": cloud_code_hash(),
        "worker_sha256": file_hash(worker), "ompl_library_sha256": file_hash(cloud_ompl_library(worker)),
        "dependency_lock_sha256": file_hash(ROOT / "configs/dependencies.lock.json"),
        "dependency_archives": archives,
        "patch_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted((ROOT / "patches").glob("*.patch"))},
        "map_manifest_sha256": digest(maps), "map_hashes": {m["map_id"]: m["sha256"] for m in maps["maps"]}}

