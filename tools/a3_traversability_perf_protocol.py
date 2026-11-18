"""Frozen schema-9 smoke: cache, same-field builder, and exact flat certificate."""
import copy

from a3_traversability_protocol import (DEFAULT_TRAVERSABILITY, FAMILIES,
    PLANNERS as OLD_PLANNERS, VARIANTS as OLD_VARIANTS, _generate_maps,
    validate_config as validate_v8)

PROTOCOL_ID = "a3_traversability_perf_smoke_v1"
GENERATOR_VERSION = "9.0.0"
CERTIFICATE_POLICY = "flat_direct_v1"
PERF_DEFAULTS = {"sampler_query_cache": False, "traversability_build_policy": "legacy",
                 "exact_flat_certificate": False}
PLANNERS = ("a3_perf_baseline", "a3_perf_sampler", "a3_perf_preprocessing",
            "a3_perf_both_kernels", "a3_perf_certificate", "a3_perf_all")
VARIANTS = {}
for name, cache, builder, certificate in zip(PLANNERS,
        (False, True, False, True, False, True),
        (False, False, True, True, False, True),
        (False, False, False, False, True, True)):
    VARIANTS[name] = dict(planner_id="bitstar_a3", state_cache=True,
        scalar_cost=True, collision_index=True, sampler_query_cache=cache,
        traversability_build_policy="skip_dominated" if builder else "legacy",
        exact_flat_certificate=certificate)


def validate_config(config, profile="smoke"):
    if profile != "smoke":
        raise ValueError("schema v9 authorizes smoke only; pilot/full unsupported")
    if config.get("schema_version") != 9:
        raise ValueError("performance protocol requires schema 9")
    for key, default in PERF_DEFAULTS.items():
        value = config.get("algorithm", {}).get(key)
        if type(value) is not type(default) or value != default:
            raise ValueError("schema v9 algorithm default differs: " + key)
    if config.get("planner_variants") != VARIANTS:
        raise ValueError("schema v9 requires the six fixed performance variants")
    # Python bool/int equality must not silently admit 0/1 instead of booleans.
    for name, variant in config["planner_variants"].items():
        for key, expected in VARIANTS[name].items():
            if type(variant[key]) is not type(expected):
                raise ValueError("invalid performance option type: " + key)
    if config.get("profiles", {}).get("smoke", {}).get("planners") != list(PLANNERS):
        raise ValueError("schema v9 planner order/design differs")
    projected = copy.deepcopy(config)
    projected["schema_version"] = 8
    for key in PERF_DEFAULTS:
        del projected["algorithm"][key]
    projected["planner_variants"] = copy.deepcopy(OLD_VARIANTS)
    projected["profiles"]["smoke"]["planners"] = list(OLD_PLANNERS)
    validate_v8(projected, profile)
    if config["algorithm"]["informed_sample_attempts_per_call"] != 100:
        raise ValueError("schema v9 requires 100 sampler attempts")
    return config


def generate_maps(config, profile, output):
    validate_config(config, profile)
    return _generate_maps(config, profile, output, schema_version=9,
                          protocol_id=PROTOCOL_ID, generator_version=GENERATOR_VERSION)
