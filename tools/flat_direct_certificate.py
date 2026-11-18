"""Independent algebraic certificate reconstructed from raw map and path inputs.

No tolerance, objective equality, rounded risk product, or worker raster is used.
Hard feasibility is evaluated by the independent continuous geometry oracle.
"""
import math
import numpy as np

POLICY = "flat_direct_v1"
TERMINATION = "EXACT_FLAT_DIRECT_CERTIFICATE"


def reconstruct(oracle, xy, start, goal, traversability):
    heights = oracle.heights.ravel()
    complete_constant = bool(heights.size and np.isfinite(heights).all()
                             and np.all(heights == heights[0]))
    direct = (len(xy) == 2 and start != goal and
              (xy == [start, goal] or xy == [goal, start]))
    # Branch on inputs, not lambda * risk, which can underflow to zero.
    zero_risk = (traversability["lambda"] == 0 or
                 (complete_constant and
                  (traversability["clearance_weight"] == 0 or not oracle.rectangles)))
    hard_valid = False
    if direct:
        oriented = xy if xy[0] == start else list(reversed(xy))
        hard_valid = bool(oracle.validate_path(oriented, start, goal)["valid"])
    conditions = {"complete_constant_height": complete_constant,
                  "direct_endpoint_path": bool(direct), "zero_active_risk": bool(zero_risk),
                  "hard_validity_checked": bool(direct), "hard_valid": hard_valid}
    return {"policy": POLICY, "certified": all(conditions.values()), "conditions": conditions,
            "proof": "constant height; direct feasible XY connector; zero active density; J=dxy<=J(any feasible path)"}


def verify_claim(oracle, request, trial, snapshots):
    """Validate provenance and proof of a claimed certificate; return errors."""
    certificate = trial.get("exact_certificate")
    trigger = trial.get("termination_reason") == TERMINATION
    expected_enabled = request.get("exact_flat_certificate", False)
    if certificate is None:
        return {"valid": not (expected_enabled or trigger), "claimed": trigger,
                "errors": ["missing_exact_certificate_provenance"] if expected_enabled or trigger else []}
    errors = []
    if (not isinstance(certificate, dict) or certificate.get("policy") != POLICY or
            certificate.get("enabled") is not expected_enabled or
            type(certificate.get("certified")) is not bool):
        return {"valid": False, "claimed": trigger, "errors": ["invalid_exact_certificate_policy"]}
    claimed = certificate["certified"]
    if claimed and not expected_enabled:
        errors.append("certificate_claimed_when_disabled")
    if trigger != claimed:
        errors.append("certificate_trigger_provenance_mismatch")
    if (type(certificate.get("triggered_termination")) is not bool or
            certificate["triggered_termination"] != trigger):
        errors.append("certificate_actual_trigger_metadata_mismatch")
    for key in ("check_wall_s",):
        value = certificate.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            errors.append("invalid_certificate_" + key)
    checks = certificate.get("checks")
    if type(checks) is not int or checks < 0 or (claimed and checks == 0):
        errors.append("invalid_certificate_check_count")
    reconstructed = None
    if claimed:
        sid = certificate.get("snapshot_id")
        if not isinstance(sid, str) or sid not in snapshots:
            errors.append("certificate_snapshot_missing")
        else:
            snapshot = snapshots[sid]
            reconstructed = reconstruct(oracle, snapshot["path_xy"], request["start"],
                                        request["goal"], request["traversability"])
            if not snapshot["valid"] or not reconstructed["certified"]:
                errors.append("certificate_independent_reconstruction_failed")
            if certificate.get("conditions") != reconstructed["conditions"]:
                errors.append("certificate_conditions_mismatch")
    return {"valid": not errors, "claimed": claimed, "errors": errors,
            "snapshot_id": certificate.get("snapshot_id"), "reconstruction": reconstructed}
