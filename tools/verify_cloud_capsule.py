"""Read-only verifier for an A3-vs-all Cloud validation capsule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from run_cloud_posthoc_validation import IntegrityError, verify_capsule


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capsule", required=True, type=Path)
    parser.add_argument("--sha256", type=Path)
    args = parser.parse_args()
    try:
        capsule = verify_capsule(args.capsule, args.sha256)
    except IntegrityError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS",
                "capsule_sha256": capsule.sha256,
                "capsule_manifest_sha256": capsule.manifest_sha256,
                "pending_trials": len(capsule.manifest["pending_trial_ids"]),
                "control_trials": len(capsule.manifest["control_trial_ids"]),
                "source_at_run_sha256": capsule.manifest["primary"]["baseline_hashes"]["source_at_run.zip"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
