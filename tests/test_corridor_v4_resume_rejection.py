"""Synthetic regression for a rejected resume before a new session exists.

This file is intentionally separate from the registered CTest runner while
the original smoke campaign is frozen. It never launches a planner.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from corridor_v4_execution import atomic_json, file_hash, read
from run_corridor_v4 import prepare, run


class RejectedResumeTests(unittest.TestCase):
    def test_rejected_resume_preserves_every_historical_session_field(self):
        with tempfile.TemporaryDirectory(prefix="corridor_v4_rejected_resume_synthetic_",
                                         dir=ROOT / "tests/fixtures") as temporary:
            directory = Path(temporary)
            output = directory / "campaign"
            manifest = prepare(output, dry_run=True)
            worker, bridge = directory / "synthetic_worker.exe", directory / "synthetic_bridge.dll"
            worker.write_bytes(b"synthetic fixture; never executed")
            bridge.write_bytes(b"synthetic fixture; never loaded")
            historical = [dict(session_index=0, started_utc="2026-11-11T00:00:00+00:00",
                finished_utc="2026-11-11T00:01:00+00:00", session_wall_s=60.,
                workers=4, validation_workers=4, timing_mode="concurrent_trials",
                collection_s=40., validation_s=20., max_concurrency=4)]
            manifest.update(status="INTERRUPTED", sessions=copy.deepcopy(historical),
                binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge),
                sem_python=str(Path(sys.executable).resolve()))
            atomic_json(output / "manifest.json", manifest)
            with mock.patch("run_corridor_v4._reconcile",
                            side_effect=ValueError("synthetic_invalid_resume_before_session")), \
                 mock.patch("subprocess.Popen", side_effect=AssertionError("unexpected planner launch")):
                result = run(output, worker, bridge, resume=True)
            self.assertEqual(result["status"], "FAILED_INFRASTRUCTURE")
            saved = read(output / "manifest.json")
            self.assertEqual(saved["error"], "synthetic_invalid_resume_before_session")
            self.assertEqual(saved["sessions"], historical)


if __name__ == "__main__":
    unittest.main()
