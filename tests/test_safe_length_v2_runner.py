"""Synthetic preparation/guard tests; do not launch scientific planners."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from run_safe_length_v2 import prepare, run, execution_gate


class RunnerTests(unittest.TestCase):
    def test_pilot_full_require_explicit_flags_before_io(self):
        for profile in ("pilot","full"):
            with self.assertRaisesRegex(ValueError,"requires"):
                run("does-not-exist","does-not-exist",profile=profile)
        execution_gate("pilot",allow_pilot=True)
        execution_gate("full",allow_full=True)

    def test_inventory_is_no_execution_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="v2_inventory_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            output = Path(tmp)/"inventory"
            with mock.patch("subprocess.Popen",side_effect=AssertionError("inventory launched subprocess")):
                manifest = prepare(output,profile="pilot",dry_run=True)
            self.assertEqual(manifest["status"],"PREPARED_INVENTORY")
            self.assertEqual(manifest["planned_trials"],210)
            self.assertEqual(manifest["maps"],[])
            self.assertFalse((output/"inputs").exists())
            with self.assertRaises(FileExistsError):
                prepare(output,profile="pilot",dry_run=True)
            with self.assertRaisesRegex(ValueError,"not an unattempted"):
                run(output,"does-not-exist",profile="pilot",allow_pilot=True)

    def test_full_preparation_needs_selection_but_inventory_does_not(self):
        with tempfile.TemporaryDirectory(prefix="v2_full_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            output = Path(tmp)/"full"
            with self.assertRaisesRegex(ValueError,"selection"):
                prepare(output,profile="full")
            self.assertFalse(output.exists())
            manifest = prepare(output,profile="full",dry_run=True)
            self.assertEqual(manifest["planned_trials"],1500)
            self.assertTrue(manifest["requires_selected_pilot"])


if __name__ == "__main__":
    unittest.main()
