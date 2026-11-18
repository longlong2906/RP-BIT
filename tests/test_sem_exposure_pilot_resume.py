"""Check the recovery notebook and its supervisor adapter without planner runs."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from build_sem_exposure_pilot_resume import CHECKPOINT_SHA256, LAUNCHER, SOURCE_SHA256, make_notebook


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = {"__name__": "test_adapter"}
        exec(compile(LAUNCHER, "resume-adapter-test", "exec"), self.adapter)

    def test_notebook_uses_bound_checkpoint_and_original_source(self):
        notebook = make_notebook()
        setup = "".join(notebook["cells"][1]["source"])
        self.assertIn(CHECKPOINT_SHA256, setup)
        self.assertIn(SOURCE_SHA256, setup)
        self.assertIn("STAGE = 'all'", setup)
        restore = "".join(notebook["cells"][2]["source"])
        self.assertIn("restore_checkpoint(RESUME_INDEX", restore)
        self.assertIn("attempted != 2400", restore)
        self.assertIn("source_input.zip", restore)
        self.assertIn("--restored-runtime", "".join(notebook["cells"][3]["source"]))

    def test_only_exact_resource_tracker_and_pid_identity_are_recognized(self):
        identify = self.adapter["tracker_identity"]
        snapshot = {7: {"start_ticks": 81}}
        command = b"/venv/bin/python\0-c\0from multiprocessing.resource_tracker import main;main(8)\0"
        self.assertTrue(identify(7, 81, snapshot, lambda _: command))
        self.assertFalse(identify(7, 80, snapshot, lambda _: command))
        self.assertFalse(identify(7, 81, snapshot, lambda _: command+b"extra"))
        self.assertFalse(identify(7, 81, snapshot, lambda _: b"/repo/terrain_benchmark\0"))

    def test_supervision_still_fails_for_an_unknown_terminated_child(self):
        with tempfile.TemporaryDirectory(prefix="resume-adapter-", dir=ROOT/"tests/fixtures") as folder:
            workspace = Path(folder)
            supervisor = SimpleNamespace(linux_host=lambda: True,
                process_snapshot=lambda: {7: {"start_ticks": 81}},
                track_descendants=lambda *_: {7}, quiesce_descendants=lambda *_: [7])
            self.adapter["install_drain"](supervisor, workspace, grace_s=0.)
            with patch.object(Path, "read_bytes", return_value=b"/repo/terrain_benchmark\0"):
                self.assertEqual([7], supervisor.quiesce_descendants(10, {7: 81}, set()))
            with patch.object(Path, "read_bytes", return_value=
                              b"/venv/bin/python\0-c\0from multiprocessing.resource_tracker import main;main(8)\0"):
                self.assertEqual([], supervisor.quiesce_descendants(10, {7: 81}, set()))
            self.assertIn("cleaned_verified_python_resource_tracker",
                          (workspace/"logs/resume_tracker_cleanup.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
