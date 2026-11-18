"""Interpreter-selection regressions; synthetic packages, no planner campaign."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import venv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_terrain_compare_fixed_work as fixed_work


class FixedWorkRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="synthetic-fixed-work-runtime-", dir=ROOT / "tests/fixtures")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_registry_executes_venv_aliases_without_resolving_them(self):
        controller = self.root / "controller-venv/bin/python"
        sgp = self.root / "sgp-venv/bin/python"
        worker = self.root / "worker"
        # All hosts can exercise this even when Windows lacks symlink rights.
        # A resolve call would select the base interpreter, as on Linux.
        original_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            if path in (controller, sgp):
                self.fail("execution must preserve the venv alias")
            return original_resolve(path, *args, **kwargs)
        with patch.object(Path, "resolve", resolve), patch.object(fixed_work.sys, "executable", str(controller)):
            registry = fixed_work.backend_registry(worker, sgp)
        self.assertEqual(registry["putn"]["argv"][0], str(controller))
        self.assertEqual(registry["sgp"]["argv"][0], str(sgp))
        for backend in ("putn", "sgp"):
            self.assertEqual(registry[backend]["argv"][1], str(ROOT / "tools/terrain_compare_external_worker.py"))
            self.assertEqual(registry[backend]["cwd"], str(ROOT))
        for backend in ("a3", "sem"):
            self.assertEqual(registry[backend]["argv"], [str(worker)])

    def test_relative_interpreters_become_absolute_without_changing_aliases(self):
        controller = Path("synthetic-controller-venv/bin/python")
        sgp = Path("synthetic-sgp-venv/bin/python")
        registry = fixed_work.backend_registry(self.root / "worker", sgp, controller)
        self.assertEqual(registry["putn"]["argv"][0], str(Path.cwd() / controller))
        self.assertEqual(registry["sgp"]["argv"][0], str(Path.cwd() / sgp))

    @unittest.skipUnless(sys.platform.startswith("linux") and os.name != "nt", "real Linux venv symlink execution required")
    def test_linux_selected_interpreters_keep_their_own_site_packages(self):
        interpreters = {}
        marker = "terrain_compare_synthetic_venv_marker"
        for role in ("controller", "sgp"):
            directory = self.root / role
            venv.EnvBuilder(with_pip=False, symlinks=True).create(directory)
            interpreter = directory / "bin/python"
            self.assertTrue(interpreter.is_symlink())
            probe = subprocess.run([str(interpreter), "-I", "-c",
                "import json,sysconfig;print(json.dumps(sysconfig.get_path('purelib')))"],
                check=True, capture_output=True, text=True, timeout=30)
            package = Path(json.loads(probe.stdout)) / (marker + ".py")
            package.write_text("ROLE = " + repr(role) + "\n", encoding="utf-8")
            interpreters[role] = interpreter
        registry = fixed_work.backend_registry(self.root / "worker", interpreters["sgp"], interpreters["controller"])
        for backend, role in (("putn", "controller"), ("sgp", "sgp")):
            command = registry[backend]["argv"][0]
            probe = subprocess.run([command, "-I", "-c", f"import {marker};print({marker}.ROLE)"],
                check=True, capture_output=True, text=True, timeout=30)
            self.assertEqual(probe.stdout.strip(), role)
            # The resolved executable cannot see the package installed only in
            # this venv. This distinguishes namespace preservation from hashing.
            base_probe = subprocess.run([str(interpreters[role].resolve()), "-I", "-c", f"import {marker}"],
                check=False, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(base_probe.returncode, 0)
            self.assertIn("ModuleNotFoundError", base_probe.stderr)


if __name__ == "__main__":
    unittest.main()
