"""Synthetic bootstrap regressions; no network, planner, or system installation."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import kaggle_a3 as runtime


class DownloadResponse(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.headers = {"Content-Length": str(len(content))}
        self.status = 200

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def geturl(self):
        return "https://files.pythonhosted.org/synthetic/uv.whl"


class KaggleBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="synthetic-kaggle-bootstrap-",
                                               dir=ROOT / "tests/fixtures")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.output = self.root / "output"
        (self.output / "setup").mkdir(parents=True)
        self.started = time.monotonic()
        self.args = argparse.Namespace(output=self.output, resume_archive=None,
                                       notebook_started_monotonic=self.started)
        self.host = self.source / "host-python3.13"
        self.native = self.source / "native-python3.12"
        self.uv = self.source / ".bootstrap-uv/uv"
        self.managed = self.source / ".managed-python/fixture/bin/python3.12"
        self.venv_python = self.source / ".venv-kaggle/bin/python"
        for path in (self.host, self.native, self.uv, self.managed):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"synthetic executable; must never be executed")
        self.freeze = "numpy==1.26.4\npip==25.1\npsutil==5.9.0"
        (self.source / "requirements-cloud-benchmark.txt").write_text(
            "numpy==1.26.4\npsutil==5.9.0\n", encoding="utf-8")
        self.commands = []
        self.probes = []

    @staticmethod
    def probe_value(version="3.12.3", full="3.12.3 (synthetic build)", headers=True):
        return {"version": version, "full": full, "headers_available": headers}

    def command(self, name, argv, *_args, **kwargs):
        command = list(map(str, argv))
        self.commands.append((name, command, kwargs))
        if "python" in command and "find" in command:
            return str(self.managed)
        if "venv" in command:
            self.venv_python.parent.mkdir(parents=True, exist_ok=True)
            self.venv_python.write_bytes(b"synthetic virtualenv interpreter")
        if name == "python_freeze":
            return self.freeze
        if "--version" in command:
            return "uv " + runtime.UV_VERSION
        return ""

    @contextlib.contextmanager
    def isolate(self, native=True, candidate=None, isolated=None, command=None):
        candidate = self.probe_value() if candidate is None else candidate
        isolated = self.probe_value() if isolated is None else isolated

        def probe(name, executable, *_args, **_kwargs):
            executable = Path(executable)
            self.probes.append((name, executable))
            return dict(isolated if executable == self.venv_python else candidate)

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "ROOT", self.source))
            stack.enter_context(patch.object(runtime.sys, "version_info", (3, 13, 0)))
            stack.enter_context(patch.object(runtime.sys, "executable", str(self.host)))
            stack.enter_context(patch.object(runtime.shutil, "which",
                side_effect=lambda name: str(self.native) if native and name == "python3.12" else None))
            stack.enter_context(patch.object(runtime, "bootstrap_uv", return_value=self.uv))
            stack.enter_context(patch.object(runtime, "python_probe", side_effect=probe))
            stack.enter_context(patch.object(runtime, "run_command", side_effect=command or self.command))
            yield

    def assert_workspace_bootstrap(self, expected_python):
        creators = [command for _, command, _ in self.commands if "venv" in command]
        self.assertEqual(len(creators), 1)
        creation = creators[0]
        self.assertEqual(creation[0], str(self.uv))
        self.assertEqual(creation[creation.index("--python") + 1], str(expected_python))
        for flag in ("--seed", "--no-python-downloads", "--no-config"):
            self.assertIn(flag, creation)
        for _, command, _ in self.commands:
            self.assertNotEqual(command[:3], [str(self.host), "-m", "pip"])
            self.assertFalse(any(command[index:index + 2] == ["-m", "venv"]
                                 for index in range(len(command) - 1)))
            self.assertNotIn("apt", command)
            self.assertNotIn("apt-get", command)
        installs = [command for name, command, _ in self.commands if name == "install_locked_requirements"]
        self.assertEqual(len(installs), 1)
        self.assertEqual(installs[0][0], str(self.venv_python))

    def test_native_python312_under_host313_uses_uv_without_ensurepip_or_host_pip(self):
        with self.isolate():
            result = runtime.isolated_python(self.args)
        self.assertEqual(result, self.venv_python)
        self.assert_workspace_bootstrap(self.native)
        self.assertFalse(any("python" in command and "install" in command
                             for _, command, _ in self.commands))
        self.assertIn(self.venv_python, [executable for _, executable in self.probes])
        provenance = runtime.read(self.output / "setup/python_bootstrap.json")
        self.assertIn("3.12.3", json.dumps(provenance))
        self.assertEqual((self.output / "setup/python-freeze.txt").read_text().strip(), self.freeze)

    def assert_managed_selected(self):
        self.assert_workspace_bootstrap(self.managed)
        provisions = [command for _, command, _ in self.commands
                      if "python" in command and "install" in command]
        self.assertEqual(len(provisions), 1)
        self.assertIn("3.12", provisions[0])
        finds = [command for _, command, _ in self.commands if "python" in command and "find" in command]
        self.assertEqual(len(finds), 1)
        self.assertIn("--managed-python", finds[0])
        self.assertIn("--no-python-downloads", finds[0])

    def test_missing_native_interpreter_selects_workspace_managed_python(self):
        with self.isolate(native=False):
            runtime.isolated_python(self.args)
        self.assert_managed_selected()

    def test_native_missing_development_headers_selects_managed_before_install(self):
        with self.isolate(candidate=self.probe_value(headers=False)):
            runtime.isolated_python(self.args)
        self.assert_managed_selected()

    def test_native_wrong_minor_version_never_receives_benchmark_dependencies(self):
        with self.isolate(candidate=self.probe_value(version="3.13.1", full="3.13.1 (fixture)")):
            runtime.isolated_python(self.args)
        self.assert_managed_selected()

    def prior(self):
        self.args.resume_archive = self.root / "synthetic-checkpoint.zip"
        runtime.write(self.output / "setup/runtime.json", {
            "python_version": "3.12.3", "python_full": "3.12.3 (synthetic build)"})
        (self.output / "setup/python-freeze.txt").write_text(self.freeze + "\n", encoding="utf-8")

    def test_resume_exact_interpreter_and_all_frozen_packages_are_preserved(self):
        self.prior()
        before = (self.output / "setup/python-freeze.txt").read_bytes()
        with self.isolate():
            runtime.isolated_python(self.args)
        self.assert_workspace_bootstrap(self.native)
        install = next(command for name, command, _ in self.commands if name == "install_locked_requirements")
        self.assertIn("--no-deps", install)
        self.assertEqual(install[install.index("-r") + 1], str(self.output / "setup/python-freeze.txt"))
        freeze = next(command for name, command, _ in self.commands if name == "python_freeze")
        self.assertIn("--all", freeze)
        self.assertEqual((self.output / "setup/python-freeze.txt").read_bytes(), before)

    def test_resume_rejects_python_full_mismatch_before_installing_dependencies(self):
        self.prior()
        with self.isolate(isolated=self.probe_value(full="3.12.3 (different compiler build)")):
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.isolated_python(self.args)
        self.assertFalse(any(name == "install_locked_requirements" for name, _, _ in self.commands))

    def test_created_interpreter_wrong_version_is_rejected_before_dependencies(self):
        with self.isolate(isolated=self.probe_value(version="3.13.1", full="3.13.1 (fixture)")):
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.isolated_python(self.args)
        self.assertFalse(any(name == "install_locked_requirements" for name, _, _ in self.commands))

    def test_resume_rejects_package_drift_and_retains_original_freeze(self):
        self.prior()
        before = (self.output / "setup/python-freeze.txt").read_bytes()

        def changed(name, argv, *args, **kwargs):
            result = self.command(name, argv, *args, **kwargs)
            return result + "\nunrequested-package==1.0" if name == "python_freeze" else result

        with self.isolate(command=changed):
            with self.assertRaisesRegex(ValueError, "packages differ"):
                runtime.isolated_python(self.args)
        self.assertEqual((self.output / "setup/python-freeze.txt").read_bytes(), before)

    def test_locked_requirement_failure_does_not_retry_or_switch_interpreter(self):
        def fail(name, argv, *args, **kwargs):
            result = self.command(name, argv, *args, **kwargs)
            if name == "install_locked_requirements":
                raise RuntimeError("FAILED: pinned requirement fixture failure")
            return result

        with self.isolate(command=fail):
            with self.assertRaisesRegex(RuntimeError, "pinned requirement fixture failure"):
                runtime.isolated_python(self.args)
        self.assertEqual(sum(name == "install_locked_requirements" for name, _, _ in self.commands), 1)
        self.assertFalse(any("python" in command and "install" in command
                             for _, command, _ in self.commands))
        self.assertFalse(any(name == "python_freeze" for name, _, _ in self.commands))

    def test_pip_install_and_freeze_ignore_inherited_redirects_and_keep_workspace_cache(self):
        inherited_redirects = {
            "PIP_TARGET": "/outside/target", "PIP_PREFIX": "/outside/prefix",
            "PIP_USER": "1", "PIP_CACHE_DIR": "/outside/cache",
            "PIP_CONFIG_FILE": "/outside/pip.conf", "PYTHONPATH": "/outside/modules",
            "PYTHONHOME": "/outside/python",
        }
        with patch.dict(os.environ, inherited_redirects), self.isolate():
            runtime.isolated_python(self.args)
            # Isolation applies to child processes; never rewrite global
            # settings merely to make a notebook bootstrap work.
            for key, value in inherited_redirects.items():
                self.assertEqual(os.environ[key], value)
        pip_commands = [(name, kwargs["env"]) for name, _, kwargs in self.commands
                        if name in ("install_locked_requirements", "python_freeze")]
        self.assertEqual(len(pip_commands), 2)
        for name, environment in pip_commands:
            with self.subTest(command=name):
                self.assertEqual(environment["PIP_CACHE_DIR"], str(self.source / ".pip-cache"))
                self.assertEqual(environment["PIP_CONFIG_FILE"], os.devnull)
                self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
                for key in ("PIP_TARGET", "PIP_PREFIX", "PIP_USER", "PYTHONPATH", "PYTHONHOME"):
                    self.assertNotIn(key, environment)

    @contextlib.contextmanager
    def wheel(self, *, bad_archive_hash=False, bad_binary_hash=False):
        binary = b"\x7fELF synthetic uv fixture, not executable code"
        member = "uv-0.6.17.data/scripts/uv"
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w") as archive:
            archive.writestr(member, binary)
            # Even an unrelated traversal member must never be extracted;
            # bootstrap reads only the explicitly pinned binary member.
            archive.writestr("../must-not-be-created.txt", "synthetic unrelated wheel member")
        data = payload.getvalue()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "ROOT", self.source))
            stack.enter_context(patch.object(runtime.platform, "machine", return_value="x86_64"))
            for name, value in {
                "UV_WHEEL_URL": "https://files.pythonhosted.org/synthetic/uv.whl",
                "UV_WHEEL_SHA256": "0" * 64 if bad_archive_hash else hashlib.sha256(data).hexdigest(),
                "UV_WHEEL_SIZE": len(data),
                "UV_BINARY_MEMBER": member,
                "UV_BINARY_SHA256": "0" * 64 if bad_binary_hash else hashlib.sha256(binary).hexdigest(),
            }.items():
                stack.enter_context(patch.object(runtime, name, value))
            stack.enter_context(patch("urllib.request.urlopen", side_effect=lambda *_a, **_kw: DownloadResponse(data)))
            command = stack.enter_context(patch.object(runtime, "run_command", return_value="uv " + runtime.UV_VERSION))
            stack.enter_context(patch.object(runtime, "guard_disk"))
            # The synthetic executable used by selection tests is not a
            # bootstrap result. Remove only this known fixture before download.
            self.uv.unlink()
            self.uv.parent.rmdir()
            yield binary, command

    def test_verified_wheel_extracts_only_pinned_binary_and_records_provenance(self):
        with self.wheel() as (binary, command):
            result = runtime.bootstrap_uv(self.output, self.started)
        self.assertTrue(self.source in result.resolve().parents)
        self.assertEqual(result.read_bytes(), binary)
        self.assertFalse((self.source / "must-not-be-created.txt").exists())
        self.assertFalse((self.root / "must-not-be-created.txt").exists())
        evidence = runtime.read(self.output / "setup/uv_bootstrap.json")
        self.assertIn(hashlib.sha256(binary).hexdigest(), json.dumps(evidence))
        self.assertTrue(any("--version" in list(map(str, call.args[1])) for call in command.call_args_list))

    def test_corrupted_wheel_is_rejected_before_executing_uv(self):
        with self.wheel(bad_archive_hash=True) as (_, command):
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.bootstrap_uv(self.output, self.started)
            command.assert_not_called()
        self.assertFalse(self.uv.exists())

    def test_corrupted_member_is_rejected_before_executing_uv(self):
        with self.wheel(bad_binary_hash=True) as (_, command):
            with self.assertRaises((ValueError, RuntimeError)):
                runtime.bootstrap_uv(self.output, self.started)
            command.assert_not_called()
        evidence = runtime.read(self.output / "setup/uv_bootstrap.json")
        self.assertEqual(evidence["status"], "FAILED")


if __name__ == "__main__":
    unittest.main()
