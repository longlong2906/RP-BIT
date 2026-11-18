"""Synthetic native-linkage fixtures; no invented Linux runtime measurements."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import terrain_compare_runtime as r


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-native-linkage-", dir=ROOT / "tests/fixtures")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.worker = self.root / "worker"
        self.worker.write_bytes(b"synthetic root")
        self.first = self.root / "libfirst.so.1"
        self.second = self.root / "libsecond.so.1"
        self.first.write_bytes(b"first library")
        self.second.write_bytes(b"second library")

    def capture(self, library=None):
        path = library or self.first
        output = "linux-vdso.so.1 (0x12345)\nlibtest.so.1 => " + str(path) + " (0x12345)\n"
        result = subprocess.CompletedProcess([], 0, output, "")
        with patch.object(r.sys, "platform", "linux"), patch.object(r.subprocess, "run", return_value=result):
            return r.capture_native_dependencies([self.worker])

    def test_ldd_closure_has_hashes_and_stable_mapping_without_addresses(self):
        result = self.capture()
        self.assertEqual(result["binary_hashes"][str(self.first.resolve())], r._hash(self.first))
        self.assertIn("linux_ldd", result["linkage"]["scope"])
        self.assertNotIn("0x12345", json.dumps(result))
        self.assertEqual(result["linkage"]["roots"][str(self.worker)]["virtual_objects"], ["linux-vdso.so.1"])

    def test_missing_dependency_fails_closed(self):
        for output in ("libmissing.so => not found", "libmissing.so => /missing/path.so (0x1234)"):
            with self.assertRaises(r.NativeDependencyError):
                r.parse_ldd(output)

    def test_ldd_failure_is_not_an_empty_successful_closure(self):
        result = subprocess.CompletedProcess([], 1, "", "bad ELF")
        with patch.object(r.sys, "platform", "linux"), patch.object(r.subprocess, "run", return_value=result):
            with self.assertRaisesRegex(r.NativeDependencyError, "ldd failed"):
                r.capture_native_dependencies([self.worker])

    def test_substitution_detected_even_if_original_library_unchanged(self):
        first = self.capture(self.first)
        second = self.capture(self.second)
        self.assertNotEqual(first["linkage"], second["linkage"])
        self.assertEqual(r._hash(self.first), first["binary_hashes"][str(self.first)])

    def test_changed_library_in_place_changes_hash_not_mapping(self):
        first = self.capture()
        self.first.write_bytes(b"changed in place")
        second = self.capture()
        self.assertEqual(first["linkage"], second["linkage"])
        self.assertNotEqual(first["binary_hashes"], second["binary_hashes"])

    def test_windows_scope_never_claims_linux_or_complete_loader_audit(self):
        runtime = self.root / "runtime"
        runtime.mkdir()
        dll = runtime / "libstdc++-6.dll"
        dll.write_bytes(b"synthetic Windows runtime")
        with patch.object(r.sys, "platform", "win32"), patch.object(r, "ROOT", self.root), \
             patch.object(r, "WINDOWS_RUNTIME_DIR", runtime), patch.object(r.subprocess, "run") as run:
            result = r.capture_native_dependencies([self.worker])
        run.assert_not_called()
        self.assertIn("windows_explicit", result["linkage"]["scope"])
        self.assertIn("not_complete_OS_loader_closure", result["linkage"]["scope"])
        self.assertIn(str(dll), result["binary_hashes"])

    def test_selected_python_libraries_hashed_without_importing_torch(self):
        interpreter = self.root / "python3.10"
        interpreter.write_bytes(b"synthetic interpreter")
        package = self.root / "site/torch/lib"
        package.mkdir(parents=True)
        library = package / "libtorch_cpu.so"
        library.write_bytes(b"synthetic torch")
        outputs = [subprocess.CompletedProcess([], 0, "", ""),
                   subprocess.CompletedProcess([], 0, json.dumps({"sites": [str(self.root / "site")],
                                                                 "base_prefix": str(self.root)}), "")]
        with patch.object(r.sys, "platform", "linux"), patch.object(r.subprocess, "run", side_effect=outputs) as run:
            result = r.capture_native_dependencies([interpreter])
        self.assertIn(str(library), result["binary_hashes"])
        self.assertEqual(result["linkage"]["python_extensions"][str(interpreter)], [str(library)])
        code = run.call_args_list[1].args[0][-1]
        self.assertNotIn("import torch", code)
        self.assertIn("site.getsitepackages", code)

    def make_venv_layout(self):
        alias = self.root / "venv/bin/python3.12"
        target = self.root / "managed-python/bin/python3.12"
        library = self.root / "managed-python/lib/libpython3.12.so.1.0"
        extension = self.root / "venv/lib/python3.12/site-packages/numpy/core.so"
        for path, content in ((alias, b"synthetic interpreter"), (target, b"synthetic interpreter"),
                              (library, b"synthetic libpython"), (extension, b"synthetic venv package")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        # Windows need not grant symlink privileges: only Path.resolve for this
        # synthetic alias is replaced; all file hashing and ldd parsing are real.
        original_resolve = Path.resolve
        def resolve(path, *args, **kwargs):
            return target if path == alias else original_resolve(path, *args, **kwargs)
        return alias, target, library, extension, resolve

    def test_venv_ldd_uses_real_elf_but_python_probe_keeps_venv(self):
        alias, target, library, extension, resolve = self.make_venv_layout()
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            self.assertNotIn("env", kwargs)  # Do not mask resolution by changing LD_LIBRARY_PATH.
            if argv[:1] == ["ldd"]:
                # Model the observed loader failure for $ORIGIN via the alias.
                output = ("libpython3.12.so.1.0 => " + str(library) + " (0x1234)\n"
                          if argv[1] == str(target) else "libpython3.12.so.1.0 => not found\n")
            else:
                self.assertEqual(argv[0], str(alias))
                output = json.dumps({"sites": [str(extension.parent.parent)], "base_prefix": str(target.parent.parent)})
            return subprocess.CompletedProcess(argv, 0, output, "")
        with patch.object(Path, "resolve", resolve), patch.object(r.sys, "platform", "linux"), \
             patch.object(r.subprocess, "run", side_effect=run):
            observed = r.capture_native_dependencies([alias])
        self.assertEqual(calls[0], ["ldd", str(target)])
        self.assertEqual(calls[1][0], str(alias))
        root_identity = observed["linkage"]["roots"][str(alias)]
        self.assertEqual(root_identity["requested_root"], str(alias))
        self.assertEqual(root_identity["resolved_target"], str(target))
        self.assertEqual(observed["binary_hashes"][str(alias)], r._hash(alias))
        self.assertEqual(observed["binary_hashes"][str(target)], r._hash(target))
        self.assertEqual(observed["binary_hashes"][str(library)], r._hash(library))
        self.assertEqual(observed["binary_hashes"][str(extension)], r._hash(extension))
        self.assertEqual(observed["linkage"]["python_extensions"][str(alias)], [str(extension)])

    def test_same_byte_interpreter_retarget_changes_recorded_linkage(self):
        alias, target, library, _, resolve = self.make_venv_layout()
        replacement = self.root / "another-managed-python/bin/python3.12"
        replacement.parent.mkdir(parents=True)
        replacement.write_bytes(target.read_bytes())
        original_resolve = Path.resolve
        def retarget(path, *args, **kwargs):
            return replacement if path == alias else original_resolve(path, *args, **kwargs)
        result = subprocess.CompletedProcess([], 0, "libpython.so => " + str(library) + " (0x1234)\n", "")
        with patch.object(r.sys, "platform", "linux"), patch.object(r.subprocess, "run", return_value=result), \
             patch.object(r, "_python_libraries", return_value=[]):
            with patch.object(Path, "resolve", resolve):
                before = r.capture_native_dependencies([alias])
            with patch.object(Path, "resolve", retarget):
                after = r.capture_native_dependencies([alias])
        self.assertEqual(before["binary_hashes"][str(alias)], after["binary_hashes"][str(alias)])
        self.assertEqual(before["binary_hashes"][str(target)], after["binary_hashes"][str(replacement)])
        self.assertNotEqual(before["linkage"], after["linkage"])
        self.assertEqual(after["linkage"]["roots"][str(alias)]["resolved_target"], str(replacement))

    def test_missing_dependency_of_real_elf_still_fails_without_alias_fallback(self):
        alias, target, _, _, resolve = self.make_venv_layout()
        missing = str(target.parent / "../lib/libpython3.12.so.1.0") + " => not found\n"
        result = subprocess.CompletedProcess([], 0, missing, "")
        with patch.object(Path, "resolve", resolve), patch.object(r.sys, "platform", "linux"), \
             patch.object(r.subprocess, "run", return_value=result) as run:
            with self.assertRaisesRegex(r.NativeDependencyError, "unresolved native library"):
                r.capture_native_dependencies([alias])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0], ["ldd", str(target)])

    @unittest.skipUnless(sys.platform.startswith("linux") and os.name != "nt", "real Linux ELF/$ORIGIN integration required")
    def test_linux_origin_elf_via_venv_alias(self):
        compiler = shutil.which("cc") or shutil.which("gcc")
        if not compiler or not shutil.which("ldd"):
            self.skipTest("Linux compiler and ldd required for the synthetic ELF fixture")
        native = self.root / "native"
        (native / "bin").mkdir(parents=True)
        (native / "lib").mkdir()
        alias = self.root / "venv/bin/python3.12"
        alias.parent.mkdir(parents=True)
        library_source, main_source = self.root / "fixture-lib.c", self.root / "fixture-main.c"
        library_source.write_text("int fixture(void) { return 0; }\n")
        main_source.write_text('#include <stdio.h>\nextern int fixture(void);\n'
                               'int main(void) { puts("{\\"sites\\": [], \\"base_prefix\\": \\".\\"}"); return fixture(); }\n')
        library, target = native / "lib/libvenvfixture.so", native / "bin/python3.12"
        # Match the managed CPython failure: $ORIGIN is in DT_NEEDED itself,
        # through the library SONAME, rather than merely in an ELF RUNPATH.
        subprocess.run([compiler, "-fPIC", "-shared", str(library_source),
                        "-Wl,-soname,$ORIGIN/../lib/libvenvfixture.so", "-o", str(library)], check=True,
                       capture_output=True, text=True, timeout=30)
        subprocess.run([compiler, str(main_source), "-L" + str(native / "lib"), "-lvenvfixture",
                        "-o", str(target)], check=True,
                       capture_output=True, text=True, timeout=30)
        alias.symlink_to(target)
        # This is an intentionally synthetic ELF, not a real Python interpreter.
        # Its dependency needs the real executable's $ORIGIN; execution through
        # the venv alias must still work before capture is attempted.
        executed = subprocess.run([str(alias)], check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(json.loads(executed.stdout)["sites"], [])
        observed = r.capture_native_dependencies([alias])
        identity = observed["linkage"]["roots"][str(alias)]
        self.assertEqual(identity["resolved_target"], str(target))
        self.assertIn(str(library), identity["libraries"].values())
        self.assertIn(str(library), observed["binary_hashes"])


if __name__ == "__main__":
    unittest.main()
