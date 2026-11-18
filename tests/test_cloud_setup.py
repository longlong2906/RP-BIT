"""Portability and source-package regression tests; all generated inputs are test fixtures."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_dependencies import build_commands, build_environment
from setup_cloud_benchmark import observed_dependency_versions, project_configure
from package_cloud_benchmark import LICENSE_INPUTS, ROOT_FILES, SOURCE_DIRECTORIES, package, verify


class LinuxBuildRecipeTests(unittest.TestCase):
    def test_linux_recipe_preserves_pins_and_uses_pic_for_shared_ompl(self):
        commands = {name: [str(value) for value in argv] for name, argv, _ in
                    build_commands(windows=False, compiler="/usr/bin/g++", root=Path("/workspace/project"))}
        self.assertEqual(commands["boost_bootstrap"], ["bash", "bootstrap.sh", "--with-toolset=gcc"])
        self.assertIn("cxxflags=-fPIC", commands["boost_libraries"])
        self.assertIn("-DCMAKE_DISABLE_FIND_PACKAGE_flann=ON", commands["configure_ompl"])
        self.assertIn("-DCMAKE_DISABLE_FIND_PACKAGE_Triangle=ON", commands["configure_ompl"])
        self.assertIn("-DCMAKE_DISABLE_FIND_PACKAGE_spot=ON", commands["configure_ompl"])
        self.assertIn("-DOMPL_REGISTRATION=OFF", commands["configure_ompl"])
        self.assertLess(list(commands).index("apply_aitstar_paper_formulas_patch"), list(commands).index("configure_ompl"))
        self.assertFalse(any("mingw" in item or "b2.exe" in item or "cmd" == item for items in commands.values() for item in items))

    def test_windows_recipe_retains_mingw_bootstrap_and_no_linux_flags(self):
        commands = {name: [str(value) for value in argv] for name, argv, _ in
                    build_commands(windows=True, compiler="C:/mingw64/bin/g++.exe")}
        self.assertEqual(commands["boost_bootstrap"], ["cmd", "/c", "bootstrap.bat", "gcc"])
        self.assertTrue(commands["boost_libraries"][0].endswith("b2.exe"))
        self.assertNotIn("cxxflags=-fPIC", commands["boost_libraries"])
        self.assertIn("-DCMAKE_CXX_COMPILER=C:/mingw64/bin/g++.exe", commands["configure_ompl"])
        self.assertIn("C:\\mingw64\\bin;", build_environment(windows=True)["PATH"])

    def test_project_release_config_uses_current_python_and_local_dependencies(self):
        command = [str(value) for value in project_configure(ROOT / "build-test-cloud", "/usr/bin/g++")]
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", command)
        self.assertIn(f"-DPython3_EXECUTABLE={sys.executable}", command)
        self.assertIn("-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", command)

    def test_header_version_gate_uses_observed_versions_and_rejects_optional_features(self):
        with tempfile.TemporaryDirectory(prefix="cloud_version_fixture_") as temporary:
            root = Path(temporary)
            headers = {
                ".deps/install/include/ompl-1.7/ompl/config.h": '#define OMPL_VERSION "1.7.0"\n#define OMPL_HAVE_FLANN 0\n#define OMPL_HAVE_SPOT 0\n',
                ".deps/boost_1_82_0/boost/version.hpp": "#define BOOST_VERSION 108200\n",
                ".deps/install/include/eigen3/Eigen/src/Core/util/Macros.h": "#define EIGEN_WORLD_VERSION 3\n#define EIGEN_MAJOR_VERSION 4\n#define EIGEN_MINOR_VERSION 0\n",
            }
            for relative, text in headers.items():
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            evidence = observed_dependency_versions(root)
            self.assertEqual(evidence["versions"], {"ompl": "1.7.0", "boost": "1.82.0", "eigen": "3.4.0"})
            self.assertEqual(len(evidence["header_sha256"]), 3)
            target = root / ".deps/install/include/ompl-1.7/ompl/config.h"
            target.write_text(headers[target.relative_to(root).as_posix()].replace("OMPL_HAVE_FLANN 0", "OMPL_HAVE_FLANN 1"))
            with self.assertRaisesRegex(ValueError, "optional"):
                observed_dependency_versions(root)
            target.write_text(headers[target.relative_to(root).as_posix()].replace('"1.7.0"', '"1.8.0"'))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                observed_dependency_versions(root)


class SourcePackageTests(unittest.TestCase):
    def fixture(self, root):
        for directory in SOURCE_DIRECTORIES:
            (root / directory).mkdir(parents=True)
        for name in ROOT_FILES:
            (root / name).write_bytes(b"synthetic source-package fixture\r\n")
        (root / "src/test.cpp").write_bytes(b"// synthetic fixture\r\n")
        (root / "tools/__pycache__").mkdir()
        (root / "tools/__pycache__/hidden.pyc").write_bytes(b"fixture")
        (root / "results/old").mkdir(parents=True)
        (root / "results/old/trial.json").write_text('{"fixture":true}')
        for relative in LICENSE_INPUTS.values():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic license packaging fixture", encoding="utf-8")

    def test_packages_source_and_licenses_without_old_runs_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory(prefix="cloud_package_fixture_") as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            self.fixture(root)
            output = Path(temporary) / "package"
            manifest = package(output, root)
            self.assertEqual(manifest, verify(output))
            self.assertEqual((output / "src/test.cpp").read_bytes(), b"// synthetic fixture\r\n")
            self.assertFalse((output / "results").exists())
            self.assertFalse((output / ".deps").exists())
            self.assertFalse((output / "tools/__pycache__").exists())
            self.assertEqual(manifest["copied_planner_binaries"], 0)
            self.assertEqual(manifest["copied_historic_trials"], 0)
            self.assertTrue((output / "licenses/OMPL-LICENSE.txt").is_file())
            self.assertIn("* -text", (output / ".gitattributes").read_text())
            with self.assertRaisesRegex(ValueError, "new or empty"):
                package(output, root)
            (output / "src/test.cpp").write_text("tampered fixture")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify(output)

    def test_no_package_over_source_or_parent(self):
        with tempfile.TemporaryDirectory(prefix="cloud_package_boundary_fixture_") as temporary:
            root = Path(temporary) / "source"
            root.mkdir()
            with self.assertRaisesRegex(ValueError, "cannot contain"):
                package(root, root)
            with self.assertRaisesRegex(ValueError, "cannot contain"):
                package(root.parent, root)


if __name__ == "__main__":
    unittest.main()
