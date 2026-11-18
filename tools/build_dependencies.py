"""Build pinned dependencies locally on Linux/Windows; retain every command."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"


def build_environment(windows=None):
    windows = os.name == "nt" if windows is None else windows
    result = dict(os.environ)
    if windows:
        result["PATH"] = "C:\\mingw64\\bin;" + result.get("PATH", "")
    return result


def run(name, argv, cwd=ROOT, *, log_dir=None):
    directory = Path(log_dir or ROOT / "logs/build")
    base, index = name, 1
    while (directory / (name + ".json")).exists():
        name = base + f"_retry{index}"
        index += 1
    subprocess.run([sys.executable, str(ROOT / "tools/record_command.py"),
                    "--log-dir", str(directory), "--name", name, "--", *map(str, argv)],
                   cwd=cwd, env=build_environment(), check=True)


def build_commands(*, windows, compiler, jobs=2, root=ROOT):
    """Return the audited platform recipe without executing it."""
    deps = Path(root) / ".deps"
    boost = deps / "boost_1_82_0"
    common = [f"-DCMAKE_CXX_COMPILER={compiler}",
              f"-DCMAKE_INSTALL_PREFIX={deps / 'install'}",
              "-DCMAKE_EXPORT_NO_PACKAGE_REGISTRY=ON"]
    bootstrap = ["cmd", "/c", "bootstrap.bat", "gcc"] if windows else ["bash", "bootstrap.sh", "--with-toolset=gcc"]
    boost_args = [boost / ("b2.exe" if windows else "b2"), f"-j{jobs}", "toolset=gcc", "address-model=64",
                  "variant=release", "link=static", "runtime-link=shared", "threading=multi", "--layout=system",
                  "--with-system", "--with-filesystem", "--with-atomic", "--with-serialization", "--with-program_options"]
    if not windows:
        boost_args.append("cxxflags=-fPIC")
    boost_args.append("stage")
    ompl_args = ["cmake", "-S", deps / "ompl-1.7.0", "-B", deps / "ompl-build", "-G", "Ninja", *common,
                 "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={deps / 'install'}",
                 f"-DBOOST_ROOT={boost}", f"-DBOOST_LIBRARYDIR={boost / 'stage/lib'}",
                 "-DBoost_NO_SYSTEM_PATHS=ON", "-DBoost_NO_BOOST_CMAKE=ON", "-DBoost_USE_STATIC_LIBS=ON",
                 "-DOMPL_BUILD_DEMOS=OFF", "-DOMPL_BUILD_TESTS=OFF", "-DOMPL_BUILD_PYBINDINGS=OFF",
                 "-DOMPL_REGISTRATION=OFF"]
    if not windows:
        ompl_args += ["-DCMAKE_DISABLE_FIND_PACKAGE_flann=ON", "-DCMAKE_DISABLE_FIND_PACKAGE_Triangle=ON",
                      "-DCMAKE_DISABLE_FIND_PACKAGE_spot=ON", "-DCMAKE_INSTALL_LIBDIR=lib"]
    return [
        ("boost_bootstrap", bootstrap, boost),
        ("boost_libraries", boost_args, boost),
        ("configure_eigen", ["cmake", "-S", deps / "eigen-3.4.0", "-B", deps / "eigen-build", "-G", "Ninja",
                             *common, "-DBUILD_TESTING=OFF"], Path(root)),
        ("install_eigen", ["cmake", "--install", deps / "eigen-build"], Path(root)),
        ("apply_rrtstar_admissible_patch", [sys.executable, Path(root) / "tools/apply_rrtstar_patch.py"], Path(root)),
        ("apply_aitstar_paper_formulas_patch", [sys.executable, Path(root) / "tools/apply_aitstar_patch.py"], Path(root)),
        ("configure_ompl", ompl_args, Path(root)),
        ("build_ompl", ["cmake", "--build", deps / "ompl-build", "--parallel", str(jobs)], Path(root)),
        ("install_ompl", ["cmake", "--install", deps / "ompl-build"], Path(root)),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--compiler")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "logs/build")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    windows = os.name == "nt"
    if not windows and not sys.platform.startswith("linux"):
        parser.error("only Windows and Linux have an audited recipe")
    compiler = args.compiler or ("C:/mingw64/bin/g++.exe" if windows else shutil.which("g++"))
    if not compiler:
        raise RuntimeError("BLOCKED: g++ missing")
    for dependency in ("boost_1_82_0", "eigen-3.4.0", "ompl-1.7.0"):
        if not (DEPS / dependency).is_dir():
            raise RuntimeError("BLOCKED: missing pinned source " + dependency)
    if windows:
        source = DEPS / "ompl-1.7.0/src/ompl/CMakeLists.txt"
        content = source.read_text()
        before, after = "target_link_libraries(ompl psapi ws2_32)", "target_link_libraries(ompl PRIVATE psapi ws2_32)"
        if before in content:
            source.write_text(content.replace(before, after))
        elif after not in content:
            raise RuntimeError("OMPL differs from audited patch base")
    for name, argv, cwd in build_commands(windows=windows, compiler=compiler, jobs=args.jobs):
        if name == "boost_bootstrap" and (DEPS / "boost_1_82_0" / ("b2.exe" if windows else "b2")).exists():
            continue
        run(name, argv, cwd, log_dir=args.log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
