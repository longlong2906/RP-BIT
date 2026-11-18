"""Prepare a Linux Cloud build and correctness gate; never launch the full campaign."""
import argparse
import datetime as dt
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from build_dependencies import run as recorded
from experiment_common import cloud_code_hash, cloud_environment, cloud_ompl_library, dump, file_hash, read_json, worker_env

PLANNERS = ("bitstar_a3", "abitstar", "aitstar", "lazyprmstar_hauser2015", "sem_rrtstar_common_v1")
PINNED_PACKAGES = {"numpy": "1.26.4", "matplotlib": "3.10.7", "PyYAML": "6.0.1", "psutil": "5.9.0"}


def require_prerequisites():
    if not sys.platform.startswith("linux"):
        raise RuntimeError("BLOCKED: setup_cloud_benchmark.py requires Linux; do not run Cloud collection on Windows")
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("BLOCKED: this environment must use Python 3.12")
    missing = [name for name in ("g++", "cmake", "ninja", "bash", "ldd", "git") if not shutil.which(name)]
    if missing:
        raise RuntimeError("BLOCKED: missing environment tools: " + ", ".join(missing))
    for name, expected in PINNED_PACKAGES.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"BLOCKED: {name}=={expected} required, observed {actual}")


def observed_dependency_versions(root=ROOT):
    """Read installed/source headers rather than reusing the dependency labels."""
    root = Path(root)
    ompl_config = root / ".deps/install/include/ompl-1.7/ompl/config.h"
    boost_header = root / ".deps/boost_1_82_0/boost/version.hpp"
    eigen_header = root / ".deps/install/include/eigen3/Eigen/src/Core/util/Macros.h"
    ompl_text, boost_text, eigen_text = [path.read_text(encoding="utf-8") for path in (ompl_config, boost_header, eigen_header)]
    ompl = re.search(r'^#define\s+OMPL_VERSION\s+"([^"]+)"', ompl_text, re.MULTILINE)
    boost = re.search(r'^#define\s+BOOST_VERSION\s+(\d+)', boost_text, re.MULTILINE)
    eigen = [re.search(r'^#define\s+EIGEN_' + name + r'_VERSION\s+(\d+)', eigen_text, re.MULTILINE)
             for name in ("WORLD", "MAJOR", "MINOR")]
    if not ompl or not boost or not all(eigen):
        raise ValueError("cannot identify installed dependency versions")
    b = int(boost.group(1))
    versions = {"ompl": ompl.group(1), "boost": f"{b // 100000}.{b // 100 % 1000}.{b % 100}",
                "eigen": ".".join(part.group(1) for part in eigen)}
    if versions != {"ompl": "1.7.0", "boost": "1.82.0", "eigen": "3.4.0"}:
        raise ValueError("installed dependency version mismatch: " + str(versions))
    for feature in ("FLANN", "SPOT"):
        if not re.search(r'^#define\s+OMPL_HAVE_' + feature + r'\s+0\s*$', ompl_text, re.MULTILINE):
            raise ValueError("unexpected optional OMPL feature: " + feature)
    return {"versions": versions, "header_sha256": {path.relative_to(root).as_posix(): file_hash(path)
                                                     for path in (ompl_config, boost_header, eigen_header)}}


def project_configure(build, compiler):
    boost = ROOT / ".deps/boost_1_82_0"
    # The worker and local .deps layout are retained in the cached Cloud snapshot.
    return ["cmake", "-S", ROOT, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CXX_COMPILER={compiler}", f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_PREFIX_PATH={ROOT / '.deps/install'}", f"-DBOOST_ROOT={boost}",
            f"-DBOOST_LIBRARYDIR={boost / 'stage/lib'}", "-DBoost_NO_SYSTEM_PATHS=ON",
            "-DBoost_NO_BOOST_CMAKE=ON", "-DBoost_USE_STATIC_LIBS=ON",
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"]


def smoke(worker, build):
    """Five short real planners plus independent final/snapshot verification."""
    from terrain_io import write_map
    from validate_results import validate_trial
    fixtures = build / "cloud_smoke"
    fixtures.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="run_", dir=fixtures))
    map_path = directory / "tilted.t25d"
    write_map(map_path, [[.2 * i + .1 * j for i in range(11)] for j in range(11)],
              radius=.1, margin=0, slope=70, metadata={"fixture": True, "purpose": "cloud_five_planner_smoke"})
    checks = []
    for planner in PLANNERS:
        target = directory / planner
        target.mkdir()
        request = {"map_path": str(map_path), "planner_id": planner, "planner_seed": 771,
                   "start": [1., 5.], "goal": [9., 5.], "planning_budget_s": 3., "samples_per_batch": 40,
                   "rewire_factor": 1.1, "sample_attempts": 100, "progress_interval_s": .05,
                   "work_iterations": 500 if planner in ("abitstar", "aitstar", "lazyprmstar_hauser2015") else 2000,
                   "cloud_phase_handshake": True}
        if planner == "sem_rrtstar_common_v1":
            request.update(sem_range_m=1., sem_neighbor_radius_m=3.)
        dump(target / "request.json", request)
        command = [str(worker), str(target / "request.json"), str(target)]
        began = time.perf_counter()
        utc = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                    env=worker_env(), timeout=30)
            exit_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        except (OSError, subprocess.TimeoutExpired) as error:
            exit_code, stdout, stderr = 124, "", str(error)
        (target / "stdout.log").write_text(stdout, encoding="utf-8")
        (target / "stderr.log").write_text(stderr, encoding="utf-8")
        dump(target / "command.json", {"argv": command, "utc": utc, "exit_code": exit_code,
                                      "wall_s": time.perf_counter() - began})
        if exit_code != 0:
            raise RuntimeError(f"smoke worker failed: {planner}; logs retained at {target}")
        trial = read_json(target / "worker.json")
        trial.update(map_hash=file_hash(map_path), trial_id=planner)
        dump(target / "trial.json", trial)
        validated = validate_trial(target)
        if not validated.get("success"):
            raise RuntimeError(f"smoke independent correctness failed: {planner}; logs retained at {target}")
        checks.append({"planner_id": planner, "exact_goal_reached": True, "independent_valid": True,
                       "evidence": str(target), "effective_policy": validated["effective_policy"]})
    dump(directory / "summary.json", {"status": "PASS", "tests_only": True, "planners": checks})
    return {"status": "PASS", "tests_only": True, "evidence_directory": str(directory), "planners": checks}


def verify_pinned_archives():
    lock = read_json(ROOT / "configs/dependencies.lock.json")
    for name, entry in lock["archives"].items():
        archive = ROOT / ".deps" / (name + ".tar.gz")
        if not archive.is_file() or file_hash(archive) != entry["sha256"]:
            raise ValueError("pinned dependency archive absent or changed: " + name)
    return {name: entry["sha256"] for name, entry in lock["archives"].items()}


def verify_gate(build):
    gate = read_json(build / "cloud_setup.json")
    worker = build / "terrain_benchmark"
    actual = {"source_sha256": cloud_code_hash(), "worker_sha256": file_hash(worker),
              "ompl_library_sha256": file_hash(cloud_ompl_library(worker))}
    if (gate.get("status") != "PASS" or not gate.get("ctest_passed") or not gate.get("smoke_passed")
            or gate.get("smoke_planner_ids") != list(PLANNERS)
            or gate.get("dependency_lock_sha256") != file_hash(ROOT / "configs/dependencies.lock.json")
            or any(gate.get(key) != value for key, value in actual.items())):
        raise ValueError("Cloud setup gate absent/stale: source/worker/library changed")
    verify_pinned_archives()
    observed_dependency_versions()
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build-cloud")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--verify-only", action="store_true", help="verify prepared snapshot without rebuilding")
    parser.add_argument("--smoke-only", action="store_true", help="run only the five-planner correctness smoke")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    require_prerequisites()
    build = args.build_dir.resolve()
    if ROOT not in build.parents:
        parser.error("--build-dir must stay inside this checkout")
    if args.verify_only:
        gate = verify_gate(build)
        print(json.dumps({"status": "PASS", "worker_sha256": gate["worker_sha256"], "source_sha256": gate["source_sha256"]}))
        return 0
    build.mkdir(parents=True, exist_ok=True)
    logs = build / "setup_logs"
    if args.smoke_only:
        print(json.dumps(smoke(build / "terrain_benchmark", build)))
        return 0
    # An already prepared cached environment is verified, never silently rebuilt.
    if (build / "cloud_setup.json").exists():
        verify_gate(build)
        print("PASS: unchanged prepared Cloud build reused")
        return 0
    try:
        recorded("fetch_pinned_dependencies", [sys.executable, ROOT / "tools/fetch_dependencies.py"], log_dir=logs)
        verify_pinned_archives()
        recorded("build_pinned_dependencies", [sys.executable, ROOT / "tools/build_dependencies.py",
                  "--jobs", args.jobs, "--log-dir", logs], log_dir=logs)
        recorded("configure_release", project_configure(build, shutil.which("g++")), log_dir=logs)
        recorded("build_release", ["cmake", "--build", build, "--parallel", args.jobs], log_dir=logs)
        recorded("ctest_correctness", ["ctest", "--test-dir", build, "--output-on-failure"], log_dir=logs)
        smoke_result = smoke(build / "terrain_benchmark", build)
        versions = observed_dependency_versions()
        worker = build / "terrain_benchmark"
        environment = cloud_environment(worker)
        gate = {"schema_version": 1, "status": "PASS", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "source_sha256": cloud_code_hash(), "code_sha256": cloud_code_hash(),
                "worker_sha256": file_hash(worker), "ompl_library_sha256": file_hash(cloud_ompl_library(worker)),
                "dependency_lock_sha256": file_hash(ROOT / "configs/dependencies.lock.json"),
                "dependency_versions": versions, "dependency_archive_sha256": verify_pinned_archives(), "ctest_passed": True, "smoke_passed": True,
                "smoke": smoke_result, "smoke_planner_ids": list(PLANNERS), "environment": environment, "command_logs": str(logs),
                "benchmark_trials_launched": 0, "correctness_smoke_trials": 5}
        dump(build / "cloud_setup.json", gate)
        print(json.dumps({"status": "PASS", "gate": str(build / "cloud_setup.json"), "worker_sha256": gate["worker_sha256"]}))
        return 0
    except Exception as error:
        dump(build / "cloud_setup_failure.json", {"status": "BLOCKED" if "BLOCKED" in str(error) else "FAILED",
             "utc": dt.datetime.now(dt.timezone.utc).isoformat(), "error": str(error), "command_logs": str(logs)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
