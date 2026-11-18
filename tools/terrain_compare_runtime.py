"""Capture native dependency identity without claiming complete dlopen coverage.

Linux ldd observes the loader closure of the supplied trusted build/interpreter
roots. Selected Python native libraries are additionally content-hashed without
importing numerical packages or running ldd on arbitrary extension modules.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WINDOWS_RUNTIME_DIR = Path("C:/mingw64/bin")
WINDOWS_RUNTIME_NAMES = ("libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll")
NATIVE_PACKAGES = ("numpy", "numpy.libs", "torch", "gpytorch", "linear_operator",
                   "scipy", "scipy.libs", "psutil")


class NativeDependencyError(ValueError):
    pass


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _absolute(path):
    # Keep the interpreter's venv path for execution. resolve() would select the
    # base interpreter on Linux and silently lose its installed site-packages.
    return Path(os.path.abspath(path))


def parse_ldd(text):
    libraries, virtual = {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if "=> not found" in line:
            raise NativeDependencyError("unresolved native library: " + line)
        if line in ("statically linked", "not a dynamic executable"):
            continue
        if "=>" in line:
            name, target = (value.strip() for value in line.split("=>", 1))
            target = re.sub(r"\s+\(0x[0-9a-fA-F]+\)\s*$", "", target)
        else:
            target = re.sub(r"\s+\(0x[0-9a-fA-F]+\)\s*$", "", line)
            if target.startswith(("linux-vdso", "linux-gate")):
                virtual.append(target)
                continue
            name = Path(target).name
        path = Path(target)
        if not path.is_absolute() or not path.is_file():
            raise NativeDependencyError("invalid/missing resolved native library: " + line)
        resolved = str(path.resolve())
        if name in libraries and libraries[name] != resolved:
            raise NativeDependencyError("ambiguous native resolution: " + name)
        libraries[name] = resolved
    return {"libraries": dict(sorted(libraries.items())), "virtual_objects": sorted(virtual)}


def _python_libraries(interpreter):
    code = ("import json,site,sys;print(json.dumps({'sites':site.getsitepackages(),"
            "'base_prefix':sys.base_prefix}))")
    result = subprocess.run([str(interpreter), "-I", "-c", code], capture_output=True,
                            text=True, check=False, timeout=30)
    if result.returncode:
        raise NativeDependencyError("cannot inspect interpreter native packages: " + str(interpreter))
    try:
        observed = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise NativeDependencyError("invalid interpreter package probe") from error
    paths = set()
    for site in observed["sites"]:
        for package in NATIVE_PACKAGES:
            directory = Path(site) / package
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and (path.name.endswith((".so", ".pyd", ".dll")) or ".so." in path.name):
                    paths.add(str(path.resolve()))
    if os.name == "nt":
        for path in Path(observed["base_prefix"]).glob("python*.dll"):
            if path.is_file():
                paths.add(str(path.resolve()))
    return sorted(paths)


def capture_native_dependencies(paths):
    """Return {binary_hashes, linkage}; compare BOTH when rechecking resume.

    File contents may remain unchanged while LD_LIBRARY_PATH changes resolution.
    Recapturing the linkage mapping catches that substitution. A library changed
    in place is caught by binary_hashes. Requested paths and resolved targets are
    both recorded, so retargeting a venv symlink is detected even with equal bytes.
    Missing libraries fail closed on Linux.
    """
    roots = sorted({_absolute(path) for path in paths})
    for path in roots:
        if not path.is_file():
            raise NativeDependencyError("missing native root: " + str(path))
    hashes = {str(path): _hash(path) for path in roots}
    linux = sys.platform.startswith("linux")
    windows = sys.platform == "win32"
    if not linux and not windows:
        raise NativeDependencyError("native dependency capture supports Linux and Windows only")
    linkage = {
        "scope": ("linux_ldd_transitive_closure_of_roots_plus_selected_python_native_file_hashes_not_complete_dlopen_proof"
                  if linux else "windows_explicit_local_runtime_and_python_native_hashes_not_complete_OS_loader_closure"),
        "roots": {}, "python_extensions": {}, "known_windows_runtime": [],
    }
    for root in roots:
        resolved = root.resolve(strict=True)
        hashes[str(resolved)] = _hash(resolved)
        if linux:
            # ldd can expand $ORIGIN relative to a venv symlink instead of the
            # executable's actual directory. Inspect the real ELF, but keep the
            # lexical venv path below when probing Python's installed packages.
            result = subprocess.run(["ldd", str(resolved)], capture_output=True, text=True,
                                    check=False, timeout=30)
            if result.returncode:
                raise NativeDependencyError("ldd failed for " + str(resolved) +
                                            " (requested " + str(root) + "): " + result.stderr.strip())
            closure = parse_ldd(result.stdout)
            linkage["roots"][str(root)] = closure
            for path in closure["libraries"].values():
                hashes[path] = _hash(path)
        else:
            linkage["roots"][str(root)] = {"libraries": {}, "virtual_objects": [],
                                          "status": "EXPLICIT_HASHES_ONLY_NO_WINDOWS_LOADER_AUDIT"}
        linkage["roots"][str(root)].update(requested_root=str(root), resolved_target=str(resolved))
        if root.name.lower().startswith("python"):
            extension_paths = _python_libraries(root)
            linkage["python_extensions"][str(root)] = extension_paths
            for path in extension_paths:
                hashes[path] = _hash(path)
    if windows:
        candidates = [WINDOWS_RUNTIME_DIR / name for name in WINDOWS_RUNTIME_NAMES]
        for directory in (ROOT / ".deps/install/lib", ROOT / ".deps/install/bin"):
            if directory.exists():
                candidates.extend(directory.rglob("*.dll"))
        for path in sorted(set(candidates)):
            if path.is_file():
                resolved = str(path.resolve())
                hashes[resolved] = _hash(path)
                linkage["known_windows_runtime"].append(resolved)
    return {"binary_hashes": dict(sorted(hashes.items())), "linkage": linkage}
