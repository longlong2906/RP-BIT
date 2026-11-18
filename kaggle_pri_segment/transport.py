"""Standard-library archive checks, also embedded verbatim in the notebook."""
from pathlib import Path, PurePosixPath, PureWindowsPath
import base64
import hashlib
import json
import os
import shutil
import stat
import tarfile
import zipfile


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    value = PurePosixPath(name)
    if (not name or '\\' in name or '\x00' in name or value.is_absolute()
            or '..' in value.parts or PureWindowsPath(name).drive
            or ':' in name or not value.parts):
        raise ValueError('Unsafe archive path: ' + repr(name))
    return value


def verify_catalog(repo, manifest_path):
    repo = Path(repo).resolve()
    catalog = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    files = catalog['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty source catalog')
    for name, expected in files.items():
        path = repo.joinpath(*safe_name(name).parts)
        if not path.resolve().is_relative_to(repo) or not path.is_file() or sha256(path) != expected:
            raise ValueError('Source catalog mismatch: ' + name)
    return catalog


def decode_base64(path, destination, expected_archive_sha256):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with Path(path).open('rb') as source, destination.open('xb') as target:
        while chunk := source.read(4 * 1024**2):
            target.write(base64.b64decode(chunk, validate=True))
    if sha256(destination) != expected_archive_sha256:
        raise ValueError('Decoded archive SHA-256 differs')
    return destination


def safe_extract_zip(archive, destination, expected_sha256):
    archive, destination = Path(archive), Path(destination)
    if sha256(archive) != expected_sha256:
        raise ValueError('ZIP SHA-256 differs')
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as bundle:
        seen = set()
        for member in bundle.infolist():
            # ZipInfo normalizes separators on Windows. Validate the raw
            # stored name as well, so host and Kaggle reject identical inputs.
            safe_name(member.orig_filename)
            name = safe_name(member.filename).as_posix()
            if name in seen or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Duplicate/link ZIP member: ' + name)
            seen.add(name)
        destination.mkdir(parents=True)
        bundle.extractall(destination)
    return destination


def safe_extract_tar(archive, destination, expected_sha256, allow_runtime_links=False,
                     allowed_prefixes=None, original_runtime_root=None):
    archive, destination = Path(archive), Path(destination).resolve()
    if sha256(archive) != expected_sha256:
        raise ValueError('TAR SHA-256 differs')
    prefixes = [destination.joinpath(*safe_name(p).parts) for p in (allowed_prefixes or ())]
    def allowed(path):
        return path.is_relative_to(destination) and (not prefixes or any(path.is_relative_to(p) for p in prefixes))
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as bundle:
        seen, members = set(), []
        for original in bundle.getmembers():
            name = safe_name(original.name).as_posix()
            target = destination.joinpath(*PurePosixPath(name).parts)
            if name in seen or not allowed(target.resolve()):
                raise ValueError('Duplicate/outside TAR member: ' + name)
            seen.add(name)
            member = original
            if not (member.isfile() or member.isdir() or
                    (allow_runtime_links and (member.issym() or member.islnk()))):
                raise ValueError('Forbidden TAR member type: ' + name)
            if member.issym() or member.islnk():
                link = Path(member.linkname)
                if link.is_absolute():
                    prior = Path(original_runtime_root or destination).resolve()
                    if not link.is_relative_to(prior):
                        raise ValueError('Absolute runtime link escapes its repository')
                    resolved = destination / link.relative_to(prior)
                else:
                    resolved = (target.parent if member.issym() else destination) / link
                resolved = resolved.resolve()
                if not allowed(resolved):
                    raise ValueError('Runtime link escapes allowed runtime roots')
                member = member.replace(linkname=os.path.relpath(resolved, target.parent) if member.issym()
                                        else str(resolved.relative_to(destination)))
            if target.exists() and not (member.isdir() and target.is_dir()):
                raise FileExistsError('Refusing to overwrite extracted content: ' + str(target))
            members.append(member)
        # Check every resolved path again while extracting: earlier links may
        # have changed the interpretation of later member ancestors.
        def checked_filter(member, directory):
            target = destination.joinpath(*safe_name(member.name).parts).resolve()
            if not allowed(target):
                raise ValueError('TAR path follows a link outside its allowed root')
            return tarfile.data_filter(member, directory)
        bundle.extractall(destination, members=members, filter=checked_filter)
    return destination
