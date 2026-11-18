"""Selectively stage a v4 checkpoint from an uploaded, read-only Output ZIP.

Only the index, its payload chain, referenced raw shards and original source
archive are copied. The frozen restore_checkpoint still verifies the chain.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile

PROTOCOL = "a3t_old_safe_sem_kaggle_v4"
INDEX_KIND = "TERRAIN_COMPARE_NATIVE_PAPER_SCHEMA13_CHECKPOINT_INDEX"
MARKER = ".native_paper_v4_zip_input.json"
GIB = 1024**3
MAX_OUTPUT_BYTES = 16*GIB
RESERVE_BYTES = 512*1024**2
INDEX_LIMIT = 64*1024**2
BLOCK = 1024**2


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def _sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(BLOCK),b""):result.update(chunk)
    return result.hexdigest()


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch("[0-9a-f]{64}",value):
        raise ValueError("invalid SHA-256")
    return value


def _name(value):
    if (not isinstance(value,str) or not value or "\\" in value or ":" in value or "\x00" in value or
            PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts or str(PurePosixPath(value))!=value):
        raise ValueError("unsafe selected ZIP path")
    for part in PurePosixPath(value).parts:
        if part.endswith(("."," ")) or part.split(".")[0].upper() in {"CON","PRN","AUX","NUL",*[f"COM{i}" for i in range(1,10)],*[f"LPT{i}" for i in range(1,10)]}:
            raise ValueError("unsafe selected ZIP component")
    return value


def _link(path):
    return path.is_symlink() or getattr(path,"is_junction",lambda:False)()


def _guard(path,base=None,*,regular=False):
    path=Path(path).absolute()
    if base is not None and (not path.is_relative_to(base) or not path.resolve().is_relative_to(base)):
        raise ValueError("staging path escapes owned root")
    for item in (path,*path.parents):
        if _link(item):raise ValueError("staging path contains a link")
    if regular:
        try:mode=path.lstat().st_mode
        except FileNotFoundError:return path
        if not stat.S_ISREG(mode):raise ValueError("staging file is not regular")
    return path


def _record(value,path):
    if not isinstance(value,dict) or value.get("path",path)!=path or type(value.get("size_bytes")) is not int or value["size_bytes"]<0:
        raise ValueError("invalid indexed file record")
    return dict(size_bytes=value["size_bytes"],sha256=_hash(value.get("sha256")))


def _member(members,name,size=None):
    _name(name)
    entries=members.get(name,[])
    if len(entries)!=1:raise ValueError("missing or duplicate selected ZIP member: "+name)
    info=entries[0];mode=info.external_attr>>16
    if info.is_dir() or stat.S_IFMT(mode) not in (0,stat.S_IFREG) or info.flag_bits&1:
        raise ValueError("selected ZIP member is linked, encrypted or not regular")
    if size is not None and info.file_size!=size:raise ValueError("selected ZIP member size mismatch")
    return info


def _checksum(data,name):
    try:parts=data.decode("ascii").strip().split()
    except UnicodeError as error:raise ValueError("invalid checksum sidecar") from error
    if len(parts)!=2 or parts[1].lstrip("*")!=name:raise ValueError("invalid checksum sidecar filename")
    return _hash(parts[0])


def _scan(archive,source_sha256,expected_index_sha256):
    members={}
    for item in archive.infolist():members.setdefault(item.filename,[]).append(item)
    candidates=[]
    for name in members:
        if not name.endswith("campaign/exports/checkpoint_index.json"):continue
        info=_member(members,name)
        if info.file_size>INDEX_LIMIT:raise ValueError("checkpoint index exceeds size limit")
        data=archive.read(info)
        try:value=json.loads(data)
        except (UnicodeError,ValueError) as error:raise ValueError("invalid checkpoint index JSON") from error
        if value.get("protocol_id")==PROTOCOL and value.get("source_input_sha256")==source_sha256:
            candidates.append((name,data,value))
    if len(candidates)!=1:raise ValueError("expected exactly one unambiguous matching checkpoint index")
    index_name,data,index=candidates[0]
    index_sha=hashlib.sha256(data).hexdigest()
    if expected_index_sha256 is not None and index_sha!=expected_index_sha256:raise ValueError("checkpoint index SHA-256 mismatch")
    if index.get("kind")!=INDEX_KIND or index.get("inventory_sha256")!=_digest(index.get("inventory")):
        raise ValueError("checkpoint inventory digest mismatch")
    root=PurePosixPath(index.get("root_path",""));workspace=PurePosixPath(index.get("workspace_path",""))
    if not root.is_absolute() or root.name!="repo" or workspace!=root.parent/"campaign":
        raise ValueError("checkpoint root/workspace identity is inconsistent")
    prefix=index_name[:-len("campaign/exports/checkpoint_index.json")]
    if prefix and PurePosixPath(prefix.rstrip("/")).name!=root.parent.name:
        raise ValueError("ZIP prefix differs from checkpoint root identity")
    selected={index_name:dict(size_bytes=len(data),sha256=index_sha)}
    def add(name,record):
        _member(members,name,record["size_bytes"])
        if name in selected:raise ValueError("duplicate indexed selection")
        selected[name]=record
    def checksum(name,expected):
        side=name+".sha256";info=_member(members,side)
        if info.file_size>512:raise ValueError("oversized checksum sidecar")
        payload=archive.read(info)
        if _checksum(payload,PurePosixPath(name).name)!=expected:raise ValueError("checksum sidecar mismatch")
        add(side,dict(size_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
    checksum(index_name,index_sha)
    payloads=index.get("payloads")
    if not isinstance(payloads,list) or not payloads:raise ValueError("checkpoint payload chain is empty")
    for payload in payloads:
        name=_name(payload.get("path"))
        if "/" in name:raise ValueError("payload must be directly in exports")
        record=_record(payload,name);full=prefix+"campaign/exports/"+name
        # The index binds payload hashes. Original v4 outputs do not require
        # separate payload checksum files; the index checksum is retained.
        add(full,record)
    inventory=index["inventory"]
    if not isinstance(inventory,dict):raise ValueError("invalid checkpoint inventory")
    for name,record in inventory.items():
        _name(name);checked=_record(record,name)
        if name.startswith("campaign/shards/") and name.endswith(".zip"):
            if len(PurePosixPath(name).parts)!=3:raise ValueError("invalid external shard location")
            add(prefix+name,checked)
    source_record=_record(inventory.get("source_input.zip"),"source_input.zip")
    if source_record["sha256"]!=source_sha256:raise ValueError("source archive inventory hash mismatch")
    add(prefix+"source_input.zip",source_record)
    folded=[name.casefold() for name in selected]
    if len(folded)!=len(set(folded)):raise ValueError("selected ZIP paths collide by case")
    return index_name,index_sha,selected


def _tree_bytes(root):
    total=0
    if root.exists():
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[name for name in dirs if not _link(Path(folder)/name)]
            for name in files:
                path=Path(folder)/name
                if not _link(path) and path.is_file():total+=path.stat().st_size
    return total


def _preflight(destination,needed):
    probe=destination
    while not probe.exists():probe=probe.parent
    kaggle=Path("/kaggle/working").absolute()
    scope=kaggle if destination.is_relative_to(kaggle) else destination
    if shutil.disk_usage(probe).free<needed+RESERVE_BYTES:
        raise RuntimeError("BLOCKED_DISK: selective ZIP import requires remaining selected bytes plus 512 MiB")
    if _tree_bytes(scope)+needed>MAX_OUTPUT_BYTES:
        raise RuntimeError("BLOCKED_DISK: selective ZIP import would exceed 16 GiB cap")


@contextmanager
def _output(path,mode):
    _guard(path,regular=True)
    flags=os.O_WRONLY|os.O_CREAT|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0)
    flags|=os.O_APPEND if mode=="ab" else os.O_EXCL
    descriptor=os.open(path,flags,0o600)
    with os.fdopen(descriptor,mode) as stream:yield stream


def _save_marker(destination,value):
    path=destination/MARKER;temporary=destination/(MARKER+".pending")
    _guard(path,destination,regular=True);_guard(temporary,destination,regular=True)
    data=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    # A complete marker is deterministic. Never replace an unrelated pending file.
    if temporary.exists():
        if temporary.read_bytes()!=data:raise ValueError("unrelated staging marker pending bytes")
    else:
        with _output(temporary,"xb") as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def _files(destination):
    result=set()
    if destination.exists():
        for item in destination.rglob("*"):
            _guard(item,destination)
            if item.is_file():result.add(item.relative_to(destination).as_posix())
            elif not item.is_dir():raise ValueError("nonregular staging entry")
    return result


def _restore(archive,info,path,record,destination):
    _guard(path,destination,regular=True);pending=path.with_name(path.name+".zip-partial")
    _guard(pending,destination,regular=True)
    if path.exists():
        if path.stat().st_size!=record["size_bytes"] or _sha(path)!=record["sha256"]:
            raise ValueError("existing selected staging file differs")
        if pending.exists():raise ValueError("both complete and pending staging files exist")
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with archive.open(info) as source:
        if pending.exists():
            if pending.stat().st_size>record["size_bytes"]:raise ValueError("staging partial exceeds selected size")
            with pending.open("rb") as prefix:
                for chunk in iter(lambda:prefix.read(BLOCK),b""):
                    if source.read(len(chunk))!=chunk:raise ValueError("staging partial prefix differs")
            mode="ab"
        else:mode="xb"
        with _output(pending,mode) as output:
            shutil.copyfileobj(source,output,BLOCK);output.flush();os.fsync(output.fileno())
    if pending.stat().st_size!=record["size_bytes"] or _sha(pending)!=record["sha256"]:
        raise ValueError("selected staging file hash/size mismatch")
    os.replace(pending,path)


def prepare_zip_input(zip_path,destination,source_sha256,expected_index_sha256=None):
    """Return the staged checkpoint_index.json without extracting raw trial trees."""
    source_sha256=_hash(source_sha256)
    if expected_index_sha256 is not None:_hash(expected_index_sha256)
    zip_path=_guard(Path(zip_path).absolute(),regular=True)
    destination=_guard(Path(destination).absolute())
    if destination==Path(destination.anchor) or zip_path.is_relative_to(destination):
        raise ValueError("staging root must be separate from the uploaded ZIP")
    source_stat=zip_path.stat();outer_sha=_sha(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        index_name,index_sha,selected=_scan(archive,source_sha256,expected_index_sha256)
        marker=dict(kind="NATIVE_PAPER_V4_SELECTIVE_ZIP_INPUT_V1",zip_sha256=outer_sha,
                    source_input_sha256=source_sha256,index_path=index_name,index_sha256=index_sha,
                    inventory=selected,inventory_sha256=_digest(selected))
        allowed={MARKER,MARKER+".pending",*selected,*[name+".zip-partial" for name in selected]}
        existing=_files(destination)
        if existing-allowed:raise ValueError("unrelated files in staging destination")
        marker_path=destination/MARKER
        if marker_path.exists():
            if json.loads(marker_path.read_bytes())!=marker:raise ValueError("staging ownership/source identity differs")
        elif existing and existing!={MARKER+".pending"}:
            raise ValueError("existing staging files have no ownership marker")
        remaining=0
        for name,record in selected.items():
            full=destination/name;pending=destination/(name+".zip-partial")
            if full.exists():
                if full.stat().st_size!=record["size_bytes"] or _sha(full)!=record["sha256"]:
                    raise ValueError("existing selected staging file differs")
            else:remaining+=max(0,record["size_bytes"]-(pending.stat().st_size if pending.exists() else 0))
        _preflight(destination,remaining)
        destination.mkdir(parents=True,exist_ok=True)
        _save_marker(destination,marker)
        for name,record in selected.items():
            _restore(archive,archive.getinfo(name),destination/name,record,destination)
    after=zip_path.stat()
    if (source_stat.st_size,source_stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError("uploaded ZIP changed during staging")
    return destination/index_name


def cleanup_zip_input(destination):
    """Verify and remove only this helper's complete, unchanged staged files."""
    destination=_guard(Path(destination).absolute())
    marker_path=_guard(destination/MARKER,destination,regular=True)
    value=json.loads(marker_path.read_bytes())
    inventory=value["inventory"]
    if value.get("kind")!="NATIVE_PAPER_V4_SELECTIVE_ZIP_INPUT_V1" or value.get("inventory_sha256")!=_digest(inventory):
        raise ValueError("invalid staging cleanup ownership marker")
    _hash(value["zip_sha256"]);_hash(value["source_input_sha256"]);_hash(value["index_sha256"])
    expected={MARKER,*inventory}
    if _files(destination)!=expected:raise ValueError("incomplete or unrelated files prevent staging cleanup")
    for name,record in inventory.items():
        path=_guard(destination/_name(name),destination,regular=True);_record(record,name)
        if path.stat().st_size!=record["size_bytes"] or _sha(path)!=record["sha256"]:
            raise ValueError("staged bytes changed; refusing cleanup")
    if value["index_path"] not in inventory or inventory[value["index_path"]]["sha256"]!=value["index_sha256"]:
        raise ValueError("staging index ownership mismatch")
    receipt=dict(kind=value["kind"],zip_sha256=value["zip_sha256"],index_sha256=value["index_sha256"],
                 files=len(inventory),bytes=sum(record["size_bytes"] for record in inventory.values()),verified=True)
    for name in inventory:_guard(destination/name,destination,regular=True).unlink()
    marker_path.unlink()
    for path in sorted(destination.rglob("*"),key=lambda p:len(p.parts),reverse=True):
        _guard(path,destination)
        if path.is_dir():path.rmdir()
    destination.rmdir()
    return receipt
