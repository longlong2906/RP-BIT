"""Bounded-memory, hash-verified Base64 transport for an existing resume ZIP.

Each raw chunk is encoded independently. Uploaded parts remain read-only;
restoration concatenates decoded bytes and never extracts the transported ZIP.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import zipfile

KIND = "NATIVE_PAPER_V4_BASE64_CHUNK_INPUT_V1"
PROTOCOL = "a3t_old_safe_sem_kaggle_v4"
MANIFEST = "native_paper_v4_resume_chunks.json"
BUILD_OWNER = ".native_paper_v4_chunk_build.json"
GIB = 1024**3
RAW_CHUNK_BYTES = 128*1024**2
MAX_OUTPUT_BYTES = 16*GIB
RESERVE_BYTES = 512*1024**2
ENCODE_BLOCK = 3*1024**2
DECODE_BLOCK = 4*1024**2
MANIFEST_LIMIT = 2*1024**2
MAX_PARTS = 4096


def _sha(path):
    result=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024**2),b""):result.update(chunk)
    return result.hexdigest()


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch(r"[0-9a-f]{64}",value):raise ValueError("invalid SHA-256")
    return value


def _integer(value,minimum=0):
    if type(value) is not int or value<minimum:raise ValueError("invalid byte count or sequence")
    return value


def _name(value):
    if not isinstance(value,str) or not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",value) or value.endswith(".") or value in (".",".."):
        raise ValueError("unsafe chunk filename")
    if value.split(".")[0].upper() in {"CON","PRN","AUX","NUL",*[f"COM{i}" for i in range(1,10)],*[f"LPT{i}" for i in range(1,10)]}:
        raise ValueError("unsafe chunk filename")
    return value


def _link(path):return path.is_symlink() or getattr(path,"is_junction",lambda:False)()


def _guard(path,*,regular=False):
    path=Path(path).absolute()
    if ".." in path.parts:raise ValueError("chunk transport path contains traversal")
    for current in (path,*path.parents):
        if _link(current):raise ValueError("chunk transport path contains a link")
    if regular:
        try:mode=path.lstat().st_mode
        except FileNotFoundError:return path
        if not stat.S_ISREG(mode):raise ValueError("chunk transport file is not regular")
    return path


def _open_output(path,append=False):
    _guard(path,regular=True)
    flags=os.O_WRONLY|os.O_CREAT|getattr(os,"O_NOFOLLOW",0)|getattr(os,"O_BINARY",0)
    flags|=os.O_APPEND if append else os.O_EXCL
    descriptor=os.open(path,flags,0o600)
    return os.fdopen(descriptor,"ab" if append else "wb")


class _PrefixWriter:
    """Compare a retained prefix before appending; never replace unknown bytes."""
    def __init__(self,path,expected_bytes):
        self.path=_guard(path,regular=True)
        self.pending=_guard(path.with_name(path.name+".chunk-partial"),regular=True)
        self.complete=self.path.exists()
        if self.complete and self.pending.exists():raise ValueError("both complete and partial chunk outputs exist")
        retained=self.path if self.complete else self.pending
        self.remaining=retained.stat().st_size if retained.exists() else 0
        if self.remaining>expected_bytes:raise ValueError("retained output exceeds declared size")
        self.prefix=retained.open("rb") if retained.exists() else None
        self.output=None;self.written=0;self.expected=expected_bytes
        self.path.parent.mkdir(parents=True,exist_ok=True)

    def __enter__(self):return self

    def write(self,data):
        if self.written+len(data)>self.expected:raise ValueError("decoded/encoded bytes exceed declared size")
        count=min(self.remaining,len(data))
        if count:
            if self.prefix.read(count)!=data[:count]:raise ValueError("retained output prefix differs")
            self.remaining-=count
        tail=data[count:]
        if tail:
            if self.complete:raise ValueError("existing final output differs")
            if self.output is None:self.output=_open_output(self.pending,append=self.pending.exists())
            self.output.write(tail)
        self.written+=len(data)

    def finish(self,expected_sha256):
        if self.remaining or self.written!=self.expected:raise ValueError("retained output length mismatch")
        if self.output is not None:
            self.output.flush();os.fsync(self.output.fileno());self.output.close();self.output=None
        if self.prefix is not None:self.prefix.close();self.prefix=None
        retained=self.path if self.complete else self.pending
        if not retained.exists() and self.expected==0:
            with _open_output(retained):pass
        if _sha(retained)!=expected_sha256:raise ValueError("completed output SHA-256 mismatch")
        if not self.complete:os.replace(self.pending,self.path)

    def __exit__(self,*args):
        if self.prefix is not None:self.prefix.close()
        if self.output is not None:self.output.close()


def _json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()


def _commit_json(path,value):
    payload=_json_bytes(value)
    with _PrefixWriter(path,len(payload)) as output:
        output.write(payload);output.finish(hashlib.sha256(payload).hexdigest())


def _tree_bytes(root):
    total=0
    if root.exists():
        for folder,dirs,files in os.walk(root,followlinks=False):
            dirs[:]=[name for name in dirs if not _link(Path(folder)/name)]
            for name in files:
                path=Path(folder)/name
                if not _link(path) and path.is_file():total+=path.stat().st_size
    return total


def _preflight(path,needed):
    probe=path
    while not probe.exists():probe=probe.parent
    kaggle=Path("/kaggle/working").absolute()
    scope=kaggle if path.is_relative_to(kaggle) else path if path.is_dir() else path.parent
    if shutil.disk_usage(probe).free<needed+RESERVE_BYTES:
        raise RuntimeError("BLOCKED_DISK: chunk transport requires remaining bytes plus 512 MiB")
    if _tree_bytes(scope)+needed>MAX_OUTPUT_BYTES:
        raise RuntimeError("BLOCKED_DISK: chunk transport would exceed 16 GiB cap")


def _part_name(sequence,count):return f"native_paper_v4_resume_part{sequence+1:03d}.b64"


def _source_identity(path):
    found=[]
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith("campaign/exports/checkpoint_index.json"):continue
            if info.file_size>64*1024**2:raise ValueError("transported checkpoint index is too large")
            payload=archive.read(info);value=json.loads(payload)
            if value.get("protocol_id")==PROTOCOL:
                found.append((_hash(value.get("source_input_sha256")),hashlib.sha256(payload).hexdigest()))
    if len(found)!=1:raise ValueError("transport ZIP must contain exactly one v4 checkpoint index")
    return found[0]


def _existing_files(root):
    result=set()
    if root.exists():
        for item in root.rglob("*"):
            _guard(item)
            if not item.is_file():raise ValueError("chunk build directory contains a non-file entry")
            result.add(item.name)
    return result


def build_chunk_input(zip_path,outputdir,raw_chunk_bytes=RAW_CHUNK_BYTES):
    """Encode an unchanged resume ZIP into independent .b64 files and a manifest."""
    zip_path=_guard(zip_path,regular=True);outputdir=_guard(outputdir)
    raw_chunk_bytes=_integer(raw_chunk_bytes,1)
    if zip_path.is_relative_to(outputdir):raise ValueError("chunk output must be separate from source ZIP")
    size=zip_path.stat().st_size
    if not size or size>MAX_OUTPUT_BYTES:raise ValueError("transport source size is outside policy")
    count=(size+raw_chunk_bytes-1)//raw_chunk_bytes
    if count>MAX_PARTS:raise ValueError("too many transport parts")
    sizes=[min(raw_chunk_bytes,size-i*raw_chunk_bytes) for i in range(count)]
    encoded_sizes=[4*((value+2)//3) for value in sizes]
    source_sha=_sha(zip_path);source_input,index_sha=_source_identity(zip_path)
    source_name=_name(zip_path.name)
    owner=dict(kind=KIND,protocol_id=PROTOCOL,source_archive=dict(name=source_name,sha256=source_sha,size_bytes=size),
               source_input_sha256=source_input,checkpoint_index_sha256=index_sha,
               raw_chunk_bytes=raw_chunk_bytes,part_count=count)
    names=[_part_name(i,count) for i in range(count)]
    allowed={BUILD_OWNER,MANIFEST,*names,*[name+".chunk-partial" for name in names],
             BUILD_OWNER+".chunk-partial",MANIFEST+".chunk-partial"}
    existing=_existing_files(outputdir)
    if existing-allowed:raise ValueError("unrelated files in chunk build output")
    owner_path=outputdir/BUILD_OWNER
    if owner_path.exists():
        if json.loads(owner_path.read_bytes())!=owner:raise ValueError("chunk build ownership/source differs")
    elif existing and existing!={BUILD_OWNER+".chunk-partial"}:
        raise ValueError("existing chunk files have no ownership marker")
    needed=MANIFEST_LIMIT
    for name,encoded in zip(names,encoded_sizes):
        path=outputdir/name;pending=path.with_name(path.name+".chunk-partial")
        retained=path if path.exists() else pending
        needed+=max(0,encoded-(retained.stat().st_size if retained.exists() else 0))
    _preflight(outputdir,needed)
    outputdir.mkdir(parents=True,exist_ok=True);_commit_json(owner_path,owner)
    parts=[];whole=hashlib.sha256()
    with zip_path.open("rb") as source:
        for sequence,(name,raw_size,encoded_size) in enumerate(zip(names,sizes,encoded_sizes)):
            decoded_hash=hashlib.sha256();encoded_hash=hashlib.sha256();remaining=raw_size
            with _PrefixWriter(outputdir/name,encoded_size) as output:
                while remaining:
                    data=source.read(min(ENCODE_BLOCK,remaining))
                    if not data:raise ValueError("source ZIP shortened during encoding")
                    remaining-=len(data);whole.update(data);decoded_hash.update(data)
                    encoded=base64.b64encode(data);encoded_hash.update(encoded);output.write(encoded)
                output.finish(encoded_hash.hexdigest())
            parts.append(dict(sequence=sequence,path=name,encoded_sha256=encoded_hash.hexdigest(),
                              decoded_sha256=decoded_hash.hexdigest(),encoded_bytes=encoded_size,decoded_bytes=raw_size))
        if source.read(1) or whole.hexdigest()!=source_sha:raise ValueError("source ZIP changed during encoding")
    manifest=dict(owner,parts=parts,encoded_total_bytes=sum(encoded_sizes))
    if len(_json_bytes(manifest))>MANIFEST_LIMIT:raise ValueError("chunk manifest exceeds size limit")
    manifest_path=outputdir/MANIFEST;_commit_json(manifest_path,manifest)
    # The marker is local build state and is not one of the uploaded files.
    return manifest_path


def _manifest(path,expected):
    path=_guard(path,regular=True)
    if path.stat().st_size>MANIFEST_LIMIT:raise ValueError("chunk manifest exceeds size limit")
    actual=_sha(path)
    if expected is not None and actual!=_hash(expected):raise ValueError("chunk manifest SHA-256 mismatch")
    value=json.loads(path.read_bytes())
    if value.get("kind")!=KIND or value.get("protocol_id")!=PROTOCOL:raise ValueError("wrong chunk manifest protocol")
    source=value.get("source_archive",{});_name(source.get("name"));_hash(source.get("sha256"));size=_integer(source.get("size_bytes"),1)
    raw=_integer(value.get("raw_chunk_bytes"),1);count=_integer(value.get("part_count"),1)
    _hash(value.get("source_input_sha256"))
    if value.get("checkpoint_index_sha256") is not None:_hash(value["checkpoint_index_sha256"])
    if size>MAX_OUTPUT_BYTES or count>MAX_PARTS or count!=(size+raw-1)//raw:raise ValueError("invalid chunk cardinality")
    parts=value.get("parts")
    if not isinstance(parts,list) or len(parts)!=count:raise ValueError("missing or extra chunk records")
    names=set();encoded_total=0
    for sequence,part in enumerate(parts):
        if not isinstance(part,dict) or type(part.get("sequence")) is not int or part["sequence"]!=sequence:
            raise ValueError("chunk sequence was reordered or is incomplete")
        name=_name(part.get("path"))
        if name!=_part_name(sequence,count) or name.casefold() in names:raise ValueError("chunk filename/order mismatch")
        names.add(name.casefold())
        decoded=min(raw,size-sequence*raw);encoded=4*((decoded+2)//3)
        if _integer(part.get("decoded_bytes"),1)!=decoded or _integer(part.get("encoded_bytes"),1)!=encoded:
            raise ValueError("chunk size/cardinality mismatch")
        _hash(part.get("decoded_sha256"));_hash(part.get("encoded_sha256"));encoded_total+=encoded
        member=_guard(path.parent/name,regular=True)
        if not member.is_file() or member.stat().st_size!=encoded:raise ValueError("missing chunk or encoded size mismatch")
    if _integer(value.get("encoded_total_bytes"),1)!=encoded_total:raise ValueError("encoded total size mismatch")
    return value,actual


def _decoded_parts(directory,parts):
    for part in parts:
        encoded_hash=hashlib.sha256();decoded_hash=hashlib.sha256();encoded_size=0;decoded_size=0;padded=False
        with (directory/part["path"]).open("rb") as source:
            while data:=source.read(DECODE_BLOCK):
                if padded:raise ValueError("data after Base64 padding within a chunk")
                encoded_hash.update(data);encoded_size+=len(data)
                try:decoded=base64.b64decode(data,validate=True)
                except ValueError as error:raise ValueError("invalid strict Base64 chunk") from error
                if base64.b64encode(decoded)!=data:raise ValueError("noncanonical Base64 chunk")
                padded=b"=" in data;decoded_hash.update(decoded);decoded_size+=len(decoded)
                if decoded_size>part["decoded_bytes"]:raise ValueError("decoded chunk exceeds declared size")
                yield decoded
        if (encoded_size!=part["encoded_bytes"] or decoded_size!=part["decoded_bytes"] or
                encoded_hash.hexdigest()!=part["encoded_sha256"] or decoded_hash.hexdigest()!=part["decoded_sha256"]):
            raise ValueError("encoded/decoded chunk SHA-256 or size mismatch")


def restore_chunk_input(manifest_path,destination_zip,expected_manifest_sha256=None):
    """Verify parts and atomically reconstruct their exact source ZIP; return a receipt."""
    manifest_path=_guard(manifest_path,regular=True);destination_zip=_guard(destination_zip,regular=True)
    value,manifest_sha=_manifest(manifest_path,expected_manifest_sha256)
    if destination_zip==manifest_path or destination_zip.parent==manifest_path.parent:
        raise ValueError("restore output must be separate from the uploaded manifest/parts")
    source=value["source_archive"];pending=destination_zip.with_name(destination_zip.name+".chunk-partial")
    owner_path=destination_zip.with_name(destination_zip.name+".chunk-owner.json")
    _guard(pending,regular=True);_guard(owner_path,regular=True)
    owner=dict(kind=KIND,manifest_sha256=manifest_sha,source_sha256=source["sha256"],size_bytes=source["size_bytes"])
    if owner_path.exists():
        if json.loads(owner_path.read_bytes())!=owner:raise ValueError("restore ownership/manifest differs")
    elif pending.exists():raise ValueError("unowned partial restore file")
    if destination_zip.exists() and (destination_zip.stat().st_size!=source["size_bytes"] or _sha(destination_zip)!=source["sha256"]):
        raise ValueError("refusing to overwrite unrelated destination ZIP")
    retained=destination_zip if destination_zip.exists() else pending
    needed=max(0,source["size_bytes"]-(retained.stat().st_size if retained.exists() else 0))+65536
    _preflight(destination_zip,needed)
    destination_zip.parent.mkdir(parents=True,exist_ok=True)
    if not destination_zip.exists():_commit_json(owner_path,owner)
    whole=hashlib.sha256();size=0
    with _PrefixWriter(destination_zip,source["size_bytes"]) as output:
        for data in _decoded_parts(manifest_path.parent,value["parts"]):
            whole.update(data);size+=len(data);output.write(data)
        if size!=source["size_bytes"] or whole.hexdigest()!=source["sha256"]:
            raise ValueError("reconstructed source archive SHA-256 mismatch")
        output.finish(source["sha256"])
    return dict(path=str(destination_zip),sha256=source["sha256"],size_bytes=size,parts=len(value["parts"]),
                verified=True,manifest_sha256=manifest_sha,source_input_sha256=value["source_input_sha256"],
                checkpoint_index_sha256=value.get("checkpoint_index_sha256"))
