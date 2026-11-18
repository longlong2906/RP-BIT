"""Synthetic selective-import fixtures; no benchmark or planner execution."""
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import native_paper_v4_zip_input as importer


def digest(data):return hashlib.sha256(data).hexdigest()


class ZipInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='zip_input_');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.zip=self.base/'uploaded.zip';self.destination=self.base/'stage'
        self.prefix='outer/native_paper_v4/'
        self.source=b'SYNTHETIC_SOURCE_ARCHIVE';self.source_sha=digest(self.source)
        self.members={self.prefix+'source_input.zip':self.source,
                      self.prefix+'campaign/exports/base.payload':b'SYNTHETIC_BASE_PAYLOAD',
                      self.prefix+'campaign/exports/delta.payload':b'SYNTHETIC_DELTA_PAYLOAD',
                      self.prefix+'campaign/shards/smoke__map.zip':b'SYNTHETIC_RAW_SHARD',
                      self.prefix+'campaign/active/not_selected/worker.json':b'UNSELECTED_RAW_DUPLICATE'}
        def record(name,data):return dict(path=name,size_bytes=len(data),sha256=digest(data),mode=0o644)
        self.index=dict(protocol_id=importer.PROTOCOL,kind=importer.INDEX_KIND,
            source_input_sha256=self.source_sha,root_path='/kaggle/working/native_paper_v4/repo',
            workspace_path='/kaggle/working/native_paper_v4/campaign',
            inventory={'source_input.zip':record('source_input.zip',self.source),
                       'campaign/shards/smoke__map.zip':record('campaign/shards/smoke__map.zip',self.members[self.prefix+'campaign/shards/smoke__map.zip'])},
            payloads=[record(name,self.members[self.prefix+'campaign/exports/'+name]) for name in ('base.payload','delta.payload')])
        self.write_zip()

    def write_zip(self,extra=()):
        self.index['inventory_sha256']=importer._digest(self.index['inventory'])
        data=json.dumps(self.index,sort_keys=True).encode();self.index_sha=digest(data)
        self.index_name=self.prefix+'campaign/exports/checkpoint_index.json'
        self.members[self.index_name]=data
        self.members[self.index_name+'.sha256']=(self.index_sha+'  checkpoint_index.json\n').encode()
        with zipfile.ZipFile(self.zip,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for name,payload in self.members.items():archive.writestr(name,payload)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                for name,payload in extra:archive.writestr(name,payload)

    def prepare(self):return importer.prepare_zip_input(self.zip,self.destination,self.source_sha,self.index_sha)

    def test_nested_selective_idempotent_import_and_verified_cleanup(self):
        before=digest(self.zip.read_bytes());index=self.prepare()
        self.assertEqual(index,self.destination/self.index_name)
        self.assertEqual(index,self.prepare())
        self.assertFalse((self.destination/(self.prefix+'campaign/active')).exists())
        marker=json.loads((self.destination/importer.MARKER).read_bytes())
        self.assertEqual(len(marker['inventory']),6)
        for name,record in marker['inventory'].items():self.assertEqual(digest((self.destination/name).read_bytes()),record['sha256'])
        receipt=importer.cleanup_zip_input(self.destination)
        self.assertEqual(receipt['files'],6);self.assertTrue(receipt['verified'])
        self.assertFalse(self.destination.exists());self.assertEqual(digest(self.zip.read_bytes()),before)

    def test_selected_hash_and_index_checksum_tamper_fail_closed(self):
        self.members[self.prefix+'source_input.zip']=b'X'*len(self.source)
        self.write_zip()
        with self.assertRaisesRegex(ValueError,'size mismatch|hash/size mismatch'):self.prepare()
        self.members[self.prefix+'source_input.zip']=self.source;self.write_zip()
        with self.assertRaisesRegex(ValueError,'index SHA-256'):importer.prepare_zip_input(self.zip,self.destination,self.source_sha,'0'*64)

    def test_duplicate_selected_member_and_ambiguous_index_rejected(self):
        self.write_zip(extra=[(self.prefix+'campaign/exports/base.payload',b'SYNTHETIC_BASE_PAYLOAD')])
        with self.assertRaisesRegex(ValueError,'duplicate selected'):self.prepare()
        self.write_zip(extra=[('copy/native_paper_v4/campaign/exports/checkpoint_index.json',json.dumps(self.index).encode())])
        with self.assertRaisesRegex(ValueError,'unambiguous'):self.prepare()

    def test_indexed_traversal_and_selected_symlink_rejected(self):
        self.index['payloads'][0]['path']='../escape.payload';self.write_zip()
        with self.assertRaisesRegex(ValueError,'unsafe'):self.prepare()
        self.index['payloads'][0]['path']='base.payload';self.write_zip()
        info=zipfile.ZipInfo(self.prefix+'campaign/exports/base.payload');info.create_system=3;info.external_attr=0o120777<<16
        data=self.members.pop(info.filename)
        with zipfile.ZipFile(self.zip,'w') as archive:
            for name,payload in self.members.items():archive.writestr(name,payload)
            archive.writestr(info,data)
        with self.assertRaisesRegex(ValueError,'not regular'):self.prepare()

    def test_interruption_resumes_only_verified_prefix_and_rejects_unknown_bytes(self):
        original=importer._restore;counter=0
        def interrupt(archive,info,path,record,destination):
            nonlocal counter
            counter+=1
            if counter==3:
                path.parent.mkdir(parents=True,exist_ok=True)
                path.with_name(path.name+'.zip-partial').write_bytes(archive.read(info)[:5])
                raise RuntimeError('SYNTHETIC_INTERRUPTION')
            return original(archive,info,path,record,destination)
        with patch.object(importer,'_restore',side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError,'SYNTHETIC'):self.prepare()
        partial=next(self.destination.rglob('*.zip-partial'));saved=partial.read_bytes()
        partial.write_bytes(b'WRONG')
        with self.assertRaisesRegex(ValueError,'prefix differs'):self.prepare()
        partial.write_bytes(saved);self.prepare()
        self.assertFalse(list(self.destination.rglob('*.zip-partial')))

    def test_disk_preflight_blocks_before_staging_and_cleanup_refuses_mutation(self):
        with patch.object(importer.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_DISK'):self.prepare()
        self.assertFalse(self.destination.exists())
        with patch.object(importer,'MAX_OUTPUT_BYTES',0):
            with self.assertRaisesRegex(RuntimeError,'16 GiB'):self.prepare()
        self.prepare();source=self.destination/(self.prefix+'source_input.zip');source.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'refusing cleanup'):importer.cleanup_zip_input(self.destination)
        self.assertTrue((self.destination/self.index_name).exists())

    def test_unowned_destination_and_link_guard_fail_closed(self):
        self.destination.mkdir();(self.destination/'unrelated').write_bytes(b'USER_BYTES')
        with self.assertRaisesRegex(ValueError,'unrelated'):self.prepare()
        original=Path.is_symlink
        with patch.object(Path,'is_symlink',lambda path:path==self.destination or original(path)):
            with self.assertRaisesRegex(ValueError,'contains a link'):self.prepare()
        self.assertEqual((self.destination/'unrelated').read_bytes(),b'USER_BYTES')


if __name__=='__main__':unittest.main()
