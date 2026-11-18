"""Synthetic chunk transport fixtures; no planner, oracle or network calls."""
import base64
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import native_paper_v4_chunk_input as chunks


def sha(data):return hashlib.sha256(data).hexdigest()


class ChunkInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='chunk_transport_');self.addCleanup(self.temp.cleanup)
        self.base=Path(self.temp.name);self.source=self.base/'synthetic_resume.zip'
        self.output=self.base/'parts';self.destination=self.base/'restored/resume.zip'
        index=dict(protocol_id=chunks.PROTOCOL,source_input_sha256=sha(b'SYNTHETIC_SOURCE_INPUT'))
        with zipfile.ZipFile(self.source,'w') as archive:
            archive.writestr('native_paper_v4/campaign/exports/checkpoint_index.json',json.dumps(index))
            archive.writestr('synthetic_data',bytes(range(256)))
        self.before=self.source.read_bytes()

    def build(self):return chunks.build_chunk_input(self.source,self.output,raw_chunk_bytes=128)

    def restore(self,manifest=None,destination=None):
        manifest=manifest or self.output/chunks.MANIFEST
        return chunks.restore_chunk_input(manifest,destination or self.destination,sha(manifest.read_bytes()))

    def test_roundtrip_padding_per_part_idempotence_and_unrelated_input_ignored(self):
        manifest=self.build();value=json.loads(manifest.read_bytes())
        self.assertGreaterEqual(len(value['parts']),3)
        for part in value['parts'][:2]:
            self.assertEqual(part['decoded_bytes'],128)
            self.assertTrue((self.output/part['path']).read_bytes().endswith(b'='))
        before={p.name:p.read_bytes() for p in self.output.iterdir()}
        (self.output/'unrelated_recovery_patch.b64').write_bytes(b'IGNORED_EXTERNAL_PATCH')
        receipt=self.restore(manifest)
        self.assertTrue(receipt['verified']);self.assertEqual(receipt['sha256'],sha(self.before))
        self.assertEqual(self.destination.read_bytes(),self.before)
        self.assertEqual(self.restore(manifest),receipt)
        for name,data in before.items():self.assertEqual((self.output/name).read_bytes(),data)
        self.assertEqual(self.source.read_bytes(),self.before)

    def test_builder_is_idempotent_but_preserves_unrelated_output_files(self):
        manifest=self.build();before=manifest.read_bytes()
        self.assertEqual(self.build().read_bytes(),before)
        unrelated=self.output/'user_file';unrelated.write_bytes(b'USER_BYTES')
        with self.assertRaisesRegex(ValueError,'unrelated'):self.build()
        self.assertEqual(unrelated.read_bytes(),b'USER_BYTES')

    def test_encoded_and_decoded_corruption_never_publish_final_zip(self):
        manifest=self.build();value=json.loads(manifest.read_bytes());part=value['parts'][0]
        path=self.output/part['path'];original=path.read_bytes()
        path.write_bytes((b'A' if original[:1]!=b'A' else b'B')+original[1:])
        with self.assertRaisesRegex(ValueError,'SHA-256'):self.restore(manifest)
        self.assertFalse(self.destination.exists())
        path.write_bytes(original);value['parts'][0]['decoded_sha256']='0'*64
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'SHA-256'):self.restore(manifest,self.base/'other/resume.zip')
        self.assertFalse((self.base/'other/resume.zip').exists())

    def test_manifest_hash_missing_parts_order_cardinality_and_paths_are_checked(self):
        manifest=self.build();original=manifest.read_bytes();value=json.loads(original)
        with self.assertRaisesRegex(ValueError,'manifest SHA-256'):
            chunks.restore_chunk_input(manifest,self.destination,'0'*64)
        missing=self.output/value['parts'][-1]['path'];data=missing.read_bytes();missing.unlink()
        with self.assertRaisesRegex(ValueError,'missing chunk'):self.restore(manifest)
        missing.write_bytes(data)
        for mutation,reason in ((lambda v:v['parts'].reverse(),'sequence'),
                (lambda v:v.update(part_count=True),'byte count'),
                (lambda v:v['parts'][0].update(path='../escape.b64'),'unsafe'),
                (lambda v:v['parts'].append(v['parts'][-1]),'extra chunk')):
            value=json.loads(original);mutation(value);manifest.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,reason):self.restore(manifest)
        self.assertFalse(self.destination.exists());self.assertEqual(self.source.read_bytes(),self.before)

    def test_restore_interruption_verifies_midpart_prefix_before_appending(self):
        manifest=self.build();original=chunks._PrefixWriter.write;interrupted=False
        def fail_once(output,data):
            nonlocal interrupted
            if output.path==self.destination and not interrupted:
                interrupted=True;original(output,data[:len(data)//2]);raise OSError('SYNTHETIC_DISK_INTERRUPTION')
            return original(output,data)
        with patch.object(chunks._PrefixWriter,'write',fail_once):
            with self.assertRaisesRegex(OSError,'SYNTHETIC'):self.restore(manifest)
        partial=self.destination.with_name(self.destination.name+'.chunk-partial');prefix=partial.read_bytes()
        self.assertGreater(len(prefix),0);partial.write_bytes(b'WRONG'+prefix[5:])
        bad=partial.read_bytes()
        with self.assertRaisesRegex(ValueError,'prefix differs'):self.restore(manifest)
        self.assertEqual(partial.read_bytes(),bad);self.assertFalse(self.destination.exists())
        partial.write_bytes(prefix);self.restore(manifest)
        self.assertEqual(self.destination.read_bytes(),self.before)

    def test_build_interruption_preserves_and_verifies_encoded_prefix(self):
        original=chunks._PrefixWriter.write;interrupted=False
        def fail_once(output,data):
            nonlocal interrupted
            if output.path.suffix=='.b64' and not interrupted:
                interrupted=True;original(output,data[:7]);raise OSError('SYNTHETIC_BUILD_INTERRUPTION')
            return original(output,data)
        with patch.object(chunks._PrefixWriter,'write',fail_once):
            with self.assertRaisesRegex(OSError,'SYNTHETIC'):self.build()
        partial=next(self.output.glob('*.b64.chunk-partial'));before=partial.read_bytes()
        partial.write_bytes(b'WRONG!!')
        with self.assertRaisesRegex(ValueError,'prefix differs'):self.build()
        partial.write_bytes(before);manifest=self.build();self.restore(manifest)
        self.assertEqual(self.destination.read_bytes(),self.before)

    def test_disk_preflight_cap_and_unknown_final_or_partial_are_not_overwritten(self):
        with patch.object(chunks.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_DISK'):self.build()
        self.assertFalse(self.output.exists())
        manifest=self.build()
        with patch.object(chunks.shutil,'disk_usage',return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError,'BLOCKED_DISK'):self.restore(manifest)
        with patch.object(chunks,'_tree_bytes',return_value=chunks.MAX_OUTPUT_BYTES):
            with self.assertRaisesRegex(RuntimeError,'16 GiB'):self.restore(manifest)
        self.assertFalse(self.destination.exists())
        self.destination.parent.mkdir();self.destination.write_bytes(b'USER_ZIP_BYTES')
        with self.assertRaisesRegex(ValueError,'unrelated destination'):self.restore(manifest)
        self.assertEqual(self.destination.read_bytes(),b'USER_ZIP_BYTES');self.destination.unlink()
        partial=self.destination.with_name(self.destination.name+'.chunk-partial');partial.write_bytes(b'UNKNOWN_PREFIX')
        with self.assertRaisesRegex(ValueError,'unowned partial'):self.restore(manifest)
        self.assertEqual(partial.read_bytes(),b'UNKNOWN_PREFIX')

    def test_noncanonical_base64_and_part_links_fail_closed(self):
        path=self.base/'noncanonical.b64';path.write_bytes(b'YR==')
        part=dict(path=path.name,encoded_bytes=4,decoded_bytes=1,encoded_sha256=sha(b'YR=='),decoded_sha256=sha(b'a'))
        with self.assertRaisesRegex(ValueError,'noncanonical'):list(chunks._decoded_parts(self.base,[part]))
        manifest=self.build();value=json.loads(manifest.read_bytes());linked=self.output/value['parts'][0]['path']
        original=Path.is_symlink
        with patch.object(Path,'is_symlink',lambda path:path==linked or original(path)):
            with self.assertRaisesRegex(ValueError,'contains a link'):self.restore(manifest)
        self.assertFalse(self.destination.exists())


if __name__=='__main__':unittest.main()
