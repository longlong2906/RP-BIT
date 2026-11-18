"""Synthetic receipt/parser fixtures; no cached path or experimental result."""
import copy
import json
import math
from pathlib import Path
import platform
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
sys.path.insert(0,str(ROOT/"tests"))
import corridor_v4_l0_cache as cache
import test_corridor_v3_validation as legacy
from corridor_v3_validation import record_corridor_receipt,validate_trial,shortest_graph_length
from native_paper_oracle import NativePaperOracle
from run_safe_length import dump


class SmokeL0CacheTests(unittest.TestCase):
    def fixture(self,root):
        directory=root/"trials/synthetic_trial"
        directory.mkdir(parents=True)
        request,info,_,oracle,field=legacy.CorridorValidationTests().fixture(directory)
        dump(directory/"request.json",request)
        record_corridor_receipt(directory)
        verdict=validate_trial(directory,request,info,dict(status="COMPLETED"),oracle=oracle,field=field)
        self.assertTrue(verdict["success"])
        # A deliberately labeled synthetic A* receipt exercises cache parsing;
        # the numerical length is independently obtained from this tiny map.
        verdict["synthetic_fixture"]=True
        verdict["corridor"]["L0_certificate"].update(method="independent_registered_graph_Astar_XY_lower_bound",
            cached=False,settled_nodes=2,synthetic_fixture=True)
        dump(directory/"validation.json",verdict)
        inventory=cache._raw_inventory(directory)
        dump(directory/"raw_inventory.json",inventory)
        info.update(map_id="synthetic_map",start=request["start"],goal=request["goal"])
        manifest=dict(profile="smoke",status="COMPLETE",synthetic_fixture=True,maps=[info],
            environment=dict(python=sys.version,platform=platform.platform()),
            sources={"tools/"+name:value for name,value in cache._oracle_hashes().items()},
            entries=[dict(method="a3t_corridor_v3",status="COMPLETED",trial_id="synthetic_trial",
                          map_id=info["map_id"],validation_sha256=cache._file_hash(directory/"validation.json"),
                          raw_inventory_sha256=cache._digest(inventory))])
        dump(root/"manifest.json",manifest)
        return directory,info,field

    def test_guarded_preload_and_actual_exact_key_marker(self):
        with tempfile.TemporaryDirectory(prefix="l0_cache_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            root=Path(tmp);_,info,field=self.fixture(root)
            bundle=cache.prepare_smoke_l0_cache(root)
            self.assertEqual(len(bundle["proofs"]),1)
            oracle=NativePaperOracle(info["map_path"])
            evidence=cache.seed_map_l0_cache(bundle,info,oracle,field,expected_sha256=cache._digest(bundle))
            self.assertEqual(len(evidence["seeded_certificates"]),1)
            result=shortest_graph_length(oracle,field,info["start"],info["goal"],1.5)
            self.assertTrue(result["cached"])
            self.assertEqual(result["smoke_reuse_binding_sha256"],bundle["proofs"][0]["binding_sha256"])
            self.assertFalse(result["search_executed_in_this_process"])
            # A one-ULP different upper bound cannot hit the seeded cache.
            different=shortest_graph_length(oracle,field,info["start"],info["goal"],math.nextafter(1.5,2.))
            self.assertFalse(different["cached"])
            self.assertNotIn("smoke_reuse_binding_sha256",different)
            self.assertEqual(cache._read(root/"manifest.json")["status"],"COMPLETE")

    def test_query_bounds_and_independent_field_mismatch_do_not_seed(self):
        with tempfile.TemporaryDirectory(prefix="l0_cache_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
            root=Path(tmp);_,info,field=self.fixture(root);bundle=cache.prepare_smoke_l0_cache(root)
            for key,value in (("start",[.8,1.2]),("goal",[2.1,1.2]),("bounds",[0.,0.,3.,3.1])):
                with self.subTest(key=key):
                    changed=dict(info,**{key:value});oracle=NativePaperOracle(info["map_path"])
                    evidence=cache.seed_map_l0_cache(bundle,changed,oracle,field)
                    self.assertEqual(evidence["seeded_certificates"],[])
            field=copy.copy(field)
            field.risk=field.risk.copy();field.risk[10,10]=math.nextafter(float(field.risk[10,10]),1.)
            oracle=NativePaperOracle(info["map_path"])
            self.assertEqual(cache.seed_map_l0_cache(bundle,info,oracle,field)["seeded_certificates"],[])

    def test_profile_and_content_or_source_tampering_rejected(self):
        for kind in ("profile","content","certificate","source","raw","runtime"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(prefix="l0_cache_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
                root=Path(tmp);directory,info,field=self.fixture(root);bundle=cache.prepare_smoke_l0_cache(root)
                oracle=NativePaperOracle(info["map_path"])
                if kind=="profile":
                    with self.assertRaisesRegex(ValueError,"only for smoke"):
                        cache.seed_map_l0_cache(bundle,info,oracle,field,profile="pilot")
                    continue
                if kind=="content":
                    with self.assertRaisesRegex(ValueError,"content hash"):
                        cache.validate_smoke_l0_cache(bundle,"0"*64)
                    continue
                if kind=="certificate":bundle["proofs"][0]["certificate"]["length"]+=1.
                if kind=="source":bundle["oracle_sources"]["independent_verifier.py"]="0"*64
                if kind=="runtime":bundle["runtime"]["numpy"]="unregistered"
                if kind=="raw":(directory/"corridor_guide_xy.csv").write_text("tampered\n")
                with self.assertRaises(ValueError):cache.validate_smoke_l0_cache(bundle)

    def test_source_validation_hash_and_map_bytes_are_mandatory(self):
        for kind in ("verdict","map"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(prefix="l0_cache_synthetic_",dir=ROOT/"tests/fixtures") as tmp:
                root=Path(tmp);directory,info,field=self.fixture(root)
                if kind=="verdict":(directory/"validation.json").write_text("{}")
                else:Path(info["map_path"]).write_text("changed terrain")
                with self.assertRaises(ValueError):cache.prepare_smoke_l0_cache(root)


if __name__=="__main__":unittest.main()
