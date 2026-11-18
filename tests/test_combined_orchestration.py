"""Combined five-planner orchestration regressions; no planner trials are run."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))

from experiment_common import ROOT,digest,dump,load_config
from generate_scenarios import generate_combined
from run_advanced_campaign import (CampaignClock,campaign_descriptor,planner_blocks,
                                   resolve_campaign_inputs,run_analysis_with_deadline)
from run_combined_campaign import require_explicit_allow
from run_experiments import (COMBINED_LATIN_SQUARE,COMBINED_PLANNERS,
                             TRIAL_ORDER_COMBINED,audit_combined_plan,plan,
                             request_for)


class CombinedOrchestrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=load_config(ROOT/"configs/a3_vs_all_v1.yaml","combined")

    def manifest_fixture(self):
        maps=[]
        settings=self.config["profiles"]["combined"]
        for family in settings["families"]:
            for seed in settings["map_seeds"]:
                maps.append({"map_id":f"{family}_{seed}","family":family,"map_seed":seed,
                    "path":f"fixture/{family}_{seed}.t25d","sha256":f"sha-{family}-{seed}",
                    "queries":[{"query_id":"q0","start":[4.,20.],"goal":[36.,20.],
                        "geometry_label":"fixture_geometry",
                        "feasibility":"feasible_witness_validated",
                        "reference":{"validation":{"valid":True}}}]})
        return {"maps":maps}

    @staticmethod
    def block_orders(trials):
        return {group[0]["map_id"]:tuple(entry["planner_id"] for entry in group)
                for group in (trials[offset:offset+len(COMBINED_PLANNERS)]
                              for offset in range(0,len(trials),len(COMBINED_PLANNERS)))}

    def test_locked_250_trial_plan_has_predeclared_five_way_latin_balance(self):
        maps=self.manifest_fixture();settings=self.config["profiles"]["combined"]
        first=plan(self.config,"combined",maps,order_policy=TRIAL_ORDER_COMBINED)
        second=plan(self.config,"combined",maps,order_policy=TRIAL_ORDER_COMBINED)
        self.assertEqual(first,second)
        self.assertEqual(len(first),250)
        self.assertEqual(len(planner_blocks(first,5)),50)
        audit=audit_combined_plan(first,settings)
        self.assertEqual(audit["latin_square"],list(map(list,COMBINED_LATIN_SQUARE)))
        self.assertEqual(set(audit["latin_row_counts"].values()),{10})
        self.assertTrue(all(set(counts)=={10} for counts in audit["position_counts"].values()))
        for family in settings["families"]:
            self.assertEqual(set(audit["family_latin_row_counts"][family].values()),{2})
            self.assertTrue(all(set(counts)=={2}
                                for counts in audit["family_position_counts"][family].values()))

        # Changing only the block-shuffle seed changes block order, never the
        # row already assigned to a given declared map/planner-seed slot.
        reshuffled=copy.deepcopy(self.config)
        reshuffled["execution"]["shuffle_trial_order_seed"]+=1
        changed=plan(reshuffled,"combined",maps,order_policy=TRIAL_ORDER_COMBINED)
        self.assertNotEqual([entry["map_id"] for entry in first[::5]],
                            [entry["map_id"] for entry in changed[::5]])
        self.assertEqual(self.block_orders(first),self.block_orders(changed))

        corrupted=copy.deepcopy(first)
        corrupted[0]["planner_id"],corrupted[1]["planner_id"]=(
            corrupted[1]["planner_id"],corrupted[0]["planner_id"])
        with self.assertRaisesRegex(ValueError,"Latin row mismatch"):
            audit_combined_plan(corrupted,settings)

    def test_combined_requests_use_twenty_seconds_and_combined_sem_parameters(self):
        trials=plan(self.config,"combined",self.manifest_fixture(),order_policy=TRIAL_ORDER_COMBINED)
        sem=next(entry for entry in trials if entry["planner_id"]=="sem_rrtstar_common_v1")
        request=request_for(sem,self.config,"combined")
        self.assertEqual(request["planning_budget_s"],20.)
        self.assertEqual(request["sem_range_m"],self.config["combined_benchmark"]["sem_range_m"])
        self.assertEqual(request["sem_neighbor_radius_m"],
                         self.config["combined_benchmark"]["sem_neighbor_radius_m"])

    def test_combined_descriptor_resolve_schedule_and_protocol_hash(self):
        descriptor=campaign_descriptor(self.config)
        self.assertEqual(descriptor,{"profile":"combined","protocol_key":"combined_benchmark",
            "protocol":self.config["combined_benchmark"],"dataset_schema_version":5,
            "generator_version":"5.0.0","order_policy":TRIAL_ORDER_COMBINED,
            "classification":"LOCKED_A3_VS_ALL_V1",
            "protocol_hash_key":"combined_protocol_sha256"})
        with tempfile.TemporaryDirectory(prefix="combined_resolve_") as directory:
            manifest_path=Path(directory)/"manifest.json";dump(manifest_path,self.manifest_fixture())
            with patch("run_advanced_campaign.verify_dataset"), \
                 patch("run_advanced_campaign.provenance_identity",return_value={"fixture":True}):
                _,_,trials,schedule,identity=resolve_campaign_inputs(
                    ROOT/"configs/a3_vs_all_v1.yaml",manifest_path,Path("fixture-worker"),"combined")
        self.assertEqual(len(trials),250)
        self.assertEqual(schedule["planner_block_size"],5)
        self.assertEqual(schedule["classification"],"LOCKED_A3_VS_ALL_V1")
        self.assertEqual(identity["trial_order_policy"],TRIAL_ORDER_COMBINED)
        self.assertEqual(identity["combined_protocol_sha256"],digest(self.config["combined_benchmark"]))

    def test_combined_generator_is_schema_five_and_never_replaces_seeds(self):
        config=copy.deepcopy(self.config)
        # Keep all 50 declared maps while making this orchestration unit test
        # independent of expensive 0.1 m raster generation.
        config["profiles"]["combined"]["grid_spacing_m"]=40.
        geometry=lambda family,seed,width,height: (
            (lambda x,y:0.),[],[("q0",[4.,20.],[36.,20.],
              [[[4.,20.],[36.,20.]]],"easy_control")],{"fixture":True})
        def write_map(path,*args,metadata=None,**kwargs):
            Path(path).write_text("fixture map\n",encoding="utf-8")
            return {**metadata,"data_sha256":"sha-"+Path(path).stem}
        oracle=SimpleNamespace(validate_path=lambda *args,**kwargs:{"valid":True,
                                                                    "cost_terrain_m":32.})
        with tempfile.TemporaryDirectory(prefix="combined_generator_") as directory, \
             patch("generate_scenarios.pilot_geometry",side_effect=geometry), \
             patch("generate_scenarios.write_map",side_effect=write_map) as writer, \
             patch("generate_scenarios.Oracle",return_value=oracle):
            manifest=generate_combined(config,Path(directory))
        self.assertEqual((manifest["schema_version"],manifest["generator_version"]),(5,"5.0.0"))
        self.assertEqual(len(manifest["maps"]),50)
        self.assertEqual(writer.call_count,50)
        self.assertEqual(manifest["declared_map_seeds"],list(range(5001,5011)))
        self.assertEqual(manifest["seed_replacement_policy"],"forbidden")
        self.assertEqual(manifest["seed_replacements"],[])

    def test_combined_analysis_uses_generic_analyzer_without_plots(self):
        protocol=self.config["combined_benchmark"]
        with tempfile.TemporaryDirectory(prefix="combined_analysis_") as directory:
            root=Path(directory);raw=root/"raw";raw.mkdir();dump(raw/"manifest.json",{"fixture":True})
            with patch("run_advanced_campaign.subprocess.run",
                       return_value=SimpleNamespace(returncode=7,stdout="",stderr="fixture")):
                result=run_analysis_with_deadline(raw,root/"report",protocol,CampaignClock(0.))
        command=result["command"]
        self.assertIn(str(ROOT/"tools/analyze_results.py"),command)
        self.assertIn("--no-plots",command)
        self.assertIn(str(protocol["bootstrap_replicates"]),command)
        self.assertIn(str(protocol["bootstrap_seed"]),command)

    def test_combined_wrapper_requires_explicit_allow(self):
        with self.assertRaisesRegex(ValueError,"--allow-combined-benchmark"):
            require_explicit_allow(False)
        require_explicit_allow(True)


if __name__=="__main__":unittest.main()
