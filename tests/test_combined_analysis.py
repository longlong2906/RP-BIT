"""Synthetic-only regression tests for the locked five-planner analyzer path."""
from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from analyze_results import analyze
from experiment_common import ROOT,digest,dump,file_hash,read_json


PLANNERS=(
    "bitstar_a3",
    "abitstar",
    "aitstar",
    "lazyprmstar_hauser2015",
    "sem_rrtstar_common_v1",
)


def write_raw_inventory(source):
    excluded={"raw_artifact_inventory.json","campaign_state.json","analysis_attempt.json"}
    entries=[]
    for path in sorted((item for item in source.rglob("*") if item.is_file()),
                       key=lambda item:item.relative_to(source).as_posix()):
        relative=path.relative_to(source).as_posix()
        if relative in excluded or relative.endswith(".tmp"):
            continue
        entries.append({"path":relative,"size_bytes":path.stat().st_size,"sha256":file_hash(path)})
    inventory={"schema_version":1,"entries":entries,"inventory_sha256":digest(entries)}
    dump(source/"raw_artifact_inventory.json",inventory)
    return inventory["inventory_sha256"]


def make_fixture(root,protocol_id="a3_vs_all_v1",with_inventory=True):
    source=Path(root)/"raw";source.mkdir()
    config={
        "analysis":{"bootstrap_unit_order":["map","query","seed"],"confidence_level":.95,
                    "target_relative_tolerances":[.10,.05,.02]},
        "model":{"numerical_abs_tolerance_m":1e-8,"numerical_rel_tolerance":1e-8},
        "profiles":{"combined":{"planning_budget_s":20.0,"planners":list(PLANNERS)}},
        "combined_benchmark":{
            "protocol_id":protocol_id,
            "primary_planner_id":"bitstar_a3",
            "comparator_planner_ids":list(PLANNERS[1:]),
            "informed_planner_ids":["bitstar_a3","abitstar","aitstar","sem_rrtstar_common_v1"],
            "bootstrap_replicates":20,
            "bootstrap_seed":73137,
            "sem_comparator_label":"SEM-RRT*-adapted common-objective",
            "official_sem_rrtstar_implementation":False,
            "terrain_risk_cost":False,
            "turning_angle_cost":False,
            "rolling_optimization":False,
            "report_metrics":["validated_success","first_solution_time_s",
                              "cost_terrain_m_at_budget","length_xy_m_final",
                              "actual_solve_wall_s","coverage"],
        },
    }
    dump(source/"resolved_config.json",config)
    entries=[]
    planner_offsets={planner:index*.1 for index,planner in enumerate(PLANNERS)}
    time_offsets={planner:.1+index*.01 for index,planner in enumerate(PLANNERS)}
    for family in ("family_alpha","family_beta"):
        for map_index in range(2):
            map_id=f"{family}_{5001+map_index}"
            for planner in PLANNERS:
                trial_id=f"{map_id}_q0_{planner}_s93001"
                entry={"trial_id":trial_id,"family":family,"map_id":map_id,
                       "map_hash":"synthetic_map_hash_"+map_id,"query_id":"q0",
                       "planner_id":planner,"planner_seed":93001}
                entries.append(entry)
                directory=source/trial_id;snapshot=directory/"snapshots"/"s0"
                snapshot.mkdir(parents=True)
                (snapshot/"path_xy.csv").write_text("x_m,y_m\n0,0\n1,0\n",encoding="utf-8")
                (snapshot/"path_surface.csv").write_text(
                    "x_m,y_m,z_m\n0,0,0\n1,0,0\n",encoding="utf-8")
                cost=10.0+map_index+planner_offsets[planner]
                dump(snapshot/"validation.json",{
                    "valid":True,"cost_terrain_m":cost,
                    "xy_sha256":file_hash(snapshot/"path_xy.csv"),
                    "surface_sha256":file_hash(snapshot/"path_surface.csv"),
                    "synthetic_test_fixture":True,
                })
                event={"snapshot_id":"s0","elapsed_s":time_offsets[planner],
                       "best_cost_terrain_m":cost,"independent_valid":True,
                       "counters":{"motion_validity_calls":10,"state_validity_calls":20}}
                dump(directory/"validated_events.json",[event])
                dump(directory/"validation.json",{
                    "valid":True,"cost_terrain_m":cost,"length_xy_m":1.0,
                    "synthetic_test_fixture":True,
                })
                dump(directory/"trial.json",dict(entry,
                    execution_status="COMPLETED",planner_status="EXACT_SOLUTION",
                    independent_valid=True,path_available=True,time_budget_s=20.0,
                    best_cost_terrain_m=cost,first_solution_time_s=time_offsets[planner],
                    length_xy_m=1.0,independently_recomputed_cost_m=cost,
                    validation_wall_s=.01,actual_solve_wall_s=20.0,worker_wall_s=20.1,
                    termination_reason="TIME_BUDGET",
                    counters={"motion_validity_calls":100,"state_validity_calls":200,
                              "sample_proposals":300,"samples_returned_before_validity":150}))
    dump(source/"manifest.json",{
        "schema_version":3,"experiment_id":"synthetic_a3_vs_all_fixture",
        "profile":"combined","planned_trials":len(entries),"status":"COMPLETE_VALIDATED",
        "identity":{"synthetic_test_fixture":True},"trials":entries,
    })
    inventory_hash=write_raw_inventory(source) if with_inventory else None
    return source,inventory_hash


class CombinedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="combined_analysis_",dir=ROOT/"tests/fixtures")
        self.root=Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_combined_protocol_adds_overall_paired_bootstrap_and_disclosures(self):
        source,inventory_hash=make_fixture(self.root)
        output=self.root/"report"
        summary=analyze(source,output,make_plots=False)

        self.assertEqual(summary["protocol_id"],"a3_vs_all_v1")
        self.assertEqual(summary["planner_ids"],list(PLANNERS))
        self.assertEqual(summary["primary_planner_id"],"bitstar_a3")
        self.assertEqual(summary["comparator_planner_ids"],list(PLANNERS[1:]))
        self.assertEqual(summary["raw_artifact_inventory_sha256"],inventory_hash)
        self.assertEqual(summary["combined_design"]["planning_budget_s"],20.0)
        self.assertEqual(summary["combined_design"]["planner_seed_labels_per_map"],1)
        self.assertFalse(summary["sem_comparator_scope"]["official_sem_rrtstar_implementation"])
        self.assertFalse(summary["figures"])
        self.assertIn("PILLOW_MATPLOTLIB_DLL_RUNTIME",summary["figure_status"])

        paired=read_json(output/"paired_bootstrap.json")
        overall=[item for item in paired if item["family"]=="ALL"]
        self.assertEqual(len(overall),12)
        self.assertEqual({(item["planner_a"],item["planner_b"]) for item in overall},
                         {("bitstar_a3",planner) for planner in PLANNERS[1:]})
        self.assertEqual({item["metric"] for item in overall},
                         {"success","cost_terrain_m","first_solution_time_s"})
        self.assertTrue(all(item["units"]["maps"]==4 for item in overall))
        self.assertTrue(all(item["units"]["paired_seed_slots"]==4 for item in overall))
        self.assertTrue(all(item["units"]["distinct_seed_labels"]==1 for item in overall))
        self.assertTrue(all(item["raw_artifact_inventory_sha256"]==inventory_hash for item in overall))

        report=(output/"REPORT.md").read_text(encoding="utf-8")
        for phrase in ("250 trial = 50 map × 5 planner","budget solve 20 s",
                       "một planner seed 93001","SEM-RRT*-adapted common-objective",
                       "`official_sem_rrtstar_implementation=false`",
                       "`terrain_risk_cost=false`","`turning_angle_cost=false`",
                       "`rolling_optimization=false`",
                       "không phải implementation SEM-RRT* chính thức và không paper-faithful",
                       "`cost_terrain_m` là chiều dài đường bám bề mặt",
                       "`length_xy_m` là chiều dài chiếu bằng phụ trợ",
                       "`EXACT_SOLUTION` không phải chứng minh tối ưu toàn cục",
                       "timeout không phải chứng minh vô nghiệm","Không tạo PNG"):
            self.assertIn(phrase,report)

    def test_combined_analysis_requires_frozen_raw_inventory(self):
        source,_=make_fixture(self.root,with_inventory=False)
        with self.assertRaisesRegex(ValueError,"locked comparison analysis requires raw_artifact_inventory"):
            analyze(source,self.root/"missing_inventory_report",make_plots=False)

    def test_overall_scope_is_protocol_id_gated(self):
        source,_=make_fixture(self.root,protocol_id="synthetic_other_protocol")
        output=self.root/"other_report"
        summary=analyze(source,output,make_plots=False)
        self.assertEqual(summary["protocol_id"],"synthetic_other_protocol")
        self.assertFalse(any(item["family"]=="ALL" for item in read_json(output/"paired_bootstrap.json")))
        self.assertNotIn("combined_design",summary)


if __name__=="__main__":
    unittest.main()
