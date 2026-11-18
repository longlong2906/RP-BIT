"""Independent traversability fixtures; generated only below tests/fixtures."""
import argparse
from decimal import Decimal,localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from independent_verifier import Oracle
from terrain_io import write_map
from experiment_common import worker_env

ROOT=Path(__file__).resolve().parents[1]
CONFIG={"feature_version":"cell_envelope_v1","lambda":1.,"slope_weight":.5,
        "clearance_weight":.5,"clearance_comfort_m":.4}
CLI=None


class TraversabilityOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="traversability_oracle_",dir=ROOT/"tests/fixtures")
        self.folder=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def terrain(self,name="plane",heights=None,**kwargs):
        path=self.folder/(name+".t25d")
        write_map(path,heights or [[float(i) for i in range(6)] for _ in range(6)],radius=.1,margin=0.,slope=60.,
                  metadata={"synthetic_test_fixture":True},**kwargs)
        return path,Oracle(path)
    def test_decimal_plane_and_constant_penalty(self):
        _,oracle=self.terrain()
        a,b=[1.25,1.25],[1.75,1.5]
        got=oracle.traversability_edge(a,b,CONFIG)
        with localcontext() as ctx:
            ctx.prec=70
            length=(Decimal(".5")**2*2+Decimal(".25")**2).sqrt()
            rs=length*Decimal(".5625")
            objective=length+rs/2
        self.assertAlmostEqual(got["cost_terrain_m"],float(length),places=13)
        self.assertAlmostEqual(got["risk_slope_integral_m"],float(rs),places=13)
        self.assertAlmostEqual(got["objective_value"],float(objective),places=13)
        self.assertEqual(got["risk_clearance_integral_m"],0.)
        zero=oracle.traversability_edge(a,b,dict(CONFIG,**{"lambda":0.}))
        self.assertEqual(zero["objective_value"],zero["cost_terrain_m"])
    def test_clearance_is_obstacle_only_and_squared(self):
        _,oracle=self.terrain(heights=[[0.]*6 for _ in range(6)],rectangles=[(2.4,1.,2.8,2.)])
        result=oracle.traversability_edge([1.25,1.25],[1.75,1.25],CONFIG)
        self.assertAlmostEqual(result["risk_clearance_integral_m"],.5*.0625,places=13)
        self.assertEqual(result["risk_slope_integral_m"],0.)
        checked=oracle.validate_path([[1.25,1.25],[1.75,1.25]],[1.25,1.25],[1.75,1.25],
                                     traversability=CONFIG)
        self.assertTrue(checked["valid"],checked)
        self.assertAlmostEqual(checked["minimum_obstacle_footprint_clearance_m"],.55)
    def test_boundary_component_max_split_and_reverse(self):
        _,oracle=self.terrain(heights=[[.1*abs(i-2)+.02*j for i in range(6)] for j in range(6)],
                              rectangles=[(3.4,1.,3.8,2.)])
        a,b,m=[2.,1.25],[2.,3.75],[2.,2.5]
        whole=oracle.traversability_edge(a,b,CONFIG);reverse=oracle.traversability_edge(b,a,CONFIG)
        left=oracle.traversability_edge(a,m,CONFIG);right=oracle.traversability_edge(m,b,CONFIG)
        for key in ("cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m","objective_value"):
            self.assertAlmostEqual(whole[key],reverse[key],places=12)
            self.assertAlmostEqual(whole[key],left[key]+right[key],places=12)
        cells=oracle.traversability_features(CONFIG)["cells"]
        short=oracle.traversability_edge([2.,1.25],[2.,1.75],CONFIG)
        self.assertAlmostEqual(short["risk_clearance_integral_m"],short["cost_terrain_m"]*max(cells[6]["clearance_penalty"],cells[7]["clearance_penalty"]))
    def test_missing_stencil_does_not_expand_hard_mask(self):
        heights=[[0.]*6 for _ in range(6)];heights[0][0]=None
        _,oracle=self.terrain(heights=heights)
        cells=oracle.traversability_features(CONFIG)["cells"]
        self.assertFalse(cells[0]["available"]);self.assertTrue(cells[1]["available"])
        self.assertFalse(oracle.traversability_edge([.2,.2],[.4,.4],CONFIG)["has_data"])
        checked=oracle.validate_path([[1.5,1.5],[2.5,2.5]],[1.5,1.5],[2.5,2.5],traversability=CONFIG)
        self.assertTrue(checked["valid"],checked)
        self.assertIsNone(checked["minimum_obstacle_footprint_clearance_m"])
        self.assertEqual(checked["obstacle_clearance_status"],"NO_EXPLICIT_OBSTACLES")
    def test_independent_claim_rejection_and_swept_slope(self):
        _,oracle=self.terrain()
        path=[[1.25,1.25],[2.75,2.75]]
        metrics=oracle.traversability_edge(*path,CONFIG);metrics["length_xy_m"]=math.dist(*path)
        checked=oracle.validate_path(path,*path,traversability=CONFIG,claimed_metrics=metrics)
        self.assertTrue(checked["valid"],checked)
        self.assertAlmostEqual(checked["maximum_swept_footprint_slope_deg"],45.)
        metrics["objective_value"]+=.01
        bad=oracle.validate_path(path,*path,traversability=CONFIG,claimed_metrics=metrics)
        self.assertIn("objective_value_mismatch",bad["errors"])
    def test_decimal_grid_shared_diagonal_does_not_create_nodata_gap(self):
        path=self.folder/"shared_diagonal.t25d"
        write_map(path,[[0.]*101 for _ in range(101)],spacing=(.2,.2),metadata={"synthetic_test_fixture":True})
        oracle=Oracle(path)
        witness=[[2.,8.],[8.,14.],[12.,14.],[18.,8.]]
        checked=oracle.validate_path(witness,witness[0],witness[-1])
        self.assertTrue(checked["valid"],checked)
        self.assertAlmostEqual(checked["cost_terrain_m"],4.+12.*math.sqrt(2.),places=11)
    def test_canonical_decimal_boundary_and_ulp_sides_remain_distinct(self):
        path=self.folder/"decimal_boundary.t25d"
        write_map(path,[[0.]*8 for _ in range(8)],spacing=(.2,.2),radius=.1,margin=0.,
                  rectangles=[(.9,.2,1.1,1.3)],metadata={"synthetic_test_fixture":True})
        oracle=Oracle(path);canonical=.2*3
        costs=[]
        for x in (math.nextafter(canonical,0.),canonical,math.nextafter(canonical,math.inf)):
            a,b=[x,.35],[x,.45];expected=oracle.traversability_edge(a,b,CONFIG);costs.append(expected["risk_clearance_integral_m"])
            if CLI is not None:
                process=subprocess.run([str(CLI),"traversability",str(path),*map(str,[*a,*b,1.,.5,.5,.4])],
                                       capture_output=True,text=True,env=worker_env(),timeout=30)
                self.assertEqual(process.returncode,0,process.stderr)
                actual=json.loads(process.stdout)
                self.assertAlmostEqual(actual["risk_clearance_integral_m"],expected["risk_clearance_integral_m"],places=12)
        self.assertLess(costs[0],costs[1]);self.assertEqual(costs[1],costs[2])
    def test_vectorized_envelope_geometry_and_large_mesh_swept_slope(self):
        _,oracle=self.terrain()
        rect=(1.25,.75,2.25,1.75)
        batch=oracle._triangles_rect_distances(oracle.vertices,rect)
        for triangle,distance in zip(oracle.vertices,batch):
            self.assertAlmostEqual(float(distance),oracle.triangle_rect_distance(triangle,rect),places=12)
        path=self.folder/"large_slope.t25d"
        write_map(path,[[.1*i*.2+.2*j*.2 for i in range(101)] for j in range(101)],spacing=(.2,.2),
                  metadata={"synthetic_test_fixture":True})
        large=Oracle(path)
        self.assertAlmostEqual(large._swept_slope([2.,8.],[18.,8.]),math.degrees(math.atan(math.sqrt(.05))),places=10)
    def test_decimal_multiscale_translated_plane_and_64_edge_accumulation(self):
        # This checks numerical accuracy against a 70-digit analytic integral,
        # not a floating-point proof for native BIT* pruning. Tolerances are in
        # ulps of each expected value, so the small-scale case cannot pass merely
        # because the ordinary metre-level absolute tolerance dominates it.
        names=("cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m",
               "risk_integral_m","objective_value")
        for exponent in (-12,0,12):
            with self.subTest(scale_exponent=exponent):
                scale=math.ldexp(1.,exponent);origin=(17.*scale,-23.*scale);height_offset=31.*scale
                path=self.folder/f"analytic_scale_{exponent}.t25d"
                write_map(path,[[height_offset+i*scale for i in range(7)] for _ in range(7)],
                          origin=origin,spacing=(scale,scale),radius=scale/8,margin=0.,slope=60.,
                          metadata={"synthetic_test_fixture":True,"scale_exponent":exponent})
                oracle=Oracle(path)
                a=[origin[0]+1.25*scale,origin[1]+1.25*scale]
                b=[origin[0]+4.75*scale,origin[1]+3.75*scale]
                with localcontext() as ctx:
                    ctx.prec=70
                    ds=Decimal.from_float(scale)
                    length=ds*(2*Decimal("3.5")**2+Decimal("2.5")**2).sqrt()
                    rs=length*Decimal(".5625")
                    risk=rs/2
                    expected=dict(zip(names,(length,rs,Decimal(0),risk,length+risk)))
                def compare(actual):
                    self.assertTrue(actual["has_data"],actual)
                    for name in names:
                        reference=float(expected[name]);tolerance=128*math.ulp(reference) if reference else 0.
                        self.assertLessEqual(abs(actual[name]-reference),tolerance,
                            (exponent,name,actual[name],str(expected[name]),tolerance))
                whole=oracle.traversability_edge(a,b,CONFIG);compare(whole)
                samples=[[a[axis]+(b[axis]-a[axis])*index/64 for axis in (0,1)] for index in range(65)]
                parts=[oracle.traversability_edge(p,q,CONFIG) for p,q in zip(samples,samples[1:])]
                accumulated={name:sum(part[name] for part in parts) for name in names}
                compare(dict(accumulated,has_data=all(part["has_data"] for part in parts)))
                # The analytic value lies strictly between the adjacent doubles
                # around its correctly rounded representation. We do not turn
                # that bracket into an assertion about unpatched native pruning.
                for name in ("cost_terrain_m","objective_value"):
                    rounded=float(expected[name])
                    self.assertLess(Decimal.from_float(math.nextafter(rounded,-math.inf)),expected[name])
                    self.assertGreater(Decimal.from_float(math.nextafter(rounded,math.inf)),expected[name])
                if CLI is not None:
                    def native(p,q):
                        process=subprocess.run([str(CLI),"traversability",str(path),*map(str,[*p,*q,1.,.5,.5,.4])],
                                               capture_output=True,text=True,env=worker_env(),timeout=30)
                        self.assertEqual(process.returncode,0,process.stderr)
                        return json.loads(process.stdout)
                    compare(native(a,b))
                    native_parts=[native(p,q) for p,q in zip(samples,samples[1:])]
                    compare(dict(has_data=all(part["has_data"] for part in native_parts),
                                 **{name:sum(part[name] for part in native_parts) for name in names}))
    def test_cpp_independent_crosscheck(self):
        if CLI is None:self.skipTest("--cli required for actual C++ crosscheck")
        path,oracle=self.terrain(heights=[[.1*abs(i-2)+.02*j for i in range(6)] for j in range(6)],
                                  rectangles=[(3.4,1.,3.8,2.)])
        def call(command,*args):
            process=subprocess.run([str(CLI),command,str(path),*map(str,args)],capture_output=True,text=True,env=worker_env(),timeout=30)
            self.assertEqual(process.returncode,0,process.stderr)
            return json.loads(process.stdout)
        args=[1.,.5,.5,.4]
        native=call("features",*args);independent=oracle.traversability_features(CONFIG)
        self.assertEqual((native["nx_cells"],native["ny_cells"]),(5,5))
        for actual,expected in zip(native["cells"],independent["cells"]):
            for key in ("available","slope_penalty","clearance_penalty"):
                if key=="available":self.assertEqual(actual[key],expected[key])
                else:self.assertAlmostEqual(actual[key],expected[key],places=11)
        for a,b in (([1.25,1.25],[3.75,3.75]),([2.,1.25],[2.,3.75]),([2.,2.],[2.,2.]),
                    ([math.nextafter(2.,0.),1.25],[math.nextafter(2.,0.),3.75])):
            actual=call("traversability",*a,*b,*args);expected=oracle.traversability_edge(a,b,CONFIG)
            self.assertEqual(actual["has_data"],expected["has_data"])
            for key in ("cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m","objective_value"):
                self.assertAlmostEqual(actual[key],expected[key],places=10,msg=(a,b,key,actual,expected))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--cli",type=Path);args,remaining=parser.parse_known_args()
    CLI=args.cli.resolve() if args.cli else None
    unittest.main(argv=[sys.argv[0],*remaining])
