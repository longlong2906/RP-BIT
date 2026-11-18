"""Synthetic analytic fixtures and real GP/C-ABI checks, never benchmark results."""
import argparse
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from terrain_io import write_map
from native_point_cloud import generate_cloud, voxel_centroids
from sgp_paper_adapter import SGPPaperModel, feature_fields, SGPStaticPlanner
from terrain_compare_bridge import TerrainBridge, PutnBridge
from independent_verifier import Oracle

LIBRARY = None


class NativeModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="native_model_test_")
        self.root = Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()
    def map(self, name="map", f=lambda x,y: 0., rectangles=(), missing=False):
        h = [[f(i*.2,j*.2) for i in range(31)] for j in range(31)]
        if missing: h[15][15] = None
        path = self.root/(name+".t25d")
        write_map(path, h, spacing=(.2,.2), rectangles=rectangles)
        return path
    def test_cloud_plane_top_walls_and_nodata(self):
        path=self.map(f=lambda x,y:.1*x+.2*y, rectangles=[(2,2,3,3)], missing=True)
        points=generate_cloud(path)
        self.assertTrue(np.isfinite(points).all())
        interior=(points[:,0]>2)&(points[:,0]<2.8)&(points[:,1]>2)&(points[:,1]<2.8)
        np.testing.assert_allclose(points[interior,2]-.1*points[interior,0]-.2*points[interior,1],2.)
        hole=(points[:,0]>2.8)&(points[:,0]<3.2)&(points[:,1]>2.8)&(points[:,1]<3.2)
        self.assertFalse(hole.any())
        wall=(points[:,0]==2)&(points[:,1]>2)&(points[:,1]<2.8)
        offsets=points[wall,2]-.1*points[wall,0]-.2*points[wall,1]
        self.assertTrue(((offsets>0)&(offsets<2)).any())
        np.testing.assert_array_equal(points,generate_cloud(path))
    def test_voxel_reduction_deterministic_and_preserves_layers(self):
        a=np.array([[.01,.01,0],[.03,.03,.02],[.01,.01,2],[.03,.03,2.02]])
        b=voxel_centroids(a)
        np.testing.assert_allclose(b,[[.02,.02,.01],[.02,.02,2.01]])
        np.testing.assert_array_equal(b,voxel_centroids(a[::-1]))
        for bad in (np.empty((0,3)),[[0,0,float("nan")]],[[1,2]]):
            with self.assertRaises(ValueError):voxel_centroids(bad)
    def test_paper_features_analytic_and_uncertainty_boundary(self):
        xx,yy=np.meshgrid(np.arange(7)*.2,np.arange(7)*.2)
        mean=.1*xx+.2*yy
        gradient=np.zeros((*mean.shape,2));gradient[:,:,0]=xx
        variance=np.ones_like(mean)
        actual=feature_fields(mean,gradient,variance)
        np.testing.assert_allclose(actual["normalized_gradient"],xx/xx.max())
        np.testing.assert_allclose(actual["plane_tilt_rad"],math.atan(math.hypot(.1,.2)),atol=1e-14)
        self.assertAlmostEqual(actual["step_height_m"][3,3],.12)
        self.assertAlmostEqual(actual["raw_traversability"][3,3],.2*.5/.5236+.4*math.atan(math.hypot(.1,.2))/.5236+.4*.12/.3)
        constant=feature_fields(np.zeros((5,5)),np.zeros((5,5,2)),np.zeros((5,5)))
        self.assertFalse(constant["uncertainty_mask"].any())
        np.testing.assert_array_equal(constant["traversability"],0.)
        shifted=feature_fields(mean+20,gradient,variance)
        np.testing.assert_allclose(actual["traversability"],shifted["traversability"],atol=1e-13)
    def test_paper_safety_filters_are_active_in_real_planner(self):
        """Analytic prediction controls use the actual paper model and planner.

        Reuse one small real GP fit; replace only its prediction arrays through
        feature_fields to isolate uncertainty and the new weighted safety map.
        No planner/model validation method is stubbed or replaced.
        """
        import torch
        path=self.root/"paper_native_safety.t25d"
        write_map(path,[[0.]*21 for _ in range(21)],spacing=(.2,.2),
                  metadata={"test_fixture":True,"purpose":"native_SGP_safety_activation"})
        search=[str(LIBRARY.parent),"C:/mingw64/bin"] if os.name=="nt" else []
        start,goal=[1.2,2.],[2.8,2.]
        mean=np.zeros((21,21));gradient=np.zeros((21,21,2));variance=np.ones((21,21))
        try:
            with TerrainBridge(LIBRARY,path,search,risk_enabled=False) as terrain:
                model=SGPPaperModel(generate_cloud(path),terrain.bounds,11)
                def fields(mu,grad,var):
                    model.mean,model.gradient,model.variance=mu.copy(),grad.copy(),var.copy()
                    model.fields=feature_fields(mu,grad,var,terrain.bounds,.2)
                    model.traversability=model.fields["traversability"]
                def solve():
                    planner=SGPStaticPlanner(terrain,model,own_cost=True)
                    events=[]
                    status=planner.solve(start,goal,11,2.,lambda *event:events.append(event),100)
                    return planner,status,events
                self.assertTrue(terrain.valid_motion(start,goal))
                fields(mean,gradient,variance)
                free,status,events=solve()
                self.assertEqual(status,"EXACT_SOLUTION")
                self.assertTrue(events)
                free_trace=[event[1:] for event in events]
                checked=Oracle(path).validate_path(events[-1][1],start,goal)
                self.assertTrue(checked["valid"],checked)
                self.assertEqual(free.counters["iterations"],100)
                # Each endpoint is rejected in turn. There is no goal/start
                # uncertainty exemption, even though canonical geometry is free.
                for endpoint in (start,goal):
                    with self.subTest(uncertain_endpoint=endpoint):
                        high=variance.copy()
                        i,j=(round(coordinate/.2) for coordinate in endpoint)
                        high[j,i]=100.
                        fields(mean,gradient,high)
                        self.assertTrue(model.fields["uncertainty_mask"][j,i])
                        planner,status,events=solve()
                        self.assertEqual(status,"INVALID_NATIVE_ENDPOINTS")
                        self.assertEqual(planner.counters["iterations"],0)
                        self.assertFalse(events)
                # Smooth height/gradient transition produces a complete native
                # traversability wall via the actual paper-weighted features.
                xx,yy=np.meshgrid(np.arange(21)*.2,np.arange(21)*.2)
                transition=np.tanh((xx-2.)/.15)
                wall_mean=.3*(1.+transition)
                wall_gradient=np.zeros_like(gradient)
                wall_gradient[:,:,0]=2.*(1.-transition*transition)
                fields(wall_mean,wall_gradient,variance)
                self.assertFalse(model.fields["uncertainty_mask"].any())
                self.assertTrue(model.valid_state(start) and model.valid_state(goal))
                crossing=SGPStaticPlanner(terrain,model,own_cost=True)
                self.assertIsNone(crossing.edge(start,goal))
                self.assertEqual(crossing.counters["common_checker_rejections"],0)
                self.assertEqual(crossing.counters["native_filter_rejections"],1)
                blocked,status,events=solve()
                self.assertEqual(status,"TIMEOUT")  # Fixed-work failure, not a proof of infeasibility.
                self.assertEqual(blocked.counters["iterations"],100)
                self.assertGreater(blocked.counters["native_filter_rejections"],0)
                self.assertFalse(events)
                # Removing only the analytic native wall restores the same-seed
                # solution, tying the rejection to the active native filter.
                fields(mean,gradient,variance)
                _,status,restored=solve()
                self.assertEqual(status,"EXACT_SOLUTION")
                self.assertEqual([event[1:] for event in restored],free_trace)
        finally:
            torch.set_flush_denormal(False)
    def test_real_rbf_gp_and_native_filter(self):
        # Real training/inference, no mock/surrogate, over a small flat fixture.
        import torch
        self.assertTrue(torch.set_flush_denormal(False))
        probe32=torch.tensor([1e-40],dtype=torch.float32)
        probe64=torch.tensor([1e-310],dtype=torch.float64)
        x,y=np.meshgrid(np.arange(.025,1.2,.05),np.arange(.025,1.2,.05))
        points=np.column_stack((x.ravel(),y.ravel(),np.zeros(x.size)))
        try:
            model=SGPPaperModel(points,(0,0,1.2,1.2),11)
            self.assertEqual(model.metadata["kernel"],"scaled_RBF_SE")
            self.assertEqual(model.metadata["training_steps"],1)
            self.assertLess(model.metadata["training_observations"],len(points))
            self.assertTrue(np.isfinite(model.mean).all())
            self.assertTrue(np.isfinite(model.gradient).all())
            self.assertTrue(model.metadata["cpu_flush_denormal"])
            policy=model.metadata["numeric_policy"]
            self.assertEqual(policy["version"],"sgp_cpu_ftz_daz_v1")
            self.assertTrue(policy["enabled"])
            self.assertEqual(policy["scope"],"dedicated_sgp_worker_cpu_float32_float64")
            self.assertEqual(policy["float32_tiny"],float(torch.finfo(torch.float32).tiny))
            self.assertEqual(policy["float64_tiny"],float(torch.finfo(torch.float64).tiny))
            # Stored operands were created before enabling FTZ/DAZ. Their actual
            # arithmetic behavior verifies application, beyond metadata alone.
            self.assertEqual(float((probe32*1.)[0]),0.)
            self.assertEqual(float((probe64*1.)[0]),0.)
            self.assertTrue(model.valid_state((.6,.6)))
            model.traversability[3,3]=.5
            self.assertFalse(model.valid_motion((.2,.6),(1.,.6)))
        finally:
            torch.set_flush_denormal(False)
    def test_required_cpu_numeric_policy_fails_closed(self):
        from sgp_static_adapter import NativeModelError
        with patch("torch.set_flush_denormal",return_value=False):
            with self.assertRaisesRegex(NativeModelError,"flush-denormal policy is unsupported"):
                SGPPaperModel(np.array([[0.,0.,0.],[1.,1.,0.]]),(0,0,1.2,1.2),11)
    def test_cabi_paper_objective_and_features(self):
        self.assertIsNotNone(LIBRARY,"--library required; native C ABI must be tested")
        path=self.map()
        search=[str(LIBRARY.parent),"C:/mingw64/bin"] if os.name=="nt" else []
        with TerrainBridge(LIBRARY,path,search,risk_enabled=False) as terrain:
            native=PutnBridge(terrain,paper=True,points=generate_cloud(path))
            try:
                self.assertEqual(terrain.timings["risk_preprocessing_s"],0.)
                events=[]
                status=native.solve([2.1,2.1],[2.7,2.1],11,1.,True,20,lambda *e:events.append(e))
                self.assertEqual(status,1)
                self.assertTrue(events)
                self.assertAlmostEqual(events[0][3],.6)
                evidence=native.path_features(events[0][1])
                self.assertTrue(all(abs(e["traversability"])<1e-12 for e in evidence))
                cost=sum(math.dist(a["center_of_mass"],b["center_of_mass"])*(1+.1*(1/(1-a["traversability"])+1/(1-b["traversability"])-2)) for a,b in zip(evidence,evidence[1:]))
                self.assertAlmostEqual(cost,events[0][3])
                self.assertGreater(native.paper_statistics["informed_proposals"],0)
            finally:native.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--library",type=Path,required=True)
    a,rest=p.parse_known_args();LIBRARY=a.library.resolve()
    unittest.main(argv=[sys.argv[0],*rest])
