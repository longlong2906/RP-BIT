"""Synthetic exact-clipping regressions; no scientific planner observations."""
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"tools"))
from independent_verifier import Oracle
from native_paper_oracle import NativePaperOracle
from terrain_io import write_map


EDGES=(
    ([13.5,5.1000000000000005],[13.600000000000001,5.2]),
    ([14.5,6.1000000000000005],[14.600000000000001,6.2]),
    ([15.5,7.1000000000000005],[15.600000000000001,7.2]),
    ([16.5,8.1],[16.6,8.200000000000001]),
    ([17.5,9.1],[17.6,9.200000000000001]))


class ClippingRoundoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="oracle_clipping_synthetic_",dir=ROOT/"tests/fixtures")
        self.addCleanup(self.tmp.cleanup)
        self.directory=Path(self.tmp.name)

    def flat(self,origin=(0.,0.)):
        path=self.directory/("flat_"+str(origin[0])+".t25d")
        write_map(path,[[0.]*101 for _ in range(101)],spacing=(.2,.2),origin=origin,
                  metadata=dict(synthetic_test_fixture=True))
        return path

    def assert_cost(self,oracle,a,b):
        cost,_=oracle.surface(a,b)
        self.assertAlmostEqual(cost,math.dist(a,b),places=12)
        intervals=oracle.intervals(a,b)
        self.assertTrue(any(low==0. and high==1. for low,high,_ in intervals),intervals)
        self.assertTrue(oracle.valid_edge(a,b))

    def test_diagonal_vertex_endpoints_cover_full_edge_forward_reverse_split(self):
        path=self.flat()
        for oracle in (Oracle(path),NativePaperOracle(path)):
            for a,b in EDGES:
                with self.subTest(oracle=type(oracle).__name__,a=a,b=b):
                    self.assert_cost(oracle,a,b)
                    self.assert_cost(oracle,b,a)
                    middle=[(left+right)/2 for left,right in zip(a,b)]
                    total=oracle.surface(a,middle)[0]+oracle.surface(middle,b)[0]
                    self.assertAlmostEqual(total,math.dist(a,b),places=12)
                    broad=oracle.intervals(a,b,use_index=False)
                    self.assertTrue(any(low==0. and high==1. for low,high,_ in broad))

    def test_translated_grid_and_adjacent_ulp_diagonal_sides(self):
        origin=(1048576.,-1048576.)
        oracle=Oracle(self.flat(origin))
        for a,b in EDGES:
            translated_a=[v+offset for v,offset in zip(a,origin)]
            translated_b=[v+offset for v,offset in zip(b,origin)]
            self.assertAlmostEqual(oracle.surface(translated_a,translated_b)[0],
                                   math.dist(translated_a,translated_b),places=11)
        small=Oracle(self.flat())
        for direction in (-math.inf,math.inf):
            a=[13.5,math.nextafter(5.1000000000000005,direction)]
            b=[13.600000000000001,math.nextafter(5.2,direction)]
            self.assertAlmostEqual(small.surface(a,b)[0],math.dist(a,b),places=12)

    def test_true_nodata_gap_still_fails(self):
        heights=[[0.]*11 for _ in range(11)]
        heights[5][5]=None
        path=self.directory/"actual_gap.t25d"
        write_map(path,heights,spacing=(.2,.2),radius=.01,margin=0.,metadata=dict(synthetic_test_fixture=True))
        for oracle in (Oracle(path),NativePaperOracle(path)):
            with self.assertRaisesRegex(ValueError,"NoData gap"):
                oracle.surface([.5,1.],[1.5,1.])
            self.assertFalse(oracle.valid_edge([.5,1.],[1.5,1.]))

    def test_exact_clip_rejects_outside_without_parallel_tolerance(self):
        triangle=[[0.,0.],[1.,0.],[1.,1.]]
        tiny=math.nextafter(0.,-math.inf)
        self.assertIsNone(Oracle._exact_triangle_interval([.2,tiny],[.8,tiny],triangle))
        self.assertEqual(Oracle._exact_triangle_interval([.2,0.],[.8,0.],triangle),(0.,1.))


if __name__=="__main__":
    unittest.main()
