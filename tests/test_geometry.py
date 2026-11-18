import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from terrain_io import write_map
from independent_verifier import Oracle


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--cli",required=True);args=parser.parse_args()
    rng=random.Random(42017)
    with tempfile.TemporaryDirectory(prefix="terrain_geometry_") as directory:
        path=Path(directory)/"map.t25d"
        count=0; collision_count=0
        modes=((),("--scalar-cost",),("--collision-index",),("--collision-index","--scalar-cost"))
        for case in ("flat","tilted","folded","random","nodata","thin","narrow","masked","clustered","corner","contact"):
            heights=[[0. for i in range(9)] for j in range(8)]
            if case=="tilted": heights=[[.2*i-.3*j for i in range(9)] for j in range(8)]
            if case=="folded": heights=[[2-abs(i-4)*.5 for i in range(9)] for j in range(8)]
            if case=="random": heights=[[rng.uniform(-.4,.4) for i in range(9)] for j in range(8)]
            if case=="nodata": heights[4][4]=None
            rects={"thin":[(3.999,0,4.001,7)],"narrow":[(0,0,8,3.35),(0,3.65,8,7)],
                   "clustered":[(i,j,i+.05,j+.05) for i in range(1,7) for j in range(1,6)],
                   "corner":[(4,3,5,4)],"contact":[(0,0,8,3.38),(0,3.62,8,7)]}.get(case,[])
            mask=[[int(case=="masked" and (i+j)%3==0) for i in range(8)] for j in range(7)]
            write_map(path,heights,radius=.1,margin=.02,slope=35,rectangles=rects,mask=mask)
            oracle=Oracle(path)
            edges=[[[rng.uniform(.2,7.8),rng.uniform(.2,6.8)] for _ in (0,1)] for _ in range(70)]
            edges += [[[1,1],[7,6]],[[1,3.5],[7,3.5]],[[2,2],[2,2]],[[1,1],[6,6]],[[1,2],[6,2]],[[0,1],[7,2]]]
            if case=="corner":
                offset=.12/math.sqrt(2)
                for delta in (-1e-7,0,1e-7):
                    q=[4-offset-delta,3-offset-delta]
                    edges += [[q,q],[[q[0]-.5,q[1]+.5],[q[0]+.5,q[1]-.5]]]
            # Explicit reverse/split cases supplement independent checks with
            # invariance checks and include long diagonal, grid-line and vertex paths.
            for a,b in list(edges[-6:]):
                mid=[.37*a[k]+.63*b[k] for k in (0,1)]
                edges += [[b,a],[a,mid],[mid,b]]
            references=[]
            for a,b in edges:
                valid=oracle.valid_edge(a,b)
                try: cost,points=oracle.surface(a,b)
                except ValueError: cost=None
                references.append((valid,cost))
            baseline=None
            for mode in modes:
                run=subprocess.run([args.cli,str(path),*mode],input="".join(" ".join(map(str,a+b))+"\n" for a,b in edges),capture_output=True,text=True)
                assert run.returncode==0,run.stderr
                records=[json.loads(line) for line in run.stdout.splitlines()]
                assert len(records)==len(edges)
                if baseline is None:baseline=records
                for (a,b),record,original,(valid,cost) in zip(edges,records,baseline,references):
                    assert record["collision_index_enabled"]==("--collision-index" in mode)
                    assert record["scalar_cost"]==("--scalar-cost" in mode)
                    assert record["valid"]==valid,(case,mode,a,b,record)
                    assert all(record[key]==original[key] for key in ("valid","cost","faces","reason","surface")),(case,mode,a,b,record,original)
                    collision_count+=1
                    if cost is None:
                        assert record["cost"] is None,(case,mode,record)
                        continue
                    # C++ deliberately forbids a whole NoData cell even when the traversed
                    # triangle has three known vertices. Oracle follows that map contract.
                    assert record["cost"] is not None,(case,mode,record)
                    assert abs(cost-record["cost"])<=1e-8+1e-8*cost,(case,mode,cost,record)
                    if record["valid"]:
                        validation=oracle.validate_path([a,b],a,b,cost,record["surface"])
                        assert validation["valid"],(case,mode,a,b,validation)
                    count+=1
        original=path.read_text()
        for changed in (original.replace("units m","units ft"), original.replace("grid 9 8 0 0 1 1","grid 9 8 0 0 0 1"),original+"extra\n",original.replace("heights\n0", "heights\ninf",1)):
            path.write_text(changed)
            result=subprocess.run([args.cli,str(path)],input="",capture_output=True,text=True)
            assert result.returncode!=0,"invalid input accepted"
        print(f"PASS: {collision_count} independent continuous collision comparisons; {count} surface comparisons; scan/BVH x polyline/scalar agreement; invalid loader inputs")

if __name__=="__main__":main()
