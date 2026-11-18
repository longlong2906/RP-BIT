"""Independent continuous oracle: explicit triangles + half-plane clipping.

Does not call C++ geometry. Costs integrate each affine triangle over its clipped
parameter interval. Collision uses a piecewise quadratic point/AABB distance
minimized over the whole segment, rather than C++ segment/rectangle predicates.
"""
import math
import time
from fractions import Fraction
from pathlib import Path
import numpy as np


class Oracle:
    def __init__(self, filename):
        tokens = iter(Path(filename).read_text(encoding="ascii").split())
        def expect(word):
            if next(tokens) != word:
                raise ValueError("expected " + word)
        expect("T25D"); expect("1"); expect("units"); expect("m")
        expect("diagonal"); expect("SW_NE"); expect("grid")
        self.nx, self.ny = int(next(tokens)), int(next(tokens))
        self.x0, self.y0, self.dx, self.dy = (float(next(tokens)) for _ in range(4))
        expect("model")
        self.radius, self.margin, self.slope = (float(next(tokens)) for _ in range(3))
        expect("heights")
        values = [next(tokens) for _ in range(self.nx*self.ny)]
        self.heights = np.array([float(x) if x != "nodata" else np.nan for x in values]).reshape(self.ny, self.nx)
        expect("mask")
        self.mask = np.array([int(next(tokens)) for _ in range((self.nx-1)*(self.ny-1))]).reshape(self.ny-1,self.nx-1)
        expect("rectangles")
        self.rectangles = [tuple(float(next(tokens)) for _ in range(4)) for _ in range(int(next(tokens)))]
        expect("end")
        if list(tokens):
            raise ValueError("extra input")
        self.bounds = (self.x0, self.y0, self.x0+(self.nx-1)*self.dx, self.y0+(self.ny-1)*self.dy)
        self.triangles = []
        self.cell_triangle_ids = np.full((self.ny-1,self.nx-1,2),-1,dtype=np.int64)
        self.forbidden = list(self.rectangles)
        for j in range(self.ny-1):
            for i in range(self.nx-1):
                corners = [(i,j),(i+1,j),(i+1,j+1),(i,j+1)]
                xyz = np.array([(self.x0+u*self.dx,self.y0+v*self.dy,self.heights[v,u]) for u,v in corners])
                missing = not np.isfinite(xyz).all()
                forbidden = missing or bool(self.mask[j,i])
                if not missing:
                    for tri_index,ids in enumerate(([0,1,2],[0,2,3])):
                        tri = xyz[ids]
                        # Solve from vertices, independently of the C++ interpolation formula.
                        coeff = np.linalg.solve(np.column_stack((tri[:,:2],np.ones(3))),tri[:,2])
                        forbidden |= math.degrees(math.atan(math.hypot(*coeff[:2]))) > self.slope
                        self.cell_triangle_ids[j,i,tri_index]=len(self.triangles)
                        self.triangles.append((tri[:,:2], coeff))
                if forbidden:
                    self.forbidden.append((xyz[0,0],xyz[0,1],xyz[2,0],xyz[2,1]))
        self.vertices = np.array([x[0] for x in self.triangles])
        self.coefficients = np.array([x[1] for x in self.triangles])
        self.triangle_slopes=(np.degrees(np.arctan(np.hypot(self.coefficients[:,0],self.coefficients[:,1])))
                              if len(self.coefficients) else np.array([]))
        self.rect_array = np.array(self.forbidden).reshape(-1,4)

    @staticmethod
    def segment_rect_distance(a,b,rect):
        a,b = np.asarray(a),np.asarray(b)
        d=b-a
        lo=np.array(rect[:2]); hi=np.array(rect[2:])
        cuts=[0.,1.]
        for k in (0,1):
            if d[k] != 0:
                cuts += [float(t) for t in ((lo[k]-a[k])/d[k],(hi[k]-a[k])/d[k]) if 0 < t < 1]
        cuts=sorted(set(cuts))
        best=math.inf
        for left,right in zip(cuts,cuts[1:]):
            mid=a+d*((left+right)/2)
            active=(mid < lo) | (mid > hi)
            target=np.where(mid < lo,lo,hi)
            aa=d[active]; bb=(a-target)[active]
            denom=float(aa@aa)
            t=left if denom == 0 else min(right,max(left,-float(aa@bb)/denom))
            q=a+t*d
            delta=np.maximum(np.maximum(lo-q,q-hi),0)
            best=min(best,float(np.linalg.norm(delta)))
        return best

    def valid_edge(self,a,b):
        a,b=np.asarray(a),np.asarray(b)
        if not np.isfinite([a,b]).all():
            return False
        r=self.radius+self.margin
        lo=np.minimum(a,b);hi=np.maximum(a,b)
        if np.any(lo-r <= self.bounds[:2]) or np.any(hi+r >= self.bounds[2:]):
            return False
        if len(self.rect_array):
            candidate=self.rect_array[(self.rect_array[:,0] <= hi[0]+r) &
                (self.rect_array[:,2] >= lo[0]-r) & (self.rect_array[:,1] <= hi[1]+r) &
                (self.rect_array[:,3] >= lo[1]-r)]
            for rect in candidate:
                if self.segment_rect_distance(a,b,rect) <= r + 1e-13:
                    return False
        return True

    @staticmethod
    def _exact_triangle_interval(a,b,triangle):
        """Exact binary-rational clipping for numerically uncertain predicates.

        Fractions represent the actual input binary64 coordinates, not rounded
        decimal coordinates. Intersections use endpoint orientations so an end
        exactly on a triangle edge cannot become a spurious early crossing.
        """
        aa=[Fraction.from_float(float(value)) for value in a]
        bb=[Fraction.from_float(float(value)) for value in b]
        vertices=[[Fraction.from_float(float(value)) for value in point] for point in triangle]
        low,high=Fraction(0),Fraction(1)
        for k in range(3):
            p,q=vertices[k],vertices[(k+1)%3]
            ex,ey=q[0]-p[0],q[1]-p[1]
            start=ex*(aa[1]-p[1])-ey*(aa[0]-p[0])
            end=ex*(bb[1]-p[1])-ey*(bb[0]-p[0])
            if start<0 and end<0:
                return None
            if start<0:
                low=max(low,start/(start-end))
            elif end<0:
                high=min(high,start/(start-end))
            if high<low:
                return None
        return float(low),float(high)

    def intervals(self,a,b,*,use_index=True):
        """Clip a segment against every explicit triangle; handles shared-edge overlap."""
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if not np.isfinite([a,b]).all():
            raise ValueError("nonfinite path")
        # Spatial broad phase only. The independent half-plane clip and affine
        # coefficient solve below remain the oracle, including shared boundaries.
        if use_index:
            lower=(np.minimum(a,b)-[self.x0,self.y0])/[self.dx,self.dy]
            upper=(np.maximum(a,b)-[self.x0,self.y0])/[self.dx,self.dy]
            i0=max(0,math.floor(lower[0]-1e-9));j0=max(0,math.floor(lower[1]-1e-9))
            i1=min(self.nx-2,math.floor(upper[0]+1e-9));j1=min(self.ny-2,math.floor(upper[1]+1e-9))
            candidates=self.cell_triangle_ids[j0:j1+1,i0:i1+1].reshape(-1)
            candidates=candidates[candidates>=0]
        else:candidates=np.arange(len(self.vertices))
        v=self.vertices[candidates]
        low=np.zeros(len(v));high=np.ones(len(v));possible=np.ones(len(v),dtype=bool)
        uncertain=np.zeros(len(v),dtype=bool)
        for k in range(3):
            p=v[:,k]; e=v[:,(k+1)%3]-p
            left_start=e[:,0]*(a[1]-p[:,1]);right_start=e[:,1]*(a[0]-p[:,0])
            left_end=e[:,0]*(b[1]-p[:,1]);right_end=e[:,1]*(b[0]-p[:,0])
            start=left_start-right_start;end=left_end-right_end
            # This bound only selects exact computation; it never expands a
            # triangle or fills a missing interval. Ordinary non-cancelling
            # endpoint signs retain the vectorized fast path.
            start_scale=np.abs(left_start)+np.abs(right_start)
            end_scale=np.abs(left_end)+np.abs(right_end)
            uncertain |= ((np.abs(start)<=16*np.finfo(float).eps*start_scale) |
                          (np.abs(end)<=16*np.finfo(float).eps*end_scale) |
                          (start_scale<np.finfo(float).tiny) | (end_scale<np.finfo(float).tiny) |
                          ~np.isfinite(start) | ~np.isfinite(end))
            outside_start=start<0;outside_end=end<0
            possible &= ~(outside_start & outside_end)
            entering=outside_start & ~outside_end
            leaving=~outside_start & outside_end
            low[entering]=np.maximum(low[entering],start[entering]/(start[entering]-end[entering]))
            high[leaving]=np.minimum(high[leaving],start[leaving]/(start[leaving]-end[leaving]))
        for k in np.flatnonzero(uncertain):
            exact=self._exact_triangle_interval(a,b,v[k])
            possible[k]=exact is not None
            if exact is not None:
                low[k],high[k]=exact
        ids=np.flatnonzero(possible & (high >= low) & (high >= 0) & (low <= 1))
        return [(float(max(0,low[k])),float(min(1,high[k])),self.coefficients[candidates[k]]) for k in ids]

    def surface(self,a,b):
        a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
        if np.any(a<self.bounds[:2]) or np.any(b<self.bounds[:2]) or np.any(a>self.bounds[2:]) or np.any(b>self.bounds[2:]):
            raise ValueError("outside map")
        intervals=self.intervals(a,b)
        cuts=sorted(set([0.,1.]+[t for left,right,_ in intervals for t in (left,right)]))
        cost=0.;points=[]
        for left,right in zip(cuts,cuts[1:]):
            if right-left < 1e-14:
                continue
            mid=(left+right)/2
            coeff=next((c for low,high,c in intervals if low-1e-13 <= mid <= high+1e-13),None)
            if coeff is None:
                raise ValueError("NoData gap")
            p=a+(b-a)*left;q=a+(b-a)*right
            zp=float(coeff[:2]@p+coeff[2]);zq=float(coeff[:2]@q+coeff[2])
            cost+=math.hypot(float(np.linalg.norm(q-p)),zq-zp)
            if not points:
                points.append([*p,zp])
            points.append([*q,zq])
        if not points:
            coeff=next((c for l,h,c in intervals),None)
            if coeff is None:
                raise ValueError("NoData point")
            points=[[float(a[0]),float(a[1]),float(coeff[:2]@a+coeff[2])]]
        return cost,points

    @staticmethod
    def _inside_triangle(point, triangle):
        signs=[np.cross(triangle[(k+1)%3]-triangle[k],np.asarray(point)-triangle[k]) for k in range(3)]
        return min(signs)>=-1e-13 or max(signs)<=1e-13

    @classmethod
    def triangle_rect_distance(cls, triangle, rect):
        corners=[(rect[0],rect[1]),(rect[2],rect[1]),(rect[2],rect[3]),(rect[0],rect[3])]
        if any(cls._inside_triangle(c,triangle) for c in corners):return 0.
        return min(cls.segment_rect_distance(triangle[k],triangle[(k+1)%3],rect) for k in range(3))

    @staticmethod
    def _rect_distance(a,b):
        return math.hypot(max(a[0]-b[2],b[0]-a[2],0.),max(a[1]-b[3],b[1]-a[3],0.))

    @staticmethod
    def _triangles_rect_distances(triangles,rect):
        """Vectorized convex-polygon distance, independent of C++ traversal."""
        lo=np.asarray(rect[:2]);hi=np.asarray(rect[2:])
        delta=np.maximum(np.maximum(lo-triangles,triangles-hi),0.)
        best=np.sqrt(np.sum(delta*delta,axis=2)).min(axis=1)
        corners=np.asarray([[lo[0],lo[1]],[hi[0],lo[1]],[hi[0],hi[1]],[lo[0],hi[1]]])
        inside=np.ones((len(triangles),4),dtype=bool)
        for k in range(3):
            a=triangles[:,k];d=triangles[:,(k+1)%3]-a
            offset=corners[None,:,:]-a[:,None,:]
            inside &= d[:,None,0]*offset[:,:,1]-d[:,None,1]*offset[:,:,0]>=-1e-13
            length2=np.sum(d*d,axis=1)
            t=np.clip(np.sum(offset*d[:,None,:],axis=2)/length2[:,None],0.,1.)
            residual=offset-t[:,:,None]*d[:,None,:]
            best=np.minimum(best,np.sqrt(np.sum(residual*residual,axis=2)).min(axis=1))
            lower=np.zeros(len(a));upper=np.ones(len(a));possible=np.ones(len(a),dtype=bool)
            for axis in (0,1):
                nonzero=d[:,axis]!=0
                possible &= nonzero | ((a[:,axis]>=lo[axis]) & (a[:,axis]<=hi[axis]))
                den=np.where(nonzero,d[:,axis],1.)
                t0=(lo[axis]-a[:,axis])/den;t1=(hi[axis]-a[:,axis])/den
                lower=np.maximum(lower,np.where(nonzero,np.minimum(t0,t1),-np.inf))
                upper=np.minimum(upper,np.where(nonzero,np.maximum(t0,t1),np.inf))
            best=np.where(possible & (lower<=upper),0.,best)
        return np.where(inside.any(axis=1),0.,best)

    @staticmethod
    def _traversability_config(config):
        keys=("lambda","slope_weight","clearance_weight","clearance_comfort_m")
        if config.get("feature_version")!="cell_envelope_v1":raise ValueError("unsupported traversability feature version")
        values=tuple(float(config[k]) for k in keys)
        if not all(math.isfinite(v) and v>=0 for v in values) or values[3]<=0:
            raise ValueError("invalid traversability parameters")
        if values[1]+values[2]!=1.:raise ValueError("risk weights must sum to one")
        return values

    def traversability_features(self, config):
        """Reconstruct fixed cell envelopes from raw triangles, never C++ output."""
        parameters=self._traversability_config(config)
        cache=getattr(self,"_traversability_cache",{})
        feature_key=(config["feature_version"],parameters[3])
        if feature_key in cache:return cache[feature_key]
        _,_,_,comfort=parameters;r=self.radius+self.margin;cells=[]
        triangle_slopes=self.triangle_slopes
        for j in range(self.ny-1):
            for i in range(self.nx-1):
                rect=(self.x0+i*self.dx,self.y0+j*self.dy,self.x0+(i+1)*self.dx,self.y0+(j+1)*self.dy)
                available=bool(np.all(self.cell_triangle_ids[j,i]>=0));alpha=0.
                i0=max(0,i-math.ceil(r/self.dx)-1);i1=min(self.nx-2,i+math.ceil(r/self.dx)+1)
                j0=max(0,j-math.ceil(r/self.dy)-1);j1=min(self.ny-2,j+math.ceil(r/self.dy)+1)
                ids=self.cell_triangle_ids[j0:j1+1,i0:i1+1].reshape(-1)
                ids=ids[ids>=0]
                distances=self._triangles_rect_distances(self.vertices[ids],rect) if len(ids) else np.array([])
                matching=ids[distances<=r+1e-13];found=len(matching)>0
                if found:alpha=float(triangle_slopes[matching].max())
                available=available and found
                slope_penalty=min(1.,max(0.,alpha/self.slope))**2 if self.slope>0 else (1. if alpha>0 else 0.)
                clearance=min((self._rect_distance(rect,o)-r for o in self.rectangles),default=math.inf)
                clearance_penalty=max(0.,1.-max(0.,clearance)/comfort)**2
                cells.append(dict(available=available,slope_penalty=slope_penalty,
                                  clearance_penalty=clearance_penalty,maximum_slope_deg=alpha))
        result={"nx_cells":self.nx-1,"ny_cells":self.ny-1,"cells":cells}
        cache[feature_key]=result;self._traversability_cache=cache
        return result

    def _point_cells(self,q,boundary_axes=(True,True)):
        indices=[]
        for value,origin,spacing,count,constant in zip(q,(self.x0,self.y0),(self.dx,self.dy),(self.nx,self.ny),boundary_axes):
            u=(value-origin)/spacing;k=math.floor(u)
            candidates=[k]
            nearest=int(round(u))
            if constant and value==origin+nearest*spacing:candidates=[nearest-1,nearest]
            indices.append([v for v in candidates if 0<=v<count-1])
        return [(i,j) for j in indices[1] for i in indices[0]]

    def traversability_edge(self,a,b,config):
        lam,sw,cw,_=self._traversability_config(config)
        fields=self.traversability_features(config)
        boundary_axes=(a[0]==b[0],a[1]==b[1])
        try:
            length,points=self.surface(a,b)
            slope_parts=[];clearance_parts=[]
            for p,q in zip(points,points[1:]):
                midpoint=(np.asarray(p[:2])+q[:2])/2
                cells=[fields["cells"][j*(self.nx-1)+i] for i,j in self._point_cells(midpoint,boundary_axes)]
                cells=[c for c in cells if c["available"]]
                if not cells:raise ValueError("unavailable_traversability_cell")
                ell=math.dist(p,q)
                slope_parts.append(ell*max(c["slope_penalty"] for c in cells))
                clearance_parts.append(ell*max(c["clearance_penalty"] for c in cells))
            if not len(points)>1:
                cells=[fields["cells"][j*(self.nx-1)+i] for i,j in self._point_cells(a)]
                if not any(c["available"] for c in cells):raise ValueError("unavailable_traversability_cell")
            rs=math.fsum(slope_parts);rc=math.fsum(clearance_parts);risk=sw*rs+cw*rc
            return dict(has_data=True,cost_terrain_m=length,risk_slope_integral_m=rs,
                        risk_clearance_integral_m=rc,risk_integral_m=risk,objective_value=length+lam*risk)
        except ValueError as error:
            return dict(has_data=False,reason=str(error),**{k:None for k in
                ("cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m","objective_value")})

    def _swept_slope(self,a,b):
        # Capsule/triangle overlap via endpoint inclusion, edge/segment distances.
        # The radius-zero case uses the same closed-set convention as hard validity.
        r=self.radius+self.margin;maximum=0.
        lo=np.minimum(a,b)-r;hi=np.maximum(a,b)+r
        lower=(lo-[self.x0,self.y0])/[self.dx,self.dy]
        upper=(hi-[self.x0,self.y0])/[self.dx,self.dy]
        i0=max(0,math.floor(lower[0]-1e-9));j0=max(0,math.floor(lower[1]-1e-9))
        i1=min(self.nx-2,math.floor(upper[0]+1e-9));j1=min(self.ny-2,math.floor(upper[1]+1e-9))
        ids=self.cell_triangle_ids[j0:j1+1,i0:i1+1].reshape(-1);ids=ids[ids>=0]
        for tid in ids:
            triangle=self.vertices[tid]
            if self._inside_triangle(a,triangle) or self._inside_triangle(b,triangle):distance=0.
            else:
                distance=min(self._segment_distance(a,b,triangle[k],triangle[(k+1)%3]) for k in range(3))
            if distance<=r+1e-13:maximum=max(maximum,float(self.triangle_slopes[tid]))
        return maximum

    @staticmethod
    def _segment_distance(a,b,c,d):
        a,b,c,d=map(lambda p:np.asarray(p,dtype=float),(a,b,c,d))
        u=b-a;v=d-c;cross=lambda p,q:float(p[0]*q[1]-p[1]*q[0])
        denominator=cross(u,v)
        if denominator!=0:
            t=cross(c-a,v)/denominator;s=cross(c-a,u)/denominator
            if 0<=t<=1 and 0<=s<=1:return 0.
        def point_segment(p,x,y):
            w=y-x;den=float(w@w);t=0. if den==0 else min(1.,max(0.,float((p-x)@w)/den))
            return float(np.linalg.norm(p-x-t*w))
        return min(point_segment(a,c,d),point_segment(b,c,d),point_segment(c,a,b),point_segment(d,a,b))

    def validate_path(self,xy,start,goal,claimed_cost=None,surface=None,tol=1e-8,*,traversability=None,claimed_metrics=None,
                      geometric_safety_metrics=False,clearance_metrics=True):
        """Validate the full swept footprint; optional clearance is reporting only.

        Schema 12 disables clearance measurements without weakening valid_edge,
        swept-slope checks or the independently integrated native A3-T objective.
        The default preserves the historical verifier output.
        """
        began=time.perf_counter()
        errors=[];cost=0.;xy_length=0.;clearance=math.inf;max_slope=0.
        if not xy:
            return {"valid":False,"errors":["missing_path"],"cost_terrain_m":None}
        if np.linalg.norm(np.array(xy[0])-start)>1e-9 or np.linalg.norm(np.array(xy[-1])-goal)>1e-9:
            errors.append("not_exact_start_goal")
        pairs=list(zip(xy,xy[1:])) or [(xy[0],xy[0])]
        expected=[]
        for a,b in pairs:
            if not self.valid_edge(a,b):
                errors.append("forbidden_capsule")
            try:
                value,points=self.surface(a,b)
                cost+=value; expected+=points if not expected else points[1:]
                for low,high,coeff in self.intervals(a,b):
                    if high-low>1e-14:
                        max_slope=max(max_slope,math.degrees(math.atan(math.hypot(*coeff[:2]))))
            except ValueError as error:
                errors.append(str(error))
            xy_length+=math.dist(a,b)
            if clearance_metrics:
                low=np.minimum(a,b);high=np.maximum(a,b)
                edge_clearance=min(*(low-np.array(self.bounds[:2])),*(np.array(self.bounds[2:])-high))
                # Full continuous minimum, including all forbidden cells/rectangles.
                for rect in self.forbidden:edge_clearance=min(edge_clearance,self.segment_rect_distance(a,b,rect))
                clearance=min(clearance,edge_clearance-self.radius-self.margin)
        if claimed_cost is not None and abs(cost-claimed_cost)>tol+tol*max(cost,claimed_cost):
            errors.append("cost_mismatch")
        if surface is not None:
            # Check every exported point AND every oracle triangle break. Extra collinear
            # points are allowed; omission of a genuine change in slope is rejected.
            if not surface or not np.isfinite(surface).all():
                errors.append("missing_or_nonfinite_surface")
            else:
                surface_array=np.asarray(surface)
                for point in expected:
                    if not np.any(np.linalg.norm(surface_array-point,axis=1)<=tol):
                        errors.append("missing_triangle_breakpoint");break
                export_length=sum(math.dist(p,q) for p,q in zip(surface,surface[1:]))
                if abs(export_length-cost)>tol+tol*cost:
                    errors.append("surface_length_mismatch")
                for p in surface:
                    try:
                        _,lift=self.surface(p[:2],p[:2])
                        if abs(lift[0][2]-p[2])>tol: errors.append("off_surface_point");break
                    except ValueError:
                        errors.append("exported_nodata");break
        result={"valid":not errors,"errors":sorted(set(errors)),"cost_terrain_m":cost,
                "length_xy_m":xy_length,"cost_abs_error_m":None if claimed_cost is None else abs(cost-claimed_cost),
                "maximum_slope_deg":max_slope,
                "validation_wall_s":time.perf_counter()-began,
                "method":"explicit_triangle_halfplane_clipping_and_quadratic_capsule_distance"}
        if clearance_metrics:result["minimum_footprint_clearance_m"]=clearance
        if traversability is not None:
            metrics=[self.traversability_edge(a,b,traversability) for a,b in pairs]
            if not all(m["has_data"] for m in metrics):errors.append("unavailable_traversability_path")
            else:
                for key in ("objective_value","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m"):
                    result[key]=math.fsum(m[key] for m in metrics)
                for key in ("objective_value","cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m","risk_integral_m","length_xy_m"):
                    claimed=(claimed_metrics or {}).get(key)
                    if claimed_metrics is not None and (not isinstance(claimed,(int,float)) or not math.isfinite(claimed) or
                        abs(result[key]-claimed)>tol+tol*max(abs(result[key]),abs(claimed))):errors.append(key+"_mismatch")
        if traversability is not None or geometric_safety_metrics:
            if clearance_metrics:
                obstacle_clearance=min((self.segment_rect_distance(a,b,rect)-self.radius-self.margin
                                       for a,b in pairs for rect in self.rectangles),default=math.inf)
                result["minimum_obstacle_footprint_clearance_m"]=obstacle_clearance if math.isfinite(obstacle_clearance) else None
                result["obstacle_clearance_status"]="MEASURED" if self.rectangles else "NO_EXPLICIT_OBSTACLES"
            result["maximum_swept_footprint_slope_deg"]=max(self._swept_slope(a,b) for a,b in pairs)
            result["valid"]=not errors;result["errors"]=sorted(set(errors))
            result["validation_wall_s"]=time.perf_counter()-began
        return result
