# SGP static adapter provenance and semantic changes

Base: `abeleinin/gp-navigation@597bfb7dafac686248deae8a3b6897108dc2bd9c`.
The unmodified relevant source files and BSD-3-Clause license are retained under
`external/sgp/597bfb7dafac686248deae8a3b6897108dc2bd9c/`. The headless extraction is
`tools/sgp_static_adapter.py`; this manifest documents the patch, rather than
claiming an unmodified upstream build or paper reproduction.

| Component | Retained | Explicit adaptation / reason |
|---|---|---|
| GP | ExactGP, ConstantMean, ScaleKernel(RQKernel), InducingPointKernel, GaussianLikelihood; source constructor arguments; one Adam step, lr .01; sorted X, four decimal input rounding, float32 | CPU only, seed/thread control; global finite raw-H nodes, no radius-5 crop; guard inducing stride by max(1,floor(N/500)) for small fixtures |
| GP prediction | Mean, likelihood variance, autograd mean gradient | Query canonical grid in blocks of 1024 to bound global inference memory; same trained GP and no surrogate. Variance gradient was unused by planning and is not computed. |
| Coordinates | Physical XY in metres | Canonical bounds, row Y/column X; no sensor offsets (-.2,-.4,+.8), no query-frame transforms, no output coordinate rounding |
| Slope | Gradient magnitude | Read gradient, not the elevation column; atan(norm(gradient)) in radians |
| Plane/relief | Least-squares plane; local height feature | One-metre-wide centered physical window; correct clipped-neighborhood coordinates. Relief=max abs(neighbor-center), not a discontinuous step. |
| Traversability | Weights .2/.5/.3; critical .5236/.5236/.30; limit .3 | Physical-threshold sum clipped to [0,1], no per-map min-max normalization. Uncertainty >=1.4 mean masks every node, including the former exempt region. |
| RRT* | Uniform proposal, XY nearest, .5m steer, 1m neighbor radius; XY own-cost | Common objective is C++ J. Recompute parent-chain costs after rewiring and forbid cycles. Continue improving within budget rather than return at first local subgoal. |
| Native filtering | All values in 3x3 neighborhood <=.3 and complete stencil | Apply along each nearest-grid region crossed by every connector, including terminal connector; conservative extension of source endpoint filter |
| Hard safety | Additional common requirement | C++ continuous swept-footprint check before every accepted parent, rewire and goal connection. It is independent of GP interpolation and never waived. |
| Goal | Singleton XY | Full terminal connector to exact goal; no frontier/local-sensing/controller behavior. Preserve raw trajectory and canonical XY separately. |
| Runtime | Real Torch/GPyTorch | Missing dependencies/model failures retained. No surrogate GP, synthetic result or fallback planner. |

Explicit obstacles are handled by the common continuous checker; GP elevation
training uses the raw ground observations, not obstacle-top heights. Obstacle
height 2m is representation metadata, not a fabricated terrain observation.

This adapter is labeled `SGP-RRT*-static-adapted`. The common checker controls hard
safety, while the native model can reject more of the domain. Neither optimum
nor asymptotic optimality nor unchanged paper behavior is claimed. Numeric GP
prediction and chunk-invariance checks are tests, not an all-roundoff proof.
