# PUTN/PF-RRT* static adapter — controlled port

Base: `jianzhuozhuthu/putn` commit `a12c900d0965a7a7ea3d4bb7408f8ccf3afdafad`.
Unmodified input files and SHA-256 are recorded in `external/putn/PROVENANCE.json`.
The upstream GNU GPL v3 license is retained in `external/putn/upstream/LICENSE`.
The derived `PutnStatic.hpp` / `PutnStatic.cpp` are GPL-3.0-only. Distributed
bundles must include the complete corresponding sources, this notice and license.

This is an explicitly named controlled port, **not** a claim of an unmodified
paper reproduction. It extracts the global PF-RRT* search and local voxel plane
model; ROS visualization, rolling tree inheritance, local sensing, GPR/NMPC are
outside this static global benchmark. No generic RRT* implementation is relabeled.

| Source method / behavior | Port behavior |
|---|---|
| `Plane::Plane`: occupied-voxel neighborhood, local SVD, fourth-power flatness, slope, vacancy sparsity | Retained, with initialized centroid, finite rank guard and normal oriented +Z. Source column-centering covariance trace reduced algebraically to avoid its quadratic allocation. |
| `World`: 0.1 m occupancy, voxel centers, sample-based body check | Sparse voxel storage; deterministic 0.2 m ground cloud. Explicit rectangles become solid 2 m extrusions. Original 20-step, 5×5 COM-body occupancy check retained as an additional native filter. |
| `project2surface`: first occupied voxel in the XY column | Static-known-map adaptation uses exact `H(q)` to preserve the singleton support-center XY. Sparse 0.2 m cloud with 0.1 m voxels otherwise creates empty columns even on flat ground. No endpoint snapping or NoData interpolation. |
| `fitPlane`: normal and elevated COM | Retained with `COM = (q,H(q)) + 0.7 normal`; canonical output uses `Plane.init_coord` / support-center XY and separately exports raw COM XYZ. |
| `findNearest`, `steer`, `findNearNeighbors` | Global benchmark contract uses Euclidean XY at support centers. Source Manhattan-COM nearest and COM-distance neighbors are therefore declared adaptations. Fixed radius 1 m and steer step 0.4 m. |
| `findParent`, `reWire`, `updateChildrenCost`, `closeCheck`, `generatePath` | Same parent-selection / neighbor rewiring / descendant update / goal-neighbor search. Initialized node state, cycle guards, consistent double costs, every improving incumbent callback. Terminal goal connector is fully checked. |
| `calCostBetweenTwoNode` | Own-cost keeps COM distance × `[1 + 0.1(1/(1.0001-tau_u) + 1/(1.0001-tau_v))]`. Common-cost replaces this at **every** parent/rewire/goal evaluation with the shared exact `L + R` integral. |
| `sample` / source COM ellipsoid using `path.dis + 1` | Common-cost uses XY ellipse with `J_best`, whose containment follows `d_XY <= L_H <= J`; own-cost uses full-domain uniform proposals after the first solution. Pre-solution goal bias 0.15 retained. No heuristic is added to native cost. |
| `getRandomNum`: new random_device engine on each call | One explicitly seeded `mt19937_64` per solve. |
| `collisionFree` tangent frame | Fix source projection expression to `diff - dot(diff,normal)*normal`; use an orthogonal vector for identity / vertical difference. Common continuous footprint checker is also mandatory for every accepted edge. |

The source `config/for_simulation/map1.yaml` provides fixed settings:
plane radius 0.5 m; COM height 0.7 m; steer step 0.4 m; total/flatness/slope/
sparsity weights 1/1000/0.1/1; vacancy ratio min/max 0.02/0.05; convergence
threshold 0.1152. Header defaults provide goal bias 0.15, neighbor and goal
threshold 1 m. These are not tuned on the benchmark's test maps.

Sparse voxel input can saturate the source vacancy penalty even on flat terrain.
This is retained and observable in model diagnostics; it is not silently
renormalized to make the native objective resemble the common objective.

Time phases: cloud conversion/voxel creation occurs in `tc_putn_create` with
separate measured counters: `conversion_seconds` includes ground observations
and explicit-obstacle extrusion points; `preprocessing_seconds` includes their
voxel insertion. C ABI statistics indices 11 and 8 expose them respectively. Query
endpoint plane fits and the search occur in `tc_putn_solve`. The common risk
field and collision index are built separately in `tc_create`. A positive
`fixed_work` executes a fixed number of iterations for tests only; production
adapters must pass zero to use the solve budget.

A4 comparison options (2026-11-19, opt-in; defaults are unchanged bit for bit):

| Option | Behavior |
|---|---|
| `tau_slope_cap_deg` (paper policy only; C ABI `tc_putn_create_paper_slope`) | The slope term of tau is `slope_deg / cap`. The source weight 0.1 per degree is cap 10 (IEEE `1./10. == .1`), so tau already reaches 1 at 10 degrees on slope alone and the paper policy rejects every plane from about 10 degrees, while the A4 safety domain admits 35. The A4 PUTN arm declares cap = 35, the protocol slope cap, not a value tuned on data; flatness, sparsity and all other source parameters stay fixed. The source arm keeps 10. |
| Convergence stop (`setConvergenceStop`; C ABI `tc_putn_set_convergence_stop`, `tc_putn_convergence`) | The SEM-RRT* rule of the A4 self-stop comparison: at each iteration boundary after the first solution, a published cost below `reference * (1 - relative_eps)` becomes the reference; once `window_iterations` iterations pass without one, the search stops. It reads only the published cost, so a stopped search is an exact prefix of the unstopped search. Off by default. |

Verification: C++ tests cover analytic planar common cost, active plane model,
native cost, deterministic fixed-work paths/rewiring, thin obstacles, final
connector, slope and NoData; the paper tests also cover cap 10 = source weight,
a 16.7-degree plane admitted at cap 35, and the stop point and exact-prefix property
of the convergence stop. Python C ABI tests use analytic length expectations
and exact singleton XY. The campaign independent verifier remains required for
all reported successes; these tests do not establish global optimality or robot
execution safety.
