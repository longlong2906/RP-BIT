"""Paper-first SGP static reconstruction, opt-in; legacy source port is unchanged.

The SE kernel follows paper Eq. 1, safety features Eq. 3-7. Missing settings use
the pinned author's husky_env_a config. Gradient min-max normalization and a
cell-center height reference are explicit static reconstruction conventions.
"""
import math
import hashlib
import numpy as np
from native_point_cloud import voxel_centroids
from sgp_static_adapter import (NativeModelError, SGPStaticModel, SGPStaticPlanner,
                                global_grid, UPSTREAM_COMMIT, PREDICTION_BLOCK_SIZE)

MODEL_VERSION = "sgp_paper_first_static_v2"
TRAVERSABILITY_LIMIT = .4


def feature_fields(mean, gradient, variance, bounds=None, resolution=.2):
    mean, gradient, variance = [np.asarray(a, dtype=np.float64) for a in (mean, gradient, variance)]
    if (mean.ndim != 2 or min(mean.shape) < 3 or gradient.shape != (*mean.shape, 2)
            or variance.shape != mean.shape or not all(np.isfinite(a).all() for a in (mean, gradient, variance))
            or (variance < 0).any() or not math.isfinite(resolution) or resolution <= 0):
        raise NativeModelError("invalid paper GP prediction arrays/resolution")
    magnitude = np.linalg.norm(gradient, axis=2)
    low, high = float(magnitude.min()), float(magnitude.max())
    normalized = (magnitude-low)/(high-low) if high > low else np.zeros_like(mean)
    tilt, step = np.empty_like(mean), np.empty_like(mean)
    for j in range(mean.shape[0]):
        for i in range(mean.shape[1]):
            j0, j1, i0, i1 = max(0, j-2), min(mean.shape[0], j+3), max(0, i-2), min(mean.shape[1], i+3)
            dx, dy = np.meshgrid((np.arange(i0, i1)-i)*resolution, (np.arange(j0, j1)-j)*resolution)
            local = mean[j0:j1, i0:i1]
            matrix = np.column_stack((dx.ravel(), dy.ravel(), np.ones(local.size)))
            coefficients, _, rank, _ = np.linalg.lstsq(matrix, local.ravel(), rcond=None)
            if rank != 3:
                raise NativeModelError("rank-deficient paper plane fit")
            tilt[j, i] = math.atan(math.hypot(*coefficients[:2]))
            step[j, i] = max(0., float(local.max()-mean[j, i]))
    raw = .2*normalized/.5236 + .4*tilt/.5236 + .4*step/.3
    penalty = np.clip(raw, 0., 1.)
    threshold = 1.4*float(variance.mean())
    mask = variance > threshold  # strict inequality in paper Eq. 6
    penalty[mask] = 1.
    return dict(gradient_magnitude=magnitude, normalized_gradient=normalized,
                plane_tilt_rad=tilt, step_height_m=step, raw_traversability=raw,
                traversability=penalty, uncertainty_mask=mask, uncertainty_limit=threshold,
                gradient_min=low, gradient_max=high)


class SGPPaperModel(SGPStaticModel):
    """Real inducing-point GP; compatible with SGPStaticPlanner's model interface."""
    def __init__(self, points, bounds, seed=0):
        import torch
        import gpytorch
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        # Explicit CPU numeric convention for this dedicated SGP worker. RBF
        # tails otherwise create subnormal operands with severe GEMM overhead.
        # This is not a claim of bitwise equivalence for arbitrary inputs.
        if not torch.set_flush_denormal(True):
            raise NativeModelError("required SGP CPU flush-denormal policy is unsupported")
        numeric_policy = dict(version="sgp_cpu_ftz_daz_v1", torch_api="torch.set_flush_denormal(True)",
            enabled=True, scope="dedicated_sgp_worker_cpu_float32_float64",
            float32_tiny=float(torch.finfo(torch.float32).tiny),
            float64_tiny=float(torch.finfo(torch.float64).tiny), unsupported="fail_closed")
        torch.manual_seed(int(seed))
        self.bounds, self.resolution = tuple(bounds), .2
        cloud = voxel_centroids(points, .2)
        cloud = np.round(cloud, 4)
        cloud = cloud[np.lexsort((cloud[:, 2], cloud[:, 1], cloud[:, 0]))]
        grid, shape = global_grid(bounds, self.resolution)
        x = torch.tensor(cloud[:, :2], dtype=torch.float32)
        y = torch.tensor(cloud[:, 2], dtype=torch.float32)
        # Exactly min(500,N) deterministic, spatially spread indices; no adaptive subsampling.
        indices = np.linspace(0, len(cloud)-1, min(500, len(cloud)), dtype=np.int64)
        inducing = x[indices]

        class PaperGP(gpytorch.models.ExactGP):
            def __init__(self, tx, ty, likelihood):
                super().__init__(tx, ty, likelihood)
                self.mean_module = gpytorch.means.ConstantMean()
                self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
                self.base_covar_module.base_kernel.initialize(lengthscale=.7)
                self.base_covar_module.initialize(outputscale=math.log(2.))
                self.covar_module = gpytorch.kernels.InducingPointKernel(
                    self.base_covar_module, inducing_points=inducing, likelihood=likelihood)
            def forward(self, inputs):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(inputs), self.covar_module(inputs))

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=math.log(2.)+1e-4)
        model = PaperGP(x, y, likelihood).cpu()
        model.train(); likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optimizer.zero_grad()
        loss = -objective(model(x), y).mean()
        if not torch.isfinite(loss):
            raise NativeModelError("nonfinite paper GP training loss")
        loss.backward(); optimizer.step()
        model.eval(); likelihood.eval()
        from sgp_cached_kernel import install_training_feature_cache
        prediction_cache = install_training_feature_cache(model)
        means, gradients, variances = [], [], []
        for offset in range(0, len(grid), PREDICTION_BLOCK_SIZE):
            query = torch.tensor(grid[offset:offset+PREDICTION_BLOCK_SIZE], dtype=torch.float32, requires_grad=True)
            prediction = likelihood(model(query))
            gradient = torch.autograd.grad(prediction.mean, query, grad_outputs=torch.ones_like(prediction.mean))[0]
            means.append(prediction.mean.detach().numpy())
            variances.append(prediction.variance.detach().numpy())
            gradients.append(gradient.detach().numpy())
        self.mean, self.variance = np.concatenate(means).reshape(shape), np.concatenate(variances).reshape(shape)
        self.gradient = np.concatenate(gradients).reshape(*shape, 2)
        self.fields = feature_fields(self.mean, self.gradient, self.variance, bounds, self.resolution)
        self.traversability = self.fields["traversability"]
        self.metadata = dict(version=MODEL_VERSION, upstream_commit=UPSTREAM_COMMIT,
            paper="arXiv:2403.19010v1", kernel="scaled_RBF_SE", native_cost="euclidean_xy_length",
            device="cpu", torch=torch.__version__, gpytorch=gpytorch.__version__, numpy=np.__version__,
            cpu_flush_denormal=True, numeric_policy=numeric_policy,
            input_observations=len(points), training_observations=len(cloud), input_reduction="3d_voxel_centroids_0.2m",
            reduced_cloud_sha256=hashlib.sha256(cloud.astype("<f8").tobytes()).hexdigest(),
            inducing_points=len(indices), inducing_selection="uniform_lexicographic_indices_v1",
            training_steps=1, learning_rate=.01, training_loss=float(loss),
            prediction_cache=prediction_cache.cache_info(),
            prediction_cache_contract="GPyTorch_1.11_immutable_eval_query_gradients_only; train_eval_or_state_reload_invalidates",
            prediction_block_size=PREDICTION_BLOCK_SIZE, prediction_grid_shape_yx=list(shape), resolution_m=.2,
            initial_hyperparameters=dict(lengthscale_m=.7, outputscale=math.log(2.), noise=math.log(2.)+1e-4, constant_mean=0.),
            trained_hyperparameters=dict(lengthscale=model.base_covar_module.base_kernel.lengthscale.detach().tolist(),
                outputscale=float(model.base_covar_module.outputscale.detach()), noise=float(likelihood.noise.detach()),
                constant_mean=float(model.mean_module.constant.detach())),
            weights=dict(slope=.2, flatness=.4, step_height=.4),
            critical_thresholds=dict(normalized_gradient=.5236, flatness_rad=.5236, step_height_m=.3),
            gradient_normalization="per_map_minmax_constant_to_zero", combined_traversability_normalization="clip_only",
            step_height_reference="predicted_cell_center", feature_window_grid=[5, 5], native_filter_grid=[3, 3],
            traversability_limit=.4, uncertainty_mean_multiplier=1.4, uncertainty_comparison="strict_greater",
            uncertainty_limit=self.fields["uncertainty_limit"], uncertain_nodes=int(self.fields["uncertainty_mask"].sum()),
            traversability_rejected_nodes=int((self.traversability > .4).sum()),
            step_size_m=.5, neighbor_radius_m=1., goal_threshold_m=.5, nearest_metric="euclidean_xy",
            thread_count=torch.get_num_threads(), interop_thread_count=torch.get_num_interop_threads(),
            search_termination="15s_anytime_benchmark_override", representation="static_global_known_terrain")

    def valid_state(self, point):
        x, y = point
        xmin, ymin, xmax, ymax = self.bounds
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
        i, j = int(math.floor((x-xmin)/self.resolution+.5)), int(math.floor((y-ymin)/self.resolution+.5))
        if i < 1 or j < 1 or i+1 >= self.traversability.shape[1] or j+1 >= self.traversability.shape[0]:
            return False
        return bool(np.all(self.traversability[j-1:j+2, i-1:i+2] <= TRAVERSABILITY_LIMIT))
