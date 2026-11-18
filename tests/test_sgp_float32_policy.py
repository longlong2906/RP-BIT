"""Synthetic far-field controls for an explicit SGP float32 FTZ convention.

This test changes only its own process floating-point mode. Production code is
not patched. FTZ drops float32 subnormals, so universal bitwise equivalence is
not asserted; trained parameters, predictions and safety decisions are checked.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import sys
import time
import unittest

if os.environ.get("SGP_POLICY_TEST_CPU"):
    import psutil
    psutil.Process().cpu_affinity([int(os.environ["SGP_POLICY_TEST_CPU"])])

import gpytorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from sgp_cached_kernel import install_training_feature_cache
from sgp_paper_adapter import feature_fields

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)


@contextmanager
def floating_point_mode(enabled):
    supported = torch.set_flush_denormal(enabled)
    if not supported:
        raise AssertionError("this CPU does not support the requested explicit FTZ policy")
    try:
        yield
    finally:
        torch.set_flush_denormal(False)


def subnormal_count(values):
    """Inspect IEEE754 bits; floating comparisons could themselves flush them."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
    return int(np.count_nonzero(((bits & 0x7f800000) == 0) & ((bits & 0x007fffff) != 0)))


class ControlGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, inducing_count):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=math.log(2.)+1e-4)
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.base_covar_module.base_kernel.initialize(lengthscale=.7)
        self.base_covar_module.initialize(outputscale=math.log(2.))
        indices = np.linspace(0, len(x)-1, min(inducing_count, len(x)), dtype=np.int64)
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module, inducing_points=x[indices], likelihood=likelihood)

    def forward(self, inputs):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(inputs), self.covar_module(inputs))


def nonflat_fixture(nx, ny, clustered=False):
    # A 40 m domain forces distances near 10 m where the .7 m RBF produces
    # float32 subnormals, unlike a small local-plane fixture.
    xx, yy = np.meshgrid(np.linspace(0., 40., nx), np.linspace(0., 36., ny))
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    if clustered:
        # Dense support over most of the query region plus real far-field
        # observations; the unobserved right border exercises uncertainty=True.
        near_x, near_y = np.meshgrid(np.linspace(17.8, 19.4, nx), np.linspace(17.6, 19.2, ny-3))
        far_x, far_y = np.meshgrid(np.linspace(0., 40., nx), [25., 27., 36.])
        xy = np.concatenate((np.column_stack((near_x.ravel(), near_y.ravel())),
                             np.column_stack((far_x.ravel(), far_y.ravel())))).astype(np.float32)
    z = (.2*np.sin(.27*xy[:, 0]) + .13*np.cos(.31*xy[:, 1]) + .002*xy[:, 0]*xy[:, 1]
         + .15*np.exp(-((xy[:, 0]-18.)**2+(xy[:, 1]-19.)**2)/9.)).astype(np.float32)
    # Native feature spacing remains .2 m; the query patch includes multiple
    # support locations and nonzero prediction gradients.
    qx, qy = np.meshgrid(17.8+np.arange(15 if clustered else 11)*.2, 17.6+np.arange(9)*.2)
    query = np.column_stack((qx.ravel(), qy.ravel())).astype(np.float32)
    return xy, z, query, qx.shape


def evaluate_control(nx, ny, inducing_count, flush, seed=9017, clustered=False):
    began = time.perf_counter()
    with floating_point_mode(flush):
        torch.manual_seed(seed)
        xy, z, query, shape = nonflat_fixture(nx, ny, clustered)
        x, y = torch.tensor(xy), torch.tensor(z)
        model = ControlGP(x, y, inducing_count)
        model.train(); model.likelihood.train()
        initial_parameters = {name: value.detach().numpy().copy() for name, value in model.named_parameters()}
        with torch.no_grad():
            k_xu = model.base_covar_module(x, model.covar_module.inducing_points).to_dense()
            kernel_subnormals = subnormal_count(k_xu.numpy())
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        objective = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer.zero_grad()
        loss = -objective(model(x), y).mean()
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite control MLL")
        loss.backward(); optimizer.step()
        parameters = {name: value.detach().numpy().copy() for name, value in model.named_parameters()}
        model.eval(); model.likelihood.eval()
        kernel = install_training_feature_cache(model)
        means, variances, gradients = [], [], []
        for offset in range(0, len(query), 33):
            q = torch.tensor(query[offset:offset+33], dtype=torch.float32, requires_grad=True)
            prediction = model.likelihood(model(q))
            gradient = torch.autograd.grad(prediction.mean, q, grad_outputs=torch.ones_like(prediction.mean))[0]
            means.append(prediction.mean.detach().numpy().copy())
            variances.append(prediction.variance.detach().numpy().copy())
            gradients.append(gradient.detach().numpy().copy())
        values = dict(mean=np.concatenate(means).reshape(shape),
                      variance=np.concatenate(variances).reshape(shape),
                      gradient=np.concatenate(gradients).reshape(*shape, 2))
        fields = feature_fields(values["mean"], values["gradient"], values["variance"], resolution=.2)
        return dict(loss=float(loss.detach()), initial_parameters=initial_parameters, parameters=parameters,
                    values=values, fields=fields, kernel_subnormals=kernel_subnormals,
                    cached_root_subnormals=subnormal_count(kernel._training_root.numpy()),
                    cache_info=kernel.cache_info(), wall_s=time.perf_counter()-began)


class Float32PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.set_flush_denormal(True):
            raise AssertionError("explicit FTZ is unsupported on this runtime")
        torch.set_flush_denormal(False)
        print(json.dumps(dict(control="runtime", torch=torch.__version__, gpytorch=gpytorch.__version__,
                              numpy=np.__version__, threads=torch.get_num_threads(),
                              interop_threads=torch.get_num_interop_threads(),
                              cpu=os.environ.get("SGP_POLICY_TEST_CPU"), float32_tiny=float(np.finfo(np.float32).tiny)), sort_keys=True))

    def tearDown(self):
        torch.set_flush_denormal(False)

    def _compare(self, nx, ny, inducing_count, clustered=False):
        off = evaluate_control(nx, ny, inducing_count, False, clustered=clustered)
        on = evaluate_control(nx, ny, inducing_count, True, clustered=clustered)
        self.assertGreater(off["kernel_subnormals"], 0, "fixture must exercise real RBF subnormals")
        self.assertEqual(on["kernel_subnormals"], 0)
        self.assertEqual(on["cached_root_subnormals"], 0)
        np.testing.assert_allclose(on["loss"], off["loss"], rtol=2e-6, atol=2e-6)
        parameter_errors = {}
        for key, value in off["parameters"].items():
            np.testing.assert_array_equal(off["initial_parameters"][key], on["initial_parameters"][key])
            np.testing.assert_allclose(on["parameters"][key], value, rtol=2e-6, atol=2e-6, err_msg=key)
            parameter_errors[key] = float(np.max(np.abs(on["parameters"][key].astype(np.float64)-value)))
        self.assertIn("covar_module.inducing_points", parameter_errors)
        prediction_errors = {}
        for key, value in off["values"].items():
            np.testing.assert_allclose(on["values"][key], value, rtol=2e-6, atol=2e-6, err_msg=key)
            prediction_errors[key] = float(np.max(np.abs(on["values"][key].astype(np.float64)-value)))
        self.assertGreater(float(np.max(np.abs(off["values"]["gradient"]))), 1e-4)
        field_errors = {}
        for key, value in off["fields"].items():
            if key == "uncertainty_mask":
                np.testing.assert_array_equal(on["fields"][key], value)
            else:
                np.testing.assert_allclose(on["fields"][key], value, rtol=1e-5, atol=1e-5, err_msg=key)
                field_errors[key] = float(np.max(np.abs(np.asarray(on["fields"][key])-value)))
        np.testing.assert_array_equal(on["fields"]["traversability"] > .4, off["fields"]["traversability"] > .4)
        if clustered:
            self.assertGreater(int(off["fields"]["uncertainty_mask"].sum()), 0)
            self.assertLess(int(off["fields"]["uncertainty_mask"].sum()), off["fields"]["uncertainty_mask"].size)
        for value in (off, on):
            self.assertEqual(value["cache_info"]["training_transform_builds"], 1)
            self.assertGreaterEqual(value["cache_info"]["training_transform_hits"], 3)
        print(json.dumps(dict(control="far_field_float32_policy", train_observations=nx*ny,
                              inducing_count=inducing_count, prediction_queries=off["values"]["mean"].size,
                              clustered_support=clustered,
                              mll_absolute_error=abs(on["loss"]-off["loss"]),
                              learned_parameter_max_abs_errors=parameter_errors,
                              prediction_max_abs_errors=prediction_errors, field_max_abs_errors=field_errors,
                              uncertainty_mask_exact=True, traversability_rejection_mask_exact=True,
                              uncertainty_nodes=int(off["fields"]["uncertainty_mask"].sum()),
                              rejected_nodes=int((off["fields"]["traversability"] > .4).sum()),
                              off_kernel_subnormals=off["kernel_subnormals"], on_kernel_subnormals=on["kernel_subnormals"],
                              off_cached_root_subnormals=off["cached_root_subnormals"],
                              on_cached_root_subnormals=on["cached_root_subnormals"],
                              diagnostic_wall_s=dict(off=off["wall_s"], on=on["wall_s"]),
                              scope="synthetic numerical control, not a speed or benchmark claim"), sort_keys=True))

    def test_far_field_240_observations_100_inducing(self):
        self._compare(16, 15, 100)

    def test_far_field_650_observations_500_inducing(self):
        self._compare(26, 25, 500)

    def test_clustered_support_exercises_both_uncertainty_states(self):
        self._compare(16, 15, 100, clustered=True)


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        torch.set_flush_denormal(False)
