"""Real pinned-GPyTorch equivalence controls; synthetic analytic data only."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import unittest

import gpytorch
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from sgp_cached_kernel import CachedInducingPointKernel, install_training_feature_cache
from sgp_paper_adapter import feature_fields

torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)


class CountedScaleKernel(gpytorch.kernels.ScaleKernel):
    """Count actual full K(X,U) evaluations, each followed by stock root matmul."""
    def __init__(self, base_kernel, training_inputs):
        super().__init__(base_kernel)
        self.control_training_inputs = training_inputs
        self.training_kernel_evaluations = 0

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if not diag and torch.equal(x1, self.control_training_inputs):
            self.training_kernel_evaluations += 1
        return super().forward(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)


class ControlGP(gpytorch.models.ExactGP):
    def __init__(self, x, y, inducing_count):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.initialize(noise=math.log(2.) + 1e-4)
        super().__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = CountedScaleKernel(gpytorch.kernels.RBFKernel(), x)
        self.base_covar_module.base_kernel.initialize(lengthscale=.7)
        self.base_covar_module.initialize(outputscale=math.log(2.))
        indices = np.linspace(0, len(x)-1, min(inducing_count, len(x)), dtype=np.int64)
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            self.base_covar_module, inducing_points=x[indices], likelihood=likelihood)

    def forward(self, inputs):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(inputs), self.covar_module(inputs))


def training_data(nx=26, ny=25):
    xx, yy = np.meshgrid(np.linspace(0., 5., nx), np.linspace(0., 4.8, ny))
    xy = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
    y = (.12*np.sin(1.3*xy[:, 0]) + .07*np.cos(.8*xy[:, 1]) + .02*xy[:, 0]*xy[:, 1]
         + .09*np.exp(-((xy[:, 0]-2.1)**2+(xy[:, 1]-2.7)**2)/.3)).astype(np.float32)
    return torch.tensor(xy), torch.tensor(y)


def step(model):
    model.train(); model.likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    objective = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer.zero_grad()
    loss = -objective(model(model.train_inputs[0]), model.train_targets).mean()
    loss.backward(); optimizer.step()
    model.eval(); model.likelihood.eval()
    model.base_covar_module.training_kernel_evaluations = 0
    return float(loss.detach())


def fitted_pair(nx=26, ny=25, inducing_count=500, seed=906):
    torch.manual_seed(seed)
    x, y = training_data(nx, ny)
    reference = ControlGP(x, y, inducing_count)
    initial = reference.covar_module.inducing_points.detach().clone()
    loss = step(reference)
    cached = ControlGP(x.clone(), y.clone(), inducing_count)
    cached.load_state_dict(copy.deepcopy(reference.state_dict()))
    cached.eval(); cached.likelihood.eval()
    parameter = cached.covar_module.inducing_points
    kernel = install_training_feature_cache(cached)
    assert kernel.inducing_points is parameter
    assert not torch.equal(initial, reference.covar_module.inducing_points)
    return reference, cached, loss


def predict(model, queries, block=1024):
    means, variances, gradients = [], [], []
    for offset in range(0, len(queries), block):
        q = torch.tensor(queries[offset:offset+block], dtype=torch.float32, requires_grad=True)
        prediction = model.likelihood(model(q))
        grad = torch.autograd.grad(prediction.mean, q, grad_outputs=torch.ones_like(prediction.mean))[0]
        means.append(prediction.mean.detach().numpy())
        variances.append(prediction.variance.detach().numpy())
        gradients.append(grad.detach().numpy())
    return dict(mean=np.concatenate(means), variance=np.concatenate(variances), gradient=np.concatenate(gradients))


class CacheTests(unittest.TestCase):
    def assert_predictions_equal(self, reference, cached, label):
        errors = {}
        for key in reference:
            errors[key] = float(np.max(np.abs(reference[key]-cached[key])))
            np.testing.assert_allclose(cached[key], reference[key], rtol=2e-6, atol=2e-6, err_msg=label+":"+key)
        print(json.dumps(dict(control=label, maximum_absolute_errors=errors), sort_keys=True))

    def test_nonflat_650_observations_500_inducing_three_1024_blocks(self):
        self.assertLess(650, gpytorch.settings.max_cholesky_size.value())
        self._assert_nonflat_large_control(26, 25)

    def test_nonflat_1200_observations_above_cholesky_threshold(self):
        self.assertGreater(1200, gpytorch.settings.max_cholesky_size.value())
        self._assert_nonflat_large_control(40, 30)

    def _assert_nonflat_large_control(self, nx, ny):
        reference, cached, loss = fitted_pair(nx, ny)
        count = nx * ny
        xx, yy = np.meshgrid(np.linspace(-1., 6., 49), np.linspace(-1., 5.8, 47))
        grid = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float32)
        self.assertFalse(gpytorch.settings.fast_pred_var.on())
        self.assertTrue(gpytorch.settings.sgpr_diagonal_correction.on())
        stock = predict(reference, grid)
        actual = predict(cached, grid)
        self.assert_predictions_equal(stock, actual, f"nonflat_{count}x500_three_blocks")
        self.assertIsInstance(cached.prediction_strategy, gpytorch.models.exact_prediction_strategies.SGPRPredictionStrategy)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, cached.state_dict()[name], rtol=0., atol=0.)
        info = cached.covar_module.cache_info()
        self.assertEqual(info["training_transform_builds"], 1)
        self.assertGreaterEqual(info["training_transform_hits"], 3)
        self.assertEqual(cached.base_covar_module.training_kernel_evaluations, 1)
        self.assertGreaterEqual(reference.base_covar_module.training_kernel_evaluations, 4)
        self.assertEqual(info["cached_bytes"], count*500*4)
        self.assertFalse(cached.covar_module._training_root.requires_grad)
        self.assertIsNone(cached.covar_module._training_root.grad_fn)
        fields = []
        for values in (stock, actual):
            fields.append(feature_fields(values["mean"].reshape(xx.shape),
                                         values["gradient"].reshape(*xx.shape, 2),
                                         values["variance"].reshape(xx.shape)))
        for key in fields[0]:
            if key == "uncertainty_mask":
                np.testing.assert_array_equal(fields[0][key], fields[1][key])
            else:
                np.testing.assert_allclose(fields[0][key], fields[1][key], rtol=1e-5, atol=1e-5, err_msg=key)
        self.assertGreater(np.count_nonzero(fields[0]["uncertainty_mask"]), 0)
        self.assertLess(np.count_nonzero(fields[0]["uncertainty_mask"]), grid.shape[0])
        np.testing.assert_array_equal(fields[0]["traversability"] > .4, fields[1]["traversability"] > .4)
        print(json.dumps(dict(control="training_transform_counts", training_observations=count,
                              max_cholesky_size=gpytorch.settings.max_cholesky_size.value(),
                              stock=reference.base_covar_module.training_kernel_evaluations,
                              cached=info, training_loss=loss), sort_keys=True))

    def test_query_equal_all_training_points_keeps_live_gradient(self):
        reference, cached, _ = fitted_pair(10, 9, 40)
        query = reference.train_inputs[0].numpy().copy()
        stock, actual = predict(reference, query), predict(cached, query)
        self.assert_predictions_equal(stock, actual, "query_equals_training")
        self.assertGreater(float(np.max(np.abs(actual["gradient"]))), .01)

    def test_train_eval_invalidation_and_unmodified_training_step(self):
        reference, cached, _ = fitted_pair(10, 9, 40)
        query = np.array([[.3, .7], [1.9, 2.6], [4.4, 3.8]], dtype=np.float32)
        predict(cached, query, 2)
        kernel = cached.covar_module
        self.assertEqual(kernel.training_transform_builds, 1)
        stock_loss, cached_loss = step(reference), step(cached)
        self.assertEqual(stock_loss, cached_loss)
        self.assertIsNone(kernel._training_root)
        self.assertIsNone(cached.prediction_strategy)
        for name, value in reference.state_dict().items():
            torch.testing.assert_close(value, cached.state_dict()[name], rtol=0., atol=0.)
        self.assert_predictions_equal(predict(reference, query, 2), predict(cached, query, 2), "retrained")
        self.assertEqual(kernel.training_transform_builds, 2)
        self.assertGreaterEqual(kernel.cache_invalidations, 1)

    def test_mutation_fails_closed_and_state_reload_invalidates_full_model(self):
        for mutation in ("parameter", "load_state_dict", "training_input"):
            with self.subTest(mutation=mutation):
                reference, cached, _ = fitted_pair(8, 7, 20)
                query = np.array([[.3, .7], [1.9, 2.6]], dtype=np.float32)
                predict(cached, query)
                if mutation == "parameter":
                    with torch.no_grad():
                        cached.covar_module.inducing_points.add_(.01)
                elif mutation == "load_state_dict":
                    cached.load_state_dict(copy.deepcopy(reference.state_dict()))
                else:
                    cached.train_inputs[0].add_(.01)
                if mutation == "load_state_dict":
                    # GPyTorch Module._load_from_state_dict already clears
                    # both ExactGP's strategy and this kernel's virtual cache.
                    self.assertIsNone(cached.covar_module._training_root)
                    self.assertIsNone(cached.prediction_strategy)
                    self.assert_predictions_equal(predict(reference, query), predict(cached, query), "state_reload")
                else:
                    with self.assertRaisesRegex(RuntimeError, "parameters changed"):
                        predict(cached, query)
                    cached.train(); cached.eval()
                values = predict(cached, query)
                self.assertTrue(all(np.isfinite(a).all() for a in values.values()))

    def test_diagonal_correction_setting_and_likelihood_noise_preserved(self):
        for enabled in (True, False):
            with self.subTest(diagonal_correction=enabled):
                reference, cached, _ = fitted_pair(8, 7, 20)
                with gpytorch.settings.sgpr_diagonal_correction(enabled):
                    # Direct training prior exercises the exact correction/clamp.
                    prior = reference.covar_module(reference.train_inputs[0]).to_dense()
                    actual = cached.covar_module(cached.train_inputs[0]).to_dense()
                    torch.testing.assert_close(prior, actual, rtol=0., atol=0.)
                    query = np.array([[.3, .7], [1.9, 2.6], [6., -1.]], dtype=np.float32)
                    self.assert_predictions_equal(predict(reference, query, 2), predict(cached, query, 2), "diagonal_"+str(enabled))
                torch.testing.assert_close(reference.likelihood.noise, cached.likelihood.noise, rtol=0., atol=0.)

    def test_same_seed_repeats_exactly_and_installation_is_guarded(self):
        query = np.array([[.3, .7], [1.9, 2.6], [4.4, 3.8]], dtype=np.float32)
        first = fitted_pair(8, 7, 20, seed=77)[1]
        second = fitted_pair(8, 7, 20, seed=77)[1]
        a, b = predict(first, query, 2), predict(second, query, 2)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        reference, _, _ = fitted_pair(8, 7, 20)
        reference.train()
        with self.assertRaisesRegex(ValueError, "evaluated"):
            install_training_feature_cache(reference)
        reference.eval()
        predict(reference, query)
        with self.assertRaisesRegex(ValueError, "first prediction"):
            install_training_feature_cache(reference)


if __name__ == "__main__":
    unittest.main(verbosity=2)
