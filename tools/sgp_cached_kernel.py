"""Evaluation-only training-feature reuse for the pinned GPyTorch 1.11 kernel.

This changes neither the inducing approximation nor the prediction strategy.
It reuses the exact ``K(X,U) @ K(U,U)^(-1/2)`` tensor which the stock kernel
recomputes for every query block. Only training-input/parameter gradients are
discarded; query-input autograd remains live. Install after training and before
the first prediction. Equivalence and invalidation are exercised by
tests/test_sgp_prediction_cache.py against the pinned stock kernel.
"""
from __future__ import annotations

import math

import gpytorch
import torch
from linear_operator import to_dense
from linear_operator.operators import (DiagLinearOperator,
                                      LowRankRootAddedDiagLinearOperator,
                                      LowRankRootLinearOperator,
                                      MatmulLinearOperator)


CACHE_VERSION = "gpytorch_1_11_eval_training_root_v1"


class CachedInducingPointKernel(gpytorch.kernels.InducingPointKernel):
    """Stock training, exact cached evaluation for a fixed training tensor.

    This deliberately inherits InducingPointKernel so GPyTorch's existing
    SGPRPredictionStrategy retains its special treatment of test covariance.
    Query points equal in value to training points still use a live query root.
    Parameter changes during populated evaluation fail closed: callers must
    use model.train()/model.eval(), which also clears ExactGP prediction caches.
    In-place mutation through ``.data`` is outside this immutable-eval contract.
    """

    def __init__(self, base_kernel, inducing_points, likelihood, active_dims=None,
                 training_inputs=None):
        if gpytorch.__version__ != "1.11":
            raise RuntimeError("training-feature cache is verified only for GPyTorch 1.11")
        super().__init__(base_kernel, inducing_points, likelihood, active_dims)
        self._training_inputs = training_inputs
        self._training_root = None
        self._training_root_signature = None
        self.training_transform_builds = 0
        self.training_transform_hits = 0
        self.cache_invalidations = 0

    def _clear_cache(self):
        super()._clear_cache()
        if getattr(self, "_training_root", None) is not None:
            self.cache_invalidations += 1
        self._training_root = None
        self._training_root_signature = None

    def _signature(self):
        x = self._training_inputs
        return (id(x), x._version, tuple(x.shape), x.dtype, x.device,
                tuple((name, id(p), p._version, p.dtype, p.device)
                      for name, p in self.named_parameters()))

    def _check_immutable(self):
        if (self._training_root is not None and
                self._training_root_signature != self._signature()):
            raise RuntimeError("cached evaluation inputs/parameters changed; call model.train() then model.eval()")

    def _matches_training(self, values):
        x = self._training_inputs
        return (values.shape == x.shape and values.dtype == x.dtype and
                values.device == x.device and torch.equal(values, x))

    def _fixed_training_root(self):
        self._check_immutable()
        if self._training_root is None:
            # The final, trainable inducing coordinates and learned kernel
            # parameters are read here, never reconstructed from initial indices.
            with torch.no_grad():
                k_xu = to_dense(self.base_kernel(self._training_inputs, self.inducing_points))
                self._training_root = k_xu.matmul(self._inducing_inv_root).detach()
            self._training_root_signature = self._signature()
            self.training_transform_builds += 1
        else:
            self.training_transform_hits += 1
        return self._training_root

    def _get_covariance(self, x1, x2):
        if self.training or self._training_inputs is None:
            return super()._get_covariance(x1, x2)
        self._check_immutable()
        same = torch.equal(x1, x2)
        if same:
            # ExactGP's initial training prior has no input autograd. The
            # posterior query==training edge case has a live concatenated input
            # graph, and MUST NOT use the detached training root for x1.
            if not x1.requires_grad and self._matches_training(x1):
                root = self._fixed_training_root()
            else:
                root = to_dense(self.base_kernel(x1, self.inducing_points)).matmul(self._inducing_inv_root)
            covar = LowRankRootLinearOperator(root)
            if gpytorch.settings.sgpr_diagonal_correction.on():
                correction = (self.base_kernel(x1, x2, diag=True) -
                              covar.diagonal(dim1=-1, dim2=-2)).clamp(0, math.inf)
                covar = LowRankRootAddedDiagLinearOperator(covar, DiagLinearOperator(correction))
            return covar
        # Match stock arithmetic and operator classes; only the fixed right
        # factor is reused. x2 is the stored-training slice supplied by ExactGP.
        left = to_dense(self.base_kernel(x1, self.inducing_points)).matmul(self._inducing_inv_root)
        right = (self._fixed_training_root() if self._matches_training(x2) else
                 to_dense(self.base_kernel(x2, self.inducing_points)).matmul(self._inducing_inv_root))
        return MatmulLinearOperator(left, right.transpose(-1, -2))

    def cache_info(self):
        return dict(version=CACHE_VERSION, training_transform_builds=self.training_transform_builds,
                    training_transform_hits=self.training_transform_hits,
                    cache_invalidations=self.cache_invalidations,
                    populated=self._training_root is not None,
                    cached_bytes=0 if self._training_root is None else
                    self._training_root.numel() * self._training_root.element_size())


def install_training_feature_cache(model):
    """Install on an immutable fitted CPU float32 ExactGP before prediction.

    No optimizer, data, parameter, likelihood or numerical setting is changed.
    Train/eval transitions continue to use GPyTorch's ordinary invalidation.
    """
    if not isinstance(model, gpytorch.models.ExactGP) or model.training:
        raise ValueError("cache installation requires an evaluated ExactGP")
    if model.prediction_strategy is not None:
        raise ValueError("install cache before the first prediction")
    kernel = model.covar_module
    if type(kernel) is not gpytorch.kernels.InducingPointKernel:
        raise ValueError("cache installation requires the stock InducingPointKernel")
    if kernel.training or model.likelihood.training:
        raise ValueError("kernel and likelihood must be in evaluation mode")
    if len(model.train_inputs) != 1:
        raise ValueError("only the single-input SGP model is supported")
    inputs = model.train_inputs[0]
    if (inputs.ndim != 2 or inputs.requires_grad or inputs.dtype != torch.float32 or
            inputs.device.type != "cpu" or kernel.active_dims is not None):
        raise ValueError("cache requires fixed CPU float32 2D training inputs and no active_dims")
    cached = CachedInducingPointKernel(kernel.base_kernel, kernel.inducing_points,
                                      kernel.likelihood, training_inputs=inputs)
    # Retain the very same Parameter after Adam, including its final locations.
    cached.inducing_points = kernel.inducing_points
    cached.eval()
    model.covar_module = cached
    return cached
