"""ChartStateV3 -standalone chart geometry holder (no v1 dependency).

Buffers: mu, U, eigvals, sigma_perp, radius_d2 (geometry)
Lifecycle: candidate ->active ->adapter_initialized ->distilled ->frozen ->inactive
"""

import math
import torch
import torch.nn as nn


class ChartStateV3(nn.Module):
    """Chart geometry + lifecycle + residual stats + chain tracking."""

    def __init__(self, layer_id=None, chart_id=None):
        super().__init__()
        self.layer_id = layer_id
        self.chart_id = chart_id
        self.birth_task = -1

        # ---- Geometry buffers ----
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("U", torch.zeros(1, 1))
        self.register_buffer("eigvals", torch.ones(1))
        self.register_buffer("sigma_perp", torch.tensor(1.0))
        self.register_buffer("radius_d2", torch.tensor(1.0))
        self.register_buffer("feat_mean", torch.zeros(1))
        self.register_buffer("feat_std", torch.ones(1))
        self.register_buffer("standardize_flag", torch.zeros(1, dtype=torch.uint8))

        # ---- Adapters ----
        self._adapters = {}   # {task_id: ChartAdapter}
        self._adapter_meta = {}
        self._router = None

        # ---- Scalar stats ----
        self.support = 0
        self.quality = 0.0
        self.coverage = 0.0
        self.mean_d2 = 0.0
        self.rec_error = 0.0
        self.overlap_rate = 0.0
        self.residual_consistency = 0.0
        self.grassmann_stability = 0.0
        self.basis_energy = 0.0
        self.subspace_r2 = 0.0
        self.full_r2 = 0.0
        self.centered_r2 = 0.0
        self.delta_mean_ratio = 0.0
        self.status = "active"
        self.is_debug = False

        # ---- Temporary tensors (cleared after adapter built) ----
        self.R = None          # [D, s] residual output basis
        self.B = None          # [s, r_p] low-dim transform
        self.delta_mean = None # [D] residual bias
        self.P_adapter = None  # [D, r_p] input basis (for MLP fallback)

        # ---- v2: Residual statistics ----
        self.residual_mean = None
        self.residual_var = None

        # ---- v2: Lifecycle ----
        self.created_task = -1
        self.last_used_task = -1
        self.usage_count = 0
        self.frozen = False
        self.lifecycle_stage = "candidate"

        # ---- v2: Descendant chain ----
        self.parent_chart_ids = []
        self.child_chart_ids = []

        # ---- v2: Effective rank ----
        self.effective_rank = 0.0

    # ---- Geometry ----

    def set_geometry(self, mu, U, eigvals, sigma_perp, radius_d2):
        for name, val in [("mu", mu), ("U", U), ("eigvals", eigvals),
                           ("sigma_perp", sigma_perp.reshape(1)),
                           ("radius_d2", radius_d2.reshape(1))]:
            self.register_buffer(name, val.clone())

    def set_feature_standardizer(self, mean, std, enabled=True):
        self.register_buffer("feat_mean", mean.clone())
        self.register_buffer("feat_std", std.clone().clamp_min(1e-6))
        self.register_buffer(
            "standardize_flag",
            torch.tensor([1 if enabled else 0], dtype=torch.uint8, device=mean.device),
        )

    @property
    def standardize_features(self):
        return bool(self.standardize_flag.item())

    def transform_features(self, features):
        if features.device != self.feat_mean.device:
            self.to(features.device)
        if not self.standardize_features:
            return features
        return (features - self.feat_mean) / self.feat_std.clamp_min(1e-6)

    def mahalanobis_d2(self, features, eps=1e-6):
        if features.device != self.mu.device:
            self.to(features.device)
        chart_features = self.transform_features(features)
        x_mu = chart_features - self.mu
        z = torch.matmul(x_mu, self.U)
        tangent_d2 = ((z ** 2) / self.eigvals.clamp_min(eps)).sum(dim=-1)
        x_n2 = x_mu.pow(2).sum(dim=-1)
        z_n2 = z.pow(2).sum(dim=-1)
        sigma = max(float(self.sigma_perp.item()), eps)
        D, r = x_mu.shape[-1], z.shape[-1]
        normal_d2 = (x_n2 - z_n2).clamp_min(0) / (sigma * max(D - r, 1))
        return tangent_d2 + normal_d2

    def within_radius(self, features, scale=1.0):
        if features.device != self.radius_d2.device:
            self.to(features.device)
        return self.mahalanobis_d2(features) <= self.radius_d2 * scale

    def get_chart_coords(self, features):
        if features.device != self.mu.device:
            self.to(features.device)
        chart_features = self.transform_features(features)
        return torch.matmul(chart_features - self.mu, self.U)

    # ---- Adapters ----

    def attach_adapter(self, adapter, task_id=None):
        if task_id is None:
            task_id = max(self._adapters.keys(), default=-1) + 1
        self._adapters[task_id] = adapter
        self._adapter_meta[task_id] = {
            "birth_task": int(task_id),
            "support": int(getattr(adapter, "support", 0)),
            "adapter_cos": float(getattr(adapter, "adapter_cos", 0.0)),
            "full_r2": float(getattr(adapter, "full_r2", getattr(self, "full_r2", 0.0))),
            "subspace_r2": float(getattr(adapter, "subspace_r2", getattr(self, "subspace_r2", 0.0))),
            "norm_ratio": float(getattr(adapter, "norm_ratio", 0.0)),
            "usage_count": 0,
            "masked": bool(getattr(adapter, "masked", False)),
        }
        self.R = None; self.B = None; self.delta_mean = None

    @property
    def adapter(self):
        if not self._adapters:
            return None
        return self._adapters[max(self._adapters.keys())]

    def get_adapter(self, task_id=None):
        if task_id is not None and task_id in self._adapters:
            return self._adapters[task_id]
        return self.adapter

    @property
    def num_adapters(self):
        return len(self._adapters)

    @property
    def tangent_dim(self):
        ad = self.adapter
        if ad is not None:
            return ad.tangent_dim
        return self.U.shape[1] if self.U is not None else 0

    @property
    def residual_dim(self):
        ad = self.adapter
        if ad is not None:
            return ad.residual_dim
        return 0

    # ---- Lifecycle state machine ----

    def promote_to_active(self):
        self.lifecycle_stage = "active"
        self.status = "active"

    def promote_to_adapter_initialized(self):
        self.lifecycle_stage = "adapter_initialized"

    def promote_to_distilled(self):
        self.lifecycle_stage = "distilled"

    def freeze(self):
        self.frozen = True
        self.lifecycle_stage = "frozen"

    def mark_inactive(self):
        self.lifecycle_stage = "inactive"
        self.status = "inactive"

    def mark_used(self, task_id):
        self.last_used_task = task_id
        self.usage_count += 1

    # ---- Residual stats ----

    def set_residual_stats(self, residual_mean, residual_var):
        self.residual_mean = residual_mean
        self.residual_var = residual_var

    def compute_effective_rank(self):
        if self.eigvals is None or self.eigvals.numel() < 2:
            self.effective_rank = 1.0 if self.eigvals is not None else 0.0
            return self.effective_rank
        p = self.eigvals / self.eigvals.sum().clamp_min(1e-8)
        p = p[p > 1e-8]
        if p.numel() == 0:
            self.effective_rank = 0.0
        else:
            entropy = -(p * p.log()).sum().item()
            self.effective_rank = float(math.exp(entropy))
        return self.effective_rank

    # ---- Chain tracking ----

    def add_parent(self, parent_layer, parent_chart_id):
        self.parent_chart_ids.append((parent_layer, parent_chart_id))

    def add_child(self, child_layer, child_chart_id):
        self.child_chart_ids.append((child_layer, child_chart_id))

    @property
    def parent_ids(self):
        return [cid for _, cid in self.parent_chart_ids]

    @property
    def child_ids(self):
        return [cid for _, cid in self.child_chart_ids]

    # ---- CPU/GPU management ----

    def cpu(self):
        super().cpu()
        return self

    def cuda(self, device=None):
        super().cuda(device)
        return self

    # ---- Summary ----

    def summary(self):
        return {
            "chart_id": self.chart_id, "layer_id": self.layer_id,
            "birth_task": self.birth_task, "created_task": self.created_task,
            "last_used_task": self.last_used_task, "usage_count": self.usage_count,
            "frozen": self.frozen, "lifecycle_stage": self.lifecycle_stage,
            "status": self.status, "support": self.support,
            "tangent_dim": self.tangent_dim, "residual_dim": self.residual_dim,
            "effective_rank": self.effective_rank, "quality": self.quality,
            "coverage": self.coverage, "mean_d2": self.mean_d2,
            "overlap_rate": self.overlap_rate, "rec_error": self.rec_error,
            "residual_consistency": self.residual_consistency,
            "grassmann_stability": self.grassmann_stability,
            "subspace_r2": self.subspace_r2, "full_r2": self.full_r2,
            "radius_d2": float(self.radius_d2.item()) if self.radius_d2 is not None else 0.0,
            "basis_energy": self.basis_energy, "num_adapters": self.num_adapters,
            "standardize_features": self.standardize_features,
            "parent_ids": self.parent_ids, "child_ids": self.child_ids,
        }


class LayerAtlasState:
    """Atlas state for one layer (L9-L11)."""

    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.charts = []

    def register_charts(self, charts):
        self.charts.extend(charts)

    def active_charts(self):
        return [c for c in self.charts if c.status == "active"]

    @property
    def num_active(self):
        return len(self.active_charts())


class TaskFeatureBuffer:
    """Temporary buffer to collect features + residuals."""

    def __init__(self, layers):
        self.layers = layers
        self.features = {l: [] for l in layers}
        self.residuals = {l: [] for l in layers}
        self.labels = []

    def record(self, layer_id, feat, resid):
        self.features[layer_id].append(feat.detach().cpu())
        self.residuals[layer_id].append(resid.detach().cpu())

    def record_labels(self, labels):
        self.labels.append(labels.detach().cpu())

    def stack(self, layer_id):
        feats = torch.cat(self.features[layer_id], dim=0)
        resids = torch.cat(self.residuals[layer_id], dim=0)
        return feats, resids

    def clear(self):
        for l in self.layers:
            self.features[l].clear()
            self.residuals[l].clear()
        self.labels.clear()

    @property
    def total_samples(self):
        if self.labels:
            return sum(t.shape[0] for t in self.labels)
        l0 = self.layers[0]
        if self.features[l0]:
            return sum(t.shape[0] for t in self.features[l0])
        return 0


class ChartCreationDecision:
    """A/B/C chart creation decision helper."""

    def __init__(self, layer_id, chart_candidates, mahalanobis_d2, residual_error):
        self.layer_id = layer_id
        self.chart_candidates = chart_candidates
        self.mahalanobis_d2 = mahalanobis_d2
        self.residual_error = residual_error

    def classify(self, geo_threshold, residual_threshold):
        geo_hit = self.mahalanobis_d2 <= geo_threshold
        residual_hit = self.residual_error <= residual_threshold
        if geo_hit and residual_hit:
            return "reuse_chart"
        elif geo_hit and not residual_hit:
            return "update_chart_adapter"
        elif not geo_hit and not residual_hit:
            return "add_chart"
        else:
            return "fallback_free"

