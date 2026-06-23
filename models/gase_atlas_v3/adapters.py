"""Standalone adapter modules for GASE-Atlas v3 (no v1 dependency)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- TaskAdapter ----

class TaskAdapter(nn.Module):
    """Temporary per-task teacher: up(act(down(h))). Discarded after distillation."""

    def __init__(self, dim=768, bottleneck=16):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


# ---- FreeAdapter ----

class FreeAdapter(nn.Module):
    """Global fallback adapter: up(act(down(h))). Trained on uncovered samples."""

    def __init__(self, dim=768, bottleneck=16):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.down.bias)
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.act(self.down(x)))


# ---- ChartAdapter ----

class ChartAdapter(nn.Module):
    """Angular chart adapter: tangent-space residual on cosine sphere.

    A(h) = gamma * clamp(proj_tangent(b + R@B@P^T@(h-mu), h))
    """

    def __init__(self, chart_state, P=None, gain=1.0, max_delta_ratio=1.5):
        super().__init__()
        self.register_buffer("b", chart_state.delta_mean.clone())
        self.register_buffer("R", chart_state.R.clone())

        s = chart_state.R.shape[1]
        if P is not None:
            r_p = P.shape[1]
            self.register_buffer("P", P.clone())
        else:
            r_p = chart_state.U.shape[1]
            self.register_buffer("P", chart_state.U.clone())

        self.B = nn.Parameter(
            chart_state.B[:, :r_p].clone() if chart_state.B.shape[1] >= r_p
            else torch.zeros(s, r_p)
        )
        self.register_buffer("gain", torch.tensor(gain))
        self.register_buffer("max_delta_ratio", torch.tensor(max_delta_ratio))
        self._tangent_dim = r_p
        self._residual_dim = s
        self.register_buffer("key_geom", torch.zeros(chart_state.U.shape[1]))
        self.register_buffer("key_adapt", torch.zeros(r_p))
        self.register_buffer("key_resid", torch.zeros(s))

    def set_keys(self, z_geom_mean, z_adapt_mean, resid_proj_mean):
        self.key_geom.copy_(z_geom_mean)
        self.key_adapt.copy_(z_adapt_mean)
        self.key_resid.copy_(resid_proj_mean)

    def compatibility(self, z):
        """V6 router compatibility. Returns (input_d2, action_d2)."""
        input_d2 = ((z - self.key_adapt.to(z.device))
                    / self.key_adapt_scale.to(z.device).clamp_min(1e-4)).pow(2).mean(dim=-1)
        dz = torch.matmul(z, self.B.t())
        resid_key = getattr(self, 'key_resid', self.key_adapt)
        resid_scale = getattr(self, 'key_resid_scale', self.key_adapt_scale)
        action_d2 = ((dz - resid_key.to(z.device))
                     / resid_scale.to(z.device).clamp_min(1e-4)).pow(2).mean(dim=-1)
        return input_d2, action_d2

    def forward(self, x, chart):
        chart_x = chart.transform_features(x)
        z = torch.matmul(chart_x - chart.mu, self.P)
        a = torch.matmul(z, self.B.t())
        delta_raw = torch.matmul(a, self.R.t()) + self.b

        # Tangent projection: remove radial component
        x_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        x_hat = x / x_norm
        radial = (delta_raw * x_hat).sum(dim=-1, keepdim=True)
        delta_tangent = delta_raw - radial * x_hat

        # Norm clamp
        delta_norm = delta_tangent.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        max_norm = self.max_delta_ratio * x_norm
        scale = torch.min(max_norm / delta_norm, torch.ones_like(delta_norm))

        return float(self.gain) * delta_tangent * scale

    @property
    def tangent_dim(self):
        return self._tangent_dim

    @property
    def residual_dim(self):
        return self._residual_dim

    @property
    def trainable_param_count(self):
        return self.B.numel()


# ---- ChartMLPAdapter ----

class ChartMLPAdapter(nn.Module):
    """Angular MLP adapter: tangent-space residual via nonlinear mapping.

    A(h) = gamma * clamp(proj_tangent(b + scale*R@W2@gelu(W1@P^T@(h-mu)), h))
    Zero-init W2 ->safe initialization. Tangent projection removes radial.
    """

    def __init__(self, chart_state, P=None, hidden=16, init_scale=0.05,
                 gain=1.0, max_delta_ratio=1.5):
        super().__init__()
        self.register_buffer("b", chart_state.delta_mean.clone())
        self.register_buffer("R", chart_state.R.clone())

        if P is not None:
            r_p = P.shape[1]
            self.register_buffer("P", P.clone())
        else:
            r_p = chart_state.U.shape[1]
            self.register_buffer("P", chart_state.U.clone())

        s = chart_state.R.shape[1]
        h = min(hidden, max(r_p, s) * 2)
        self.W1 = nn.Parameter(torch.randn(h, r_p) * 0.01)
        self.b1 = nn.Parameter(torch.zeros(h))
        self.W2 = nn.Parameter(torch.zeros(s, h))
        self.b2 = nn.Parameter(torch.zeros(s))
        self.log_scale = nn.Parameter(torch.tensor(init_scale).log())
        self.register_buffer("gain", torch.tensor(gain))
        self.register_buffer("max_delta_ratio", torch.tensor(max_delta_ratio))
        self._tangent_dim = r_p
        self._residual_dim = s
        self.register_buffer("key_geom", torch.zeros(chart_state.U.shape[1]))
        self.register_buffer("key_adapt", torch.zeros(r_p))
        self.register_buffer("key_resid", torch.zeros(s))

    def set_keys(self, z_geom_mean, z_adapt_mean, resid_proj_mean):
        self.key_geom.copy_(z_geom_mean)
        self.key_adapt.copy_(z_adapt_mean)
        self.key_resid.copy_(resid_proj_mean)

    def forward(self, x, chart):
        chart_x = chart.transform_features(x)
        z = torch.matmul(chart_x - chart.mu, self.P)
        h = F.gelu(F.linear(z, self.W1, self.b1))
        a = F.linear(h, self.W2, self.b2)
        delta_raw = self.log_scale.exp() * torch.matmul(a, self.R.t()) + self.b

        # Tangent projection
        x_norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        x_hat = x / x_norm
        radial = (delta_raw * x_hat).sum(dim=-1, keepdim=True)
        delta_tangent = delta_raw - radial * x_hat

        # Norm clamp
        delta_norm = delta_tangent.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
        max_norm = self.max_delta_ratio * x_norm
        scale = torch.min(max_norm / delta_norm, torch.ones_like(delta_norm))

        return float(self.gain) * delta_tangent * scale

    @property
    def scale_value(self):
        return float(self.log_scale.exp().item())

    @property
    def tangent_dim(self):
        return self._tangent_dim

    @property
    def residual_dim(self):
        return self._residual_dim

    @property
    def trainable_param_count(self):
        return sum(p.numel() for p in self.parameters())
# ---- ChartRouter (adapter selector) ----

class ChartRouter(nn.Module):
    """Per-chart adapter selector: maps [z_geom, z_adapt] ->adapter_slot logits."""

    def __init__(self, num_adapters, r_geom=8, r_adapt=8, hidden=32):
        super().__init__()
        in_dim = r_geom + r_adapt
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, num_adapters),
        )

    def forward(self, z_geom, z_adapt):
        x = torch.cat([z_geom, z_adapt], dim=-1)
        return self.net(x)

