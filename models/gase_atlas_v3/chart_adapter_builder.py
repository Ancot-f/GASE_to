"""Standalone Ridge Chart Adapter Builder for GASE-Atlas v3.

Builds ChartAdapter via:
  1. PCA on residuals ->R, b (output basis)
  2. X^T @ Y SVD ->P (residual-predictive input basis)
  3. Ridge regression in P-space ->B
"""

import torch
from models.gase_atlas_v3.adapters import ChartAdapter


class RidgeChartAdapterBuilder:
    """Build chart adapters with learned P basis (X^T @ Y SVD)."""

    def __init__(self, residual_energy=0.90, residual_dim_min=1, residual_dim_max=8,
                 ridge_lambda=1e-3, p_dim=8, tune_epochs=0, cos_weight=0.1):
        self.residual_energy = residual_energy
        self.residual_dim_min = residual_dim_min
        self.residual_dim_max = residual_dim_max
        self.ridge_lambda = ridge_lambda
        self.p_dim = p_dim
        self.tune_epochs = tune_epochs
        self.cos_weight = cos_weight

    def build(self, chart, features, residuals):
        N = features.shape[0]
        if N < 2:
            return None

        orig_device = features.device
        use_gpu = torch.cuda.is_available() and orig_device.type == 'cpu'
        if use_gpu:
            features = features.cuda()
            residuals = residuals.cuda()
            chart.cuda()

        # Step 1: Residual bias + output basis R
        b = residuals.mean(dim=0)
        Delta = residuals - b

        try:
            _, S_delta, Vh_delta = torch.linalg.svd(Delta, full_matrices=False)
        except RuntimeError:
            if use_gpu:
                chart.cpu()
            return None

        S_sq = (S_delta ** 2) / max(N - 1, 1)
        total_energy = S_sq.sum()
        if total_energy < 1e-8:
            if use_gpu:
                chart.cpu()
            return None

        cumsum = torch.cumsum(S_sq, dim=0)
        s = torch.searchsorted(cumsum / total_energy, self.residual_energy).item() + 1
        s = max(self.residual_dim_min, min(self.residual_dim_max, s, len(S_sq)))
        R = Vh_delta[:s].T.clone()

        # Step 2: Residual-predictive input basis P from X^T @ Y
        chart_features = chart.transform_features(features)
        Xc = chart_features - chart.mu
        Yc = Delta
        M = torch.matmul(Xc.t(), Yc)
        try:
            U_P, S_P, _ = torch.linalg.svd(M, full_matrices=False)
        except RuntimeError:
            r_p = min(self.p_dim, chart.U.shape[1])
            P = chart.U[:, :r_p].clone()
        else:
            r_p = min(self.p_dim, len(S_P))
            P = U_P[:, :r_p].clone()

        # Step 3: Ridge regression in P-space
        Z = torch.matmul(Xc, P)
        Y_out = torch.matmul(Delta, R)
        ZtZ = torch.matmul(Z.t(), Z)
        I = torch.eye(r_p, device=Z.device)
        try:
            W = torch.linalg.solve(ZtZ + self.ridge_lambda * I, torch.matmul(Z.t(), Y_out))
        except RuntimeError:
            W = torch.zeros(r_p, s, device=Z.device)
        B = W.t().clone()

        # Store on chart (copied to adapter then cleared)
        chart.R = R.cpu().clone()
        chart.delta_mean = b.cpu().clone()
        chart.B = B.cpu().clone()
        chart.P_adapter = P.cpu().clone()

        # Compute metrics
        pred_centered = torch.matmul(torch.matmul(Z, B.t()), R.t())
        ss_centered_res = (Delta - pred_centered).pow(2).sum()
        ss_centered_tot = Delta.pow(2).sum()
        chart.centered_r2 = max(0.0, float(1.0 - ss_centered_res / max(ss_centered_tot, 1e-8)))

        pred_full = pred_centered + b
        ss_full_res = (residuals - pred_full).pow(2).sum()
        ss_full_tot = residuals.pow(2).sum()
        chart.full_r2 = max(0.0, float(1.0 - ss_full_res / max(ss_full_tot, 1e-8)))

        if len(S_P) > 0:
            chart.basis_energy = float((S_P[:r_p] ** 2).sum() / max((S_P ** 2).sum(), 1e-8))
        else:
            chart.basis_energy = 0.0

        Y_pred = torch.matmul(Z, B.t())
        ss_sub = Y_out.pow(2).sum()
        chart.subspace_r2 = max(0.0, float(1.0 - (Y_out - Y_pred).pow(2).sum() / max(ss_sub, 1e-8)))

        chart.residual_consistency = chart.centered_r2

        # Routing keys
        with torch.no_grad():
            z_geom_mean = torch.matmul(Xc, chart.U).mean(dim=0).cpu()
            z_adapt_mean = Z.mean(dim=0).cpu()
            resid_proj = torch.matmul(Delta, R).mean(dim=0).cpu()

        if use_gpu:
            chart.cpu()

        adapter = ChartAdapter(chart, P=P.cpu())
        adapter.set_keys(z_geom_mean, z_adapt_mean, resid_proj)
        with torch.no_grad():
            pred = pred_full
            cos = torch.cosine_similarity(
                pred,
                residuals,
                dim=-1,
                eps=1e-8,
            ).mean().item()
            pred_norm = pred.norm(dim=-1).mean().item()
            resid_norm = residuals.norm(dim=-1).mean().item()
        adapter.support = int(N)
        adapter.full_r2 = float(chart.full_r2)
        adapter.subspace_r2 = float(chart.subspace_r2)
        adapter.adapter_cos = float(cos)
        adapter.norm_ratio = float(pred_norm / max(resid_norm, 1e-8))
        adapter.masked = bool(
            adapter.full_r2 <= 0.0 or adapter.adapter_cos <= 0.0
        )
        return adapter

