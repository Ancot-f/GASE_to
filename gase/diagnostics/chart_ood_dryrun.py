"""
Chart OOD Dry-run Diagnostics (Phase-9.8).

Builds candidate charts from cached/distilled h_chart features using
multiple geometry-only methods, then evaluates chart quality, overlap,
purity, routing, and oracle upper bound — without modifying the model.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from gase.atlas.chart_state import ChartState
from gase.atlas.ppca import PPCAEstimator


# ------------------------------------------------------------------
#  Torch-based KMeans (no sklearn dependency)
# ------------------------------------------------------------------

def torch_kmeans(x: Tensor, k: int, num_iters: int = 50, seed: int = 1993) -> Tuple[Tensor, Tensor]:
    """KMeans clustering in PyTorch.

    Args:
        x: [N, D] float tensor (CPU or CUDA).
        k: number of clusters.
        num_iters: max iterations.
        seed: random seed.

    Returns:
        labels: [N] long tensor with cluster assignments.
        centers: [k, D] float tensor.
    """
    N = x.shape[0]
    if k <= 0 or N < k:
        return torch.zeros(N, dtype=torch.long, device=x.device), x[:1].clone()

    # Normalize input for stable distances
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True).clamp_min(1e-10)
    x_norm = (x - x_mean) / x_std

    # Initialize centers with random samples
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)
    perm = torch.randperm(N, generator=g, device=x.device)
    centers = x_norm[perm[:k]].clone()

    labels = torch.zeros(N, dtype=torch.long, device=x.device)
    for _ in range(num_iters):
        # Assignment
        dist = torch.cdist(x_norm, centers)  # [N, k]
        new_labels = dist.argmin(dim=1)

        # Check convergence
        if (new_labels == labels).all():
            break
        labels = new_labels

        # Update centers
        for c in range(k):
            mask = labels == c
            if mask.sum() > 0:
                centers[c] = x_norm[mask].mean(dim=0)
            else:
                # Reinitialize empty cluster with farthest point
                farthest_idx = dist.max(dim=1).values.argmax()
                centers[c] = x_norm[farthest_idx]

    # Map labels back to original space cluster means
    centers_orig = torch.zeros(k, x.shape[1], device=x.device)
    for c in range(k):
        mask = labels == c
        if mask.sum() > 0:
            centers_orig[c] = x[mask].mean(dim=0)
        else:
            centers_orig[c] = x[0]

    return labels, centers_orig


def _compute_accuracy_from_preds(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Safely compute 0-100 accuracy from prediction arrays.

    Handles both [N] and [N, topk] shapes for y_pred.
    """
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]  # take top-1 only
    correct = int((y_pred == y_true).sum())
    total_count = len(y_true)
    if total_count == 0:
        return 0.0
    acc = 100.0 * correct / total_count
    assert 0.0 <= acc <= 100.0, f"Invalid accuracy: {acc}"
    return acc


class ChartOODDryRunner:
    """Dry-run multi-chart diagnostics. Read-only with respect to the model."""

    def __init__(self, backbone, atlas_layers, config: dict):
        self.backbone = backbone
        self.atlas_layers = atlas_layers
        self.config = config
        self.rank = config.get("rank", 8)
        self.min_support = config.get("min_support", 32)
        self.radius_quantile = config.get("radius_quantile", 0.95)
        self.regularize_sigma = config.get("regularize_sigma", 1e-4)
        self.methods = config.get("methods", ["kmeans_pca"])
        self.k_list = config.get("k_list", [1, 2, 3])

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------

    def extract_layer_features(self, data_loader, layer_id: int) -> Tuple[Tensor, Tensor]:
        """Extract h_chart and labels at a specific layer. Returns (h [N,D], labels [N])."""
        self.backbone.eval()
        h_list, label_list = [], []
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(next(self.backbone.parameters()).device)
                h = self.backbone._extract_h_chart_at_layer(inputs, layer_id)
                h_list.append(h.cpu())
                label_list.append(targets)
        return torch.cat(h_list, dim=0), torch.cat(label_list, dim=0)

    def extract_all_layer_features(self, data_loader) -> Dict[int, Tuple[Tensor, Tensor]]:
        """Extract features for all atlas layers."""
        result = {}
        for lid in self.atlas_layers:
            h, labels = self.extract_layer_features(data_loader, lid)
            result[lid] = (h, labels)
            logging.info("[DryChartExtract] layer=%d N=%d D=%d", lid, h.shape[0], h.shape[1])
        return result

    # ------------------------------------------------------------------
    #  Chart building helpers
    # ------------------------------------------------------------------

    def _build_ppca_chart(self, h_local: Tensor, chart_id: int, layer_id: int) -> Optional[ChartState]:
        """Build a single PPCA chart from local features."""
        N = h_local.shape[0]
        if N < self.min_support:
            return None

        ppca = PPCAEstimator(dim=h_local.shape[1], rank=self.rank)
        ppca.fit(h_local, self.rank)

        cs = ChartState(chart_id=chart_id, layer_id=layer_id)
        cs.mu = ppca.mu.clone().detach()
        cs.U = ppca.U.clone().detach()
        cs.eigvals = ppca.eigvals.clone().detach()
        cs.sigma_perp = max(ppca.sigma_perp, self.regularize_sigma)
        cs.radius_d2 = ppca.radius_d2
        cs.n_support = N
        cs.Q_router = cs.U
        cs.router_eigvals = cs.eigvals.clone()
        cs.router_rank = self.rank

        return cs

    def _compute_mahalanobis_d2(self, h: Tensor, cs: ChartState) -> Tensor:
        """PPCA Mahalanobis d2, matching PPCAEstimator._ppca_mahalanobis_d2 formula.

        d^2 = sum_k (z_k^2 / (eigval_k + eps)) + ||residual||^2 / (sigma_perp^2 + eps)
        """
        X = h.to(cs.mu.device) - cs.mu.unsqueeze(0)
        z = X @ cs.U.to(h.device)  # [N, rank]
        ev = cs.eigvals if cs.eigvals is not None else torch.ones(cs.U.shape[1], device=h.device)
        ev_safe = ev.clamp_min(1e-8).to(h.device)
        tangent_term = (z ** 2 / ev_safe.unsqueeze(0)).sum(dim=-1)
        residual = X - z @ cs.U.T.to(h.device)
        residual_sq = (residual ** 2).sum(dim=-1)
        normal_term = residual_sq / max(cs.sigma_perp ** 2, 1e-8)
        return tangent_term + normal_term

    # ------------------------------------------------------------------
    #  Method 1: KMeans + local PCA
    # ------------------------------------------------------------------

    def build_kmeans_pca_charts(self, h_all: Tensor, layer_id: int, k: int) -> List[ChartState]:
        """Cluster features with torch KMeans, build one PPCA chart per cluster."""
        labels, centers = torch_kmeans(h_all, k=k, num_iters=50)

        charts = []
        for cid in range(k):
            mask = labels == cid
            n_c = int(mask.sum())
            if n_c < self.min_support:
                logging.info("[DryChartBuild][kmeans_pca] layer=%d k=%d chart=%d "
                             "support=%d < min_support=%d, skipping",
                             layer_id, k, cid, n_c, self.min_support)
                continue
            h_local = h_all[mask]
            cs = self._build_ppca_chart(h_local, chart_id=cid, layer_id=layer_id)
            if cs is not None:
                charts.append(cs)
                logging.info("[DryChartBuild][kmeans_pca] layer=%d k=%d chart=%d support=%d "
                             "radius_d2=%.1f sigma_perp=%.4f",
                             layer_id, k, cid, cs.n_support, cs.radius_d2, cs.sigma_perp)

        return charts

    # ------------------------------------------------------------------
    #  Method 2: Mahalanobis outlier split
    # ------------------------------------------------------------------

    def build_mahalanobis_outlier_charts(self, h_all: Tensor, layer_id: int,
                                          existing_chart: ChartState) -> List[ChartState]:
        """Split outliers from an existing chart by d2 threshold."""
        d2 = self._compute_mahalanobis_d2(h_all, existing_chart)
        q90 = float(torch.quantile(d2, 0.90))
        core_mask = d2 <= q90
        outlier_mask = d2 > q90

        n_core = int(core_mask.sum())
        n_out = int(outlier_mask.sum())
        logging.info("[DryChartBuild][mahalanobis_outlier_split] layer=%d "
                     "core_support=%d outlier_support=%d q90_d2=%.1f",
                     layer_id, n_core, n_out, q90)

        charts = []
        cs_core = self._build_ppca_chart(h_all[core_mask], chart_id=0, layer_id=layer_id)
        if cs_core is not None:
            charts.append(cs_core)

        if n_out >= self.min_support:
            h_out = h_all[outlier_mask]
            n_split = min(3, n_out // self.min_support)
            if n_split >= 2:
                ol, _ = torch_kmeans(h_out, k=n_split, num_iters=50)
                for cid in range(n_split):
                    h_local = h_out[ol == cid]
                    cs = self._build_ppca_chart(h_local, chart_id=len(charts), layer_id=layer_id)
                    if cs is not None:
                        charts.append(cs)
            else:
                cs = self._build_ppca_chart(h_out, chart_id=len(charts), layer_id=layer_id)
                if cs is not None:
                    charts.append(cs)

        return charts

    # ------------------------------------------------------------------
    #  Method 3: Spectral PCA split
    # ------------------------------------------------------------------

    def build_spectral_pca_charts(self, h_all: Tensor, layer_id: int, k: int) -> List[ChartState]:
        """Project to PCA space first, then torch_kmeans in PCA space."""
        # First build a single chart to get U
        cs_full = self._build_ppca_chart(h_all, chart_id=-1, layer_id=layer_id)
        if cs_full is None:
            return []

        # Project all features to PCA space
        X = h_all - cs_full.mu.unsqueeze(0)
        z = X @ cs_full.U  # [N, rank]

        actual_k = min(k, h_all.shape[0] // self.min_support)
        spec_labels, _ = torch_kmeans(z, k=actual_k, num_iters=50)

        charts = []
        for cid in range(actual_k):
            mask = spec_labels == cid
            n_c = int(mask.sum())
            if n_c < self.min_support:
                continue
            h_local = h_all[mask]
            cs = self._build_ppca_chart(h_local, chart_id=cid, layer_id=layer_id)
            if cs is not None:
                charts.append(cs)
                logging.info("[DryChartBuild][spectral_pca_split] layer=%d k=%d chart=%d support=%d "
                             "(using PCA-space torch_kmeans)",
                             layer_id, k, cid, cs.n_support)

        return charts

    # ------------------------------------------------------------------
    #  Build all charts for a method and K
    # ------------------------------------------------------------------

    def build_candidate_charts(self, h_all: Tensor, layer_id: int,
                                method: str, k: int) -> List[ChartState]:
        """Dispatch to the correct build method."""
        if method == "kmeans_pca":
            return self.build_kmeans_pca_charts(h_all, layer_id, k)
        elif method == "mahalanobis_outlier_split":
            cs0 = self._build_ppca_chart(h_all, chart_id=0, layer_id=layer_id)
            if cs0 is None:
                return []
            return self.build_mahalanobis_outlier_charts(h_all, layer_id, cs0)
        elif method == "spectral_pca_split":
            return self.build_spectral_pca_charts(h_all, layer_id, k)
        else:
            raise ValueError(f"Unknown method: {method}")

    # ------------------------------------------------------------------
    #  Quality diagnostics
    # ------------------------------------------------------------------

    def compute_quality_metrics(self, charts: List[ChartState], h_all: Tensor,
                                 layer_id: int, method: str, k: int) -> Dict:
        """Compute per-chart and aggregate quality metrics."""
        if not charts:
            return {}

        metrics = {"method": method, "layer": layer_id, "k": k, "num_charts": len(charts)}
        d2_means, d2_q50s, d2_q90s, d2_q95s, recon_means, cond_nums = [], [], [], [], [], []
        supports = []

        # Assign samples to nearest chart
        all_d2 = []
        for cs in charts:
            d2 = self._compute_mahalanobis_d2(h_all, cs)
            all_d2.append(d2)
        all_d2_stack = torch.stack(all_d2, dim=1)  # [N, num_charts]
        assignments = all_d2_stack.argmin(dim=1)

        for ci, cs in enumerate(charts):
            mask = assignments == ci
            n_assigned = int(mask.sum())
            if n_assigned == 0:
                continue
            d2_assigned = all_d2_stack[mask, ci]
            d2_means.append(float(d2_assigned.mean()))
            d2_q50s.append(float(torch.quantile(d2_assigned, 0.50)))
            d2_q90s.append(float(torch.quantile(d2_assigned, 0.90)))
            d2_q95s.append(float(torch.quantile(d2_assigned, 0.95)))
            supports.append(n_assigned)

            recon = cs.sigma_perp * self.rank
            recon_means.append(recon)

            if cs.eigvals is not None and len(cs.eigvals) > 1:
                ev = cs.eigvals.clamp_min(1e-10)
                cond = float(ev.max() / ev.min())
            else:
                cond = 1.0
            cond_nums.append(cond)

        metrics["mean_d2"] = float(np.mean(d2_means)) if d2_means else 0.0
        metrics["q50_d2"] = float(np.mean(d2_q50s)) if d2_q50s else 0.0
        metrics["q90_d2"] = float(np.mean(d2_q90s)) if d2_q90s else 0.0
        metrics["q95_d2"] = float(np.mean(d2_q95s)) if d2_q95s else 0.0
        # Coverage: fraction of samples within radius of at least one chart
        covered = torch.zeros(h_all.shape[0], dtype=torch.bool)
        for cs in charts:
            d2 = self._compute_mahalanobis_d2(h_all, cs)
            covered |= (d2 <= cs.radius_d2)
        metrics["coverage_at_radius"] = float(covered.float().mean())
        metrics["recon_error_mean"] = float(np.mean(recon_means)) if recon_means else 0.0
        metrics["recon_error_q90"] = float(np.percentile(recon_means, 90)) if len(recon_means) > 1 else (recon_means[0] if recon_means else 0.0)
        metrics["condition_number_mean"] = float(np.mean(cond_nums)) if cond_nums else 0.0
        metrics["min_support"] = int(np.min(supports)) if supports else 0
        metrics["max_support"] = int(np.max(supports)) if supports else 0

        logging.info("[DryChartQuality] method=%s layer=%d k=%d "
                     "mean_d2=%.1f q50_d2=%.1f q90_d2=%.1f q95_d2=%.1f "
                     "coverage=%.3f recon_mean=%.4f recon_q90=%.4f cond_mean=%.1f "
                     "min_support=%d max_support=%d num_charts=%d",
                     method, layer_id, k,
                     metrics["mean_d2"], metrics["q50_d2"], metrics["q90_d2"], metrics["q95_d2"],
                     metrics["coverage_at_radius"], metrics["recon_error_mean"],
                     metrics["recon_error_q90"], metrics["condition_number_mean"],
                     metrics["min_support"], metrics["max_support"], len(charts))

        return metrics

    def compute_quality_gain(self, dry_metrics: Dict, single_metrics: Dict,
                              layer_id: int, method: str, k: int):
        """Compare dry-run charts vs single chart."""
        single_d2 = single_metrics.get("mean_d2", float("inf"))
        dry_d2 = dry_metrics.get("mean_d2", float("inf"))
        gain = (single_d2 - dry_d2) / (single_d2 + 1e-10) * 100 if single_d2 > 0 else 0

        single_recon = single_metrics.get("recon_error_mean", float("inf"))
        dry_recon = dry_metrics.get("recon_error_mean", float("inf"))
        recon_gain = (single_recon - dry_recon) / (single_recon + 1e-10) * 100 if single_recon > 0 else 0

        logging.info("[DryChartQualityGain] layer=%d k=%d method=%s "
                     "single_mean_d2=%.1f dry_mean_d2=%.1f gain_d2=%.1f%% "
                     "single_recon=%.4f dry_recon=%.4f gain_recon=%.1f%%",
                     layer_id, k, method, single_d2, dry_d2, gain,
                     single_recon, dry_recon, recon_gain)

        return {"gain_d2_pct": gain, "gain_recon_pct": recon_gain}

    # ------------------------------------------------------------------
    #  Overlap diagnostics
    # ------------------------------------------------------------------

    def compute_overlap(self, charts: List[ChartState], h_all: Tensor,
                         layer_id: int, method: str, k: int) -> Dict:
        """Compute chart overlap and boundary metrics."""
        if len(charts) < 2:
            logging.info("[DryChartOverlap] method=%s layer=%d k=%d num_charts=%d skip",
                        method, layer_id, k, len(charts))
            return {"overlap_ratio": 0.0, "boundary_ratio": 0.0}

        N = h_all.shape[0]
        # Count how many charts each sample falls within radius of
        in_radius = torch.zeros(N, len(charts), dtype=torch.bool)
        d2_all = torch.zeros(N, len(charts))

        for i, cs in enumerate(charts):
            d2 = self._compute_mahalanobis_d2(h_all, cs)
            d2_all[:, i] = d2
            in_radius[:, i] = d2 <= cs.radius_d2

        n_in_radius = in_radius.sum(dim=1)  # per sample
        overlap_ratio = float((n_in_radius > 1).float().mean())

        # Boundary: samples on edge of best chart (margin small)
        best_d2 = d2_all.min(dim=1).values
        d2_filled = d2_all.clone()
        d2_filled.scatter_(1, d2_all.argmin(dim=1, keepdim=True), float("inf"))
        second_d2 = d2_filled.min(dim=1).values
        margin = second_d2 - best_d2
        boundary_ratio = float((margin < best_d2 * 0.1).float().mean())

        logging.info("[DryChartOverlap] method=%s layer=%d k=%d "
                     "overlap_ratio=%.3f boundary_ratio=%.3f "
                     "assign_margin_mean=%.1f assign_margin_q50=%.1f assign_margin_q75=%.1f",
                     method, layer_id, k, overlap_ratio, boundary_ratio,
                     float(margin.mean()), float(torch.quantile(margin, 0.50)),
                     float(torch.quantile(margin, 0.75)))

        return {"overlap_ratio": overlap_ratio, "boundary_ratio": boundary_ratio,
                "assign_margin_mean": float(margin.mean()),
                "assign_margin_q50": float(torch.quantile(margin, 0.50))}

    # ------------------------------------------------------------------
    #  Purity diagnostics
    # ------------------------------------------------------------------

    def compute_purity(self, charts: List[ChartState], h_all: Tensor,
                        labels: Tensor, layer_id: int, method: str, k: int,
                        increment: int = 10) -> Dict:
        """Per-chart source-task distribution and purity."""
        if len(charts) < 2:
            return {"mean_purity": 1.0, "chart_purities": []}

        N = h_all.shape[0]
        d2_all = torch.zeros(N, len(charts))
        for i, cs in enumerate(charts):
            d2_all[:, i] = self._compute_mahalanobis_d2(h_all, cs)
        assignments = d2_all.argmin(dim=1)  # [N]

        source_tasks = labels // increment
        purity_summary = []
        purities = []

        for cid, cs in enumerate(charts):
            mask = assignments == cid
            if mask.sum() == 0:
                continue
            src = source_tasks[mask]
            unique, counts = torch.unique(src, return_counts=True)
            hist = {int(u): int(c) for u, c in zip(unique, counts)}
            purity = float(counts.max() / counts.sum())
            purities.append(purity)
            purity_summary.append((cid, hist, purity))

        mean_purity = float(np.mean(purities)) if purities else 0.0

        logging.info("[DryChartPurity] method=%s layer=%d k=%d mean_purity=%.3f",
                     method, layer_id, k, mean_purity)
        for cid, hist, purity in purity_summary:
            logging.info("[DryChartPurity]   chart=%d hist=%s purity=%.3f", cid, hist, purity)

        return {"mean_purity": mean_purity, "chart_purities": purity_summary}

    # ------------------------------------------------------------------
    #  Chart routing diagnostics
    # ------------------------------------------------------------------

    def compute_chart_routing(self, charts: List[ChartState], h_all: Tensor,
                               labels: Tensor, layer_id: int, method: str, k: int,
                               increment: int = 10) -> Dict:
        """Compute per-sample chart assignment and margin."""
        if not charts:
            return {}

        N = h_all.shape[0]
        d2_all = torch.zeros(N, len(charts))
        for i, cs in enumerate(charts):
            d2_all[:, i] = self._compute_mahalanobis_d2(h_all, cs)

        assignments = d2_all.argmin(dim=1)
        best_d2 = d2_all.min(dim=1).values
        d2_filled = d2_all.clone()
        d2_filled.scatter_(1, assignments.unsqueeze(1), float("inf"))
        second_d2 = d2_filled.min(dim=1).values
        margin = second_d2 - best_d2

        # Overall chart histogram
        chart_ids = [cs.chart_id for cs in charts]
        hist = {int(c): int((assignments == i).sum()) for i, c in enumerate(chart_ids)}

        logging.info("[DryChartRouting] method=%s layer=%d k=%d chart_hist=%s "
                     "margin_mean=%.1f margin_q50=%.1f margin_q90=%.1f",
                     method, layer_id, k, hist,
                     float(margin.mean()), float(torch.quantile(margin, 0.50)),
                     float(torch.quantile(margin, 0.90)))

        # Per source task
        source_tasks = labels // increment
        for src in sorted(source_tasks.unique().tolist()):
            smask = source_tasks == src
            src_hist = {int(c): int((assignments[smask] == i).sum())
                       for i, c in enumerate(chart_ids)}
            logging.info("[DryChartRoutingByTask] method=%s layer=%d k=%d source=%d chart_hist=%s",
                        method, layer_id, k, src, src_hist)

        return {"chart_hist": hist, "assignments": assignments,
                "margin_mean": float(margin.mean()),
                "margin_q50": float(torch.quantile(margin, 0.50))}

    # ------------------------------------------------------------------
    #  Chart-slot routing diagnostics
    # ------------------------------------------------------------------

    def compute_chart_slot_routing(self, charts: List[ChartState], h_all: Tensor,
                                    labels: Tensor, layer_id: int, method: str, k: int,
                                    increment: int = 10) -> Dict:
        """Within each chart, compute per-sample slot routing using raw NLL."""
        from gase.routing.nll_router import CalibratedNLLSlotRouter

        if not charts:
            return {}

        blk = self.backbone.get_block(layer_id)
        available_slots = blk.get_available_slot_ids(0)
        if len(available_slots) <= 1:
            logging.info("[DryChartSlotRouting] layer=%d no multi-slot, skip", layer_id)
            return {}

        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        cs_orig = blk.chart_states.get(0)

        N = h_all.shape[0]
        # Chart assignment
        d2_all = torch.zeros(N, len(charts))
        for i, cs in enumerate(charts):
            d2_all[:, i] = self._compute_mahalanobis_d2(h_all, cs)
        chart_assign = d2_all.argmin(dim=1)

        source_tasks = labels // increment
        overall_slot_hist = {}

        for cid, cs in enumerate(charts):
            cmask = chart_assign == cid
            if cmask.sum() == 0:
                continue
            h_chart_local = h_all[cmask]

            # Slot routing within this chart using original chart's Q_router + slots
            # We use the chart's own Q_router for projection
            slot_states = {sid: blk.slot_states.get(f"0_{sid}")
                          for sid in available_slots
                          if blk.slot_states.get(f"0_{sid}") is not None}

            if not slot_states:
                continue

            # Route using the original chart's Q_router
            routing = router.route(h_chart_local, cs_orig, slot_states)
            slot_ids = routing["slot_ids"]

            local_hist = {}
            for sid in available_slots:
                cnt = int((slot_ids == sid).sum())
                if cnt > 0:
                    local_hist[int(sid)] = cnt
                    overall_slot_hist[int(sid)] = overall_slot_hist.get(int(sid), 0) + cnt

            logging.info("[DryChartSlotRouting] method=%s layer=%d k=%d chart=%d slot_hist=%s",
                        method, layer_id, k, cid, local_hist)

            # Per source task
            src_local = source_tasks[cmask]
            for src in sorted(src_local.unique().tolist()):
                smask = src_local == src
                src_sids = slot_ids[smask]
                src_hist = {int(s): int((src_sids == s).sum()) for s in available_slots if (src_sids == s).sum() > 0}
                logging.info("[DryChartSlotRoutingByTask] method=%s layer=%d k=%d chart=%d source=%d slot_hist=%s",
                            method, layer_id, k, cid, src, src_hist)

        logging.info("[DryChartSlotRouting] overall_slot_hist=%s", overall_slot_hist)
        return {"overall_slot_hist": overall_slot_hist}

    # ------------------------------------------------------------------
    #  Dry-run eval (chart reroute only)
    # ------------------------------------------------------------------

    def dryrun_chart_slot_eval(self, charts: List[ChartState], data_loader,
                                layer_id: int, method: str, k: int,
                                total_classes: int, topk: int = 1) -> Optional[Dict]:
        """Lightweight eval with dry-run chart + existing slot routing.

        Returns None if eval is not safe (adapters bound to original chart).
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self.backbone
        blk = backbone.get_block(layer_id)
        cs_orig = blk.chart_states.get(0)
        available_slots = blk.get_available_slot_ids(0)

        if not available_slots or not charts:
            return None

        # This eval is a routing-only diagnostic: we use the existing chart's
        # Q_router and slot adapters but route to the best dry-run chart first.
        # Since adapters are bound to chart=0, we can only evaluate routing
        # behavior, not actual adapter output with multi-chart.

        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        slot_states = {sid: blk.slot_states.get(f"0_{sid}")
                      for sid in available_slots
                      if blk.slot_states.get(f"0_{sid}") is not None}

        backbone.eval()
        all_preds, all_labels = [], []
        device = next(backbone.parameters()).device

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(device)
                B = inputs.shape[0]
                h = backbone._extract_h_chart_at_layer(inputs, layer_id)

                # Assign to nearest dry chart
                d2_per_chart = torch.zeros(B, len(charts))
                for i, cs in enumerate(charts):
                    d2_per_chart[:, i] = self._compute_mahalanobis_d2(h, cs)
                chart_assign = d2_per_chart.argmin(dim=1)  # [B]

                # Within each chart, route slots using existing slot adapters
                # but still bound to chart=0's adapter path
                routing = router.route(h, cs_orig, slot_states)
                slot_ids = routing["slot_ids"]  # [B]

                # Forward with path_key_slot_student
                backbone._clear_path_slot_ids()
                for lid in self.atlas_layers:
                    backbone.blocks[lid].path_slot_id = slot_ids
                backbone.set_adapter_mode("path_key_slot_student")
                try:
                    logits = backbone.forward(inputs)["logits"]
                finally:
                    backbone.set_adapter_mode("task_train")

                logits = logits[:, :total_classes]
                topk_preds = torch.topk(logits, k=topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        total_acc = _compute_accuracy_from_preds(y_pred, y_true)

        logging.info("[DryRunEval] method=%s k=%d layer=%d total=%.2f",
                    method, k, layer_id, total_acc)

        return {"top1": total_acc}

    # ------------------------------------------------------------------
    #  Oracle chart eval
    # ------------------------------------------------------------------

    def oracle_chart_eval(self, charts: List[ChartState], data_loader,
                           layer_id: int, method: str, k: int,
                           total_classes: int, raw_nll_total: float,
                           path_nll_total: float, topk: int = 1) -> Dict:
        """Oracle: pick the chart that gives the best classification for each sample.

        This is an upper bound — it uses the actual label to choose the chart.
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self.backbone
        blk = backbone.get_block(layer_id)
        cs_orig = blk.chart_states.get(0)
        available_slots = blk.get_available_slot_ids(0)

        if len(charts) < 2 or not available_slots:
            return {"total": 0, "gain_over_raw": 0, "gain_over_path": 0}

        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        slot_states = {sid: blk.slot_states.get(f"0_{sid}")
                      for sid in available_slots
                      if blk.slot_states.get(f"0_{sid}") is not None}

        backbone.eval()
        all_preds, all_labels = [], []
        device = next(backbone.parameters()).device

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(device)
                B = inputs.shape[0]
                h = backbone._extract_h_chart_at_layer(inputs, layer_id)

                # Try each chart, keep best logits per sample
                best_logits = torch.full([B, total_classes], -float("inf"), device=device)
                for cs in charts:
                    # Within this chart, route slots
                    routing = router.route(h, cs_orig, slot_states)
                    slot_ids = routing["slot_ids"]

                    backbone._clear_path_slot_ids()
                    for lid in self.atlas_layers:
                        backbone.blocks[lid].path_slot_id = slot_ids
                    backbone.set_adapter_mode("path_key_slot_student")
                    try:
                        logits = backbone.forward(inputs)["logits"][:, :total_classes]
                    finally:
                        backbone.set_adapter_mode("task_train")
                    best_logits = torch.max(best_logits, logits)

                topk_preds = torch.topk(best_logits, k=topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        increment = self.config.get("increment", 10)

        # Safe accuracy computation (handles broadcasting)
        total = _compute_accuracy_from_preds(y_pred, y_true)
        assert 0.0 <= total <= 100.0, f"[DryRunOracleChartEval][ERROR] invalid accuracy scale total={total}"

        gain_raw = total - raw_nll_total
        gain_path = total - path_nll_total

        logging.info("[DryRunOracleChartEval] method=%s k=%d layer=%d total=%.2f "
                    "OracleGainOverRaw=%+.2f OracleGainOverPathRaw=%+.2f",
                    method, k, layer_id, total, gain_raw, gain_path)

        # Per-task breakdown
        for t_start in range(0, total_classes, increment):
            t_end = t_start + increment
            t_mask = (y_true >= t_start) & (y_true < t_end)
            if t_mask.sum() > 0:
                t_acc = _compute_accuracy_from_preds(y_pred[t_mask], y_true[t_mask])
                t_key = f"{t_start:02d}-{t_end-1:02d}"
                logging.info("[DryRunOracleChartEval] %s=%.2f", t_key, t_acc)

        return {"total": total, "gain_over_raw": gain_raw, "gain_over_path": gain_path}

    # ------------------------------------------------------------------
    #  Chart creation proposal
    # ------------------------------------------------------------------

    def propose_chart_creation(self, quality_results: Dict, purity_results: Dict,
                                overlap_results: Dict, slot_bias_change: Dict,
                                layer_id: int, method: str, k: int,
                                oracle_gain_over_path: float) -> Dict:
        """Recommend whether to create multi-chart based on dry-run results."""
        quality_gain = quality_results.get("gain_d2_pct", 0)
        purity = purity_results.get("mean_purity", 0)
        overlap = overlap_results.get("overlap_ratio", 1.0)
        slot0_ratio_change = slot_bias_change.get("slot0_ratio_change", 0)

        recommend = (
            quality_gain > 20 and
            purity > 0.60 and
            overlap < 0.35 and
            slot0_ratio_change < -20 and
            oracle_gain_over_path > 0.5
        )

        reasons = []
        if quality_gain <= 20:
            reasons.append(f"d2_gain={quality_gain:.1f}% <= 20%")
        if purity <= 0.60:
            reasons.append(f"purity={purity:.3f} <= 0.60")
        if overlap >= 0.35:
            reasons.append(f"overlap={overlap:.3f} >= 0.35")
        if slot0_ratio_change >= -20:
            reasons.append(f"slot0_ratio_change={slot0_ratio_change:.1f}% > -20%")
        if oracle_gain_over_path <= 0.5:
            reasons.append(f"oracle_gain={oracle_gain_over_path:.2f} <= 0.5")

        logging.info("[ChartCreationProposal] layer=%d method=%s k=%d recommend=%s reason=%s",
                    layer_id, method, k, recommend,
                    "; ".join(reasons) if reasons else "all criteria met")

        return {"recommend": recommend, "reasons": reasons,
                "quality_gain": quality_gain, "purity": purity,
                "overlap": overlap, "slot0_ratio_change": slot0_ratio_change}

    # ------------------------------------------------------------------
    #  Phase-9.8-radius: Radius sweep and hard Voronoi
    # ------------------------------------------------------------------

    def _build_ppca_chart_with_radius(self, h_local: Tensor, chart_id: int, layer_id: int,
                                       radius_quantile: float) -> Optional[ChartState]:
        """Build PPCA chart with explicit radius quantile."""
        N = h_local.shape[0]
        if N < self.min_support:
            return None

        ppca = PPCAEstimator(dim=h_local.shape[1], rank=self.rank)
        ppca.fit(h_local, self.rank)

        cs = ChartState(chart_id=chart_id, layer_id=layer_id)
        cs.mu = ppca.mu.clone().detach()
        cs.U = ppca.U.clone().detach()
        cs.eigvals = ppca.eigvals.clone().detach()
        cs.sigma_perp = max(ppca.sigma_perp, self.regularize_sigma)
        cs.n_support = N
        cs.Q_router = cs.U
        cs.router_eigvals = cs.eigvals.clone()
        cs.router_rank = self.rank

        # Recompute radius with the specified quantile
        d2_train = self._compute_mahalanobis_d2(h_local, cs)
        cs.radius_d2 = float(torch.quantile(d2_train, radius_quantile))

        return cs

    def build_charts_with_radius(self, h_all: Tensor, layer_id: int, method: str,
                                  k: int, radius_quantile: float) -> List[ChartState]:
        """Build candidate charts with a specific radius quantile."""
        if method in ("kmeans_pca", "spectral_pca_split"):
            if method == "kmeans_pca":
                labels, _ = torch_kmeans(h_all, k=k, num_iters=50)
            else:
                cs_full = self._build_ppca_chart(h_all, chart_id=-1, layer_id=layer_id)
                if cs_full is None:
                    return []
                X = h_all - cs_full.mu.unsqueeze(0)
                z = X @ cs_full.U
                actual_k = min(k, h_all.shape[0] // self.min_support)
                labels, _ = torch_kmeans(z, k=actual_k, num_iters=50)

            charts = []
            for cid in range(k):
                mask = labels == cid
                n_c = int(mask.sum())
                if n_c < self.min_support:
                    continue
                cs = self._build_ppca_chart_with_radius(h_all[mask], chart_id=cid,
                                                        layer_id=layer_id,
                                                        radius_quantile=radius_quantile)
                if cs is not None:
                    charts.append(cs)
            return charts

        elif method == "mahalanobis_outlier_split":
            cs0 = self._build_ppca_chart_with_radius(h_all, chart_id=0, layer_id=layer_id,
                                                      radius_quantile=radius_quantile)
            if cs0 is None:
                return []
            return self.build_mahalanobis_outlier_charts(h_all, layer_id, cs0)

        return []

    def _compute_radius_scale_diag(self, charts: List[ChartState], h_all: Tensor,
                                    layer_id: int, method: str, k: int,
                                    radius_quantile: float) -> None:
        """Log per-chart train/test d2 stats and radius values."""
        for cs in charts:
            d2_all = self._compute_mahalanobis_d2(h_all, cs)
            logging.info("[RadiusScaleDiag] method=%s layer=%d k=%d chart=%d rq=%.2f "
                         "train_d2_mean=%.1f train_d2_q50=%.1f train_d2_q90=%.1f train_d2_q95=%.1f "
                         "radius_q=%.1f",
                         method, layer_id, k, cs.chart_id, radius_quantile,
                         float(d2_all.mean()), float(torch.quantile(d2_all, 0.50)),
                         float(torch.quantile(d2_all, 0.90)), float(torch.quantile(d2_all, 0.95)),
                         cs.radius_d2)

    def _compute_overlap_count_stats(self, charts: List[ChartState], h_all: Tensor,
                                      layer_id: int, method: str, k: int,
                                      radius_quantile: float) -> Dict:
        """Count how many chart radii each sample falls within."""
        if len(charts) < 2:
            return {"mean_overlap_count": 1.0}

        N = h_all.shape[0]
        in_radius = torch.zeros(N, len(charts), dtype=torch.bool)
        for i, cs in enumerate(charts):
            d2 = self._compute_mahalanobis_d2(h_all, cs)
            in_radius[:, i] = d2 <= cs.radius_d2
        n_in = in_radius.sum(dim=1).float()

        logging.info("[OverlapCountStats] method=%s layer=%d k=%d rq=%.2f "
                     "mean=%.2f q50=%.1f q90=%.1f max=%d",
                     method, layer_id, k, radius_quantile,
                     float(n_in.mean()), float(torch.quantile(n_in, 0.50)),
                     float(torch.quantile(n_in, 0.90)), int(n_in.max()))

        is_suspect = float(n_in.mean()) > 1.5 and radius_quantile <= 0.50
        if is_suspect:
            logging.info("[OverlapDiag][SUSPECT] method=%s layer=%d k=%d rq=%.2f "
                         "mean_overlap=%.2f — should be near 1.0 at low quantile",
                         method, layer_id, k, radius_quantile, float(n_in.mean()))

        return {"mean_overlap_count": float(n_in.mean()),
                "q50": float(torch.quantile(n_in, 0.50)),
                "suspected_bug": is_suspect}

    def _compute_hard_voronoi_routing(self, charts: List[ChartState], h_all: Tensor,
                                       labels: Tensor, layer_id: int, method: str, k: int,
                                       increment: int = 10) -> Dict:
        """Hard Voronoi assignment: each sample to nearest chart by d2."""
        if len(charts) < 2:
            return {}

        N = h_all.shape[0]
        d2_all = torch.zeros(N, len(charts))
        for i, cs in enumerate(charts):
            d2_all[:, i] = self._compute_mahalanobis_d2(h_all, cs)

        assignments = d2_all.argmin(dim=1)
        best_d2 = d2_all.min(dim=1).values
        d2_filled = d2_all.clone()
        d2_filled.scatter_(1, assignments.unsqueeze(1), float("inf"))
        second_d2 = d2_filled.min(dim=1).values
        margin = second_d2 - best_d2

        chart_ids = [cs.chart_id for cs in charts]
        hist = {int(c): int((assignments == i).sum()) for i, c in enumerate(chart_ids)}
        boundary_ratio = float((margin < 1.0).float().mean())

        logging.info("[HardVoronoiRouting] method=%s layer=%d k=%d chart_hist=%s "
                     "margin_mean=%.1f margin_q50=%.1f margin_q90=%.1f boundary=%.3f",
                     method, layer_id, k, hist,
                     float(margin.mean()), float(torch.quantile(margin, 0.50)),
                     float(torch.quantile(margin, 0.90)), boundary_ratio)

        source_tasks = labels // increment
        for src in sorted(source_tasks.unique().tolist()):
            smask = source_tasks == src
            src_hist = {int(c): int((assignments[smask] == i).sum())
                       for i, c in enumerate(chart_ids)}
            logging.info("[HardVoronoiRoutingByTask] method=%s layer=%d k=%d source=%d chart_hist=%s",
                        method, layer_id, k, src, src_hist)

        return {"chart_hist": hist, "margin_mean": float(margin.mean()),
                "boundary_ratio": boundary_ratio, "assignments": assignments}

    def _compute_hard_voronoi_purity(self, charts: List[ChartState], assignments: Tensor,
                                      labels: Tensor, layer_id: int, method: str, k: int,
                                      increment: int = 10) -> Dict:
        """Purity under hard Voronoi assignment."""
        if len(charts) < 2:
            return {"mean_purity": 1.0}

        source_tasks = labels // increment
        purities = []
        for cid, cs in enumerate(charts):
            mask = assignments == cid
            n = int(mask.sum())
            if n == 0:
                continue
            src = source_tasks[mask]
            unique, counts = torch.unique(src, return_counts=True)
            hist = {int(u): int(c) for u, c in zip(unique, counts)}
            purity = float(counts.max() / counts.sum())
            purities.append(purity)

        mean_purity = float(np.mean(purities)) if purities else 0.0
        logging.info("[HardVoronoiPurity] method=%s layer=%d k=%d mean_purity=%.3f",
                    method, layer_id, k, mean_purity)
        return {"mean_purity": mean_purity}

    def _compute_radius_tradeoff(self, sweep_results: List[Dict], layer_id: int,
                                  method: str, k: int) -> Dict:
        """Summarize coverage-overlap tradeoff across radius quantiles."""
        logging.info("[RadiusTradeoff] method=%s layer=%d k=%d", method, layer_id, k)
        best_rq, best_score = None, -float("inf")
        for r in sweep_results:
            rq = r["radius_quantile"]
            cov = r.get("coverage", 0)
            ovp = r.get("overlap_ratio", 1)
            bnd = r.get("boundary_ratio", 1)
            logging.info("[RadiusTradeoff]   rq=%.2f coverage=%.3f overlap=%.3f boundary=%.3f",
                        rq, cov, ovp, bnd)

            score = cov - 1.5 * ovp - 0.5 * bnd
            if score > best_score:
                best_score = score
                best_rq = rq

        logging.info("[RadiusTradeoffBest] method=%s layer=%d k=%d best_rq=%.2f score=%.3f",
                    method, layer_id, k, best_rq or 0, best_score)
        return {"best_rq": best_rq, "best_score": best_score}

    # ------------------------------------------------------------------
    #  Full dry-run pipeline (updated for radius sweep)
    # ------------------------------------------------------------------

    def run_all_diagnostics(self, data_loader, total_classes: int,
                             raw_nll_total: float, path_nll_total: float,
                             increment: int = 10) -> Dict:
        """Run complete dry-run diagnostics with radius quantile sweep."""
        import warnings

        all_results = {}
        features = self.extract_all_layer_features(data_loader)

        # Read radius_quantile_list from config
        radius_quantile_list = self.config.get("radius_quantile_list", [0.95])

        for layer_id in self.atlas_layers:
            h_all, labels_all = features[layer_id]
            all_results[layer_id] = {}

            # Single chart baseline
            single_cs = self._build_ppca_chart(h_all, chart_id=0, layer_id=layer_id)
            single_metrics = {}
            if single_cs is not None:
                single_metrics = self.compute_quality_metrics(
                    [single_cs], h_all, layer_id, "single", 1)

            for method in self.methods:
                all_results[layer_id][method] = {}
                ks = self.k_list if method != "mahalanobis_outlier_split" else [1]
                for k in ks:
                    if k == 1 and method != "mahalanobis_outlier_split":
                        continue
                    if k > h_all.shape[0] // self.min_support:
                        continue

                    logging.info("[DryChartRun] layer=%d method=%s k=%d", layer_id, method, k)

                    res = {}
                    radius_sweep_results = []

                    for rq in radius_quantile_list:
                        charts = self.build_charts_with_radius(h_all, layer_id, method, k, rq)
                        if len(charts) < 2:
                            continue

                        # Radius scale diagnostics
                        self._compute_radius_scale_diag(charts, h_all, layer_id, method, k, rq)

                        # Overlap count
                        oc = self._compute_overlap_count_stats(charts, h_all, layer_id, method, k, rq)

                        # Quality
                        dry_metrics = self.compute_quality_metrics(charts, h_all, layer_id, method, k)
                        gain = self.compute_quality_gain(dry_metrics, single_metrics, layer_id, method, k)

                        # Overlap (radius-soft)
                        overlap = self.compute_overlap(charts, h_all, layer_id, method, k)

                        sweep_entry = {
                            "radius_quantile": rq,
                            "coverage": dry_metrics.get("coverage_at_radius", 0),
                            "overlap_ratio": overlap.get("overlap_ratio", 0),
                            "boundary_ratio": overlap.get("boundary_ratio", 0),
                            "d2_gain": gain.get("gain_d2_pct", 0),
                            "recon_gain": gain.get("gain_recon_pct", 0),
                            "quality": dry_metrics,
                            "quality_gain": gain,
                            "overlap_details": overlap,
                            "overlap_count": oc,
                        }
                        radius_sweep_results.append(sweep_entry)

                        # Store best result for default rq
                        if rq == radius_quantile_list[-1]:
                            res["quality"] = dry_metrics
                            res["quality_gain"] = gain
                            res["overlap"] = overlap
                            res["charts"] = charts

                    if not radius_sweep_results:
                        logging.info("[DryChartRun] layer=%d method=%s k=%d no valid charts",
                                    layer_id, method, k)
                        continue

                    res["radius_sweep"] = radius_sweep_results

                    # Radius tradeoff
                    tradeoff = self._compute_radius_tradeoff(radius_sweep_results, layer_id, method, k)
                    res["radius_tradeoff"] = tradeoff

                    # Use best rq charts for purity, routing, oracle
                    best_rq = tradeoff.get("best_rq") or radius_quantile_list[-1]
                    best_charts = self.build_charts_with_radius(h_all, layer_id, method, k, best_rq)

                    if len(best_charts) >= 2:
                        # Purity (radius-soft)
                        purity = self.compute_purity(best_charts, h_all, labels_all, layer_id, method, k, increment)
                        res["purity"] = purity

                        # Hard Voronoi
                        hv_routing = self._compute_hard_voronoi_routing(best_charts, h_all, labels_all, layer_id, method, k, increment)
                        res["hard_voronoi_routing"] = hv_routing
                        hv_purity = self._compute_hard_voronoi_purity(
                            best_charts, hv_routing.get("assignments", torch.zeros(h_all.shape[0], dtype=torch.long)),
                            labels_all, layer_id, method, k, increment)
                        res["hard_voronoi_purity"] = hv_purity

                        # Chart-slot routing (hard Voronoi)
                        hv_sr = self.compute_chart_slot_routing(best_charts, h_all, labels_all, layer_id, method, k, increment)
                        res["slot_routing"] = hv_sr

                        # Oracle chart eval — only on primary layer; adapters bound to chart=0
                        first_atlas = min(self.atlas_layers)
                        if layer_id != first_atlas:
                            oracle = {"total": path_nll_total, "gain_over_raw": path_nll_total - raw_nll_total,
                                      "gain_over_path": 0.0}
                            logging.info("[DryRunOracleChartEval][SKIP] method=%s k=%d layer=%d "
                                         "reason=non_primary_layer", method, k, layer_id)
                        else:
                            oracle = self.oracle_chart_eval(best_charts, data_loader, layer_id, method, k,
                                                            total_classes, raw_nll_total, path_nll_total)
                        res["oracle_eval"] = oracle

                        # Proposal
                        proposal = self.propose_chart_creation(
                            res.get("quality_gain", {}), purity, res.get("overlap", {}), {},
                            layer_id, method, k, oracle.get("gain_over_path", 0))
                        res["proposal"] = proposal

                    all_results[layer_id][method][k] = res

        return all_results
