"""Legacy LayerDistiller for early V3 experiments.

The active V3 training path is implemented in ``learner.py``.  This module is
kept only for reference while V3 is treated as a modularized V2; do not use it
as the source of truth for current chart/slot/free behavior.

Pipeline: build charts ->build adapters (ridge regression) ->train free adapter ->log.
"""

import logging
import torch
import torch.nn.functional as F

from models.gase_atlas_v3.decision_log import (
    ExpansionDecision,
    DECISION_ADD_CHART, DECISION_UPDATE_CHART_ADAPTER,
    DECISION_UPDATE_FREE_ADAPTER,
    REASON_NO_EXISTING_CHARTS, REASON_OVERLAP_REJECT,
)


class LayerDistiller:
    """Unified distillation orchestrator for one layer."""

    def __init__(
        self, chart_builder, adapter_builder, decision_log,
        free_adapter_epochs=3, free_adapter_lr=1e-4,
        free_adapter_l2_to_prev=1e-3, free_usage_alarm_high=0.50,
        free_usage_alarm_low=0.02, chart_adapter_type="projected_linear",
        chart_adapter_tune_epochs=0, chart_adapter_hidden=16,
        joint_distill_epochs=3, joint_distill_lr=1e-4,
        joint_lambda_free=0.01, joint_lambda_chart=0.001,
    ):
        self.chart_builder = chart_builder
        self.adapter_builder = adapter_builder
        self.decision_log = decision_log
        self.free_adapter_epochs = free_adapter_epochs
        self.free_adapter_lr = free_adapter_lr
        self.free_adapter_l2_to_prev = free_adapter_l2_to_prev
        self.free_usage_alarm_high = free_usage_alarm_high
        self.free_usage_alarm_low = free_usage_alarm_low
        self.chart_adapter_type = chart_adapter_type
        self.chart_adapter_tune_epochs = chart_adapter_tune_epochs
        self.chart_adapter_hidden = chart_adapter_hidden
        self.joint_distill_epochs = joint_distill_epochs
        self.joint_distill_lr = joint_distill_lr
        self.joint_lambda_free = joint_lambda_free
        self.joint_lambda_chart = joint_lambda_chart

    def distill_layer(self, layer, features, residuals, task_id, max_charts=None):
        device = next(layer.free_adapter.parameters()).device
        N = features.shape[0]
        existing_charts = layer.atlas.active_charts()
        coverage_before = _compute_coverage(existing_charts, features)

        if max_charts is not None:
            self.chart_builder.max_charts = max_charts

        features_dev = features.to(device)
        residuals_dev = residuals.to(device)
        next_id = max((c.chart_id for c in layer.atlas.charts), default=-1) + 1

        charts, non_chart_idx, reuse_pairs, update_pairs = \
            self.chart_builder.build_layer_charts(
                features_dev, residuals_dev,
                existing_charts=existing_charts if existing_charts else None,
                next_chart_id=next_id, birth_task=task_id,
            )
        for c in charts:
            c.cpu()
        non_chart_idx = non_chart_idx.cpu()

        layer.register_charts(charts)

        # Log decisions ----------------------------------------------------------
        # Case C: new chart added
        for chart in charts:
            self.decision_log.record(ExpansionDecision(
                task_id=task_id, layer_id=layer.layer_id,
                decision_type=DECISION_ADD_CHART,
                reason=REASON_NO_EXISTING_CHARTS if not existing_charts else "geo_outlier",
                chart_id=chart.chart_id, num_samples=chart.support,
                geo_score=chart.mean_d2, residual_score=getattr(chart, 'full_r2', 0.0),
                coverage_before=coverage_before,
                coverage_after=_compute_coverage(layer.atlas.active_charts(), features),
            ))

        # Case A: reuse_chart
        for old_chart, assigned_mask in reuse_pairs:
            self.decision_log.record(ExpansionDecision(
                task_id=task_id, layer_id=layer.layer_id,
                decision_type=DECISION_REUSE_CHART,
                reason="geo_hit_residual_good",
                chart_id=old_chart.chart_id,
                num_samples=assigned_mask.sum().item(),
                geo_score=old_chart.mean_d2,
                residual_score=getattr(old_chart, 'full_r2', 0.0),
                coverage_before=coverage_before,
                coverage_after=_compute_coverage(layer.atlas.active_charts(), features),
            ))

        # Case B: update_chart_adapter
        for old_chart, assigned_mask in update_pairs:
            self.decision_log.record(ExpansionDecision(
                task_id=task_id, layer_id=layer.layer_id,
                decision_type=DECISION_UPDATE_CHART_ADAPTER,
                reason="geo_hit_residual_bad",
                chart_id=old_chart.chart_id,
                num_samples=assigned_mask.sum().item(),
                geo_score=old_chart.mean_d2,
                residual_score=getattr(old_chart, 'full_r2', 0.0),
                coverage_before=coverage_before,
                coverage_after=_compute_coverage(layer.atlas.active_charts(), features),
            ))

        # ---- Phase 1: Build/update chart adapters ----
        # New charts
        for chart in charts:
            if chart.status != "active":
                continue
            chart.to(device)
            d2 = chart.mahalanobis_d2(features_dev)
            assigned_idx = torch.where(d2 <= chart.radius_d2)[0]
            if assigned_idx.shape[0] < 2:
                chart.cpu(); continue
            adapter = self._build_adapter(chart, features_dev[assigned_idx],
                                          residuals_dev[assigned_idx], device)
            if adapter is not None:
                chart.attach_adapter(adapter, task_id=task_id)
                layer.chart_adapters.append(adapter)
                chart.promote_to_adapter_initialized()
                chart.mark_used(task_id)
            chart.cpu()

        # Update adapters for existing charts (Case B)
        for old_chart, assigned_mask in update_pairs:
            idx = torch.where(assigned_mask)[0]
            if idx.shape[0] < 2:
                continue
            old_chart.to(device)
            adapter = self._build_adapter(
                old_chart, features_dev[idx], residuals_dev[idx], device)
            old_chart.cpu()
            if adapter is not None:
                old_chart.attach_adapter(adapter, task_id=task_id)
                layer.chart_adapters.append(adapter)
                old_chart.mark_used(task_id)
                logging.info("[ChartAdapterUpdate:L%d#%d] rebuilt adapter on %d samples",
                             layer.layer_id, old_chart.chart_id, idx.shape[0])

        # ---- Phase 2: Joint distill (chart_adapter + free_adapter 鈮?task_adapter) ----
        joint_non_chart = self._joint_distill(
            layer, features_dev, residuals_dev, non_chart_idx, task_id)

        coverage_after = _compute_coverage(layer.atlas.active_charts(), features)

        layer.set_mode("inference")
        layer.remove_task_adapter_grads()
        torch.cuda.empty_cache()

        return {
            "num_charts": len(charts), "num_active": len(layer.atlas.active_charts()),
            "num_total": len(layer.atlas.charts),
            "non_chart_samples": joint_non_chart,
            "coverage_before": coverage_before, "coverage_after": coverage_after,
            "free_result": {"num_samples": joint_non_chart},
            "reuse_pairs": len(reuse_pairs), "update_pairs": len(update_pairs),
        }

    def _joint_distill(self, layer, features, residuals, non_chart_idx, task_id):
        """Phase 2: Jointly fine-tune chart_adapter + free_adapter.

        L = ||螖_task - 危螖_chart - 螖_free||虏 + 位f||螖_free||虏 + 位c||螖_chart||虏
        """
        if self.joint_distill_epochs <= 0:
            # Fall back to free-only training on non-chart samples
            n = non_chart_idx.shape[0]
            if n > 0:
                self._train_free_adapter_only(
                    layer, features[non_chart_idx], residuals[non_chart_idx], task_id)
            return n

        N = features.shape[0]
        device = features.device
        charts = layer.atlas.active_charts()
        lr = self.joint_distill_lr

        # Gather trainable params: all chart adapters + free adapter
        trainable = list(layer.free_adapter.parameters())
        for chart in charts:
            ad = chart.adapter
            if ad is not None:
                for p in ad.parameters():
                    if p.requires_grad:
                        trainable.append(p)

        if not trainable:
            return N

        for p in trainable:
            p.requires_grad = True

        opt = torch.optim.Adam(trainable, lr=lr)
        bs = min(256, N)

        for ep in range(self.joint_distill_epochs):
            perm = torch.randperm(N, device=device)
            total_loss = 0.0; n_batches = 0
            for b_start in range(0, N, bs):
                b_idx = perm[b_start:b_start + bs]
                fb = features[b_idx]; rb = residuals[b_idx]

                # Chart adapter contribution (sum over all charts, weighted by coverage)
                chart_delta = torch.zeros_like(rb)
                for chart in charts:
                    ad = chart.adapter
                    if ad is None:
                        continue
                    chart.to(device)
                    pred = ad(fb, chart)
                    chart.cpu()
                    # Weight by geometric proximity
                    d2 = chart.mahalanobis_d2(fb)
                    w = torch.softmax(-d2 / 0.5, dim=0).view(-1, 1)
                    chart_delta = chart_delta + w * pred

                # Free adapter contribution
                free_delta = layer.free_adapter(fb)

                # Joint loss
                combined = chart_delta + free_delta
                fit_loss = F.mse_loss(combined, rb)
                free_reg = self.joint_lambda_free * free_delta.pow(2).mean()
                chart_reg = self.joint_lambda_chart * chart_delta.pow(2).mean()
                loss = fit_loss + free_reg + chart_reg

                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += fit_loss.item(); n_batches += 1

            if n_batches > 0:
                logging.info("[JointDistill:L%d] epoch=%d/%d fit=%.4f free_reg=%.4f chart_reg=%.4f",
                             layer.layer_id, ep + 1, self.joint_distill_epochs,
                             total_loss / n_batches,
                             free_reg.item() if n_batches > 0 else 0,
                             chart_reg.item() if n_batches > 0 else 0)

        for p in trainable:
            p.requires_grad = False
        for chart in charts:
            chart.promote_to_distilled()

        del opt; torch.cuda.empty_cache()
        return non_chart_idx.shape[0] if non_chart_idx is not None else 0

    def _train_free_adapter_only(self, layer, features, residuals, task_id):
        """Train free adapter only (fallback when joint_distill_epochs=0)."""
        N = features.shape[0]
        device = features.device
        if N < 2:
            return

        epochs = self.free_adapter_epochs
        lr = self.free_adapter_lr
        l2_w = self.free_adapter_l2_to_prev
        prev_state = {k: v.clone() for k, v in layer.free_adapter.state_dict().items()}
        fa = layer.free_adapter
        fa.train()
        for p in fa.parameters():
            p.requires_grad = True

        opt = torch.optim.Adam(fa.parameters(), lr=lr)
        bs = min(256, N)
        for ep in range(epochs):
            perm = torch.randperm(N, device=device)
            total = 0.0; nb = 0
            for b_start in range(0, N, bs):
                idx = perm[b_start:b_start + bs]
                pred = fa(features[idx])
                mse = F.mse_loss(pred, residuals[idx])
                l2_loss = torch.tensor(0.0, device=device)
                if l2_w > 0:
                    for name, param in fa.named_parameters():
                        if name in prev_state:
                            l2_loss = l2_loss + (param - prev_state[name]).pow(2).sum()
                    l2_loss = l2_loss / max(sum(p.numel() for p in fa.parameters()), 1)
                loss = mse + l2_w * l2_loss
                opt.zero_grad(); loss.backward(); opt.step()
                total += mse.item(); nb += 1
        for p in fa.parameters():
            p.requires_grad = False
        fa.eval(); del opt; torch.cuda.empty_cache()
        logging.info("[FreeAdapter:L%d] trained on %d samples", layer.layer_id, N)

    def _build_adapter(self, chart, features, residuals, device):
        adapter = self.adapter_builder.build(chart, features, residuals)
        if adapter is None:
            return None

        if self.chart_adapter_type == "projected_mlp":
            from models.gase_atlas_v3.adapters import ChartMLPAdapter
            P = getattr(chart, 'P_adapter', None)
            adapter = ChartMLPAdapter(chart, P=P, hidden=self.chart_adapter_hidden)
            if self.chart_adapter_tune_epochs > 0:
                chart.to(device); adapter = adapter.to(device); adapter.train()
                opt = torch.optim.Adam(adapter.parameters(), lr=1e-4)
                for _ in range(self.chart_adapter_tune_epochs):
                    opt.zero_grad()
                    loss = F.mse_loss(adapter(features, chart), residuals)
                    loss.backward(); opt.step()
                adapter.eval(); adapter = adapter.cpu(); chart.cpu()
                del opt
        return adapter

    def _train_free_adapter(self, layer, features, residuals, task_id):
        N = features.shape[0]
        device = features.device
        if N < 2:
            return {"num_samples": N, "usage": 0.0, "distill_mse": 0.0}

        epochs = self.free_adapter_epochs
        lr = self.free_adapter_lr
        l2_w = self.free_adapter_l2_to_prev

        prev_state = {k: v.clone() for k, v in layer.free_adapter.state_dict().items()}
        fa = layer.free_adapter
        fa.train()
        for p in fa.parameters():
            p.requires_grad = True

        opt = torch.optim.Adam(fa.parameters(), lr=lr)
        bs = min(256, N)
        final_mse = 0.0

        for ep in range(epochs):
            perm = torch.randperm(N, device=device)
            total_loss = 0.0; n_batches = 0
            for b_start in range(0, N, bs):
                b_idx = perm[b_start:b_start + bs]
                pred = fa(features[b_idx])
                mse_loss = F.mse_loss(pred, residuals[b_idx])
                l2_loss = torch.tensor(0.0, device=device)
                if l2_w > 0:
                    for name, param in fa.named_parameters():
                        if name in prev_state:
                            l2_loss = l2_loss + (param - prev_state[name]).pow(2).sum()
                    l2_loss = l2_loss / sum(p.numel() for p in fa.parameters())
                loss = mse_loss + l2_w * l2_loss
                opt.zero_grad(); loss.backward(); opt.step()
                total_loss += mse_loss.item(); n_batches += 1
            if n_batches > 0:
                final_mse = total_loss / n_batches

        for p in fa.parameters():
            p.requires_grad = False
        fa.eval(); del opt; torch.cuda.empty_cache()

        with torch.no_grad():
            free_pred = fa(features)
            free_norm = free_pred.norm(dim=-1).mean().item()
            resid_norm = residuals.norm(dim=-1).mean().item()
            free_ratio = free_norm / max(resid_norm, 1e-8)

        overuse = free_ratio > self.free_usage_alarm_high

        if overuse:
            self.decision_log.record(ExpansionDecision(
                task_id=task_id, layer_id=layer.layer_id,
                decision_type=DECISION_UPDATE_FREE_ADAPTER,
                reason="high_free_ratio", chart_id=-1, num_samples=N,
            ))

        logging.info("[FreeAdapter:L%d] trained %d samples mse=%.4f free_ratio=%.3f %s",
                     layer.layer_id, N, final_mse, free_ratio,
                     "OVERUSE" if overuse else "OK")

        return {"num_samples": N, "distill_mse": final_mse,
                "free_residual_norm": free_norm, "free_ratio": free_ratio,
                "usage": free_ratio, "overuse": overuse}


def _compute_coverage(charts, features):
    if not charts:
        return 0.0
    N = features.shape[0]
    device = features.device
    covered = torch.zeros(N, dtype=torch.bool, device=device)
    for chart in charts:
        d2 = chart.mahalanobis_d2(features)
        covered = covered | (d2 <= chart.radius_d2)
    return covered.float().mean().item()

