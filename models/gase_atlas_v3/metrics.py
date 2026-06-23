"""Complete diagnostic metrics for GASE-Atlas v3 (no v1 dependency)."""

import math
import torch


# ---- CIL Performance ----

def compute_cil_metrics(acc_matrix):
    T = acc_matrix.shape[0]
    if T == 0:
        return {"average_accuracy": 0, "forgetting": 0, "last_accuracy": 0}
    avg_acc = acc_matrix[-1, :].mean().item()
    forgetting = 0.0
    for j in range(T):
        peak = acc_matrix[:j + 1, j].max().item()
        final = acc_matrix[-1, j].item()
        forgetting += max(0.0, peak - final)
    forgetting /= max(T, 1)
    return {"average_accuracy": avg_acc, "forgetting": forgetting,
            "last_accuracy": acc_matrix[-1, :].mean().item()}


# ---- Parameter Growth ----

def compute_param_growth_metrics(model, task_id):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(p.numel() for n, p in model.named_parameters()
                         if "adapter" in n.lower() or "chart" in n.lower())
    num_charts_per_layer = {}
    for lid in [9, 10, 11]:
        from models.gase_atlas_v3.atlas_layer import GASEAtlasLayerV3
        for m in model.modules():
            if isinstance(m, GASEAtlasLayerV3) and m.layer_id == lid:
                num_charts_per_layer[f"L{lid}"] = m.num_charts
    return {"total_params": total_params, "trainable_params": trainable_params,
            "adapter_params": adapter_params, "num_charts_per_layer": num_charts_per_layer,
            "growth_rate": sum(num_charts_per_layer.values()) / max(task_id + 1, 1)}


# ---- Chart Geometry ----

def compute_chart_geometry_metrics(charts, features):
    N = features.shape[0]; K = len(charts)
    if K == 0 or N < 2:
        return {"K": 0, "coverage": 0, "mean_mahalanobis": 0,
                "distance_margin": float('inf'), "fragmentation": 0}

    d2_all = []
    for chart in charts:
        d2_all.append(chart.mahalanobis_d2(features))
    d2_stack = torch.stack(d2_all, dim=1)

    covered = torch.zeros(N, dtype=torch.bool, device=features.device)
    for d2 in d2_all:
        covered = covered | (d2 <= charts[0].radius_d2)  # approximate

    min_d2, assignments = d2_stack.min(dim=1)
    coverage = covered.float().mean().item()  # simplified
    mean_maha = min_d2.mean().item()

    margin = float('inf')
    if K >= 2:
        top2 = d2_stack.topk(2, dim=1, largest=False).values
        margin = (top2[:, 1] - top2[:, 0]).mean().item()

    usage_vals = torch.zeros(K)
    for k in range(K):
        usage_vals[k] = (assignments == k).sum().float()
    usage_vals = usage_vals / usage_vals.sum().clamp_min(1e-8)
    usage_vals = usage_vals[usage_vals > 1e-8]
    fragmentation = float(math.exp(-(usage_vals * usage_vals.log()).sum().item())) if usage_vals.numel() > 0 else 0.0

    return {"K": K, "coverage": coverage, "mean_mahalanobis": mean_maha,
            "distance_margin": margin, "fragmentation": fragmentation}


# ---- Router Metrics ----

def compute_router_metrics(route_results):
    if not route_results:
        return {"route_entropy": 0.0, "route_margin": 0.0,
                "top1_distance_mean": 0.0, "uncertain_ratio": 0.0,
                "free_fallback_ratio": 0.0, "top1_chart_histogram": {},
                "num_routed": 0, "num_free_fallback": 0}
    total = len(route_results)
    entropies = [r.entropy for r in route_results]
    margins = [r.margin for r in route_results if r.margin < float('inf')]
    top1s = [r.top1_distance for r in route_results if r.top1_distance < float('inf')]
    uncertain_count = sum(1 for r in route_results if r.uncertain)
    free_count = sum(1 for r in route_results if r.fallback_free)
    chart_counts = {}
    for r in route_results:
        if r.selected_chart is not None:
            chart_counts[r.selected_chart] = chart_counts.get(r.selected_chart, 0) + 1
    return {
        "route_entropy": sum(entropies) / total,
        "route_margin": sum(margins) / len(margins) if margins else float('inf'),
        "top1_distance_mean": sum(top1s) / len(top1s) if top1s else 0,
        "uncertain_ratio": uncertain_count / total,
        "free_fallback_ratio": free_count / total,
        "top1_chart_histogram": chart_counts,
        "num_routed": total - free_count, "num_free_fallback": free_count,
    }


# ---- Residual Decomposition ----

def compute_residual_metrics(chart_adapter, free_adapter, features, residuals, chart=None):
    eps = 1e-6
    task_norm = residuals.norm(dim=-1).mean().item()
    with torch.no_grad():
        if chart_adapter is not None and chart is not None:
            chart_pred = chart_adapter(features, chart)
            chart_norm = chart_pred.norm(dim=-1).mean().item()
        else:
            chart_pred = torch.zeros_like(residuals); chart_norm = 0.0
        free_pred = free_adapter(features) if free_adapter is not None else torch.zeros_like(residuals)
        free_norm = free_pred.norm(dim=-1).mean().item()
        combined = chart_pred + free_pred
        distill_mse = (combined - residuals).pow(2).mean().item()
        relative_error = (residuals - combined).norm(dim=-1).mean().item() / max(task_norm, eps)
        cos_sim = (combined * residuals).sum(dim=-1) / (combined.norm(dim=-1) * residuals.norm(dim=-1)).clamp_min(eps)
        residual_cosine = cos_sim.mean().item()
        ss_res = (residuals - combined).pow(2).sum()
        ss_tot = residuals.pow(2).sum()
        subR2 = max(0.0, float(1.0 - ss_res / ss_tot)) if ss_tot > eps else 1.0
        free_ratio = free_norm / max(chart_norm + free_norm, eps)
    return {"task_residual_norm": task_norm, "chart_residual_norm": chart_norm,
            "free_residual_norm": free_norm, "distill_mse": distill_mse,
            "relative_error": relative_error, "residual_cosine": residual_cosine,
            "subR2": subR2, "free_ratio": free_ratio}


# ---- Descendant Chain ----

def compute_descendant_metrics(descendant_chain):
    return descendant_chain.compute_metrics()


# ---- Expansion Summary ----

def compute_expansion_summary(decision_log, task_id=None, layer_id=None):
    return decision_log.summary(task_id=task_id, layer_id=layer_id)

