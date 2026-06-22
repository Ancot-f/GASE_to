"""
Slot Quality / Router Basis / Adapter Effect Diagnostics (Phase-9.9).
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def _safe_cat_tensors(tensors: List, dim: int = 0) -> Optional[torch.Tensor]:
    tensors = [t for t in tensors if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0]
    return torch.cat([t.detach().cpu() for t in tensors], dim=dim) if tensors else None


def _safe_cat_arrays(arrays: List) -> Optional[np.ndarray]:
    arrays = [a for a in arrays if a is not None and isinstance(a, np.ndarray) and len(a) > 0]
    return np.concatenate(arrays, axis=0) if arrays else None


def _count_slots(backbone, atlas_layers) -> int:
    total = 0
    for lid in atlas_layers:
        blk = backbone.get_block(lid)
        total += len(blk.get_available_slot_ids(0))
    return total


def _limit_data_loader(data_loader, max_samples: int):
    if max_samples <= 0:
        yield from data_loader
        return
    count = 0
    for batch in data_loader:
        yield batch
        count += batch[1].shape[0] if len(batch) > 1 else 0
        if count >= max_samples:
            break


# ============================================================================
#  Diagnostic functions (all accept task_id)
# ============================================================================

def compute_oracle_error_diag(backbone, data_loader, atlas_layers,
                               task_id: int, total_classes: int,
                               increment: int = 10) -> Dict:
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    layer_stats = {lid: {"total": 0, "raw_match": 0, "path_match": 0,
                         "oracle_slot_counts": {}, "raw_slot_counts": {}, "path_slot_counts": {}}
                   for lid in atlas_layers}
    overall_raw = overall_path = overall_total = 0

    with torch.no_grad():
        for _, inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.shape[0]
            backbone._clear_path_slot_ids()
            backbone.set_nll_router(router)
            _ = backbone.compute_key_slot_logits(inputs)

            for lid in atlas_layers:
                blk = backbone.get_block(lid)
                available = blk.get_available_slot_ids(0)
                if len(available) <= 1:
                    continue
                cs = blk.chart_states.get(0)
                ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                           for sid in available if blk.slot_states.get(f"0_{sid}") is not None}
                if not ss_dict or cs is None:
                    continue

                h = backbone._extract_h_chart_at_layer(inputs, lid)
                raw_slot = router.route(h, cs, ss_dict)["slot_ids"]

                best_slot = torch.zeros(B, dtype=torch.long, device=device)
                best_score = torch.full([B], -float("inf"), device=device)
                for sid in available:
                    backbone._clear_path_slot_ids()
                    for l2 in atlas_layers:
                        backbone.blocks[l2].path_slot_id = torch.full([B], sid, device=device, dtype=torch.long)
                    backbone.set_adapter_mode("path_key_slot_student")
                    try:
                        logits = backbone.forward(inputs)["logits"][:, :total_classes]
                    finally:
                        backbone.set_adapter_mode("task_train")
                    score = logits.gather(1, targets.unsqueeze(1)).squeeze(1)
                    better = score > best_score
                    best_score = torch.where(better, score, best_score)
                    best_slot = torch.where(better, torch.full_like(best_slot, sid), best_slot)

                path_slot = router.route(h, cs, ss_dict)["slot_ids"]

                raw_match = int((raw_slot == best_slot).sum())
                path_match = int((path_slot == best_slot).sum())
                ls = layer_stats[lid]
                ls["total"] += B; ls["raw_match"] += raw_match; ls["path_match"] += path_match
                overall_total += B; overall_raw += raw_match; overall_path += path_match

                for sid in available:
                    ls["oracle_slot_counts"][sid] = ls["oracle_slot_counts"].get(sid, 0) + int((best_slot == sid).sum())
                    ls["raw_slot_counts"][sid] = ls["raw_slot_counts"].get(sid, 0) + int((raw_slot == sid).sum())
                    ls["path_slot_counts"][sid] = ls["path_slot_counts"].get(sid, 0) + int((path_slot == sid).sum())

    if overall_total > 0:
        logging.info("[OracleErrorDiag] task=%d total=%d raw_match=%.1f%% path_match=%.1f%%",
                     task_id, overall_total,
                     100 * overall_raw / overall_total, 100 * overall_path / overall_total)
    for lid in atlas_layers:
        ls = layer_stats[lid]
        if ls["total"] == 0:
            continue
        logging.info("[OracleErrorDiag][Layer] task=%d layer=%d raw_match=%.1f%% path_match=%.1f%%",
                     task_id, lid, 100 * ls["raw_match"] / ls["total"], 100 * ls["path_match"] / ls["total"])
        logging.info("[OracleErrorDiag][Layer] task=%d layer=%d oracle=%s",
                     task_id, lid, {k: v for k, v in sorted(ls["oracle_slot_counts"].items())})
        logging.info("[OracleErrorDiag][Layer] task=%d layer=%d raw=%s",
                     task_id, lid, {k: v for k, v in sorted(ls["raw_slot_counts"].items())})
        logging.info("[OracleErrorDiag][Layer] task=%d layer=%d path=%s",
                     task_id, lid, {k: v for k, v in sorted(ls["path_slot_counts"].items())})

    return {"raw_match_pct": 100 * overall_raw / max(overall_total, 1),
            "path_match_pct": 100 * overall_path / max(overall_total, 1),
            "layer_stats": layer_stats}


def compute_router_basis_diag(backbone, data_loader, atlas_layers,
                               task_id: int, increment: int = 10) -> Dict:
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    results = {}

    for lid in atlas_layers:
        blk = backbone.get_block(lid)
        available = blk.get_available_slot_ids(0)
        cs = blk.chart_states.get(0)
        if cs is None or len(available) <= 1:
            continue

        ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                   for sid in available if blk.slot_states.get(f"0_{sid}") is not None}
        results[lid] = {}

        basis_options = {}
        if cs.Q_router is not None:
            basis_options["shared_Q"] = cs.Q_router
        if cs.U is not None and (cs.Q_router is None or not torch.equal(cs.U, cs.Q_router)):
            basis_options["chart_U"] = cs.U
        for sid in available:
            ss = blk.slot_states.get(f"0_{sid}")
            if ss is not None and getattr(ss, "P", None) is not None:
                basis_options[f"adapter_P_slot{sid}"] = ss.P
                break  # one is enough for diagnostic

        for basis_name, basis in basis_options.items():
            logging.info("[RouterBasisDiag][START] task=%d layer=%d basis=%s", task_id, lid, basis_name)
            orig_Q = cs.Q_router
            orig_ev = cs.router_eigvals
            cs.Q_router = basis.to(device)
            cs.router_eigvals = None

            total_samples = slot0_count = 0
            with torch.no_grad():
                for _, inputs, _ in data_loader:
                    inputs = inputs.to(device)
                    h = backbone._extract_h_chart_at_layer(inputs, lid)
                    routing = router.route(h, cs, ss_dict)
                    slot_ids = routing["slot_ids"]
                    total_samples += slot_ids.shape[0]
                    slot0_count += int((slot_ids == 0).sum())

            cs.Q_router = orig_Q
            cs.router_eigvals = orig_ev

            if total_samples == 0:
                logging.info("[RouterBasisDiag][SKIP] task=%d layer=%d basis=%s reason=no_samples",
                            task_id, lid, basis_name)
                results[lid][basis_name] = {"total_samples": 0, "slot0_ratio": None, "skip_reason": "no_samples"}
                continue

            info = {"total_samples": total_samples,
                    "slot0_ratio": 100 * slot0_count / max(total_samples, 1)}
            results[lid][basis_name] = info
            logging.info("[RouterBasisDiag][END] task=%d layer=%d basis=%s total=%d slot0=%.1f%%",
                        task_id, lid, basis_name, total_samples, info["slot0_ratio"])

    return results


def compute_adapter_effect_diag(backbone, data_loader, atlas_layers,
                                 task_id: int, total_classes: int,
                                 increment: int = 10) -> Dict:
    backbone.eval()
    device = next(backbone.parameters()).device
    results = {}

    for lid in atlas_layers:
        blk = backbone.get_block(lid)
        available = blk.get_available_slot_ids(0)
        if len(available) <= 1:
            continue
        results[lid] = {}

        for sid in available:
            delta_norms = []
            with torch.no_grad():
                for _, inputs, _ in data_loader:
                    inputs = inputs.to(device)
                    backbone._clear_path_slot_ids()
                    backbone.set_adapter_mode("task_train")
                    logits_no = backbone.forward(inputs)["logits"][:, :total_classes]

                    backbone._clear_path_slot_ids()
                    for l2 in atlas_layers:
                        backbone.blocks[l2].path_slot_id = torch.full(
                            [inputs.shape[0]], sid, device=device, dtype=torch.long)
                    backbone.set_adapter_mode("path_key_slot_student")
                    logits_slot = backbone.forward(inputs)["logits"][:, :total_classes]
                    backbone.set_adapter_mode("task_train")

                    delta_norms.append((logits_slot - logits_no).norm(dim=-1).cpu())

            cat = _safe_cat_tensors(delta_norms)
            if cat is None:
                logging.info("[AdapterEffectDiag][SKIP_SLOT] task=%d layer=%d chart=0 slot=%d reason=no_samples",
                            task_id, lid, sid)
                results[lid][sid] = {"delta_norm_mean": None, "skip_reason": "no_samples"}
                continue

            info = {"layer_id": lid, "chart_id": 0, "slot_id": sid,
                    "delta_norm_mean": float(cat.mean()),
                    "delta_norm_q50": float(torch.quantile(cat, 0.50)),
                    "delta_norm_q90": float(torch.quantile(cat, 0.90))}
            results[lid][sid] = info
            logging.info("[AdapterEffectDiag] task=%d layer=%d chart=0 slot=%d delta=%.3f",
                        task_id, lid, sid, info["delta_norm_mean"])

    return results


def compute_slot_quality_prior_eval(backbone, data_loader, atlas_layers,
                                     task_id: int, total_classes: int,
                                     quality_weights: List[float],
                                     raw_nll_baseline: float, path_nll_baseline: float) -> Dict:
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    results = {}

    for lid in atlas_layers:
        blk = backbone.get_block(lid)
        available = blk.get_available_slot_ids(0)
        cs = blk.chart_states.get(0)
        if cs is None or len(available) <= 1:
            continue
        ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                   for sid in available if blk.slot_states.get(f"0_{sid}") is not None}
        results[lid] = {}

        for w in quality_weights:
            all_preds, all_labels = [], []
            with torch.no_grad():
                for _, inputs, targets in data_loader:
                    inputs = inputs.to(device)
                    h = backbone._extract_h_chart_at_layer(inputs, lid)

                    nll_scores = []
                    for sid in available:
                        ss = ss_dict.get(sid)
                        if ss is not None:
                            nll = router.compute_nll(h, cs, ss)
                            self_std = getattr(ss, "router_nll_std", 1.0)
                            quality = -self_std
                            nll_scores.append(-nll + w * quality)
                        else:
                            nll_scores.append(torch.full([inputs.shape[0]], -float("inf"), device=device))

                    scores = torch.stack(nll_scores, dim=1)
                    best_slot = torch.tensor([available[i.item()] for i in scores.argmax(dim=1)],
                                            device=device, dtype=torch.long)

                    backbone._clear_path_slot_ids()
                    for l2 in atlas_layers:
                        backbone.blocks[l2].path_slot_id = best_slot
                    backbone.set_adapter_mode("path_key_slot_student")
                    try:
                        logits = backbone.forward(inputs)["logits"][:, :total_classes]
                    finally:
                        backbone.set_adapter_mode("task_train")
                    all_preds.append(torch.topk(logits, k=1, dim=1)[1].cpu().numpy())
                    all_labels.append(targets.cpu().numpy())

            y_pred = _safe_cat_arrays(all_preds)
            if y_pred is None:
                logging.info("[SlotQualityPriorEval][SKIP_WEIGHT] task=%d layer=%d weight=%.2f reason=no_preds",
                            task_id, lid, w)
                results[lid][w] = {"top1": None, "skip_reason": "no_preds"}
                continue

            y_true = _safe_cat_arrays(all_labels)
            if y_pred.ndim == 2:
                y_pred = y_pred[:, 0]
            acc = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)
            results[lid][w] = {"top1": acc}
            logging.info("[SlotQualityPriorEval] task=%d layer=%d weight=%.2f total=%.2f", task_id, lid, w, acc)

    best_w, best_acc = None, 0.0
    for lid, lr in results.items():
        for w, r in lr.items():
            if r.get("top1") and r["top1"] > best_acc:
                best_acc = r["top1"]
                best_w = w

    if best_w is not None:
        logging.info("[SlotQualityPriorBest] task=%d best_weight=%.2f total=%.2f gain_raw=%+.2f gain_path=%+.2f",
                     task_id, best_w, best_acc, best_acc - raw_nll_baseline, best_acc - path_nll_baseline)
    else:
        logging.info("[SlotQualityPriorBest] task=%d no_valid_results", task_id)

    return {"sweep": results, "best_weight": best_w, "best_result": {"top1": best_acc}}


def compute_path_decision_diag(backbone, data_loader, atlas_layers,
                                task_id: int, total_classes: int,
                                increment: int = 10) -> Dict:
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device

    first_atlas = min(atlas_layers)
    l9_blk = backbone.get_block(first_atlas)
    available = l9_blk.get_available_slot_ids(0)

    path_slot_hist = {}
    per_layer_slots_hist = {lid: {} for lid in atlas_layers}
    agreement_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_samples = 0

    with torch.no_grad():
        for _, inputs, _ in data_loader:
            inputs = inputs.to(device)
            B = inputs.shape[0]
            total_samples += B

            h9 = backbone._extract_h_chart_at_layer(inputs, first_atlas)
            cs9 = l9_blk.chart_states.get(0)
            ss9 = {sid: l9_blk.slot_states.get(f"0_{sid}")
                   for sid in available if l9_blk.slot_states.get(f"0_{sid}") is not None}
            if cs9 is not None and len(ss9) >= 1:
                path_slot = router.route(h9, cs9, ss9)["slot_ids"]
            else:
                path_slot = torch.zeros(B, dtype=torch.long, device=device)

            for sid in path_slot.unique().tolist():
                path_slot_hist[int(sid)] = path_slot_hist.get(int(sid), 0) + int((path_slot == sid).sum())

            per_layer_best = torch.zeros(B, len(atlas_layers), dtype=torch.long, device=device)
            for li, lid in enumerate(atlas_layers):
                blk = backbone.get_block(lid)
                cs = blk.chart_states.get(0)
                avail = blk.get_available_slot_ids(0)
                ss_dict = {s: blk.slot_states.get(f"0_{s}") for s in avail if blk.slot_states.get(f"0_{s}") is not None}
                h_fresh = backbone._extract_h_chart_at_layer(inputs, lid)
                if cs is not None and len(ss_dict) >= 1:
                    per_layer_best[:, li] = router.route(h_fresh, cs, ss_dict)["slot_ids"]

            for li in range(len(atlas_layers)):
                pl_slot = per_layer_best[:, li]
                for s in pl_slot.unique().tolist():
                    key = int(s)
                    per_layer_slots_hist[atlas_layers[li]][key] = \
                        per_layer_slots_hist[atlas_layers[li]].get(key, 0) + int((pl_slot == s).sum())

            agree = (per_layer_best == path_slot.unsqueeze(1)).sum(dim=1)
            for a in range(4):
                agreement_counts[a] += int((agree == a).sum())

    logging.info("[PathDecisionDiag] task=%d path_slot_hist=%s", task_id, path_slot_hist)
    logging.info("[PathConsistencyDiag] task=%d agreement_hist=%s", task_id, agreement_counts)
    for lid in atlas_layers:
        logging.info("[PathConsistencyDiag] task=%d layer=%d per_layer=%s",
                    task_id, lid, per_layer_slots_hist[lid])

    return {"path_slot_hist": path_slot_hist,
            "per_layer_hist": per_layer_slots_hist,
            "agreement_hist": agreement_counts}


# ============================================================================
#  Orchestrator
# ============================================================================

def run_all_slot_quality_diagnostics(backbone, data_loader, atlas_layers,
                                      total_classes: int, raw_nll_baseline: float,
                                      path_nll_baseline: float, config: dict,
                                      task_id: int = -1) -> Dict:
    increment = config.get("increment", 10)
    max_samples = config.get("diag_max_eval_samples", 512)
    skip_single = config.get("diag_skip_single_slot_heavy", True)
    total_slots = _count_slots(backbone, atlas_layers)
    results = {}
    status = {}

    logging.info("[SlotDiagConfig] diag_max_eval_samples=%d", max_samples)
    logging.info("[SlotDiagConfig] diag_skip_single_slot_heavy=%s", skip_single)

    is_single_slot = total_slots <= len(atlas_layers)

    def _run(name, fn, is_heavy=False):
        nonlocal status
        if is_heavy and is_single_slot and skip_single:
            logging.info("[%s][SKIP] task=%d reason=single_slot", name, task_id)
            status[name] = "SKIP_SINGLE_SLOT"
            return None
        logging.info("[%s][START] task=%d", name, task_id)
        t0 = time.time()
        try:
            loader = _limit_data_loader(data_loader, max_samples) if is_heavy else data_loader
            r = fn(backbone, loader, atlas_layers, task_id=task_id, total_classes=total_classes,
                   increment=increment)
            elapsed = time.time() - t0
            logging.info("[%s][END] task=%d elapsed=%.1fs", name, task_id, elapsed)
            status[name] = "OK"
            return r
        except Exception as e:
            elapsed = time.time() - t0
            import traceback
            logging.info("[SlotDiag][ERROR] module=%s task=%d error=%s", name, task_id, str(e))
            logging.info("[SlotDiag][ERROR_TRACE] %s", traceback.format_exc().replace('\n', ' | ')[:500])
            status[name] = "ERROR"
            return None

    results["slot_score"] = _run("SlotScoreDiag", lambda bb, dl, al, **kw: _light_slot_score(bb, dl, al, task_id=task_id))

    results["oracle_error"] = _run("OracleErrorDiag",
        lambda bb, dl, al, **kw: compute_oracle_error_diag(bb, dl, al, task_id=task_id, total_classes=total_classes, increment=increment),
        is_heavy=True)
    results["router_basis"] = _run("RouterBasisDiag",
        lambda bb, dl, al, **kw: compute_router_basis_diag(bb, dl, al, task_id=task_id, increment=increment),
        is_heavy=True)
    results["adapter_effect"] = _run("AdapterEffectDiag",
        lambda bb, dl, al, **kw: compute_adapter_effect_diag(bb, dl, al, task_id=task_id, total_classes=total_classes, increment=increment),
        is_heavy=True)
    results["path_decision"] = _run("PathDecisionDiag",
        lambda bb, dl, al, **kw: compute_path_decision_diag(bb, dl, al, task_id=task_id, total_classes=total_classes, increment=increment),
        is_heavy=True)

    quality_weights = config.get("quality_weight_list", [0.0, 0.1, 0.2, 0.5, 1.0])
    results["slot_quality_prior"] = _run("SlotQualityPriorEval",
        lambda bb, dl, al, **kw: compute_slot_quality_prior_eval(
            bb, dl, al, task_id=task_id, total_classes=total_classes,
            quality_weights=quality_weights,
            raw_nll_baseline=raw_nll_baseline, path_nll_baseline=path_nll_baseline),
        is_heavy=True)

    logging.info("[SlotDiagSummary] task=%d num_layers=%d num_charts=%d num_slots=%d heavy_skipped=%s %s",
                 task_id, len(atlas_layers), len(atlas_layers), total_slots,
                 is_single_slot and skip_single,
                 " ".join(f"{k}={v}" for k, v in sorted(status.items())))

    results["_status"] = status
    return results


def _light_slot_score(backbone, data_loader, atlas_layers, task_id: int = -1, **kwargs) -> Dict:
    results = {}
    for lid in atlas_layers:
        blk = backbone.get_block(lid)
        cs = blk.chart_states.get(0)
        if cs is None:
            continue
        results[lid] = {}
        for sid in blk.get_available_slot_ids(0):
            ss = blk.slot_states.get(f"0_{sid}")
            if ss is None:
                continue
            info = {"chart_id": 0, "layer_id": lid, "slot_id": sid,
                    "support": getattr(ss, "router_support", 0),
                    "self_nll_mean": getattr(ss, "router_nll_mean", None),
                    "self_nll_std": getattr(ss, "router_nll_std", None)}
            results[lid][sid] = info
            logging.info("[SlotScoreDiag] task=%d layer=%d chart=0 slot=%d support=%d self_nll=%.1f",
                        task_id, lid, sid, info["support"],
                        info["self_nll_mean"] if info["self_nll_mean"] is not None else float("nan"))
    return results


# ============================================================================
#  JSON serialization
# ============================================================================

def safe_serialize(obj):
    """Convert to JSON-safe types; NaN/Inf -> null."""
    import math
    if obj is None:
        return None
    if isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, np.generic):
        return safe_serialize(obj.item())
    if isinstance(obj, np.ndarray):
        return [safe_serialize(x) for x in obj.tolist()]
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return safe_serialize(obj.detach().cpu().item())
        return safe_serialize(obj.detach().cpu().numpy())
    if isinstance(obj, dict):
        return {str(k): safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_serialize(x) for x in obj]
    return str(obj)
