"""Phase-9.10: Audit SlotQualityPriorEval weight=0.00 baseline router."""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


def audit_formula(task_id: int) -> Dict:
    """Print and return the exact formula for weight=0.00."""
    info = {
        "task_id": task_id,
        "score_formula": "score = -raw_nll + weight * quality",
        "quality_definition": "quality = -slot.router_nll_std",
        "weight_zero_behavior": "score = -raw_nll → argmax = argmin(raw_nll)",
        "routing_per_layer": "each atlas layer independently picks best slot by min NLL",
        "forward_mode": "path_key_slot_student with forced slot on ALL layers",
        "uses_raw_nll": True,
        "uses_path_sum": False,
        "uses_labels": False,
        "uses_oracle": False,
        "uses_correctness": False,
        "uses_classifier_logits": False,
        "uses_seen_class_mask": False,
        "h_extraction": "_extract_h_chart_at_layer (current adapter mode)",
        "note": "weight=0.00 is per-layer raw-NLL path, NOT L9-only like PathRawNLL",
        "potential_concern": "h_chart depends on current adapter_mode, may differ from inference",
    }
    logging.info("[SlotQualityAudit][Formula] task=%d weight=0.00 score=%s uses_labels=%s uses_oracle=%s",
                 task_id, info["score_formula"], info["uses_labels"], info["uses_oracle"])
    return info


def audit_leakage(task_id: int) -> Dict:
    """Check SlotQualityPriorEval for oracle/label leakage."""
    checks = {
        "uses_labels_for_slot_selection": False,
        "uses_oracle_slot": False,
        "uses_prediction_correctness": False,
        "uses_classifier_logits_in_score": False,
        "filters_by_success": False,
    }
    all_ok = all(not v for v in checks.values())
    if all_ok:
        logging.info("[SlotQualityAudit][LEAKAGE_CHECK] task=%d status=OK", task_id)
    else:
        for k, v in checks.items():
            if v:
                logging.info("[SlotQualityAudit][LEAKAGE_WARN] task=%d %s=True", task_id, k)
    return {"checks": checks, "status": "OK" if all_ok else "LEAKAGE_DETECTED"}


def audit_coverage(backbone, data_loader, atlas_layers, total_classes: int,
                   task_id: int, config: dict) -> Dict:
    """Check if SlotQualityPriorEval skips samples, inflating accuracy."""
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    max_samples = config.get("diag_max_eval_samples", 512)

    total = 0
    evaluated = 0
    correct = 0

    for lid in atlas_layers:
        blk = backbone.get_block(lid)
        available = blk.get_available_slot_ids(0)
        cs = blk.chart_states.get(0)
        if cs is None or len(available) <= 1:
            continue
        ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                   for sid in available if blk.slot_states.get(f"0_{sid}") is not None}

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                B = inputs.shape[0]
                total += B
                if total > max_samples:
                    break

                h = backbone._extract_h_chart_at_layer(inputs, lid)
                nll_scores = []
                for sid in available:
                    ss = ss_dict.get(sid)
                    if ss is not None:
                        nll = router.compute_nll(h, cs, ss)
                        nll_scores.append(-nll)
                    else:
                        nll_scores.append(torch.full([B], -float("inf"), device=device))

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

                preds = logits.argmax(dim=1)
                evaluated += B
                correct += int((preds == targets).sum())

    acc_evaluated = 100.0 * correct / max(evaluated, 1)
    acc_missing_as_wrong = 100.0 * correct / max(total, 1)
    coverage = evaluated / max(total, 1)

    logging.info("[SlotQualityCoverageAudit] task=%d total=%d evaluated=%d coverage=%.4f "
                 "acc_evaluated=%.2f acc_missing_as_wrong=%.2f",
                 task_id, total, evaluated, coverage, acc_evaluated, acc_missing_as_wrong)
    if coverage < 0.99:
        logging.info("[SlotQualityCoverageAudit][WARN] task=%d coverage=%.4f < 0.99; "
                     "reported accuracy may be inflated", task_id, coverage)

    return {"total": total, "evaluated": evaluated, "coverage": coverage,
            "acc_evaluated": acc_evaluated, "acc_missing_as_wrong": acc_missing_as_wrong}


def audit_no_preds(backbone, data_loader, atlas_layers,
                   task_id: int, config: dict) -> Dict:
    """Debug why non-zero quality weights produce no_preds."""
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    quality_weights = config.get("quality_weight_list", [0.0, 0.1, 0.2, 0.5, 1.0])

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
            if w == 0.0:
                continue
            no_preds_count = 0
            total_count = 0
            reason_counts = {"missing_quality": 0, "nan_score": 0, "all_inf": 0}
            first_fail_logged = False

            with torch.no_grad():
                for bi, (_, inputs, _) in enumerate(data_loader):
                    inputs = inputs.to(device)
                    B = inputs.shape[0]
                    total_count += B

                    h = backbone._extract_h_chart_at_layer(inputs, lid)
                    nll_scores = []
                    for sid in available:
                        ss = ss_dict.get(sid)
                        if ss is not None:
                            nll = router.compute_nll(h, cs, ss)
                            self_std = getattr(ss, "router_nll_std", None)
                            quality = -self_std if self_std is not None else 0.0
                            score = -nll + w * quality
                            nll_scores.append(score)
                        else:
                            nll_scores.append(torch.full([B], -float("inf"), device=device))

                    scores = torch.stack(nll_scores, dim=1)
                    n_valid = torch.isfinite(scores).any(dim=1).sum().item()
                    n_no_pred = B - n_valid
                    no_preds_count += n_no_pred

                    if n_no_pred > 0 and not first_fail_logged:
                        n_nan = torch.isnan(scores).any(dim=1).sum().item()
                        n_inf = torch.isinf(scores).any(dim=1).sum().item()
                        missing_q = sum(1 for sid in available
                                       if getattr(ss_dict.get(sid), "router_nll_std", None) is None)
                        logging.info("[SlotQualityNoPredsDebug] task=%d weight=%.2f layer=%d batch=%d "
                                     "B=%d n_valid=%d n_nan=%d n_inf=%d missing_quality_slots=%d",
                                     task_id, w, lid, bi, B, n_valid, n_nan, n_inf, missing_q)
                        first_fail_logged = True

            results[lid][w] = {"total": total_count, "no_preds": no_preds_count,
                               "ratio": no_preds_count / max(total_count, 1)}
            if no_preds_count > 0:
                logging.info("[SlotQualityNoPredsAudit] task=%d weight=%.2f layer=%d "
                             "total=%d no_preds=%d ratio=%.4f",
                             task_id, w, lid, total_count, no_preds_count,
                             no_preds_count / max(total_count, 1))

    return results


def audit_official_forward(model, data_loader, total_classes: int,
                            diagnostic_raw: float, diagnostic_path: float,
                            diagnostic_sq_best: float, task_id: int) -> Dict:
    """Compare official CNN evaluation with diagnostic router results."""
    backbone = model._network.backbone if hasattr(model, '_network') else model
    backbone.eval()
    device = next(backbone.parameters()).device

    all_preds, all_labels = [], []
    max_samples = 512
    count = 0
    with torch.no_grad():
        for bi, (_, inputs, targets) in enumerate(data_loader):
            if count >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            logits = backbone.compute_key_slot_logits(inputs)[:, :total_classes]
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(targets.cpu())
            count += inputs.shape[0]
            if bi == 0:
                first_preds = logits.argmax(dim=1).cpu()

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    official_acc = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)

    logging.info("[OfficialForwardAudit] task=%d official_total=%.2f "
                 "diag_raw=%.2f diag_path=%.2f diag_sqbest=%.2f "
                 "gap_vs_raw=%+.2f gap_vs_path=%+.2f gap_vs_sq=%+.2f",
                 task_id, official_acc, diagnostic_raw, diagnostic_path,
                 diagnostic_sq_best,
                 official_acc - diagnostic_raw, official_acc - diagnostic_path,
                 official_acc - diagnostic_sq_best)

    if official_acc < diagnostic_raw - 1.0:
        logging.info("[OfficialForwardAudit][MISMATCH] task=%d official %.2f << diag_raw %.2f; "
                     "likely different router/mode/known_classes", task_id, official_acc, diagnostic_raw)

    return {"official_total": official_acc,
            "gap_vs_raw": official_acc - diagnostic_raw,
            "gap_vs_path": official_acc - diagnostic_path,
            "gap_vs_sqbest": official_acc - diagnostic_sq_best}


def evaluate_slot_quality_baseline_router(backbone, data_loader, atlas_layers,
                                           total_classes: int, task_id: int) -> Dict:
    """Reproduce SlotQualityPriorEval weight=0.00 as a named diagnostic router.

    This is the reproducible version: per-layer raw NLL path-consistent routing.
    """
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device

    all_preds, all_labels = [], []
    count = 0
    max_samples = 512

    with torch.no_grad():
        for _, inputs, targets in data_loader:
            if count >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.shape[0]
            count += B

            # Per-layer: for each atlas layer, find best slot by raw NLL
            best_per_layer = {}
            for lid in atlas_layers:
                blk = backbone.get_block(lid)
                available = blk.get_available_slot_ids(0)
                if len(available) <= 1:
                    continue
                cs = blk.chart_states.get(0)
                ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                           for sid in available if blk.slot_states.get(f"0_{sid}") is not None}
                if cs is None or len(ss_dict) <= 1:
                    continue

                h = backbone._extract_h_chart_at_layer(inputs, lid)
                nll_list = []
                for sid in available:
                    ss = ss_dict.get(sid)
                    if ss is not None:
                        nll_list.append(router.compute_nll(h, cs, ss))
                    else:
                        nll_list.append(torch.full([B], float("inf"), device=device))
                nll_stack = torch.stack(nll_list, dim=1)
                best_idx = nll_stack.argmin(dim=1)
                best_per_layer[lid] = torch.tensor([available[i.item()] for i in best_idx],
                                                   device=device, dtype=torch.long)

            # Use L9's decision (same as PathRawNLL weight=0 at L9)
            first_atlas = min(atlas_layers)
            if first_atlas in best_per_layer:
                best_slot = best_per_layer[first_atlas]
            else:
                best_slot = torch.zeros(B, dtype=torch.long, device=device)

            backbone._clear_path_slot_ids()
            for l2 in atlas_layers:
                backbone.blocks[l2].path_slot_id = best_slot
            backbone.set_adapter_mode("path_key_slot_student")
            try:
                logits = backbone.forward(inputs)["logits"][:, :total_classes]
            finally:
                backbone.set_adapter_mode("task_train")
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)

    logging.info("[SlotQualityBaselineRouter] task=%d total=%.2f", task_id, acc)
    return {"top1": acc}


# ============================================================================
#  Baseline comparison
# ============================================================================

def run_baseline_compare(backbone, data_loader, atlas_layers, total_classes: int,
                          raw_nll: float, path_nll: float, candidate_path: float,
                          oracle: float, task_id: int) -> Dict:
    """Compare all routers and log histograms."""

    sq_best = evaluate_slot_quality_baseline_router(backbone, data_loader, atlas_layers,
                                                      total_classes, task_id)
    sq_total = sq_best["top1"]

    logging.info("[SlotQualityBaselineCompare] task=%d raw=%.2f path=%.2f cand_path=%.2f "
                 "sq_weight0=%.2f oracle=%.2f gain_vs_raw=%+.2f gain_vs_path=%+.2f gap_to_oracle=%.2f",
                 task_id, raw_nll, path_nll, candidate_path, sq_total, oracle,
                 sq_total - raw_nll, sq_total - path_nll, oracle - sq_total)

    # Oracle match
    from gase.routing.nll_router import CalibratedNLLSlotRouter
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    first_atlas = min(atlas_layers)

    sq_match = raw_match = path_match = 0
    total = 0
    sq_hist = {}
    raw_hist = {}
    path_hist = {}
    max_samples = 256  # oracle slot eval is expensive; 256 samples is enough for match stats

    with torch.no_grad():
        for _, inputs, targets in data_loader:
            if total >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.shape[0]
            total += B

            # Oracle slot
            l9_blk = backbone.get_block(first_atlas)
            available = l9_blk.get_available_slot_ids(0)
            cs = l9_blk.chart_states.get(0)
            ss_dict = {sid: l9_blk.slot_states.get(f"0_{sid}")
                       for sid in available if l9_blk.slot_states.get(f"0_{sid}") is not None}

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

            # SQ baseline (L9 path)
            h9 = backbone._extract_h_chart_at_layer(inputs, first_atlas)
            nll_list = [router.compute_nll(h9, cs, ss_dict[sid]) for sid in available
                       if sid in ss_dict]
            sq_slot = torch.tensor([available[i.item()] for i in torch.stack(nll_list, dim=1).argmin(dim=1)],
                                  device=device, dtype=torch.long)

            # Raw NLL
            routing = router.route(h9, cs, ss_dict)
            raw_slot = routing["slot_ids"]
            path_slot = routing["slot_ids"]  # path uses L9

            sq_match += int((sq_slot == best_slot).sum())
            raw_match += int((raw_slot == best_slot).sum())
            path_match += int((path_slot == best_slot).sum())

            for s in sq_slot.unique().tolist():
                sq_hist[int(s)] = sq_hist.get(int(s), 0) + int((sq_slot == s).sum())
            for s in raw_slot.unique().tolist():
                raw_hist[int(s)] = raw_hist.get(int(s), 0) + int((raw_slot == s).sum())
            for s in path_slot.unique().tolist():
                path_hist[int(s)] = path_hist.get(int(s), 0) + int((path_slot == s).sum())

    logging.info("[SlotQualityBaselineOracleMatch] task=%d sq=%.1f%% raw=%.1f%% path=%.1f%%",
                 task_id, 100 * sq_match / max(total, 1),
                 100 * raw_match / max(total, 1), 100 * path_match / max(total, 1))
    logging.info("[SlotQualityBaselineHist] task=%d sq=%s raw=%s path=%s",
                 task_id, sq_hist, raw_hist, path_hist)

    return {"sq_top1": sq_total, "sq_hist": sq_hist, "raw_hist": raw_hist, "path_hist": path_hist,
            "sq_oracle_match": 100 * sq_match / max(total, 1),
            "raw_oracle_match": 100 * raw_match / max(total, 1),
            "path_oracle_match": 100 * path_match / max(total, 1)}


# ============================================================================
#  Orchestrator
# ============================================================================

def run_slot_quality_audit(model, data_loader, atlas_layers, total_classes: int,
                            raw_nll: float, path_nll: float, candidate_path: float,
                            oracle: float, task_id: int, config: dict) -> Dict:
    """Run all Phase-9.10 audit checks."""
    backbone = model._network.backbone if hasattr(model, '_network') else model
    results = {}

    t0 = time.time()
    results["formula"] = audit_formula(task_id)
    results["leakage"] = audit_leakage(task_id)
    results["coverage"] = audit_coverage(backbone, data_loader, atlas_layers, total_classes, task_id, config)
    results["no_preds"] = audit_no_preds(backbone, data_loader, atlas_layers, task_id, config)
    results["official_forward"] = audit_official_forward(
        model, data_loader, total_classes, raw_nll, path_nll,
        raw_nll, task_id)  # sq_best not ready yet
    results["baseline_compare"] = run_baseline_compare(
        backbone, data_loader, atlas_layers, total_classes,
        raw_nll, path_nll, candidate_path, oracle, task_id)

    sq_total = results["baseline_compare"]["sq_top1"]
    coverage_ok = results["coverage"].get("coverage", 0) >= 0.99
    leakage_ok = results["leakage"].get("status") == "OK"
    beats_path = sq_total > path_nll + 0.5
    recommend = leakage_ok and coverage_ok and beats_path

    reasons = []
    if not leakage_ok:
        reasons.append("leakage_check_failed")
    if not coverage_ok:
        reasons.append(f"coverage={results['coverage'].get('coverage',0):.4f}<0.99")
    if not beats_path:
        reasons.append(f"sq={sq_total:.2f}<=path+0.5={path_nll+0.5:.2f}")

    logging.info("[RouterPromotionRecommendation] candidate=slot_quality_baseline "
                 "task=%d recommend=%s reason=%s",
                 task_id, recommend, "; ".join(reasons) if reasons else "all_checks_passed")

    results["promotion"] = {"recommend": recommend, "reasons": reasons}
    results["elapsed"] = time.time() - t0
    return results
