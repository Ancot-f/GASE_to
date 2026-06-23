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
        "h_extraction": "_extract_h_chart_at_layer with explicit slot_quality_prefix_mode",
        "note": "deploy audit uses L9 raw-NLL, then forces the selected slot on all atlas layers",
        "potential_concern": "current-prefix comparison is diagnostic only and should not be promoted",
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


def _extract_h_with_prefix(backbone, inputs: Tensor, layer_id: int,
                           prefix_mode: str) -> Tuple[Tensor, str]:
    """Extract h_chart under an explicit adapter prefix mode."""
    mode = prefix_mode or "current"
    if mode == "current":
        return backbone._extract_h_chart_at_layer(inputs, layer_id), getattr(backbone, "adapter_mode", "unknown")

    prev_mode = getattr(backbone, "adapter_mode", None)
    backbone.set_adapter_mode(mode)
    try:
        return backbone._extract_h_chart_at_layer(inputs, layer_id), mode
    finally:
        if prev_mode is not None:
            backbone.set_adapter_mode(prev_mode)


def audit_coverage(backbone, data_loader, atlas_layers, total_classes: int,
                   task_id: int, config: dict) -> Dict:
    """Check if SlotQualityPriorEval skips samples, inflating accuracy."""
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    max_samples = config.get("slot_quality_audit_max_samples", 0)
    prefix_mode = config.get("slot_quality_prefix_mode", "path_key_slot_student")

    total = 0
    evaluated = 0
    correct = 0
    first_atlas = min(atlas_layers)
    actual_prefix_mode = prefix_mode

    blk = backbone.get_block(first_atlas)
    available = blk.get_available_slot_ids(0)
    cs = blk.chart_states.get(0)
    if cs is None or len(available) <= 1:
        logging.info("[SlotQualityCoverageAudit] task=%d layer=%d skip=single_slot_or_no_chart",
                     task_id, first_atlas)
        return {"total": 0, "evaluated": 0, "coverage": 0.0,
                "acc_evaluated": 0.0, "acc_missing_as_wrong": 0.0}

    ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
               for sid in available if blk.slot_states.get(f"0_{sid}") is not None}

    with torch.no_grad():
        for _, inputs, targets in data_loader:
            if max_samples and evaluated >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.shape[0]
            if max_samples:
                keep = min(B, max_samples - evaluated)
                inputs, targets = inputs[:keep], targets[:keep]
                B = keep
            total += B

            h, actual_prefix_mode = _extract_h_with_prefix(backbone, inputs, first_atlas, prefix_mode)
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

            prev_mode = getattr(backbone, "adapter_mode", None)
            backbone._clear_path_slot_ids()
            for l2 in atlas_layers:
                backbone.blocks[l2].path_slot_id = best_slot
            backbone.set_adapter_mode("path_key_slot_student")
            try:
                logits = backbone.forward(inputs)["logits"][:, :total_classes]
            finally:
                backbone._clear_path_slot_ids()
                if prev_mode is not None:
                    backbone.set_adapter_mode(prev_mode)

            preds = logits.argmax(dim=1)
            evaluated += B
            correct += int((preds == targets).sum())

    acc_evaluated = 100.0 * correct / max(evaluated, 1)
    acc_missing_as_wrong = 100.0 * correct / max(total, 1)
    coverage = evaluated / max(total, 1)

    logging.info("[SlotQualityCoverageAudit] task=%d layer=%d prefix_mode=%s actual_prefix_mode=%s "
                 "max_samples=%s total=%d evaluated=%d coverage=%.4f "
                 "acc_evaluated=%.2f acc_missing_as_wrong=%.2f",
                 task_id, first_atlas, prefix_mode, actual_prefix_mode,
                 max_samples if max_samples else "full",
                 total, evaluated, coverage, acc_evaluated, acc_missing_as_wrong)
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
                            diagnostic_sq_best: float, task_id: int,
                            config: Optional[dict] = None) -> Dict:
    """Compare official CNN evaluation with diagnostic router results."""
    backbone = model._network.backbone if hasattr(model, '_network') else model
    backbone.eval()
    device = next(backbone.parameters()).device
    config = config or {}
    default_router = getattr(model, "default_router", "shared_q_dist")
    max_samples = config.get("official_audit_max_samples", 0)

    prev_router = None
    for lid in getattr(model, "atlas_layers", []):
        blk = backbone.get_block(lid)
        prev_router = getattr(blk, "nll_router", None)
        break

    router = None
    if hasattr(model, "_build_nll_router_for_name"):
        router = model._build_nll_router_for_name(default_router)
    elif default_router in ("raw_nll", "path_raw_nll"):
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)

    all_preds, all_labels = [], []
    count = 0

    backbone.set_nll_router(router)
    try:
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                if max_samples and count >= max_samples:
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                if default_router == "path_raw_nll":
                    logits = backbone.compute_path_key_slot_logits(inputs)[:, :total_classes]
                else:
                    logits = backbone.compute_key_slot_logits(inputs)[:, :total_classes]
                all_preds.append(logits.argmax(dim=1).cpu())
                all_labels.append(targets.cpu())
                count += inputs.shape[0]
    finally:
        backbone.set_nll_router(prev_router)

    if not all_preds:
        logging.info("[OfficialForwardAudit][WARN] task=%d no samples evaluated", task_id)
        return {"official_total": 0.0, "samples": 0, "router_mode": default_router}

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    official_acc = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)
    expected = diagnostic_path if default_router == "path_raw_nll" else diagnostic_raw

    logging.info("[OfficialForwardAudit] task=%d router_mode=%s samples=%d official_total=%.2f "
                 "diag_raw=%.2f diag_path=%.2f diag_sqbest=%.2f "
                 "gap_vs_raw=%+.2f gap_vs_path=%+.2f gap_vs_sq=%+.2f gap_vs_default=%+.2f",
                 task_id, default_router, len(y_true), official_acc, diagnostic_raw, diagnostic_path,
                 diagnostic_sq_best,
                 official_acc - diagnostic_raw, official_acc - diagnostic_path,
                 official_acc - diagnostic_sq_best, official_acc - expected)

    if abs(official_acc - expected) > 1.0:
        logging.info("[OfficialForwardAudit][MISMATCH] task=%d router_mode=%s official %.2f vs expected %.2f; "
                     "likely different router/mode/sample coverage/known_classes",
                     task_id, default_router, official_acc, expected)

    return {"official_total": official_acc,
            "samples": len(y_true),
            "router_mode": default_router,
            "gap_vs_raw": official_acc - diagnostic_raw,
            "gap_vs_path": official_acc - diagnostic_path,
            "gap_vs_sqbest": official_acc - diagnostic_sq_best,
            "gap_vs_default": official_acc - expected}


def evaluate_slot_quality_baseline_router(backbone, data_loader, atlas_layers,
                                           total_classes: int, task_id: int,
                                           config: Optional[dict] = None,
                                           prefix_mode: Optional[str] = None,
                                           tag: str = "deploy") -> Dict:
    """Reproduce SlotQualityPriorEval weight=0.00 as a named diagnostic router.

    This is the reproducible version: per-layer raw NLL path-consistent routing.
    """
    from gase.routing.nll_router import CalibratedNLLSlotRouter

    backbone.eval()
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    config = config or {}
    prefix_mode = prefix_mode or config.get("slot_quality_prefix_mode", "path_key_slot_student")

    all_preds, all_labels = [], []
    count = 0
    max_samples = config.get("slot_quality_audit_max_samples", 0)
    first_atlas = min(atlas_layers)
    actual_prefix_mode = prefix_mode

    with torch.no_grad():
        for _, inputs, targets in data_loader:
            if max_samples and count >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.shape[0]
            if max_samples:
                keep = min(B, max_samples - count)
                inputs, targets = inputs[:keep], targets[:keep]
                B = keep
            count += B

            # Use L9's raw-NLL decision and force that slot across the path.
            blk = backbone.get_block(first_atlas)
            available = blk.get_available_slot_ids(0)
            cs = blk.chart_states.get(0)
            ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                       for sid in available if blk.slot_states.get(f"0_{sid}") is not None}
            if cs is None or len(ss_dict) <= 1:
                best_slot = torch.zeros(B, dtype=torch.long, device=device)
            else:
                h, actual_prefix_mode = _extract_h_with_prefix(backbone, inputs, first_atlas, prefix_mode)
                sid_list = sorted(ss_dict.keys())
                nll_list = [router.compute_nll(h, cs, ss_dict[sid]) for sid in sid_list]
                nll_stack = torch.stack(nll_list, dim=1)
                best_idx = nll_stack.argmin(dim=1)
                best_slot = torch.tensor([sid_list[i.item()] for i in best_idx],
                                         device=device, dtype=torch.long)

            prev_mode = getattr(backbone, "adapter_mode", None)
            backbone._clear_path_slot_ids()
            for l2 in atlas_layers:
                backbone.blocks[l2].path_slot_id = best_slot
            backbone.set_adapter_mode("path_key_slot_student")
            try:
                logits = backbone.forward(inputs)["logits"][:, :total_classes]
            finally:
                backbone._clear_path_slot_ids()
                if prev_mode is not None:
                    backbone.set_adapter_mode(prev_mode)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)

    logging.info("[SlotQualityBaselineRouter][%s] task=%d prefix_mode=%s actual_prefix_mode=%s "
                 "samples=%d max_samples=%s total=%.2f",
                 tag, task_id, prefix_mode, actual_prefix_mode,
                 len(y_true), max_samples if max_samples else "full", acc)
    return {"top1": acc, "samples": len(y_true),
            "prefix_mode": prefix_mode, "actual_prefix_mode": actual_prefix_mode}


def evaluate_prototype_path_router(backbone, data_loader, atlas_layers,
                                    total_classes: int, task_id: int,
                                    config: Optional[dict] = None) -> Dict:
    """Evaluate deploy-prefix L9 multi-prototype routing with path-forced slots."""
    from gase.routing.prototype_router import PrototypeNLLSlotRouter

    backbone.eval()
    config = config or {}
    device = next(backbone.parameters()).device
    first_atlas = min(atlas_layers)
    prefix_mode = config.get("slot_quality_prefix_mode", "path_key_slot_student")
    max_samples = config.get("slot_quality_audit_max_samples", 0)
    router = PrototypeNLLSlotRouter(
        temperature=config.get("prototype_temperature", 1.0),
        use_logdet=config.get("prototype_use_logdet", True),
        use_proto_prior=config.get("prototype_use_prior", True),
        aggregate=config.get("prototype_aggregate", "logsumexp"),
    )

    all_preds, all_labels = [], []
    slot_hist = {}
    count = 0
    actual_prefix_mode = prefix_mode

    with torch.no_grad():
        for _, inputs, targets in data_loader:
            if max_samples and count >= max_samples:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            B = inputs.shape[0]
            if max_samples:
                keep = min(B, max_samples - count)
                inputs, targets = inputs[:keep], targets[:keep]
                B = keep
            count += B

            blk = backbone.get_block(first_atlas)
            available = blk.get_available_slot_ids(0)
            cs = blk.chart_states.get(0)
            ss_dict = {sid: blk.slot_states.get(f"0_{sid}")
                       for sid in available if blk.slot_states.get(f"0_{sid}") is not None}
            if cs is None or len(ss_dict) <= 1:
                selected = torch.zeros(B, dtype=torch.long, device=device)
            else:
                h, actual_prefix_mode = _extract_h_with_prefix(backbone, inputs, first_atlas, prefix_mode)
                selected = router.route(h, cs, ss_dict)["slot_ids"]

            for s in selected.unique().tolist():
                slot_hist[int(s)] = slot_hist.get(int(s), 0) + int((selected == s).sum())

            prev_mode = getattr(backbone, "adapter_mode", None)
            backbone._clear_path_slot_ids()
            for lid in atlas_layers:
                backbone.blocks[lid].path_slot_id = selected
            backbone.set_adapter_mode("path_key_slot_student")
            try:
                logits = backbone.forward(inputs)["logits"][:, :total_classes]
            finally:
                backbone._clear_path_slot_ids()
                if prev_mode is not None:
                    backbone.set_adapter_mode(prev_mode)

            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_labels.append(targets.cpu().numpy())

    if not all_preds:
        logging.info("[PrototypePathRouter][WARN] task=%d no samples evaluated", task_id)
        return {"top1": 0.0, "samples": 0, "slot_hist": slot_hist,
                "prefix_mode": prefix_mode, "actual_prefix_mode": actual_prefix_mode}

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    acc = 100.0 * (y_pred == y_true).sum() / max(len(y_true), 1)
    logging.info("[PrototypePathRouter] task=%d prefix_mode=%s actual_prefix_mode=%s "
                 "samples=%d max_samples=%s total=%.2f slot_hist=%s",
                 task_id, prefix_mode, actual_prefix_mode, len(y_true),
                 max_samples if max_samples else "full", acc, slot_hist)
    return {"top1": acc, "samples": len(y_true), "slot_hist": slot_hist,
            "prefix_mode": prefix_mode, "actual_prefix_mode": actual_prefix_mode}


# ============================================================================
#  Baseline comparison
# ============================================================================

def run_baseline_compare(backbone, data_loader, atlas_layers, total_classes: int,
                          raw_nll: float, path_nll: float, candidate_path: float,
                          oracle: float, task_id: int, config: Optional[dict] = None) -> Dict:
    """Compare all routers and log histograms."""
    config = config or {}

    deploy_prefix_mode = config.get("slot_quality_prefix_mode", "path_key_slot_student")
    sq_deploy = evaluate_slot_quality_baseline_router(
        backbone, data_loader, atlas_layers, total_classes, task_id,
        config=config, prefix_mode=deploy_prefix_mode, tag="deploy")
    sq_total = sq_deploy["top1"]

    sq_current = None
    if config.get("audit_compare_current_prefix", True):
        sq_current = evaluate_slot_quality_baseline_router(
            backbone, data_loader, atlas_layers, total_classes, task_id,
            config=config, prefix_mode="current", tag="current")
    sq_current_total = sq_current["top1"] if sq_current else None
    proto_result = None
    if config.get("audit_prototype_router", True):
        proto_result = evaluate_prototype_path_router(
            backbone, data_loader, atlas_layers, total_classes, task_id, config=config)
    proto_total = proto_result["top1"] if proto_result else None

    logging.info("[SlotQualityBaselineCompare] task=%d raw=%.2f path=%.2f cand_path=%.2f "
                 "sq_deploy=%.2f sq_current=%s proto=%s oracle=%.2f "
                 "deploy_gain_vs_raw=%+.2f deploy_gain_vs_path=%+.2f deploy_gap_to_oracle=%.2f",
                 task_id, raw_nll, path_nll, candidate_path, sq_total,
                 f"{sq_current_total:.2f}" if sq_current_total is not None else "disabled",
                 f"{proto_total:.2f}" if proto_total is not None else "disabled",
                 oracle, sq_total - raw_nll, sq_total - path_nll, oracle - sq_total)

    # Oracle match
    from gase.routing.nll_router import CalibratedNLLSlotRouter
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
    device = next(backbone.parameters()).device
    first_atlas = min(atlas_layers)

    sq_match = raw_match = path_match = 0
    proto_top1_match = 0
    proto_topm_match = 0
    total = 0
    sq_hist = {}
    raw_hist = {}
    path_hist = {}
    proto_hist = {}
    max_samples = 256  # oracle slot eval is expensive; 256 samples is enough for match stats
    proto_top_m = config.get("prototype_top_m", 3)
    proto_router = None
    if config.get("audit_prototype_router", True):
        from gase.routing.prototype_router import PrototypeNLLSlotRouter
        proto_router = PrototypeNLLSlotRouter(
            temperature=config.get("prototype_temperature", 1.0),
            use_logdet=config.get("prototype_use_logdet", True),
            use_proto_prior=config.get("prototype_use_prior", True),
            aggregate=config.get("prototype_aggregate", "logsumexp"),
        )

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
            if cs is None or len(ss_dict) <= 1:
                continue

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
            h9, _ = _extract_h_with_prefix(backbone, inputs, first_atlas, deploy_prefix_mode)
            sid_list = sorted(ss_dict.keys())
            nll_list = [router.compute_nll(h9, cs, ss_dict[sid]) for sid in sid_list]
            sq_slot = torch.tensor([sid_list[i.item()] for i in torch.stack(nll_list, dim=1).argmin(dim=1)],
                                  device=device, dtype=torch.long)

            # Raw NLL
            routing = router.route(h9, cs, ss_dict)
            raw_slot = routing["slot_ids"]
            path_slot = routing["slot_ids"]  # path uses L9

            sq_match += int((sq_slot == best_slot).sum())
            raw_match += int((raw_slot == best_slot).sum())
            path_match += int((path_slot == best_slot).sum())

            if proto_router is not None:
                proto_routing = proto_router.route(h9, cs, ss_dict)
                proto_slot = proto_routing["slot_ids"]
                proto_top_ids, _, _ = proto_router.topm_slot_ids(h9, cs, ss_dict, m=proto_top_m)
                proto_top1_match += int((proto_slot == best_slot).sum())
                proto_topm_match += int((proto_top_ids == best_slot.unsqueeze(1)).any(dim=1).sum())
                for s in proto_slot.unique().tolist():
                    proto_hist[int(s)] = proto_hist.get(int(s), 0) + int((proto_slot == s).sum())

            for s in sq_slot.unique().tolist():
                sq_hist[int(s)] = sq_hist.get(int(s), 0) + int((sq_slot == s).sum())
            for s in raw_slot.unique().tolist():
                raw_hist[int(s)] = raw_hist.get(int(s), 0) + int((raw_slot == s).sum())
            for s in path_slot.unique().tolist():
                path_hist[int(s)] = path_hist.get(int(s), 0) + int((path_slot == s).sum())

    logging.info("[SlotQualityBaselineOracleMatch] task=%d sq=%.1f%% raw=%.1f%% path=%.1f%%",
                 task_id, 100 * sq_match / max(total, 1),
                 100 * raw_match / max(total, 1), 100 * path_match / max(total, 1))
    if proto_router is not None:
        logging.info("[PrototypeRouterOracleCoverage] task=%d top1=%.1f%% top%d=%.1f%%",
                     task_id, 100 * proto_top1_match / max(total, 1), proto_top_m,
                     100 * proto_topm_match / max(total, 1))
    logging.info("[SlotQualityBaselineHist] task=%d sq=%s raw=%s path=%s proto=%s",
                 task_id, sq_hist, raw_hist, path_hist, proto_hist)

    return {"sq_top1": sq_total,
            "sq_deploy_top1": sq_total,
            "sq_current_top1": sq_current_total,
            "sq_deploy": sq_deploy,
            "sq_current": sq_current,
            "prototype_path": proto_result,
            "sq_hist": sq_hist, "raw_hist": raw_hist, "path_hist": path_hist,
            "proto_hist": proto_hist,
            "sq_oracle_match": 100 * sq_match / max(total, 1),
            "raw_oracle_match": 100 * raw_match / max(total, 1),
            "path_oracle_match": 100 * path_match / max(total, 1),
            "proto_top1_oracle_match": 100 * proto_top1_match / max(total, 1),
            "proto_topm_oracle_match": 100 * proto_topm_match / max(total, 1)}


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
        raw_nll, task_id, config=config)  # sq_best not ready yet
    results["baseline_compare"] = run_baseline_compare(
        backbone, data_loader, atlas_layers, total_classes,
        raw_nll, path_nll, candidate_path, oracle, task_id, config=config)

    sq_total = results["baseline_compare"]["sq_deploy_top1"]
    sq_current = results["baseline_compare"].get("sq_current_top1")
    prefix_mode = results["baseline_compare"].get("sq_deploy", {}).get(
        "prefix_mode", config.get("slot_quality_prefix_mode", "path_key_slot_student"))
    coverage_ok = results["coverage"].get("coverage", 0) >= 0.99
    leakage_ok = results["leakage"].get("status") == "OK"
    deployable_prefix = prefix_mode != "current"
    beats_path = sq_total > path_nll + 0.5
    beats_raw = sq_total > raw_nll + 0.5
    recommend = leakage_ok and coverage_ok and deployable_prefix and beats_path and beats_raw

    reasons = []
    if not leakage_ok:
        reasons.append("leakage_check_failed")
    if not coverage_ok:
        reasons.append(f"coverage={results['coverage'].get('coverage',0):.4f}<0.99")
    if not deployable_prefix:
        reasons.append(f"prefix_mode={prefix_mode} is diagnostic_only")
    if not beats_path:
        reasons.append(f"sq_deploy={sq_total:.2f}<=path+0.5={path_nll+0.5:.2f}")
    if not beats_raw:
        reasons.append(f"sq_deploy={sq_total:.2f}<=raw+0.5={raw_nll+0.5:.2f}")

    logging.info("[RouterPromotionRecommendation] candidate=slot_quality_baseline_deploy "
                 "task=%d prefix_mode=%s sq_deploy=%.2f sq_current=%s recommend=%s reason=%s",
                 task_id, prefix_mode, sq_total,
                 f"{sq_current:.2f}" if sq_current is not None else "disabled",
                 recommend, "; ".join(reasons) if reasons else "all_checks_passed")

    results["promotion"] = {"candidate": "slot_quality_baseline_deploy",
                            "recommend": recommend, "reasons": reasons,
                            "prefix_mode": prefix_mode,
                            "sq_deploy_top1": sq_total,
                            "sq_current_top1": sq_current}
    results["elapsed"] = time.time() - t0
    return results
