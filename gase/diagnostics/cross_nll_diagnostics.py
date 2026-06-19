"""Phase-9: Cross-slot NLL matrix diagnostics."""

import logging
from typing import Any, Dict, List

import torch
from torch import Tensor


def compute_cross_nll_matrix(
    h_by_source: Dict[int, Tensor],
    chart_state,
    slot_states: Dict[int, object],
    eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Compute cross-slot NLL matrix.

    For each source slot's h_chart, compute NLL to all target slots.
    This diagnoses whether slot0 naturally explains all features better
    (low NLL even for new-task features), causing slot0 adsorption.

    Args:
        h_by_source: dict source_slot_id -> h_chart [N, D].
        chart_state: ChartState.
        slot_states: dict slot_id -> SlotState.
        eps: numerical epsilon.

    Returns:
        Dict with per-source per-target NLL means.
    """
    from gase.routing.nll_router import CalibratedNLLSlotRouter
    router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False,
                                      use_logdet=True, eps=eps)

    matrix: Dict[int, Dict[int, Dict[str, float]]] = {}
    source_ids = sorted(h_by_source.keys())
    target_ids = sorted(slot_states.keys())

    for src_id in source_ids:
        h_src = h_by_source[src_id]
        matrix[src_id] = {}
        best_target = None
        best_nll = float("inf")
        for tgt_id in target_ids:
            ss = slot_states.get(tgt_id)
            if ss is None or getattr(ss, "router_key", None) is None:
                continue
            nll = router.compute_nll(h_src, chart_state, ss)
            mean_nll = float(nll.mean())
            matrix[src_id][tgt_id] = {"mean": mean_nll}
            logging.info("[CrossNLLMatrix] layer=%d source=%d target=%d mean=%.2f",
                         chart_state.layer_id, src_id, tgt_id, mean_nll)
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_target = tgt_id
        if best_target is not None:
            logging.info("[CrossNLLMatrix] layer=%d source=%d best_target=%d best_nll=%.2f",
                         chart_state.layer_id, src_id, best_target, best_nll)

    return {"matrix": matrix, "source_ids": source_ids, "target_ids": target_ids}
