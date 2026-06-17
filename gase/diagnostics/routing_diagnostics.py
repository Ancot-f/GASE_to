"""Phase-7: Routing diagnostics for per-sample slot selection."""

import logging
from typing import Any, Dict, List

import torch
from torch import Tensor


def summarize_routing(
    routing_info: Dict[int, Any],
    labels: Tensor,
    increment: int,
) -> Dict[str, Any]:
    """
    Summarize per-layer routing behavior.

    Args:
        routing_info: dict layer_id -> routing dict from KeySlotRouter.
        labels: ground-truth labels [N].
        increment: classes per task.

    Returns:
        Dict with per-layer slot_hist, entropy_mean, margin_mean, routing_acc.
    """
    oracle_task = labels // increment
    summary = {}

    for lid, info in routing_info.items():
        slot_ids = info.get("slot_ids")
        entropy = info.get("entropy")
        margin = info.get("margin")
        slot_id_list = info.get("slot_id_list", [])

        layer_summary = {
            "entropy_mean": float(entropy.mean()) if entropy is not None else None,
            "margin_mean": float(margin[margin < 1e9].mean()) if margin is not None else None,
            "slot_hist": {},
        }

        if slot_ids is not None:
            unique, counts = slot_ids.unique(return_counts=True)
            hist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
            layer_summary["slot_hist"] = hist

            if len(slot_id_list) > 0:
                routing_acc = float((slot_ids == oracle_task.to(slot_ids.device)).float().mean())
                layer_summary["routing_acc"] = routing_acc

                logging.info(
                    "[RoutingDiag] layer=%d slot_hist=%s entropy_mean=%.3f margin_mean=%.3f routing_acc=%.3f",
                    lid, hist, layer_summary["entropy_mean"] or 0,
                    layer_summary["margin_mean"] or 0, routing_acc,
                )
            else:
                logging.info(
                    "[RoutingDiag] layer=%d slot_hist=%s entropy_mean=%.3f",
                    lid, hist, layer_summary["entropy_mean"] or 0,
                )

        summary[lid] = layer_summary

    return summary
