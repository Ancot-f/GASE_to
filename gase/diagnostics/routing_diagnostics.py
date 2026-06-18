"""Phase-7.7: Routing diagnostics for per-sample slot selection."""

import logging
from collections import defaultdict
from typing import Any, Dict, List

import torch
from torch import Tensor


def summarize_routing_records(
    routing_records: List[Dict[str, Any]],
    labels: Tensor,
    increment: int,
    mode: str = "per_layer",
) -> Dict[str, Any]:
    """
    Summarize routing behavior across multiple batches.

    Args:
        routing_records: list of per-batch routing info dicts from ViTGASE.
        labels: all ground-truth labels [N] (for diagnostics only, NOT for forward).
        increment: classes per task.
        mode: "per_layer" or "path".

    Returns:
        Summary dict with per-layer slot_hist, routing_acc, entropy, margin, confusion.
    """
    oracle_task = labels // increment
    summary: Dict[str, Any] = {"mode": mode}

    N = labels.shape[0]

    if mode == "per_layer":
        # Collect all per-layer slot_ids
        all_slot_ids: Dict[int, List[Tensor]] = {}
        all_entropy: Dict[int, List[Tensor]] = {}
        all_margin: Dict[int, List[Tensor]] = {}
        slot_id_lists: Dict[int, List[int]] = {}

        for record in routing_records:
            per_layer = record.get("per_layer", {})
            for lid, info in per_layer.items():
                if lid not in all_slot_ids:
                    all_slot_ids[lid] = []
                    all_entropy[lid] = []
                    all_margin[lid] = []
                if info.get("slot_ids") is not None:
                    all_slot_ids[lid].append(info["slot_ids"])
                if info.get("entropy") is not None:
                    all_entropy[lid].append(info["entropy"])
                if info.get("margin") is not None:
                    all_margin[lid].append(info["margin"])
                if info.get("slot_id_list"):
                    slot_id_lists[lid] = info["slot_id_list"]

        layers_summary = {}
        for lid in sorted(all_slot_ids.keys()):
            cat_slot_ids = torch.cat(all_slot_ids[lid])  # should be [N]
            actual_n = min(cat_slot_ids.shape[0], N)
            cat_slot_ids = cat_slot_ids[:actual_n]
            oracle_n = oracle_task[:actual_n]

            unique, counts = cat_slot_ids.unique(return_counts=True)
            hist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}

            entropy_cat = torch.cat(all_entropy[lid])[:actual_n] if all_entropy[lid] else None
            margin_cat = torch.cat(all_margin[lid])[:actual_n] if all_margin[lid] else None

            layer_diag: Dict[str, Any] = {
                "slot_hist": hist,
                "entropy_mean": float(entropy_cat.mean()) if entropy_cat is not None else None,
                "margin_mean": float(margin_cat[margin_cat < 1e9].mean()) if margin_cat is not None else None,
                "entropy_std": float(entropy_cat.std()) if entropy_cat is not None else None,
                "margin_std": float(margin_cat[margin_cat < 1e9].std()) if margin_cat is not None else None,
            }

            if len(slot_id_lists.get(lid, [])) > 0:
                routing_acc = float((cat_slot_ids == oracle_n.to(cat_slot_ids.device)).float().mean())
                layer_diag["routing_acc"] = routing_acc

                # Confusion matrix: oracle_task -> selected_slot
                confusion = defaultdict(lambda: defaultdict(int))
                for o, s in zip(oracle_n.tolist(), cat_slot_ids.tolist()):
                    confusion[o][s] += 1
                layer_diag["confusion"] = {str(k): dict(v) for k, v in confusion.items()}

                logging.info(
                    "[RoutingDiag][%s] layer=%d slot_hist=%s routing_acc=%.3f "
                    "entropy=%.3f±%.3f margin=%.3f±%.3f",
                    mode, lid, hist, routing_acc,
                    layer_diag["entropy_mean"] or 0, layer_diag["entropy_std"] or 0,
                    layer_diag["margin_mean"] or 0, layer_diag["margin_std"] or 0,
                )
                logging.info("[RoutingDiag][%s] layer=%d confusion=%s", mode, lid,
                             {str(k): dict(v) for k, v in confusion.items()})
            else:
                logging.info("[RoutingDiag][%s] layer=%d slot_hist=%s", mode, lid, hist)

            layers_summary[lid] = layer_diag

        summary["layers"] = layers_summary

    elif mode == "path":
        # Path mode: collect decider layer's routing info
        all_path_slot_ids = []
        all_path_entropy = []
        all_path_margin = []
        path_slot_id_list = []

        for record in routing_records:
            path_info = record.get("path", {})
            if path_info.get("slot_ids") is not None:
                all_path_slot_ids.append(path_info["slot_ids"])
            if path_info.get("entropy") is not None:
                all_path_entropy.append(path_info["entropy"])
            if path_info.get("margin") is not None:
                all_path_margin.append(path_info["margin"])
            if path_info.get("slot_id_list"):
                path_slot_id_list = path_info["slot_id_list"]

        if all_path_slot_ids:
            cat_ids = torch.cat(all_path_slot_ids)[:N]
            oracle_n = oracle_task[:cat_ids.shape[0]]
            unique, counts = cat_ids.unique(return_counts=True)
            hist = {int(u.item()): int(c.item()) for u, c in zip(unique, counts)}
            entropy_cat = torch.cat(all_path_entropy)[:N] if all_path_entropy else None
            margin_cat = torch.cat(all_path_margin)[:N] if all_path_margin else None

            routing_acc = float((cat_ids == oracle_n.to(cat_ids.device)).float().mean())
            confusion = defaultdict(lambda: defaultdict(int))
            for o, s in zip(oracle_n.tolist(), cat_ids.tolist()):
                confusion[o][s] += 1

            summary["path_diag"] = {
                "decider_layer": routing_records[0].get("path", {}).get("decider_layer", 9),
                "slot_hist": hist,
                "routing_acc": routing_acc,
                "entropy_mean": float(entropy_cat.mean()) if entropy_cat is not None else None,
                "margin_mean": float(margin_cat[margin_cat < 1e9].mean()) if margin_cat is not None else None,
                "confusion": {str(k): dict(v) for k, v in confusion.items()},
            }
            logging.info(
                "[RoutingDiag][%s] decider=%d slot_hist=%s routing_acc=%.3f",
                mode, summary["path_diag"]["decider_layer"], hist, routing_acc,
            )
            logging.info("[RoutingDiag][%s] confusion=%s", mode,
                         {str(k): dict(v) for k, v in confusion.items()})

    return summary
