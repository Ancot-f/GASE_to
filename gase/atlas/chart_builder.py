"""
ChartBuilder: PPCA-based chart construction from feature geometry.

Chart = feature geometry container.
Chart uses ONLY h_chart (pre-adapter CLS feature).
Chart MUST NOT use delta_teacher, labels, or task_id.
"""

import logging
from typing import Dict, List, Optional

import torch
from torch import Tensor

from .chart_state import ChartState
from .ppca import PPCAEstimator


class ChartBuilder:
    """
    Builds charts from h_chart via PPCA/PCA.

    Phase-6.5: single-chart policy (chart_id=0 per layer).
    Future: multi-chart with MDL-based acceptance.
    """

    def __init__(self, config: dict):
        self.config = config
        self.min_support: int = config.get("min_support", 16)
        self.max_charts_per_layer: int = config.get("max_charts_per_layer", 24)
        self.ppca_rank: int = config.get("rank", 8)
        self.mdl_lambda: float = config.get("mdl_lambda", 1.0)
        self.posterior_threshold: float = config.get("posterior_threshold", 0.55)
        self.entropy_threshold: float = config.get("entropy_threshold", 1.0)

    # ------------------------------------------------------------------
    #  Phase-6.5: single-chart build-or-reuse
    # ------------------------------------------------------------------

    def build_or_reuse_single_chart_for_layer(
        self,
        h_chart: Tensor,
        layer_id: int,
        task_id: int,
        existing_charts: Optional[List[ChartState]] = None,
        chart_id: int = 0,
    ) -> ChartState:
        """
        Phase-6.5 single-chart policy.

        If chart_id already exists: reuse, do not update geometry.
        Else: fit PPCA from h_chart.

        Args:
            h_chart: pre-adapter CLS features [N, D]. MUST be permanent-path.
            layer_id: ViT block index.
            task_id: current task id (for logging only).
            existing_charts: previously built charts (optional).
            chart_id: chart id (default 0).

        Returns:
            ChartState.
        """
        if existing_charts:
            for cs in existing_charts:
                if cs.chart_id == chart_id and cs.mu is not None:
                    logging.info(
                        "[ChartContract] layer=%d chart=%d REUSE "
                        "definition=feature_geometry method=PPCA/PCA "
                        "uses_delta_teacher=False uses_label=False "
                        "support=%d rank=%d sigma_perp=%.4f slots=%s",
                        layer_id, chart_id, cs.n_support,
                        cs.U.shape[1] if cs.U is not None else 0,
                        cs.sigma_perp, cs.slot_ids,
                    )
                    return cs

        ppca = PPCAEstimator(dim=h_chart.shape[1], rank=self.ppca_rank)
        ppca.fit(h_chart, rank=self.ppca_rank)
        chart_state = ppca.to_chart_state(layer_id=layer_id, chart_id=chart_id)

        logging.info(
            "[ChartContract] layer=%d chart=%d BUILD "
            "definition=feature_geometry method=PPCA/PCA "
            "uses_delta_teacher=False uses_label=False "
            "support=%d rank=%d sigma_perp=%.4f radius_d2=%.4f",
            layer_id, chart_id, chart_state.n_support,
            ppca.rank, ppca.sigma_perp, chart_state.radius_d2,
        )
        return chart_state

    def build_single_chart_for_layer(
        self, h_chart: Tensor, layer_id: int, chart_id: int = 0,
    ) -> ChartState:
        """Legacy wrapper: always builds new chart (no reuse)."""
        return self.build_or_reuse_single_chart_for_layer(
            h_chart, layer_id, task_id=-1, existing_charts=None, chart_id=chart_id,
        )

    # ------------------------------------------------------------------
    #  Future: multi-chart evaluation (skeleton only)
    # ------------------------------------------------------------------

    def evaluate_new_chart_need(
        self,
        h_uncovered: Tensor,
        existing_charts: List[ChartState],
    ) -> Dict[str, float]:
        """
        Future only. Estimate whether uncovered samples require a new chart.

        Do not use in Phase-6.5.
        """
        raise NotImplementedError("Phase-7+ will implement multi-chart evaluation.")

    # ------------------------------------------------------------------
    #  Unimplemented (Phase-7+)
    # ------------------------------------------------------------------

    def build_candidates(self, h_chart, existing_charts):
        raise NotImplementedError("Phase-7+.")
    def split_covered_boundary_uncovered(self, h_chart, existing_charts):
        raise NotImplementedError("Phase-7+.")
    def build_candidate_components(self, h_uncovered):
        raise NotImplementedError("Phase-7+.")
    def fit_chart_from_component(self, comp, layer_id, chart_id):
        raise NotImplementedError("Phase-7+.")
    def accept_candidate(self, candidate, existing_charts):
        raise NotImplementedError("Phase-7+.")
    def compute_candidate_mdl_gain(self, candidate, h_comp, existing):
        raise NotImplementedError("Phase-7+.")
    def assign_or_create_chart(self, h_chart, existing_charts):
        raise NotImplementedError("Phase-7+.")
