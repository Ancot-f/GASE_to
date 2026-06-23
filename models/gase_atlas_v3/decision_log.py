"""ExpansionDecisionLog -records why the model expanded at each step.

Every chart creation, adapter update, or free fallback decision is logged
with scores, reasons, and before/after diagnostics.

Corresponds to design doc Section 13.7.
"""

from dataclasses import dataclass, field
from typing import Optional


# Valid decision types
DECISION_REUSE_CHART = "reuse_chart"
DECISION_UPDATE_CHART_ADAPTER = "update_chart_adapter"
DECISION_ADD_CHART = "add_chart"
DECISION_UPDATE_FREE_ADAPTER = "update_free_adapter"
DECISION_FALLBACK_IDENTITY = "fallback_identity"

# Valid reasons
REASON_GEO_COVERED_RESIDUAL_GOOD = "geo_covered_residual_good"
REASON_GEO_COVERED_RESIDUAL_BAD = "geo_covered_residual_bad"
REASON_GEO_OUTLIER = "geo_outlier"
REASON_HIGH_UNCERTAINTY = "high_uncertainty"
REASON_HIGH_FREE_RATIO = "high_free_ratio"
REASON_LOW_SUBR2 = "low_subR2"
REASON_NO_EXISTING_CHARTS = "no_existing_charts"
REASON_LOW_SAMPLES = "low_samples"
REASON_OVERLAP_REJECT = "overlap_reject"


@dataclass
class ExpansionDecision:
    """Single expansion decision record (design doc Section 13.7)."""
    task_id: int
    layer_id: int
    decision_type: str          # one of DECISION_* constants
    reason: str                 # one of REASON_* constants
    chart_id: int = -1          # affected chart (-1 if new/free)
    num_samples: int = 0
    geo_score: float = 0.0      # Mahalanobis d虏 or coverage score
    residual_score: float = 0.0  # residual error or subR2
    coverage_before: float = 0.0
    coverage_after: float = 0.0
    distill_error_before: float = 0.0
    distill_error_after: float = 0.0
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "task_id": self.task_id,
            "layer_id": self.layer_id,
            "decision_type": self.decision_type,
            "reason": self.reason,
            "chart_id": self.chart_id,
            "num_samples": self.num_samples,
            "geo_score": self.geo_score,
            "residual_score": self.residual_score,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
            "distill_error_before": self.distill_error_before,
            "distill_error_after": self.distill_error_after,
            **self.extra,
        }


class ExpansionDecisionLog:
    """Log of all expansion decisions across tasks and layers.

    Usage:
        log = ExpansionDecisionLog()
        log.record(ExpansionDecision(task_id=1, layer_id=9,
                    decision_type=DECISION_ADD_CHART, reason=REASON_GEO_OUTLIER, ...))

        # Summary after task:
        summary = log.summary(task_id=1, layer_id=9)
    """

    def __init__(self):
        self._decisions = []  # List[ExpansionDecision]

    def record(self, decision):
        """Record a single expansion decision."""
        self._decisions.append(decision)

    def record_batch(self, decisions):
        """Record multiple decisions."""
        self._decisions.extend(decisions)

    def query(self, task_id=None, layer_id=None, decision_type=None):
        """Filter decisions by task, layer, or type."""
        result = self._decisions
        if task_id is not None:
            result = [d for d in result if d.task_id == task_id]
        if layer_id is not None:
            result = [d for d in result if d.layer_id == layer_id]
        if decision_type is not None:
            result = [d for d in result if d.decision_type == decision_type]
        return result

    def summary(self, task_id=None, layer_id=None):
        """Return a summary dict for the given filters.

        Returns:
            dict with decision_type_distribution, reason_distribution,
            avg_geo_score, avg_residual_score, coverage_delta, etc.
        """
        decisions = self.query(task_id=task_id, layer_id=layer_id)
        if not decisions:
            return {"total_decisions": 0}

        type_dist = {}
        reason_dist = {}
        geo_scores = []
        residual_scores = []
        coverage_deltas = []
        distill_errors_before = []
        distill_errors_after = []

        for d in decisions:
            type_dist[d.decision_type] = type_dist.get(d.decision_type, 0) + 1
            reason_dist[d.reason] = reason_dist.get(d.reason, 0) + 1
            if d.geo_score > 0:
                geo_scores.append(d.geo_score)
            if d.residual_score > 0:
                residual_scores.append(d.residual_score)
            coverage_deltas.append(d.coverage_after - d.coverage_before)
            if d.distill_error_before > 0:
                distill_errors_before.append(d.distill_error_before)
            if d.distill_error_after > 0:
                distill_errors_after.append(d.distill_error_after)

        return {
            "total_decisions": len(decisions),
            "decision_type_distribution": type_dist,
            "reason_distribution": reason_dist,
            "avg_geo_score": sum(geo_scores) / len(geo_scores) if geo_scores else 0,
            "avg_residual_score": sum(residual_scores) / len(residual_scores) if residual_scores else 0,
            "avg_coverage_delta": sum(coverage_deltas) / len(coverage_deltas) if coverage_deltas else 0,
            "avg_distill_error_before": sum(distill_errors_before) / len(distill_errors_before) if distill_errors_before else 0,
            "avg_distill_error_after": sum(distill_errors_after) / len(distill_errors_after) if distill_errors_after else 0,
            "num_add_chart": type_dist.get(DECISION_ADD_CHART, 0),
            "num_update_adapter": type_dist.get(DECISION_UPDATE_CHART_ADAPTER, 0),
            "num_reuse_chart": type_dist.get(DECISION_REUSE_CHART, 0),
            "num_free_fallback": type_dist.get(DECISION_UPDATE_FREE_ADAPTER, 0),
        }

    def log_summary_string(self, task_id=None, layer_id=None):
        """Human-readable summary string for logging."""
        s = self.summary(task_id=task_id, layer_id=layer_id)
        if s["total_decisions"] == 0:
            return "[ExpansionLog] no decisions recorded"

        parts = [f"[ExpansionLog] task={task_id} layer={layer_id}"]
        parts.append(f"total={s['total_decisions']}")
        for k, v in s["decision_type_distribution"].items():
            parts.append(f"{k}={v}")
        parts.append(f"avg_geo={s['avg_geo_score']:.3f}")
        parts.append(f"avg_res={s['avg_residual_score']:.4f}")
        parts.append(f"cov_delta={s['avg_coverage_delta']:+.3f}")
        return " ".join(parts)

    def to_records(self, task_id=None, layer_id=None):
        """Export all decisions as list of dicts."""
        return [d.to_dict() for d in self.query(task_id=task_id, layer_id=layer_id)]

    def clear(self, task_id=None):
        """Clear all or filtered decisions."""
        if task_id is None:
            self._decisions.clear()
        else:
            self._decisions = [d for d in self._decisions if d.task_id != task_id]

    @property
    def num_decisions(self):
        return len(self._decisions)

