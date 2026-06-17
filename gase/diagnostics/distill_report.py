"""DistillReporter: summarizes distillation quality and residual fit."""

from typing import Dict, List

from torch import Tensor


class DistillReporter:
    """
    Generates reports on distillation quality.

    Tracks residual MSE, feature consistency, logit KL,
    and adapter-specific fit metrics.
    """

    def __init__(self, writer=None):
        """
        Args:
            writer: optional TensorBoard/MLflow writer.
        """
        self.writer = writer

    def summarize_teacher_fit(
        self,
        delta_student: Tensor,
        delta_teacher: Tensor,
    ) -> Dict:
        """
        Summarize how well the student matches the teacher residual.

        Args:
            delta_student: student residuals [N, D].
            delta_teacher: teacher residuals [N, D].

        Returns:
            Dict with keys: mse, cosine_mean, r2, norm_ratio.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def summarize_adapter_fit(
        self,
        layer_id: int,
        delta_chart: Tensor,
        delta_teacher: Tensor,
        chart_ids: List[int],
    ) -> Dict:
        """
        Summarize chart-adapter fit quality per chart.

        Args:
            layer_id: ViT block index.
            delta_chart: chart-adapter residuals [N, D].
            delta_teacher: teacher residuals [N, D].
            chart_ids: chart assignments [N].

        Returns:
            Dict with per-chart fit metrics.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def summarize_free_adapter_fit(
        self,
        layer_id: int,
        delta_free: Tensor,
        delta_free_target: Tensor,
    ) -> Dict:
        """
        Summarize free-adapter fit quality.

        Args:
            layer_id: ViT block index.
            delta_free: free-adapter residuals [N, D].
            delta_free_target: target residuals [N, D].

        Returns:
            Dict with free-adapter fit metrics.
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def log_residual_statistics(
        self,
        delta_teacher: Tensor,
        delta_chart: Tensor,
        delta_free: Tensor,
    ) -> None:
        """
        Log residual norm and direction statistics.

        Args:
            delta_teacher: teacher residuals [N, D].
            delta_chart: chart-adapter residuals [N, D].
            delta_free: free-adapter residuals [N, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")
