"""Standalone AtlasLoggerV3 -structured logging for all v3 diagnostics."""

import logging


class AtlasLoggerV3:
    """Centralized logger for GASE-Atlas v3 diagnostics."""

    def __init__(self, task_id=0):
        self.task_id = task_id

    def set_task(self, task_id):
        self.task_id = task_id

    # ---- Teacher flow ----

    def log_teacher_flow(self, layer_id, stats):
        logging.info(
            "[TeacherFlow:T%d:L%d] delta_norm=%.3f shift_norm=%.3f "
            "h_pre_norm=%.3f h_post_norm=%.3f samples=%d",
            self.task_id, layer_id,
            stats.get("delta_task_norm", 0), stats.get("feature_shift_norm", 0),
            stats.get("h_pre_norm", 0), stats.get("h_post_norm", 0),
            stats.get("num_samples", 0),
        )

    def log_teacher_spectral(self, layer_id, spectral):
        top5 = spectral.get("spectral_top5", [])
        spec_str = " spect=[%s]" % ",".join("%.3f" % v for v in top5) if top5 else ""
        logging.info(
            "[TeacherSpectral:T%d:L%d] e1=%.3f e4=%.3f e8=%.3f%s",
            self.task_id, layer_id,
            spectral.get("energy_at_1", 0), spectral.get("energy_at_4", 0),
            spectral.get("energy_at_8", 0), spec_str,
        )

    # ---- Charts ----

    def log_chart(self, layer_id, chart_metrics):
        birth = chart_metrics.get("birth_task", -1)
        logging.info(
            "[Chart:T%d:L%d#%d birth=T%d] support=%d dim=%d rec_error=%.4f "
            "radius=%.1f coverage=%.3f quality=%.2f stage=%s",
            self.task_id, layer_id,
            chart_metrics.get("chart_id", -1), birth,
            chart_metrics.get("support", 0), chart_metrics.get("tangent_dim", 0),
            chart_metrics.get("rec_error", 0), chart_metrics.get("radius_d2", 0),
            chart_metrics.get("coverage", 0), chart_metrics.get("quality", 0),
            chart_metrics.get("lifecycle_stage", "?"),
        )

    def log_chart_geometry(self, layer_id, metrics):
        logging.info(
            "[ChartGeom:T%d:L%d] K=%d coverage=%.3f mean_maha=%.1f margin=%.1f frag=%.1f",
            self.task_id, layer_id,
            metrics.get("K", 0), metrics.get("coverage", 0),
            metrics.get("mean_mahalanobis", 0), metrics.get("distance_margin", 0),
            metrics.get("fragmentation", 0),
        )

    # ---- Adapter ----

    def log_adapter(self, layer_id, chart_id, metrics):
        logging.info(
            "[ChartAdapter:T%d:L%d#%d] r_p=%d s=%d params=%d "
            "fR2=%.3f subR2=%.3f cos=%.2f",
            self.task_id, layer_id, chart_id,
            metrics.get("input_dim", 0), metrics.get("residual_dim", 0),
            metrics.get("trainable_params", 0),
            metrics.get("full_r2", 0), metrics.get("subspace_r2", 0),
            metrics.get("distill_cos", 0),
        )

    # ---- Free adapter ----

    def log_free(self, layer_id, metrics):
        usage = metrics.get("usage", 0)
        alarm = "OVERUSE" if usage > 0.50 else ("UNDERUSE" if usage < 0.02 else "OK")
        logging.info(
            "[FreeAdapter:T%d:L%d] usage=%.3f samples=%d mse=%.4f alarm=%s",
            self.task_id, layer_id, usage,
            metrics.get("num_samples", 0), metrics.get("distill_mse", 0), alarm,
        )

    # ---- Router ----

    def log_router(self, layer_id, metrics):
        hist = metrics.get("top1_chart_histogram", {})
        hist_str = " charts=%s" % str({k: v for k, v in sorted(hist.items())}) if hist else ""
        logging.info(
            "[Router:T%d:L%d] entropy=%.3f margin=%.1f top1_dist=%.1f "
            "uncertain=%.3f free_fb=%.3f routed=%d free=%d%s",
            self.task_id, layer_id,
            metrics.get("route_entropy", 0), metrics.get("route_margin", 0),
            metrics.get("top1_distance_mean", 0),
            metrics.get("uncertain_ratio", 0), metrics.get("free_fallback_ratio", 0),
            metrics.get("num_routed", 0), metrics.get("num_free_fallback", 0),
            hist_str,
        )

    # ---- Residual decomposition ----

    def log_residual(self, layer_id, metrics):
        logging.info(
            "[Residual:T%d:L%d] task_norm=%.3f chart_norm=%.3f free_norm=%.3f "
            "mse=%.4f rel=%.3f cos=%.3f subR2=%.3f free_ratio=%.3f",
            self.task_id, layer_id,
            metrics.get("task_residual_norm", 0), metrics.get("chart_residual_norm", 0),
            metrics.get("free_residual_norm", 0), metrics.get("distill_mse", 0),
            metrics.get("relative_error", 0), metrics.get("residual_cosine", 0),
            metrics.get("subR2", 0), metrics.get("free_ratio", 0),
        )

    # ---- Descendant chain ----

    def log_descendant(self, metrics):
        logging.info(
            "[Descendant:T%d] L9->L10: ent=%.3f purity=%.2f | "
            "L10->L11: ent=%.3f purity=%.2f | switch=%.3f",
            self.task_id,
            metrics.get("L9_to_L10_transition_entropy", 0),
            metrics.get("L9_to_L10_chain_purity", 0),
            metrics.get("L10_to_L11_transition_entropy", 0),
            metrics.get("L10_to_L11_chain_purity", 0),
            metrics.get("chain_switch_rate", 0),
        )

    # ---- Expansion decisions ----

    def log_expansion(self, layer_id, summary):
        dist = summary.get("decision_type_distribution", {})
        logging.info(
            "[Expansion:T%d:L%d] total=%d add=%d update=%d reuse=%d free=%d "
            "avg_geo=%.3f avg_res=%.4f cov_delta=%+.3f",
            self.task_id, layer_id,
            summary.get("total_decisions", 0), summary.get("num_add_chart", 0),
            summary.get("num_update_adapter", 0), summary.get("num_reuse_chart", 0),
            summary.get("num_free_fallback", 0),
            summary.get("avg_geo_score", 0), summary.get("avg_residual_score", 0),
            summary.get("avg_coverage_delta", 0),
        )

    # ---- CIL ----

    def log_cil(self, metrics):
        logging.info(
            "[CIL:T%d] avg_acc=%.2f forgetting=%.2f last_acc=%.2f",
            self.task_id,
            metrics.get("average_accuracy", 0) * 100,
            metrics.get("forgetting", 0) * 100,
            metrics.get("last_accuracy", 0) * 100,
        )

    # ---- Param growth ----

    def log_params(self, metrics):
        charts = metrics.get("num_charts_per_layer", {})
        chart_str = " ".join(f"{k}={v}" for k, v in sorted(charts.items()))
        logging.info(
            "[Params:T%d] total=%d trainable=%d adapter=%d growth=%.1f charts=[%s]",
            self.task_id,
            metrics.get("total_params", 0), metrics.get("trainable_params", 0),
            metrics.get("adapter_params", 0), metrics.get("growth_rate", 0), chart_str,
        )


