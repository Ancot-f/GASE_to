"""V2 chart builder transplanted into V3 for debugging."""
import logging
import torch


class PPCAChartBuilderV2:
    def __init__(
        self, max_charts=8, min_samples=64, seed_sample_size=2048,
        fit_sample_size=4096, knn_size=256, pca_energy=0.90,
        dim_min=2, dim_max=16, radius_quantile=0.95, radius_scale=1.0,
        max_support_ratio=0.50, quality_active=0.60, quality_candidate=0.30,
        quality_mode="weighted_sum", overlap_max=0.30, rec_error_scale=1.0,
        grassmann_tau=0.5, force_one_debug=False, standardize_features=False, eps=1e-6,
    ):
        self.max_charts = max_charts
        self.min_samples = min_samples
        self.seed_sample_size = seed_sample_size
        self.fit_sample_size = fit_sample_size
        self.knn_size = knn_size
        self.pca_energy = pca_energy
        self.dim_min = dim_min
        self.dim_max = dim_max
        self.radius_quantile = radius_quantile
        self.radius_scale = radius_scale
        self.max_support_ratio = max_support_ratio
        self.quality_active = quality_active
        self.quality_candidate = quality_candidate
        self.quality_mode = quality_mode
        self.overlap_max = overlap_max
        self.rec_error_scale = rec_error_scale
        self.grassmann_tau = grassmann_tau
        self.force_one_debug = force_one_debug
        self.overlap_reject_new = overlap_max
        self.standardize_features = bool(standardize_features)
        self.eps = eps

    def build_layer_charts(self, features, residuals, labels=None,
                           existing_charts=None, next_chart_id=0, birth_task=0):
        from models.gase_atlas_v3.chart_state import ChartStateV3

        N, D = features.shape
        device = features.device

        feat_mean = features.mean(dim=0)
        feat_std = features.std(dim=0).clamp_min(self.eps)
        features_norm = (features - feat_mean) / feat_std

        charts = []
        reuse_pairs = []
        update_pairs = []  # [(chart, mask)] -- charts whose adapter needs update
        covered = torch.zeros(N, dtype=torch.bool, device=device)

        # ---- A/B/C classification: reuse existing charts by residual fit ----
        if existing_charts:
            class_result = self._classify_samples(
                features_norm, residuals, existing_charts, device)
            # Case A: geo hit + residual good -> reuse, mark covered
            for chart, mask in class_result.get("reuse", []):
                reuse_pairs.append((chart, mask.clone()))
                covered = covered | mask
            # Case B: geo hit + residual bad -> update adapter, mark covered
            for chart, mask in class_result.get("update", []):
                update_pairs.append((chart, mask.clone()))
                covered = covered | mask
            # Case D: geo miss + residual good -> reuse by residual
            for chart, mask in class_result.get("reuse_by_residual", []):
                reuse_pairs.append((chart, mask.clone()))
                covered = covered | mask

            n_classified = covered.sum().item()
            if n_classified > 0:
                logging.info("[ABC:L] classified %d/%d samples: reuse=%d update=%d residual_reuse=%d",
                             n_classified, N,
                             sum(m.sum().item() for _, m in class_result.get("reuse", [])),
                             sum(m.sum().item() for _, m in class_result.get("update", [])),
                             sum(m.sum().item() for _, m in class_result.get("reuse_by_residual", [])))

        if self.force_one_debug and N >= self.min_samples:
            chart = self._build_debug_chart(features_norm, residuals, device)
            if chart is not None:
                chart.chart_id = next_chart_id
                chart.birth_task = birth_task
                chart.promote_to_active()
                chart.is_debug = True
                chart.quality = 1.0
                chart.coverage = 1.0
                chart.support = N
                charts.append(chart)
                covered[:] = True
            return charts, torch.where(~covered)[0], reuse_pairs, update_pairs

        quality_eval = ChartQualityEvaluator(
            min_samples=self.min_samples, rec_error_scale=self.rec_error_scale,
            overlap_max=self.overlap_max, grassmann_tau=self.grassmann_tau,
            quality_mode=self.quality_mode,
        )

        rejected_seeds = set()
        local_fit_size = min(self.knn_size, N)

        for attempt in range(self.max_charts):
            uncovered_mask = ~covered
            n_remaining = uncovered_mask.sum().item()
            if n_remaining < self.min_samples:
                break

            seed_pool_size = min(self.seed_sample_size, n_remaining)
            uncovered_idx = torch.where(uncovered_mask)[0]
            perm = torch.randperm(n_remaining, device=device)[:seed_pool_size]
            seed_pool_idx = uncovered_idx[perm]
            seed_pool = features_norm[seed_pool_idx]

            seed = self._select_seed_avoiding(seed_pool, seed_pool_idx, rejected_seeds)
            if seed is None:
                break
            rejected_seeds.add(seed)

            chart = self._fit_local_chart(seed, features_norm, local_fit_size, device)
            if chart is None:
                continue

            d2_all = chart.mahalanobis_d2(features_norm)
            assigned = (d2_all <= chart.radius_d2 * self.radius_scale) & ~covered
            n_assigned = assigned.sum().item()

            if n_assigned < self.min_samples:
                continue

            # Auto-shrink if too global
            support_ratio = n_assigned / N
            radius = chart.radius_d2.item()
            shrink_count = 0
            while support_ratio > self.max_support_ratio and shrink_count < 10:
                radius *= 0.7
                assigned = (d2_all <= radius * self.radius_scale) & ~covered
                n_assigned = assigned.sum().item()
                support_ratio = n_assigned / N
                shrink_count += 1
            if n_assigned < self.min_samples:
                continue
            if shrink_count > 0:
                chart.register_buffer("radius_d2", torch.tensor(radius, device=device).reshape(1))

            self._refit_chart(chart, features_norm[assigned], residuals[assigned])

            quality_info = quality_eval.evaluate(chart, features_norm, residuals, assigned, charts)
            chart.quality = quality_info["quality"]
            chart.grassmann_stability = quality_info.get("q_stable", 0.0)
            chart.residual_consistency = quality_info.get("q_residual", 0.0)
            chart.rec_error = quality_info.get("rec_error", 0.0)
            chart.overlap_rate = quality_info.get("overlap_rate", 0.0)
            chart.coverage = n_assigned / N
            chart.mean_d2 = d2_all[assigned].mean().item()
            chart.support = n_assigned

            # Cross-task overlap: if new chart overlaps old chart, don't create --
            # add adapter slot to old chart instead (same chart, new residual mode)
            if existing_charts:
                best_old, best_old_ol = None, 0.0
                for old_c in existing_charts:
                    old_within = old_c.within_radius(features_norm)
                    old_ol = (assigned & old_within).float().sum().item() / max(n_assigned, 1)
                    if old_ol > best_old_ol:
                        best_old_ol = old_ol; best_old = old_c
                if best_old is not None and best_old_ol > 0.6:
                    update_pairs.append((best_old, assigned.clone()))
                    covered = covered | assigned
                    logging.info("[ChartCandidate] attempt=%d cross_task_ol=%.3f "
                                 "-> add adapter slot to old chart=%d (n=%d)",
                                 attempt, best_old_ol, best_old.chart_id, n_assigned)
                    continue

            # Peer overlap: merge into same-task peer chart
            if chart.overlap_rate > self.overlap_max and len(charts) > 0:
                # Find best peer chart to absorb these samples
                best_peer, best_peer_ol = None, 0.0
                for peer in charts:
                    peer_within = peer.within_radius(features_norm)
                    peer_ol = (assigned & peer_within).float().sum().item() / max(n_assigned, 1)
                    if peer_ol > best_peer_ol:
                        best_peer_ol = peer_ol; best_peer = peer
                if best_peer is not None:
                    reuse_pairs.append((best_peer, assigned.clone()))
                    covered = covered | assigned
                    logging.info("[ChartCandidate] attempt=%d peer_ol=%.3f > %.2f -> assign to chart=%d (n=%d)",
                                 attempt, chart.overlap_rate, self.overlap_max,
                                 best_peer.chart_id, n_assigned)
                else:
                    covered = covered | assigned  # mark covered to avoid non_chart
                continue

            quality_val = chart.quality
            if quality_val >= self.quality_active:
                chart.status = "active"
                chart.lifecycle_stage = "active"
            elif quality_val >= self.quality_candidate:
                chart.status = "candidate"
            else:
                chart.status = "rejected"

            if chart.status == "rejected":
                continue

            # Cross-task overlap check
            if existing_charts:
                max_ol = 0.0
                best_old = None
                for old_c in existing_charts:
                    try:
                        old_within = old_c.within_radius(features_norm)
                        ol = (assigned & old_within).float().sum().item() / max(n_assigned, 1)
                        if ol > max_ol:
                            max_ol = ol; best_old = old_c
                    except Exception:
                        pass
                if max_ol > self.overlap_reject_new and best_old is not None:
                    # Check residual fit: compatible -> reuse, incompatible -> new adapter slot
                    if best_old.adapter is not None:
                        with torch.no_grad():
                            pred = best_old.adapter(features_norm[assigned], best_old)
                            err = (pred - residuals[assigned]).norm(dim=-1)
                            resid_norm = residuals[assigned].norm(dim=-1).clamp_min(1e-6)
                            rel_err = (err / resid_norm).mean().item()
                            cos_sim = ((pred * residuals[assigned]).sum(dim=-1) /
                                       (pred.norm(dim=-1) * resid_norm).clamp_min(1e-6)).mean().item()
                        if rel_err < 0.5 and cos_sim > 0.7:
                            # Case A: geo + residual compatible -> reuse
                            reuse_pairs.append((best_old, assigned.clone()))
                            logging.info("[OverlapReuse] chart=%d ol=%.2f rel_err=%.3f cos=%.3f -> reuse",
                                         best_old.chart_id, max_ol, rel_err, cos_sim)
                        else:
                            # Case B: geo hit + residual bad -> same chart, new adapter slot
                            update_pairs.append((best_old, assigned.clone()))
                            logging.info("[OverlapNewSlot] chart=%d ol=%.2f rel_err=%.3f cos=%.3f -> new adapter slot",
                                         best_old.chart_id, max_ol, rel_err, cos_sim)
                    else:
                        reuse_pairs.append((best_old, assigned.clone()))
                    covered = covered | assigned
                    continue

            chart.chart_id = next_chart_id + len(charts)
            chart.birth_task = birth_task
            chart.created_task = birth_task
            charts.append(chart)
            covered = covered | assigned

        return charts, torch.where(~covered)[0], reuse_pairs, update_pairs

    def _classify_samples(self, features_norm, residuals, existing_charts, device,
                          geo_scale=1.0, residual_err_threshold=0.5, cos_threshold=0.7):
        """A/B/C/D classification per sample against existing charts.

        Returns dict with keys: 'reuse', 'update', 'reuse_by_residual'
        Each value is list of (chart, mask) tuples.
        """
        N = features_norm.shape[0]
        result = {"reuse": [], "update": [], "reuse_by_residual": []}
        if not existing_charts:
            return result

        handled = torch.zeros(N, dtype=torch.bool, device=device)

        for chart in existing_charts:
            if chart.adapter is None:
                continue
            # Geometric distance
            d2 = chart.mahalanobis_d2(features_norm)
            geo_hit = d2 <= chart.radius_d2 * geo_scale

            # Residual fit
            with torch.no_grad():
                pred = chart.adapter(features_norm, chart)
                err = (pred - residuals).norm(dim=-1)
                resid_norm = residuals.norm(dim=-1).clamp_min(1e-6)
                rel_err = err / resid_norm
                cos_sim = (pred * residuals).sum(dim=-1) / (
                    pred.norm(dim=-1) * resid_norm).clamp_min(1e-6)

            # Case A: geo hit + residual good -> reuse
            reuse_mask = geo_hit & (rel_err < residual_err_threshold) & (
                cos_sim > cos_threshold) & ~handled
            if reuse_mask.sum() > 0:
                result["reuse"].append((chart, reuse_mask))
                handled = handled | reuse_mask

            # Case B: geo hit + residual bad -> update adapter
            update_mask = geo_hit & ~(rel_err < residual_err_threshold) & ~handled
            if update_mask.sum() >= max(2, self.min_samples // 4):
                result["update"].append((chart, update_mask))
                handled = handled | update_mask

            # Case D: geo miss + residual good -> reuse by residual (avoids chart bloat)
            residual_good = (
                (rel_err < residual_err_threshold * 1.5) &
                (cos_sim > cos_threshold * 0.8) &
                ~geo_hit & ~handled
            )
            if residual_good.sum() >= max(2, self.min_samples // 4):
                result["reuse_by_residual"].append((chart, residual_good))
                handled = handled | residual_good

        # Per-decision trace logging
        for decision_key in ["reuse", "update", "reuse_by_residual"]:
            for chart, mask in result[decision_key]:
                n = mask.sum().item()
                if n > 0:
                    avg_d2 = chart.mahalanobis_d2(features_norm[mask]).mean().item()
                    logging.info(
                        "[ChartDecision:T] decision=%s chart=%d samples=%d "
                        "avg_d2=%.2f radius=%.1f",
                        decision_key, chart.chart_id, n, avg_d2,
                        float(chart.radius_d2.item()) if chart.radius_d2 is not None else 0)

        return result

    def _fit_local_chart(self, seed_idx, features_norm, fit_size, device):
        from models.gase_atlas_v3.chart_state import ChartStateV3

        seed_vec = features_norm[seed_idx:seed_idx + 1]
        dists = torch.cdist(seed_vec, features_norm).squeeze(0)
        _, neighbors = torch.topk(dists, k=min(fit_size, features_norm.shape[0]), largest=False)

        X = features_norm[neighbors]
        M, D = X.shape

        if M < max(2, self.dim_min):
            return None

        mu = X.mean(dim=0)
        Xc = X - mu

        try:
            _, S_full, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except RuntimeError:
            return None

        S_sq = (S_full ** 2) / max(M - 1, 1)
        total_energy = S_sq.sum()
        if total_energy < self.eps:
            return None

        cumsum = torch.cumsum(S_sq, dim=0)
        r = torch.searchsorted(cumsum / total_energy, self.pca_energy).item() + 1
        r = max(self.dim_min, min(self.dim_max, r, len(S_sq)))

        chart = ChartStateV3()
        chart.register_buffer("mu", mu.clone())
        chart.register_buffer("U", Vh[:r].T.clone())
        chart.register_buffer("eigvals", S_sq[:r].clone())

        total_var = S_sq.sum()
        remaining_var = S_sq[r:].mean() if len(S_sq) > r else torch.tensor(0.0, device=device)
        floor = max(total_var.item() / max(D, 1) * 0.01, self.eps)
        chart.register_buffer("sigma_perp", remaining_var.clamp_min(floor).clone().reshape(1))

        d2_local = chart.mahalanobis_d2(X)
        chart.register_buffer("radius_d2", d2_local.quantile(self.radius_quantile).clamp_min(1.0).clone().reshape(1))

        return chart

    def _build_debug_chart(self, features_norm, residuals, device):
        from models.gase_atlas_v3.chart_state import ChartStateV3

        N, D = features_norm.shape
        mu = features_norm.mean(dim=0)
        Xc = features_norm - mu
        try:
            _, S_full, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except RuntimeError:
            return None

        S_sq = (S_full ** 2) / max(N - 1, 1)
        total_energy = S_sq.sum()
        if total_energy < self.eps:
            return None

        cumsum = torch.cumsum(S_sq, dim=0)
        r = torch.searchsorted(cumsum / total_energy, self.pca_energy).item() + 1
        r = max(self.dim_min, min(self.dim_max, r, len(S_sq)))

        chart = ChartStateV3()
        chart.register_buffer("mu", mu.clone())
        chart.register_buffer("U", Vh[:r].T.clone())
        chart.register_buffer("eigvals", S_sq[:r].clone())

        total_var = S_sq.sum()
        remaining_var = S_sq[r:].mean() if len(S_sq) > r else torch.tensor(0.0, device=device)
        floor = max(total_var.item() / max(D, 1) * 0.01, self.eps)
        chart.register_buffer("sigma_perp", remaining_var.clamp_min(floor).clone().reshape(1))

        d2 = chart.mahalanobis_d2(features_norm)
        chart.register_buffer("radius_d2", d2.quantile(self.radius_quantile).clamp_min(1.0).clone().reshape(1))
        chart.support = N
        return chart

    def _select_seed_avoiding(self, seed_pool, seed_pool_idx, rejected):
        M = seed_pool.shape[0]
        if M < 2:
            s = seed_pool_idx[0].item()
            return s if s not in rejected else None

        k = min(self.knn_size, M - 1)
        dists = torch.cdist(seed_pool, seed_pool)
        knn_dists, _ = torch.topk(dists, k=k + 1, dim=1, largest=False)
        knn_dists = knn_dists[:, 1:]
        density = -knn_dists.mean(dim=1)
        sorted_idx = density.argsort(descending=True)
        for i in sorted_idx:
            s = seed_pool_idx[i].item()
            if s not in rejected:
                return s
        return None

    def _refit_chart(self, chart, features_norm, residuals):
        N = features_norm.shape[0]
        if N < 2:
            return
        chart.mu = features_norm.mean(dim=0).clone()
        Xc = features_norm - chart.mu
        D = Xc.shape[1]
        try:
            _, S_full, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except RuntimeError:
            return
        S_sq = (S_full ** 2) / max(N - 1, 1)
        total_energy = S_sq.sum()
        if total_energy < self.eps:
            return
        cumsum = torch.cumsum(S_sq, dim=0)
        r = torch.searchsorted(cumsum / total_energy, self.pca_energy).item() + 1
        r = max(self.dim_min, min(self.dim_max, r, len(S_sq)))
        chart.U = Vh[:r].T.clone()
        chart.eigvals = S_sq[:r].clone()
        total_var = S_sq.sum()
        remaining_var = S_sq[r:].mean() if len(S_sq) > r else torch.tensor(0.0, device=chart.mu.device)
        floor = max(total_var.item() / max(D, 1) * 0.01, self.eps)
        chart.register_buffer("sigma_perp", remaining_var.clamp_min(floor).clone().reshape(1))
        d2_local = chart.mahalanobis_d2(features_norm)
        chart.register_buffer("radius_d2", d2_local.quantile(self.radius_quantile).clamp_min(1.0).clone().reshape(1))
        chart.support = N


class ChartQualityEvaluator:
    """Weighted-sum quality evaluator."""

    def __init__(self, min_samples=64, rec_error_scale=1.0, overlap_max=0.30,
                 grassmann_tau=0.5, quality_mode="weighted_sum"):
        self.min_samples = min_samples
        self.rec_error_scale = rec_error_scale
        self.overlap_max = overlap_max
        self.grassmann_tau = grassmann_tau
        self.quality_mode = quality_mode

    def evaluate(self, chart, features_norm, residuals, assigned, existing_charts):
        assigned_idx = torch.where(assigned)[0]
        n = assigned_idx.shape[0]
        q_support = min(1.0, n / self.min_samples)

        X = features_norm[assigned_idx]
        x_mu = X - chart.mu
        z = torch.matmul(x_mu, chart.U)
        recon = torch.matmul(z, chart.U.t())
        rec_err = (x_mu - recon).pow(2).sum(dim=-1).mean().item()
        total_var = x_mu.pow(2).sum(dim=-1).mean().item()
        rec_err_norm = rec_err / max(total_var, 1e-6)
        rec_err_scaled = rec_err_norm / max(self.rec_error_scale, 1e-6)
        q_compact = float(torch.exp(torch.tensor(-rec_err_scaled)).item())

        stab = self._compute_stability(chart, X)
        q_stable = stab["q_stable"]

        q_residual = self._compute_residual_consistency(chart, residuals, assigned_idx)

        q_nonoverlap = 1.0
        for other in existing_charts:
            other_assigned = other.within_radius(features_norm)
            overlap = (assigned & other_assigned).float().mean().item()
            overlap_rate = overlap / max(assigned.float().mean().item(), 1e-6)
            if overlap_rate > self.overlap_max:
                q_nonoverlap = min(q_nonoverlap, 1.0 - overlap_rate)

        if self.quality_mode == "weighted_sum":
            quality = (0.20 * q_support + 0.25 * q_compact + 0.25 * q_stable
                       + 0.10 * q_residual + 0.20 * q_nonoverlap)
        else:
            q_stable_floor = max(q_stable, 0.05)
            quality = q_support * q_compact * q_stable_floor * max(q_residual, 0.1) * q_nonoverlap

        return {
            "quality": quality, "q_support": q_support, "q_compact": q_compact,
            "q_stable": q_stable, "q_residual": q_residual, "q_nonoverlap": q_nonoverlap,
            "grassmann_raw": stab.get("grassmann_raw", 0),
            "grassmann_norm": stab.get("grassmann_norm", 0),
            "rec_error": rec_err_norm, "overlap_rate": 1.0 - q_nonoverlap,
        }

    def _compute_stability(self, chart, X):
        N = X.shape[0]
        r = chart.tangent_dim
        if N < max(4, r * 2):
            return {"q_stable": 0.5, "grassmann_raw": 0.0, "grassmann_norm": 0.0}
        perm = torch.randperm(N, device=X.device)
        half = N // 2
        X1 = X[perm[:half]]; X2 = X[perm[half:]]
        try:
            _, _, Vh1 = torch.linalg.svd(X1 - X1.mean(dim=0), full_matrices=False)
            _, _, Vh2 = torch.linalg.svd(X2 - X2.mean(dim=0), full_matrices=False)
        except RuntimeError:
            return {"q_stable": 0.5, "grassmann_raw": 0.0, "grassmann_norm": 0.0}
        r_actual = min(r, Vh1.shape[0], Vh2.shape[0])
        U1 = Vh1[:r_actual].T; U2 = Vh2[:r_actual].T
        cross = torch.matmul(U1.t(), U2)
        dG_raw = r_actual - cross.pow(2).sum().item()
        dG_norm = dG_raw / max(r_actual, 1)
        q_stable = float(torch.exp(torch.tensor(-dG_norm / self.grassmann_tau)).item())
        q_stable = max(0.0, min(1.0, q_stable))
        return {"q_stable": q_stable, "grassmann_raw": dG_raw, "grassmann_norm": dG_norm}

    def _compute_residual_consistency(self, chart, residuals, assigned_idx):
        if assigned_idx.shape[0] < 2:
            return 0.5
        delta = residuals[assigned_idx]
        delta_mean = delta.mean(dim=0)
        total_var = delta.pow(2).sum(dim=-1).mean().item()
        within_var = (delta - delta_mean).pow(2).sum(dim=-1).mean().item()
        if total_var < 1e-8:
            return 1.0
        return max(0.0, 1.0 - within_var / total_var)
