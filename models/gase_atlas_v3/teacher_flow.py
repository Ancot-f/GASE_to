"""TeacherFlowCache -records layer-wise dependent teacher flow.

Replaces v1 TaskFeatureBuffer. Key difference: records full flow
(h_pre, delta_task, h_post) with layer-wise dependency:
  h10_pre = L9(h9_pre + delta9_task) after L9 adapter
  h11_pre = L10(h10_pre + delta10_task) after L10 adapter

Corresponds to design doc Sections 2 (TaskAdapter lifecycle) and 3 (TeacherFlowCache).
"""

from dataclasses import dataclass

import torch


@dataclass
class LayerFlow:
    """Recorded flow for one layer."""
    h_pre: torch.Tensor       # [N, D] input to task adapter
    delta_task: torch.Tensor  # [N, D] task adapter residual
    h_post: torch.Tensor      # [N, D] h_pre + delta_task
    labels: torch.Tensor      # [N] class labels

    @property
    def N(self):
        return self.h_pre.shape[0]

    @property
    def D(self):
        return self.h_pre.shape[1]


class TeacherFlowCache:
    """Records L9->L10->L11 teacher flow with full layer-wise dependency.

    Usage:
        cache = TeacherFlowCache(layers=[9, 10, 11])

        # During teacher forward pass:
        cache.record(layer_id=9, h_pre=h9, delta_task=d9, h_post=h9+d9, labels=labels)
        cache.record(layer_id=10, h_pre=h10, delta_task=d10, h_post=h10+d10, labels=labels)
        cache.record(layer_id=11, h_pre=h11, delta_task=d11, h_post=h11+d11, labels=labels)

        # Retrieve per-layer flow:
        flow9 = cache.stack(9)   # LayerFlow
        flow10 = cache.stack(10)
        flow11 = cache.stack(11)
    """

    def __init__(self, layers=(9, 10, 11)):
        self.layers = list(layers)
        self._h_pre = {l: [] for l in self.layers}
        self._delta_task = {l: [] for l in self.layers}
        self._h_post = {l: [] for l in self.layers}
        self._labels = []

    def record(self, layer_id, h_pre, delta_task, h_post, labels=None):
        """Record one batch's flow for a given layer.

        All inputs are expected as CPU tensors or will be detached.
        """
        self._h_pre[layer_id].append(_detach_cpu(h_pre))
        self._delta_task[layer_id].append(_detach_cpu(delta_task))
        self._h_post[layer_id].append(_detach_cpu(h_post))
        if labels is not None and layer_id == self.layers[0]:
            self._labels.append(_detach_cpu(labels))

    def record_labels(self, labels):
        self._labels.append(_detach_cpu(labels))

    def has_records(self, layer_id):
        return layer_id in self._h_pre and len(self._h_pre[layer_id]) > 0

    def stack(self, layer_id):
        """Return stacked LayerFlow for a given layer."""
        if not self.has_records(layer_id):
            raise ValueError(f"No teacher flow records for layer {layer_id}")
        h_pre = torch.cat(self._h_pre[layer_id], dim=0)
        delta_task = torch.cat(self._delta_task[layer_id], dim=0)
        h_post = torch.cat(self._h_post[layer_id], dim=0)
        labels = torch.cat(self._labels, dim=0) if self._labels else torch.zeros(h_pre.shape[0], dtype=torch.long)
        return LayerFlow(h_pre=h_pre, delta_task=delta_task, h_post=h_post, labels=labels)

    def stack_pre_and_delta(self, layer_id):
        """Return (h_pre, delta_task) tuple -v1 compatibility."""
        flow = self.stack(layer_id)
        return flow.h_pre, flow.delta_task

    def get_features_and_targets(self, layer_id):
        """Return (features=h_pre, targets=delta_task) for chart building."""
        flow = self.stack(layer_id)
        return flow.h_pre, flow.delta_task

    @property
    def total_samples(self):
        if self._labels:
            return sum(t.shape[0] for t in self._labels)
        l0 = self.layers[0]
        if self._h_pre[l0]:
            return sum(t.shape[0] for t in self._h_pre[l0])
        return 0

    def clear(self):
        for l in self.layers:
            self._h_pre[l].clear()
            self._delta_task[l].clear()
            self._h_post[l].clear()
        self._labels.clear()

    def compute_layer_statistics(self, layer_id):
        """Compute per-layer flow statistics for diagnostics.

        Returns dict with: delta_task_norm, feature_shift_norm, h_pre_norm, h_post_norm.
        """
        flow = self.stack(layer_id)
        with torch.no_grad():
            delta_norm = flow.delta_task.norm(dim=-1).mean().item()
            shift_norm = (flow.h_post - flow.h_pre).norm(dim=-1).mean().item()
            pre_norm = flow.h_pre.norm(dim=-1).mean().item()
            post_norm = flow.h_post.norm(dim=-1).mean().item()
        return {
            "delta_task_norm": delta_norm,
            "feature_shift_norm": shift_norm,
            "h_pre_norm": pre_norm,
            "h_post_norm": post_norm,
            "num_samples": flow.N,
        }

    def compute_spectral(self, layer_id, top_k=5):
        """Compute spectral statistics of delta_task for diagnostics."""
        flow = self.stack(layer_id)
        delta = flow.delta_task.float()
        if delta.shape[0] < 2:
            return {"energy_at_1": 0, "energy_at_4": 0, "energy_at_8": 0, "spectral_top5": []}
        try:
            S = torch.linalg.svdvals(delta)
            s_sq = (S ** 2) / max(delta.shape[0] - 1, 1)
            s_total = s_sq.sum().item()
            if s_total > 1e-8:
                s_norm = s_sq / s_total
                top = s_norm[:min(top_k, len(s_norm))]
                return {
                    "energy_at_1": float(top[0]) if len(top) > 0 else 0,
                    "energy_at_4": float(s_norm[:min(4, len(s_norm))].sum()),
                    "energy_at_8": float(s_norm[:min(8, len(s_norm))].sum()),
                    "spectral_top5": [float(x) for x in top],
                }
        except RuntimeError:
            pass
        return {"energy_at_1": 0, "energy_at_4": 0, "energy_at_8": 0, "spectral_top5": []}


def _detach_cpu(t):
    if isinstance(t, torch.Tensor):
        return t.detach().cpu()
    return torch.tensor(t)

