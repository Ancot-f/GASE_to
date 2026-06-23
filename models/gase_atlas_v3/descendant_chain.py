"""DescendantChain -tracks L9->L10->L11 chart inheritance relations.

Records which chart in layer L induces which chart in layer L+1,
based on the feature flow after adapter application.

Corresponds to design doc Section 9.
"""

import math
from collections import defaultdict

import torch


class DescendantChain:
    """Records and queries layer-to-layer chart transitions.

    Usage:
        chain = DescendantChain()
        chain.record_transition(9, chart9_id, 10, chart10_id)
        chain.record_transition(10, chart10_id, 11, chart11_id)

        # Get transition matrix from L9 to L10
        matrix = chain.get_transition_matrix(9, 10)  # [num_c9, num_c10]

        metrics = chain.compute_metrics()
    """

    def __init__(self):
        # {(parent_layer, child_layer): {(parent_cid, child_cid): count}}
        self._transitions = defaultdict(lambda: defaultdict(int))
        # {(parent_layer, child_layer): set of parent_cid}
        self._parent_ids = defaultdict(set)
        # {(parent_layer, child_layer): set of child_cid}
        self._child_ids = defaultdict(set)

    def record_transition(self, parent_layer, parent_chart_id,
                          child_layer, child_chart_id, count=1):
        """Record that parent_chart_id in parent_layer led to child_chart_id in child_layer."""
        key = (parent_layer, child_layer)
        pair = (parent_chart_id, child_chart_id)
        self._transitions[key][pair] += count
        self._parent_ids[key].add(parent_chart_id)
        self._child_ids[key].add(child_chart_id)

    def record_batch(self, parent_layer, parent_assignments, child_layer, child_assignments):
        """Record transitions from a batch of per-sample chart assignments.

        Args:
            parent_layer: int, source layer
            parent_assignments: [N] chart IDs at parent layer
            child_layer: int, target layer
            child_assignments: [N] chart IDs at child layer
        """
        for pc, cc in zip(parent_assignments, child_assignments):
            if pc >= 0 and cc >= 0:
                self.record_transition(parent_layer, int(pc), child_layer, int(cc))

    def get_transition_matrix(self, parent_layer, child_layer):
        """Return normalized transition matrix: P(child | parent).

        Returns:
            Tensor [num_parents, num_children] of probabilities.
        """
        key = (parent_layer, child_layer)
        if key not in self._transitions:
            return torch.zeros(0, 0)

        parents = sorted(self._parent_ids[key])
        children = sorted(self._child_ids[key])
        pid_to_idx = {p: i for i, p in enumerate(parents)}
        cid_to_idx = {c: i for i, c in enumerate(children)}

        matrix = torch.zeros(len(parents), len(children))
        for (pc, cc), count in self._transitions[key].items():
            matrix[pid_to_idx[pc], cid_to_idx[cc]] = count

        # Normalize rows
        row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        matrix = matrix / row_sums
        return matrix

    def compute_metrics(self):
        """Compute all descendant chain metrics.

        Returns dict with:
            transition_entropy: avg entropy of P(child|parent) distribution
            dominant_child_ratio: fraction of parent->child pairs where one child dominates (>50%)
            chain_purity: how often parent maps to single dominant child
            chain_switch_rate: fraction of samples that switch dominant chain
            per_layer_transition_matrix: dict of (parent,child) ->matrix
        """
        result = {
            "L9_to_L10_transition_entropy": 0.0,
            "L10_to_L11_transition_entropy": 0.0,
            "L9_to_L10_dominant_child_ratio": 0.0,
            "L10_to_L11_dominant_child_ratio": 0.0,
            "L9_to_L10_chain_purity": 0.0,
            "L10_to_L11_chain_purity": 0.0,
            "chain_switch_rate": 0.0,
            "per_layer_transition_matrix": {},
        }

        for (pl, cl) in [(9, 10), (10, 11)]:
            matrix = self.get_transition_matrix(pl, cl)
            if matrix.numel() == 0:
                continue
            result[f"L{pl}_to_L{cl}_transition_matrix"] = matrix

            # Transition entropy: mean entropy of each parent's child distribution
            row_entropies = []
            dominant_count = 0
            for row in matrix:
                row = row[row > 1e-8]
                if row.numel() > 0:
                    ent = -(row * row.log()).sum().item()
                    row_entropies.append(ent)
                if row.numel() > 0 and row.max().item() > 0.5:
                    dominant_count += 1

            if row_entropies:
                result[f"L{pl}_to_L{cl}_transition_entropy"] = sum(row_entropies) / len(row_entropies)
            if matrix.shape[0] > 0:
                result[f"L{pl}_to_L{cl}_dominant_child_ratio"] = dominant_count / matrix.shape[0]

            # Chain purity: max over children / total for each parent
            row_max = matrix.max(dim=1).values.sum().item()
            result[f"L{pl}_to_L{cl}_chain_purity"] = row_max / max(matrix.shape[0], 1)

        # Chain switch rate: fraction of pairs where parent->child isn't the dominant path
        if (9, 10) in self._transitions and (10, 11) in self._transitions:
            mat_9_10 = self.get_transition_matrix(9, 10)
            mat_10_11 = self.get_transition_matrix(10, 11)
            # Matrices may have different L10 chart sets; align by common child IDs
            if mat_9_10.numel() > 0 and mat_10_11.numel() > 0:
                # Get L10 chart IDs from each transition set
                l10_from_9 = sorted(self._child_ids[(9, 10)])
                l10_from_11 = sorted(self._parent_ids[(10, 11)])
                common_l10 = set(l10_from_9) & set(l10_from_11)
                if common_l10 and mat_9_10.shape[1] == mat_10_11.shape[0]:
                    two_step = torch.matmul(mat_9_10, mat_10_11)
                    if two_step.numel() > 0:
                        result["chain_switch_rate"] = 1.0 - two_step.max(dim=1).values.mean().item()

        result["per_layer_transition_matrix"] = {
            f"L{pl}_to_L{cl}": self.get_transition_matrix(pl, cl).tolist()
            for (pl, cl) in [(9, 10), (10, 11)]
        }
        return result

    def clear(self):
        self._transitions.clear()
        self._parent_ids.clear()
        self._child_ids.clear()

    @property
    def num_transitions(self):
        return sum(len(pairs) for pairs in self._transitions.values())

