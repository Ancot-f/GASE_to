"""Classifier wrapper for GASE-Atlas v3.

Supports cosine, prototype, and linear classifiers with prototype update
for continual learning. Handles class expansion across tasks.

Corresponds to design doc Section 11.
"""

import math
import torch
import torch.nn as nn

from backbone.linears import CosineLinear, ProtoCalibratedCosine


class AtlasClassifier(nn.Module):
    """Wraps classifier head with prototype management for CIL.

    Modes:
      - "cosine": CosineLinear with learnable scale
      - "cosine_imprint": CosineLinear plus end-of-task weight imprint
      - "prototype": ProtoCalibratedCosine with running prototypes
      - "linear": Standard linear classifier
    """

    def __init__(self, in_dim, out_dim, classifier_type="prototype",
                 cosine_scale=24.0, prototype_alpha=0.8, prototype_mode="add"):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.classifier_type = classifier_type
        self.prototype_alpha = prototype_alpha
        self.prototype_mode = prototype_mode

        self.fc = self._build(in_dim, out_dim, classifier_type, cosine_scale,
                              prototype_alpha, prototype_mode)

    def _build(self, in_dim, out_dim, ctype, cos_scale, proto_alpha, proto_mode):
        if ctype == "prototype":
            fc = ProtoCalibratedCosine(in_dim, out_dim, alpha=proto_alpha, mode=proto_mode)
            if fc.sigma is not None:
                fc.sigma.data.fill_(cos_scale)
        elif ctype in ("cosine", "cosine_imprint"):
            fc = CosineLinear(in_dim, out_dim)
            if fc.sigma is not None:
                fc.sigma.data.fill_(cos_scale)
        else:
            fc = nn.Linear(in_dim, out_dim)
            nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
            nn.init.zeros_(fc.bias)
        return fc

    def forward(self, features):
        """Forward pass: features ->logits tensor."""
        out = self.fc(features)
        if isinstance(out, dict):
            return out["logits"]
        return out

    def expand(self, new_out_dim):
        """Expand classifier to accommodate new classes.

        Preserves old class weights and prototypes.
        """
        old_out = self.out_dim
        if new_out_dim <= old_out:
            return

        old_weight = self.fc.weight.data.clone()
        old_bias = self.fc.bias.data.clone() if self.fc.bias is not None else None

        if self.classifier_type == "prototype":
            new_fc = ProtoCalibratedCosine(self.in_dim, new_out_dim,
                                           alpha=self.fc.alpha, mode=self.fc.mode)
            if new_fc.sigma is not None and self.fc.sigma is not None:
                new_fc.sigma.data.copy_(self.fc.sigma.data)
            # Copy old prototypes
            new_fc.proto_features.data[:old_out] = self.fc.proto_features.data[:old_out]
            new_fc.proto_counts.data[:old_out] = self.fc.proto_counts.data[:old_out]
        elif self.classifier_type in ("cosine", "cosine_imprint"):
            new_fc = CosineLinear(self.in_dim, new_out_dim)
            if new_fc.sigma is not None and self.fc.sigma is not None:
                new_fc.sigma.data.copy_(self.fc.sigma.data)
        else:
            new_fc = nn.Linear(self.in_dim, new_out_dim)
            nn.init.kaiming_uniform_(new_fc.weight, a=math.sqrt(5))
            nn.init.zeros_(new_fc.bias)

        new_fc.weight.data[:old_out] = old_weight
        if old_bias is not None and new_fc.bias is not None:
            new_fc.bias.data[:old_out] = old_bias

        self.fc = new_fc
        self.out_dim = new_out_dim

    def update_prototypes(self, features, targets):
        """Update running prototypes for seen classes.

        Only valid for prototype classifier.
        """
        if self.classifier_type == "prototype":
            self.fc.update_prototypes(features, targets)

    def get_classifier_metrics(self, features, targets, old_class_count=0):
        """Compute classifier diagnostics.

        Returns dict with:
            classifier_weight_norm, prototype_norm, class_margin,
            old_new_logit_bias, old_class_acc, new_class_acc
        """
        with torch.no_grad():
            raw_out = self.fc(features)
            logits = raw_out["logits"] if isinstance(raw_out, dict) else raw_out
            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean().item()

            weight_norm = self.fc.weight.norm(dim=1).mean().item()

            # Old/new logit bias
            old_new_bias = 0.0
            if old_class_count > 0 and old_class_count < self.out_dim:
                old_logits = logits[:, :old_class_count].max(dim=1).values.mean().item()
                new_logits = logits[:, old_class_count:].max(dim=1).values.mean().item()
                old_new_bias = new_logits - old_logits

            # Per-group accuracy
            old_mask = targets < old_class_count
            new_mask = targets >= old_class_count
            old_acc = (preds[old_mask] == targets[old_mask]).float().mean().item() if old_mask.any() else 0.0
            new_acc = (preds[new_mask] == targets[new_mask]).float().mean().item() if new_mask.any() else 0.0

        return {
            "classifier_weight_norm": weight_norm,
            "classifier_accuracy": acc,
            "old_new_logit_bias": old_new_bias,
            "old_class_acc": old_acc,
            "new_class_acc": new_acc,
        }

