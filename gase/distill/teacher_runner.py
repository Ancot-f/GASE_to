"""TeacherRunner: manages teacher mode for computing teacher residuals."""

from torch import Tensor


class TeacherRunner:
    """
    Manages the teacher (task-adapter) mode of the model.

    The teacher runner:
    1. Enables task-adapters on the model.
    2. Computes teacher residuals (delta_teacher) at each atlas layer.
    3. Computes teacher logits for logit-level distillation.
    4. Disables task-adapters after collection.

    Teacher residuals are computed as:
        delta_teacher = h_with_adapter - h_without_adapter
    where h is the post-block feature at a given layer.
    """

    def __init__(self, model, atlas_layers: list):
        """
        Args:
            model: ViTGASE model.
            atlas_layers: list of layer indices with task-adapters.
        """
        self.model = model
        self.atlas_layers = atlas_layers

    def enable_teacher_mode(self) -> None:
        """Enable task-adapters on all atlas layers."""
        raise NotImplementedError("Phase-0 skeleton only.")

    def disable_teacher_mode(self) -> None:
        """Disable task-adapters on all atlas layers."""
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_teacher_residual(
        self,
        h_chart: Tensor,
        layer_id: int,
    ) -> Tensor:
        """
        Compute teacher residual at a specific layer.

        delta_teacher = task_adapter(h_chart)

        Args:
            h_chart: pre-adapter features [B, D] from permanent path.
            layer_id: ViT block index.

        Returns:
            delta_teacher of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_teacher_logits(self, images: Tensor) -> Tensor:
        """
        Compute teacher logits from full forward pass with task-adapters.

        Args:
            images: batch of images [B, C, H, W].

        Returns:
            Teacher logits of shape [B, num_classes].
        """
        raise NotImplementedError("Phase-0 skeleton only.")

    def compute_teacher_features(
        self,
        images: Tensor,
        layer_id: int,
    ) -> Tensor:
        """
        Compute teacher features at a specific layer.

        Args:
            images: batch of images [B, C, H, W].
            layer_id: ViT block index.

        Returns:
            Teacher features of shape [B, D].
        """
        raise NotImplementedError("Phase-0 skeleton only.")
