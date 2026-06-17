"""Forgetting and accuracy metrics for continual learning evaluation."""

from typing import List

import numpy as np


def compute_forgetting(
    accuracy_matrix: np.ndarray,
) -> float:
    """
    Compute average forgetting across tasks.

    Forgetting for task i = max_{j <= i} acc_{j,i} - acc_{T,i}
    where T is the final task.

    Args:
        accuracy_matrix: matrix of shape [num_tasks, num_tasks] where
            accuracy_matrix[i, j] = accuracy on task j after learning task i.

    Returns:
        Average forgetting (lower is better).
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_average_accuracy(
    accuracy_matrix: np.ndarray,
) -> float:
    """
    Compute average accuracy across all tasks after final training.

    Args:
        accuracy_matrix: matrix of shape [num_tasks, num_tasks].

    Returns:
        Average accuracy over all tasks.
    """
    raise NotImplementedError("Phase-0 skeleton only.")


def compute_taskwise_accuracy_table(
    accuracy_matrix: np.ndarray,
) -> List[float]:
    """
    Compute per-task accuracy after final training.

    Args:
        accuracy_matrix: matrix of shape [num_tasks, num_tasks].

    Returns:
        List of per-task accuracies (length num_tasks).
    """
    raise NotImplementedError("Phase-0 skeleton only.")
