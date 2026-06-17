"""AdapterFactory: construct TaskAdapter, ChartAdapter, and FreeAdapter."""

from typing import Union

from .task_adapter import TaskAdapter
from .chart_adapter import LinearChartAdapter, MLPChartAdapter
from .free_adapter import FreeAdapter


def build_task_adapter(config: dict, dim: int) -> TaskAdapter:
    """
    Build a TaskAdapter from config.

    Args:
        config: dict with keys bottleneck_dim, dropout, scale.
        dim: feature dimension D.

    Returns:
        TaskAdapter instance.
    """
    task_cfg = config.get("task_adapter", {})
    return TaskAdapter(
        dim=dim,
        bottleneck_dim=task_cfg.get("bottleneck_dim", 16),
        dropout=task_cfg.get("dropout", 0.0),
        scale=task_cfg.get("scale", 1.0),
    )


def build_chart_adapter(
    config: dict,
    dim: int,
    layer_id: int,
    chart_id: int,
    slot_id: int,
) -> Union[LinearChartAdapter, MLPChartAdapter]:
    """
    Build a ChartAdapter from config.

    Args:
        config: dict with chart_adapter.type and slot config.
        dim: feature dimension D.
        layer_id: ViT block index.
        chart_id: chart id.
        slot_id: slot id.

    Returns:
        LinearChartAdapter or MLPChartAdapter instance.
    """
    adapter_cfg = config.get("chart_adapter", {})
    adapter_type = adapter_cfg.get("type", "linear")
    slot_cfg = config.get("slot", {})
    input_rank = slot_cfg.get("input_rank", 8)
    output_rank = slot_cfg.get("output_rank", 4)

    if adapter_type == "linear":
        return LinearChartAdapter(
            dim=dim,
            input_rank=input_rank,
            output_rank=output_rank,
            chart_id=chart_id,
            slot_id=slot_id,
            layer_id=layer_id,
        )
    elif adapter_type == "mlp":
        hidden_dim = adapter_cfg.get("hidden_dim", 8)
        return MLPChartAdapter(
            dim=dim,
            input_rank=input_rank,
            output_rank=output_rank,
            hidden_dim=hidden_dim,
            chart_id=chart_id,
            slot_id=slot_id,
            layer_id=layer_id,
        )
    else:
        raise ValueError(f"Unknown chart_adapter type: {adapter_type}")


def build_free_adapter(config: dict, dim: int) -> FreeAdapter:
    """
    Build a FreeAdapter from config.

    Args:
        config: dict with free_adapter config.
        dim: feature dimension D.

    Returns:
        FreeAdapter instance.
    """
    free_cfg = config.get("free_adapter", {})
    return FreeAdapter(
        dim=dim,
        bottleneck_dim=free_cfg.get("bottleneck_dim", 16),
        dropout=free_cfg.get("dropout", 0.0),
        scale=free_cfg.get("scale", 1.0),
    )
