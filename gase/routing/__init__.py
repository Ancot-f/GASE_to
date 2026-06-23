from .chart_router import ProbabilisticChartRouter
from .pair_router import ChartSlotPairRouter
from .prototype_router import ChartSlotPrototypeRouter, PrototypeNLLSlotRouter
from .uncertainty import (
    compute_entropy,
    compute_top_margin,
    is_uncertain,
    compute_chart_uncertainty,
    compute_slot_uncertainty,
)
from .soft_mixture import (
    mix_chart_outputs,
    mix_slot_outputs,
    normalize_topk_probs,
    apply_topk_mask,
)
from .fallback import IdentityFallback, FreeAdapterFallback
