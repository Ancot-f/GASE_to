"""GASE-Atlas v3 -Layer-wise Geometric Memory System for Continual Learning.

All components are standalone -zero dependency on v1.
"""

from models.gase_atlas_v3.teacher_flow import TeacherFlowCache, LayerFlow
from models.gase_atlas_v3.chart_state import ChartStateV3, LayerAtlasState, ChartCreationDecision
from models.gase_atlas_v3.adapters import TaskAdapter, FreeAdapter, ChartAdapter, ChartMLPAdapter, ChartRouter
from models.gase_atlas_v3.chart_builder import PPCAChartBuilder, ChartQualityEvaluator
from models.gase_atlas_v3.chart_adapter_builder import RidgeChartAdapterBuilder
from models.gase_atlas_v3.descendant_chain import DescendantChain
from models.gase_atlas_v3.decision_log import (
    ExpansionDecision, ExpansionDecisionLog,
    DECISION_REUSE_CHART, DECISION_UPDATE_CHART_ADAPTER, DECISION_ADD_CHART,
    DECISION_UPDATE_FREE_ADAPTER, DECISION_FALLBACK_IDENTITY,
    REASON_GEO_COVERED_RESIDUAL_GOOD, REASON_GEO_COVERED_RESIDUAL_BAD,
    REASON_GEO_OUTLIER, REASON_HIGH_UNCERTAINTY, REASON_HIGH_FREE_RATIO,
    REASON_LOW_SUBR2, REASON_NO_EXISTING_CHARTS,
)
from models.gase_atlas_v3.router import ChartRouterV3, RouteResult
from models.gase_atlas_v3.distiller import LayerDistiller
from models.gase_atlas_v3.atlas_layer import GASEAtlasLayerV3
from models.gase_atlas_v3.classifier import AtlasClassifier
from models.gase_atlas_v3.metrics import (
    compute_cil_metrics, compute_param_growth_metrics,
    compute_chart_geometry_metrics, compute_router_metrics,
    compute_residual_metrics, compute_descendant_metrics,
    compute_expansion_summary,
)
from models.gase_atlas_v3.logger import AtlasLoggerV3


