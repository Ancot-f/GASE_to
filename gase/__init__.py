"""
GASE: Geometry-Aware Slot-Enhanced Atlas for Class-Incremental Learning.

This package provides the full GASE-Atlas framework including:
- atlas: chart construction, lifecycle, and metrics
- adapters: task, chart, and free adapters
- slots: slot state, construction, routing, and lifecycle
- distill: teacher-guided distillation pipeline
- routing: chart routing, slot routing, and fallback mechanisms
- geometry: PCA, Grassmann, Mahalanobis, and MDL utilities
- diagnostics: reporting and statistics tools
"""

# GASE registration constants
ADAPTER_MODE_TASK_TRAIN = "task_train"
ADAPTER_MODE_DISTILL = "distill"
ADAPTER_MODE_INFER = "infer"

CHART_STATE_CANDIDATE = "candidate"
CHART_STATE_PROVISIONAL = "provisional"
CHART_STATE_ACTIVE = "active"
CHART_STATE_MATURE = "mature"
CHART_STATE_SATURATED = "saturated"
CHART_STATE_DORMANT = "dormant"

SLOT_STATE_CANDIDATE = "candidate"
SLOT_STATE_ACTIVE = "active"
SLOT_STATE_MATURE = "mature"
SLOT_STATE_MERGED = "merged"
SLOT_STATE_RETIRED = "retired"
