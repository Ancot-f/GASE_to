"""
GASE Learner: Geometry-Aware Slot-Enhanced Atlas for Class-Incremental Learning.

Phase-2: TaskAdapter-only teacher training loop.
Trains L9-L11 task adapters + classifier on each task.
No chart/slot/distill/PPCA yet.
"""

import logging
import numpy as np
import math
from typing import Any, Dict, List, Optional

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.base import BaseLearner
from utils.inc_net import GASEVitNet
from utils.toolkit import tensor2numpy
from gase.adapters.adapter_factory import build_task_adapter

num_workers = 8


class GASELearner(BaseLearner):
    """
    GASE learner for class-incremental learning.

    Phase-2 implements: task-adapter training only.
    Creates a ViTGASE backbone, trains L9-L11 TaskAdapters + classifier
    on each task, and evaluates using teacher (task_train) mode.
    """

    def __init__(self, args: Dict[str, Any]):
        super().__init__(args)

        self.atlas_layers: List[int] = args.get("atlas_layers", [9, 10, 11])
        self.adapter_dim: int = args.get("adapter_dim", 16)
        self.task_adapters: Dict[int, nn.Module] = {}
        self.chart_atlases: Dict[int, List] = {}
        self.free_adapters: Dict[int, nn.Module] = {}
        self.distill_cache = None
        self.current_task_id: int = -1

        self.use_soft_chart_routing: bool = args.get("routing", {}).get(
            "use_soft_chart_routing", True
        )
        self.use_slot_router: bool = args.get("slot", {}).get(
            "use_teacher_guided_router", False
        )
        self.use_free_adapter: bool = args.get("free_adapter", {}).get("enabled", True)
        self.chart_lifecycle_config: dict = args.get("chart", {}).get("lifecycle", {})
        self.slot_lifecycle_config: dict = {}

        # Build the network
        self._network = GASEVitNet(args, True)

        # Phase flag
        self.phase: str = args.get("phase", "task_adapter_only")
        self.phase4_report: Optional[Dict] = None
        self.phase5_report: Optional[Dict] = None
        self.phase6_report: Optional[Dict] = None
        self.phase6_oracle_eval: Optional[Dict] = None
        self.phase6_key_eval: Optional[Dict] = None
        self.debug_max_tasks: int = args.get("debug_max_tasks", -1)
        self.stop_after_task: int = args.get("stop_after_task", -1)

        # Phase-6.5: bootstrap config
        self.task0_train_all_adapters: bool = args.get("task0_train_all_adapters", True)
        self.freeze_base_after_task0: bool = args.get("freeze_base_after_task0", True)
        self.bootstrap_adapter_layers: List[int] = args.get("bootstrap_adapter_layers", list(range(12)))
        self.should_stop_training: bool = False

        # Phase-8.6: classifier head stabilization
        cls_cfg = args.get("classifier", {})
        self.head_stabilization: bool = cls_cfg.get("head_stabilization", False)
        self.save_head_snapshots: bool = cls_cfg.get("save_head_snapshots", False)
        self.freeze_old_classes: bool = cls_cfg.get("freeze_old_classes", False)
        self.freeze_sigma_after_task0: bool = cls_cfg.get("freeze_sigma_after_task0", False)
        self.weight_norm_align: bool = cls_cfg.get("weight_norm_align", False)
        self.snapshot_eval: bool = cls_cfg.get("snapshot_eval", False)
        self.head_snapshots: Dict[int, Dict] = {}
        self.head_diag_enabled: bool = cls_cfg.get("diagnostics", False)

        # Phase-9: default router
        self.default_router: str = args.get("routing", {}).get("default_router", "shared_q_distance")
        self.phase6_path_nll_eval: Optional[Dict] = None
        self.phase6_cand_path_eval: Optional[Dict] = None
        self.phase6_hybrid_eval: Optional[Dict] = None

        logging.info(
            "GASELearner Phase-2 initialized. atlas_layers=%s, adapter_dim=%d",
            self.atlas_layers, self.adapter_dim,
        )
        routing_cfg = args.get("routing", {})
        actual_router = routing_cfg.get("default_router", routing_cfg.get("slot_router", "shared_q_mahalanobis"))
        logging.info("[RouterConfig] default_router=%s", actual_router)
        logging.info("[RouterConfig] use_shared_router_basis=%s", routing_cfg.get("use_shared_router_basis", True))
        logging.info("[RouterConfig] per_sample=%s", routing_cfg.get("per_sample", True))
        logging.info("[RouterConfig] use_logdet=%s", routing_cfg.get("use_logdet", True))
        logging.info("[RouterConfig] calibrate_nll=%s", routing_cfg.get("calibrate_nll", False))
        logging.info("[RouterConfig] compare_router_variants=%s", routing_cfg.get("compare_router_variants", True))
        logging.info("[RouterConfig] variants=%s", routing_cfg.get("router_variants", ["shared_q_dist", "raw_nll"]))

    # ==================================================================
    #  Incremental training entry point (called by trainer.py)
    # ==================================================================

    def incremental_train(self, data_manager):
        """Called by trainer.py for each task."""
        self._cur_task += 1
        self.current_task_id = self._cur_task

        # Phase-5.5: stop_after_task / debug_max_tasks protection
        if self.stop_after_task >= 0 and self._cur_task > self.stop_after_task:
            logging.info(
                "[GASE] stop_after_task=%d reached at task %d. Skipping.",
                self.stop_after_task, self._cur_task,
            )
            return
        if self.debug_max_tasks > 0 and self._cur_task >= self.debug_max_tasks:
            logging.info(
                "[GASE] debug_max_tasks=%d reached at task %d. Stopping.",
                self.debug_max_tasks, self._cur_task,
            )
            self.should_stop_training = True
            return

        if self._cur_task == 0:
            from backbone.linears import CosineLinear
            self._network.backbone.head = CosineLinear(
                self._network.backbone.embed_dim, data_manager.nb_classes, sigma=True
            )
            self._network.backbone.head.sigma.data.fill_(16.0)  # cosine needs high sigma

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"],
            shuffle=True, num_workers=num_workers,
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test",
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"],
            shuffle=False, num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    # ==================================================================
    #  Training
    # ==================================================================

    def _train(self, train_loader, test_loader):
        """Phase-2 training: freeze backbone, create task adapters, train."""
        self._network.to(self._device)

        # 1. Create and inject TaskAdapters at GASE layers
        self._create_task_adapters()

        # 2. Freeze backbone, unfreeze only task adapters + head
        self._freeze_backbone_except_task_adapters_and_classifier()
        self._log_trainable_parameters()

        # 3. Set model to task_train mode
        self._network.backbone.enable_task_adapters()

        # 4. Optimizer
        trainable = [p for p in self._network.parameters() if p.requires_grad]
        epochs = self.args.get("epochs", self.args.get("func_epoch", 20))
        lr = self.args.get("lr", self.args.get("init_lr", 0.001))
        wd = self.args.get("weight_decay", 0.0001)

        if self.args.get("optimizer", "adam").lower() == "sgd":
            optimizer = optim.SGD(trainable, momentum=0.9, lr=lr, weight_decay=wd)
        else:
            optimizer = optim.AdamW(trainable, lr=lr, weight_decay=wd)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        # 5. Training loop
        prog_bar = tqdm(range(epochs))
        for epoch in range(epochs):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]
                logits = logits[:, :self._total_classes]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()

                # Phase-8.6: freeze old class weights gradient
                if self.freeze_old_classes and self._cur_task > 0:
                    head = self._network.backbone.head
                    if hasattr(head, "weight") and head.weight.grad is not None:
                        head.weight.grad[:self._known_classes] = 0
                    if self.freeze_sigma_after_task0 and hasattr(head, "sigma") and head.sigma.grad is not None:
                        head.sigma.grad = None

                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs,
                    losses / len(train_loader), train_acc, test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, epochs,
                    losses / len(train_loader), train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    # ==================================================================
    #  Task adapter management
    # ==================================================================

    def _create_task_adapters(self) -> None:
        """Create task_adapters: Task0→L0-L11, Task1+→L9-L11."""
        backbone = self._network.backbone
        dim = backbone.embed_dim

        if self._cur_task == 0 and self.task0_train_all_adapters:
            layer_ids = self.bootstrap_adapter_layers
            mode = "task0_bootstrap"
            for lid in layer_ids:
                blk = backbone.get_block(lid)
                blk.task_adapter = build_task_adapter(self.args, dim)
                blk.task_adapter.to(self._device)
        else:
            layer_ids = self.atlas_layers
            mode = "base_plus_task_train"
            for lid in layer_ids:
                blk = backbone.get_block(lid)
                blk.task_adapter = build_task_adapter(self.args, dim)
                blk.task_adapter.to(self._device)

        backbone.set_adapter_mode(mode)
        logging.info("[GASE] Created task_adapters for layers %s, mode=%s", layer_ids, mode)

    # ==================================================================
    #  Parameter freezing
    # ==================================================================

    def _freeze_backbone_except_task_adapters_and_classifier(self) -> None:
        """
        Freeze entire model, then unfreeze trainable modules.

        Strategy:
          1. Freeze all parameters.
          2. Task0: unfreeze L0-L8 SEMA adapters, L9-L11 task_adapters, head.
          3. Task1+: L0-L8 adapters stay frozen (only L9-L11 task_adapters + head).
        """
        backbone = self._network.backbone

        # 1. Freeze all
        for p in backbone.parameters():
            p.requires_grad = False

        # 2. Unfreeze task adapters at L9-L11
        for blk in backbone.get_atlas_blocks():
            if blk.task_adapter is not None:
                for p in blk.task_adapter.parameters():
                    p.requires_grad = True

        # 3. Unfreeze head
        for p in backbone.head.parameters():
            p.requires_grad = True

        # 4. Task0: unfreeze task_adapters on ALL layers (L0-L11)
        if self._cur_task == 0 and self.task0_train_all_adapters:
            for lid in self.bootstrap_adapter_layers:
                blk = backbone.get_block(lid)
                if blk.task_adapter is not None:
                    for p in blk.task_adapter.parameters():
                        p.requires_grad = True

    def _log_trainable_parameters(self) -> None:
        """Log all trainable parameter names and total count."""
        trainable_names = []
        total_trainable = 0
        for name, p in self._network.named_parameters():
            if p.requires_grad:
                trainable_names.append(name)
                total_trainable += p.numel()

        total_all = sum(p.numel() for p in self._network.parameters())
        logging.info(
            "Trainable params: %d / %d (%.2f%%)",
            total_trainable, total_all,
            100.0 * total_trainable / max(total_all, 1),
        )
        logging.info("Trainable parameter names: %s", trainable_names)

    # ==================================================================
    #  Evaluation
    # ==================================================================

    def eval_task(self):
        """Evaluate using teacher (task_train) mode."""
        self._network.eval()
        self._network.backbone.enable_task_adapters()
        return super().eval_task()

    def _eval_cnn(self, loader):
        """Override to handle GASE dict output."""
        self._network.eval()
        self._network.backbone.enable_task_adapters()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)
                logits = outputs["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        """Compute accuracy using GASE dict output."""
        model.eval()
        model.backbone.enable_task_adapters()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outcome = model(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    # ==================================================================
    #  Task lifecycle
    # ==================================================================

    def _check_debug_limits(self) -> bool:
        """Return True if debug limits say we should skip this task."""
        if self.debug_max_tasks > 0 and self._cur_task >= self.debug_max_tasks:
            logging.info("[GASE] skipping task %d (debug_max_tasks=%d)",
                         self._cur_task, self.debug_max_tasks)
            return True
        if self.stop_after_task >= 0 and self._cur_task > self.stop_after_task:
            logging.info("[GASE] skipping task %d (stop_after_task=%d)",
                         self._cur_task, self.stop_after_task)
            return True
        return False

    def before_task(self, task_id: int) -> None:
        """Prepare for a new task (Phase-2: no-op)."""
        pass

    def after_task(self):
        """Post-task cleanup."""
        pre_known = self._known_classes
        self._known_classes = self._total_classes
        logging.info("Task %d completed. known_classes=%d (pre=%d)", self._cur_task, self._known_classes, pre_known)

        logging.info("[HeadRange] task=%d previous_known=%d current_known=%d old_range=0-%d new_range=%d-%d",
                     self._cur_task, pre_known, self._total_classes, pre_known - 1 if pre_known > 0 else -1,
                     pre_known, self._total_classes - 1)

        # Phase-3: run L9 feature collection if configured
        if self.phase == "feature_collect" and self._cur_task == 0:
            logging.info("[GASE] Phase-3: collecting L9 features for task 0...")
            self.collect_chart_features_for_task(
                task_id=self._cur_task, data_loader=self.train_loader
            )

        # Phase-4: collect + distill + eval
        if self.phase == "l9_one_chart_one_slot" and self._cur_task == 0:
            logging.info("[GASE] Phase-4: L9 one-chart one-slot distillation...")
            self.collect_chart_features_for_task(
                task_id=self._cur_task, data_loader=self.train_loader
            )
            self.distill_l9_one_chart_one_slot(task_id=self._cur_task)
            self.evaluate_l9_student_vs_teacher(self.test_loader)

        # Phase-5: sequential L9→L10→L11 one-chart one-slot
        if self.phase == "sequential_one_chart_one_slot":
            if not self._check_debug_limits():
                logging.info("[GASE] Phase-5: sequential L9→L10→L11 distillation...")
                self.distill_sequential_one_chart_one_slot(
                    task_id=self._cur_task, train_loader=self.train_loader
                )
                eval_metrics = self.evaluate_sequential_student_vs_teacher(self.test_loader)
                if self.phase5_report is not None:
                    self.phase5_report["eval"] = eval_metrics

        # Phase-6: multi-slot with oracle + key eval
        if self.phase in ("multi_slot_oracle", "bootstrap_multislot_oracle"):
            if not self._check_debug_limits():
                # Phase-6.5: Task0 → commit base adapters first
                if self._cur_task == 0 and self.freeze_base_after_task0:
                    logging.info("[GASE] Phase-6.5: committing Task0 base adapters...")
                    self._network.backbone.commit_task0_base_adapters()

                slot_id = self._cur_task
                logging.info("[GASE] Phase-6: multi-slot distillation slot=%d...", slot_id)
                self.distill_sequential_multi_slot_for_task(
                    task_id=self._cur_task, train_loader=self.train_loader,
                )
                oracle_result = self.evaluate_oracle_slot_student(self.test_loader)
                key_result = self.evaluate_key_slot_student(self.test_loader)
                path_result = self.evaluate_path_key_slot_student(self.test_loader)
                agg_result = self.evaluate_aggregated_path_key_slot_student(self.test_loader)
                path_nll_result = self.evaluate_path_raw_nll_slot_student(self.test_loader)
                cand_path_result = self.evaluate_candidate_path_raw_nll_slot_student(self.test_loader)
                hybrid_result = self.evaluate_hybrid_path_raw_nll_slot_student(self.test_loader)
                self.phase6_path_nll_eval = path_nll_result
                self.phase6_cand_path_eval = cand_path_result
                self.phase6_hybrid_eval = hybrid_result

                if oracle_result and key_result and path_result:
                    logging.info(
                        "[EvalCompare] Oracle=%.2f PerLayerKey=%.2f "
                        "PathRawNLL=%.2f CandidatePathRawNLL=%.2f HybridBest=%.2f "
                        "Oracle-gap=%.2f DefaultRouter=%s",
                        oracle_result.get("top1", 0), key_result.get("top1", 0),
                        path_nll_result.get("top1", 0) if path_nll_result else 0,
                        cand_path_result.get("top1", 0) if cand_path_result else 0,
                        hybrid_result.get("top1", 0) if hybrid_result else 0,
                        oracle_result.get("top1", 0) - key_result.get("top1", 0),
                        self.default_router,
                    )
                # Phase-8.7: head stabilization with pre_known range
                if self.weight_norm_align and self._cur_task > 0:
                    self._align_new_class_weight_norms(pre_known)
                if self.head_diag_enabled:
                    self._log_head_diagnostics(pre_known)
                if self.save_head_snapshots or self.snapshot_eval:
                    self.save_head_snapshot()
                if self.snapshot_eval and self._cur_task >= 1:
                    for snap_id in range(self._cur_task + 1):
                        self.evaluate_oracle_with_snapshot_head(self.test_loader, snap_id)
                        self.evaluate_hybrid_snapshot_head(self.test_loader, snap_id)

                if self.default_router == "raw_nll":
                    from gase.routing.nll_router import CalibratedNLLSlotRouter
                    r = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
                    self._network.backbone.set_nll_router(r)
                self._eval_router_variants()
                self._save_metrics_json()

    # ==================================================================
    #  Reports (minimal debug versions)
    # ==================================================================

    def report_atlas_state(self) -> Dict:
        return {
            "phase": self.phase,
            "num_layers": len(self.atlas_layers),
            "atlas_layers": self.atlas_layers,
            "charts_per_layer": {
                lid: len(charts) for lid, charts in self.chart_atlases.items()
            },
        }

    def report_routing_state(self) -> Dict:
        return {
            "phase": self.phase,
            "use_soft_chart_routing": self.use_soft_chart_routing,
            "use_slot_router": self.use_slot_router,
            "use_free_adapter": self.use_free_adapter,
        }

    def report_distillation_state(self) -> Dict:
        return {
            "phase": self.phase,
            "distill_cache_samples": (
                self.distill_cache.summary() if self.distill_cache is not None else {}
            ),
        }

    # ==================================================================
    #  Unimplemented (Phase-3+)
    # ==================================================================

    # ==================================================================
    #  Phase-6: Multi-slot distillation per layer
    # ==================================================================

    def distill_one_chart_one_slot_for_layer(
        self, task_id: int, layer_id: int, slot_id: Optional[int] = None,
    ) -> Dict:
        """
        Build/reuse one chart, create one new slot, fit one adapter.

        Phase-6: chart_id=0 is reused across tasks. Each task creates
        a new slot_id (defaults to task_id). Old slots are frozen.

        Args:
            task_id: current task id.
            layer_id: target atlas layer (9, 10, or 11).
            slot_id: slot id (defaults to task_id).

        Returns:
            Dict with chart, slot, distill metrics.
        """
        from gase.atlas.chart_builder import ChartBuilder
        from gase.slots.slot_builder import SlotBuilder
        from gase.distill.chart_distiller import ChartAdapterDistiller

        slot_id = task_id if slot_id is None else slot_id

        batch = self.distill_cache.get_layer_cache(layer_id)
        h_chart = batch.h_chart.to(self._device)
        delta_teacher = batch.delta_teacher.to(self._device)

        # Reuse or build chart
        blk = self._network.backbone.get_block(layer_id)
        if len(blk.chart_states) > 0 and 0 in blk.chart_states:
            chart_state = blk.chart_states[0]
            logging.info(
                "[L%dChart][slot=%d] reusing existing chart, support=%d",
                layer_id, slot_id, chart_state.n_support,
            )
        else:
            chart_builder = ChartBuilder(self.args.get("chart", {}))
            chart_state = chart_builder.build_single_chart_for_layer(
                h_chart, layer_id=layer_id, chart_id=0,
            )
            blk.register_chart(chart_state)
        self.chart_atlases[layer_id] = [chart_state]

        # Build slot
        slot_builder = SlotBuilder(self.args.get("slot", {}))
        slot_state = slot_builder.create_single_slot_from_residuals(
            chart_state, h_chart, delta_teacher, task_id=task_id, slot_id=slot_id,
        )

        # Fit adapter
        distiller = ChartAdapterDistiller(self.args.get("distill", {}))
        adapter, metrics = distiller.fit_linear_chart_adapter(
            chart_state, slot_state, h_chart, delta_teacher,
        )

        # Register adapter + slot (freeze=True keeps old slots frozen)
        blk.register_slot(slot_state)
        blk.register_chart_adapter(chart_id=0, slot_id=slot_id, adapter=adapter, freeze=True)

        # Phase-6.5: fit free adapter for leftover residual
        free_metrics = {}
        if self.args.get("free_adapter", {}).get("enabled", True):
            from gase.distill.free_distiller import FreeAdapterDistiller
            with torch.no_grad():
                delta_chart = adapter(h_chart, chart_state.mu)
            free_distiller = FreeAdapterDistiller(self.args)
            free_adapter_obj, free_metrics = free_distiller.fit_free_adapter_for_layer_slot(
                h_chart, delta_teacher, delta_chart,
                layer_id=layer_id, slot_id=slot_id,
            )
            blk.register_free_adapter(slot_id, free_adapter_obj, freeze=True)
            metrics.update(free_metrics)

        committed_slots = blk.get_available_slot_ids(0)
        logging.info(
            "[CommittedSlots] layer=%d slots=%s", layer_id, committed_slots,
        )

        # Phase-9.2: CrossNLL matrix for current source slot
        if self.args.get("routing", {}).get("cross_nll_diagnostics", False):
            from gase.diagnostics.cross_nll_diagnostics import compute_cross_nll_matrix
            all_slot_states = {}
            for sid in committed_slots:
                ss = blk.slot_states.get(f"0_{sid}")
                if ss is not None:
                    all_slot_states[sid] = ss
            if all_slot_states:
                compute_cross_nll_matrix({slot_id: h_chart}, chart_state, all_slot_states)

        return {"chart": chart_state.to_dict(), "slot": slot_state.to_dict(),
                "distill": metrics}

    # ------------------------------------------------------------------
    #  Phase-6: Sequential multi-slot pipeline for one task
    # ------------------------------------------------------------------

    def distill_sequential_multi_slot_for_task(
        self, task_id: int, train_loader,
    ) -> Dict:
        """
        Phase-6: sequential L9→L10→L11 distillation with slot_id=task_id.

        Lower-layer prefix uses CURRENT_SLOT_STUDENT mode to ensure
        that L10 collection sees L9's current-task slot, etc.

        Args:
            task_id: current task id.
            train_loader: DataLoader for feature collection.

        Returns:
            Dict with per-layer metrics.
        """
        from gase.distill.feature_collector import FeatureCollector
        from gase.distill.cache import DistillCache

        if self.distill_cache is None:
            self.distill_cache = DistillCache()

        collector = FeatureCollector(
            model=self._network, atlas_layers=[9],
            device=self._device, collect_mode="sequential",
        )

        slot_id = task_id
        all_metrics: Dict[int, Dict] = {}
        backbone = self._network.backbone

        logging.info("[MultiSlotDistill] task=%d slot_id=%d", task_id, slot_id)

        for layer_id in self.atlas_layers:
            committed = [
                lid for lid in self.atlas_layers if lid < layer_id
                and backbone.get_block(lid).has_active_chart_adapter()
            ]
            logging.info(
                "[SequentialDistill] Start layer=%d, prefix committed=%s (using slot=%d)",
                layer_id, committed, slot_id,
            )

            # Use current_slot_student for prefix layers
            backbone.set_adapter_mode("current_slot_student")
            backbone.set_active_slot_id(slot_id)

            batch = collector.collect_layer_features(
                train_loader, layer_id=layer_id, task_id=task_id,
            )
            self.distill_cache.add_layer_batch(layer_id, batch)

            metrics = self.distill_one_chart_one_slot_for_layer(
                task_id=task_id, layer_id=layer_id, slot_id=slot_id,
            )
            all_metrics[layer_id] = metrics

        self.phase6_report = all_metrics
        return all_metrics

    # ------------------------------------------------------------------
    #  Phase-6: Oracle-slot eval (upper-bound diagnostic)
    # ------------------------------------------------------------------

    def evaluate_oracle_slot_student(self, data_loader) -> Dict:
        """
        Evaluate using true labels to infer task_id → slot_id.

        This is NOT task-agnostic — it only verifies that slots
        preserve old tasks correctly (storage upper bound).

        Groups samples by oracle slot_id and runs separate forward passes.
        """
        backbone = self._network.backbone
        backbone.eval()

        all_preds, all_labels_list = [], []
        increment = self.args.get("increment", 10)

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                targets_np = targets.cpu().numpy()
                oracle_slot = targets_np // increment  # simple mapping
                unique_slots = np.unique(oracle_slot)

                batch_logits = None
                for sid in unique_slots:
                    mask = oracle_slot == sid
                    if not mask.any():
                        continue
                    x_sub = inputs[mask]
                    logits_sub = backbone.compute_oracle_slot_logits(x_sub, int(sid))
                    logits_sub = logits_sub[:, :self._total_classes]
                    if batch_logits is None:
                        batch_logits = torch.zeros(len(targets_np), logits_sub.shape[1],
                                                   device=logits_sub.device)
                    batch_logits[torch.tensor(mask, device=logits_sub.device)] = logits_sub

                topk_preds = torch.topk(batch_logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels_list.append(targets_np)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[OracleSlotEval] total=%.2f", result["top1"])
        for key, val in sorted(result["grouped"].items()):
            if "-" in key:
                logging.info("[OracleSlotEval] %s=%.2f", key, val)

        self.phase6_oracle_eval = result
        return result

    # ------------------------------------------------------------------
    #  Phase-6: Key-slot eval (simple baseline)
    # ------------------------------------------------------------------

    def evaluate_key_slot_student(self, data_loader) -> Dict:
        """
        Evaluate using per-sample per-layer key slot routing (default GASE inference).

        Collects routing diagnostics and logs slot histograms, routing_acc, etc.
        """
        backbone = self._network.backbone
        backbone.eval()

        all_preds, all_labels_list = [], []
        routing_records: List[Dict] = []

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                logits = backbone.compute_key_slot_logits(inputs)
                routing_info = backbone.collect_last_routing_info()
                routing_records.append(routing_info)

                logits = logits[:, :self._total_classes]
                topk_preds = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[PerLayerKeySlotEval] total=%.2f", result["top1"])
        for key, val in sorted(result["grouped"].items()):
            if "-" in key:
                logging.info("[PerLayerKeySlotEval] %s=%.2f", key, val)

        # Routing diagnostics
        if routing_records and routing_records[0].get("per_layer"):
            from gase.diagnostics.routing_diagnostics import summarize_routing_records
            summarize_routing_records(routing_records, torch.from_numpy(y_true),
                                      self.args.get("increment", 10), mode="per_layer")

        self.phase6_key_eval = result
        return result

    # ------------------------------------------------------------------
    #  Phase-7.5: Path-level consistent slot routing eval
    # ------------------------------------------------------------------

    def _save_metrics_json(self):
        """Save Oracle/Key/Path metrics to JSON after each task."""
        import json, os
        try:
            log_dir = f"logs/gase_metrics/{self.args.get('prefix', 'gase')}"
            os.makedirs(log_dir, exist_ok=True)
            path = f"{log_dir}/task{self._cur_task}_metrics.json"
            data = {
                "task_id": self._cur_task, "known_classes": self._known_classes,
                "oracle": {"top1": float(self.phase6_oracle_eval["top1"]) if self.phase6_oracle_eval else None},
                "per_layer_key": {"top1": float(self.phase6_key_eval["top1"]) if self.phase6_key_eval else None},
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            logging.info("[MetricsJSON] saved to %s", path)
        except Exception as e:
            logging.warning("[MetricsJSON] save failed: %s", e)

    # ==================================================================
    #  Phase-8.6: Head stabilization
    # ==================================================================

    def save_head_snapshot(self) -> None:
        """Save classifier head snapshot after current task."""
        import copy, os
        head = self._network.backbone.head
        snap = {
            "task_id": self._cur_task,
            "known_classes": self._known_classes,
            "weight": copy.deepcopy(head.weight.data.cpu()),
        }
        if hasattr(head, "sigma"):
            snap["sigma"] = copy.deepcopy(head.sigma.data.cpu())
        self.head_snapshots[self._cur_task] = snap

        if self.save_head_snapshots:
            d = f"logs/gase_head_snapshots/{self.args.get('prefix', 'gase')}"
            os.makedirs(d, exist_ok=True)
            p = f"{d}/task{self._cur_task}_head.pt"
            torch.save(snap, p)
            logging.info("[HeadSnapshot] saved task=%d known_classes=%d", self._cur_task, self._known_classes)

    def _apply_head_snapshot(self, snap) -> Dict:
        """Temporarily swap current head with a snapshot. Returns save dict for restore."""
        head = self._network.backbone.head
        save = {"weight": head.weight.data.clone(), "sigma": None}
        if hasattr(head, "sigma"):
            save["sigma"] = head.sigma.data.clone()
            head.sigma.data.copy_(snap["sigma"].to(head.sigma.device))
        head.weight.data[:snap["weight"].shape[0]].copy_(snap["weight"].to(head.weight.device))
        return save

    def _restore_head(self, save: Dict) -> None:
        head = self._network.backbone.head
        head.weight.data.copy_(save["weight"])
        if save["sigma"] is not None and hasattr(head, "sigma"):
            head.sigma.data.copy_(save["sigma"])

    def evaluate_oracle_with_snapshot_head(self, data_loader, snap_task_id: int) -> Dict:
        """Pure snapshot eval: only evaluate classes within snapshot range."""
        if snap_task_id not in self.head_snapshots:
            return {}
        snap = self.head_snapshots[snap_task_id]
        snap_known = snap["known_classes"]
        save = self._apply_head_snapshot(snap)
        try:
            backbone = self._network.backbone
            backbone.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for _, inputs, targets in data_loader:
                    # Only keep samples with labels < snap_known
                    mask = targets < snap_known
                    if not mask.any():
                        continue
                    inputs = inputs[mask].to(self._device)
                    logits = self._compute_oracle_logits(inputs, targets[mask])
                    logits = logits[:, :snap_known]
                    topk = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                    all_preds.append(topk.cpu().numpy())
                    all_labels.append(targets[mask].cpu().numpy())
            y_pred = np.concatenate(all_preds) if all_preds else np.zeros((0, self.topk))
            y_true = np.concatenate(all_labels) if all_labels else np.zeros(0)
            if len(y_true) == 0:
                result = {"top1": 0, "grouped": {}}
            else:
                orig_total = self._total_classes
                self._total_classes = snap_known
                result = self._evaluate(y_pred, y_true)
                self._total_classes = orig_total
        finally:
            self._restore_head(save)

        logging.info("[PureSnapshotHeadEval] snap=%d known=%d total=%.2f",
                     snap_task_id, snap_known, result.get("top1", 0))
        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[PureSnapshotHeadEval] snap=%d %s=%.2f", snap_task_id, key, val)
        return result

    def _compute_oracle_logits(self, inputs, labels):
        """Compute logits using oracle slot path for given labels."""
        backbone = self._network.backbone
        increment = self.args.get("increment", 10)
        oracle_slots = (labels // increment).cpu().numpy()
        unique_slots = np.unique(oracle_slots)
        batch_logits = None
        for sid in unique_slots:
            mask = oracle_slots == sid
            x_sub = inputs[mask]
            logits_sub = backbone.compute_oracle_slot_logits(x_sub, int(sid))
            if batch_logits is None:
                batch_logits = torch.zeros(len(labels), logits_sub.shape[1], device=logits_sub.device)
            batch_logits[torch.tensor(mask, device=logits_sub.device)] = logits_sub
        return batch_logits

    def evaluate_hybrid_snapshot_head(self, data_loader, snap_task_id: int) -> Dict:
        """Hybrid eval: old classes use snapshot weights, new classes use current weights."""
        if snap_task_id not in self.head_snapshots:
            return {}
        snap = self.head_snapshots[snap_task_id]
        snap_known = snap["known_classes"]
        head = self._network.backbone.head

        cur_weight = head.weight.data.clone()
        cur_sigma = head.sigma.data.clone() if hasattr(head, "sigma") else None

        # Slice snapshot weight to snap_known rows
        head.weight.data[:snap_known] = snap["weight"][:snap_known].to(head.weight.device)

        try:
            result = self.evaluate_oracle_slot_student(data_loader)
        finally:
            head.weight.data.copy_(cur_weight)
            if cur_sigma is not None and hasattr(head, "sigma"):
                head.sigma.data.copy_(cur_sigma)

        logging.info("[HybridSnapshotHeadEval] snap=%d weight=old:snap,new:current sigma=current total=%.2f",
                     snap_task_id, result.get("top1", 0))
        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[HybridSnapshotHeadEval] snap=%d %s=%.2f", snap_task_id, key, val)
        return result

    def _get_pre_update_known(self) -> int:
        """Return known_classes BEFORE after_task update."""
        return self._total_classes - self.args.get("increment", 10)

    def _align_new_class_weight_norms(self, pre_known: int) -> None:
        """Align new class weight norms to old class mean."""
        head = self._network.backbone.head
        old_start, old_end = 0, pre_known
        new_start, new_end = pre_known, self._total_classes

        if new_end <= new_start:
            logging.info("[HeadNormAlign] task=%d no new classes (new=%d:%d)", self._cur_task, new_start, new_end)
            return

        old_w = head.weight[old_start:old_end]
        new_w = head.weight[new_start:new_end]
        if new_w.numel() == 0:
            return

        old_norm = old_w.norm(dim=1).mean()
        new_norm_before = new_w.norm(dim=1).mean()
        logging.info("[HeadNormAlign] task=%d old=%d:%d new=%d:%d old_norm=%.4f new_norm_before=%.4f",
                     self._cur_task, old_start, old_end, new_start, new_end,
                     float(old_norm), float(new_norm_before))

        new_norm_per = new_w.norm(dim=1, keepdim=True).clamp_min(1e-8)
        new_w_aligned = new_w / new_norm_per * old_norm
        head.weight.data[new_start:new_end] = new_w_aligned

        new_norm_after = head.weight[new_start:new_end].norm(dim=1).mean()
        logging.info("[HeadNormAlign] new_norm_after=%.4f", float(new_norm_after))

    def _log_head_diagnostics(self, pre_known: int) -> None:
        """Log classifier head norm statistics."""
        head = self._network.backbone.head
        old_start, old_end = 0, pre_known
        new_start, new_end = pre_known, self._total_classes

        old_w = head.weight[old_start:old_end] if old_end > old_start else None
        new_w = head.weight[new_start:new_end] if new_end > new_start else None

        old_nm = float(old_w.norm(dim=1).mean()) if old_w is not None and old_w.numel() > 0 else None
        old_ns = float(old_w.norm(dim=1).std(unbiased=False)) if old_w is not None and old_w.numel() > 1 else None
        new_nm = float(new_w.norm(dim=1).mean()) if new_w is not None and new_w.numel() > 0 else None
        new_ns = float(new_w.norm(dim=1).std(unbiased=False)) if new_w is not None and new_w.numel() > 1 else None
        ratio = new_nm / (old_nm + 1e-8) if old_nm and new_nm else None
        sigma_val = float(head.sigma.item()) if hasattr(head, "sigma") else None

        logging.info("[HeadDiag] task=%d known=%d old=%d:%d new=%d:%d old_norm=%s new_norm=%s ratio=%s sigma=%s",
                     self._cur_task, self._total_classes, old_start, old_end, new_start, new_end,
                     f"{old_nm:.4f}±{old_ns:.4f}" if old_nm else "none",
                     f"{new_nm:.4f}±{new_ns:.4f}" if new_nm else "none",
                     f"{ratio:.3f}" if ratio else "none",
                     f"{sigma_val:.4f}" if sigma_val else "none")

    # ==================================================================
    #  Phase-9.3: Path-consistent raw NLL routing
    # ==================================================================

    def evaluate_path_raw_nll_slot_student(self, data_loader) -> Dict:
        """
        Use raw NLL at L9 to select one slot per sample, then force
        L9/L10/L11 all to use that same slot (path-consistent).
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        backbone.eval()
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)

        all_preds, all_labels_list = [], []
        first_atlas = min(self.atlas_layers)

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                B = inputs.shape[0]

                # Get L9 chart and slots for path routing
                l9_blk = backbone.get_block(first_atlas)
                available = l9_blk.get_available_slot_ids(0)
                cs_l9 = l9_blk.chart_states.get(0)

                if not available or cs_l9 is None:
                    logits = backbone.compute_key_slot_logits(inputs)
                else:
                    # Collect slot states
                    slot_states = {}
                    for sid in available:
                        ss = l9_blk.slot_states.get(f"0_{sid}")
                        if ss is not None:
                            slot_states[sid] = ss

                    # Extract L9 h_chart for routing
                    h9 = backbone._extract_h_chart_at_layer(inputs, first_atlas)

                    # Route with raw NLL at L9
                    routing = router.route(h9, cs_l9, slot_states)
                    path_slot = routing["slot_ids"]  # [B]

                    # Force-consistent path forward
                    backbone._clear_path_slot_ids()
                    for lid in self.atlas_layers:
                        backbone.blocks[lid].path_slot_id = path_slot
                    backbone.set_adapter_mode("path_key_slot_student")
                    try:
                        out = backbone.forward(inputs)
                        logits = out["logits"]
                    finally:
                        backbone.set_adapter_mode("task_train")

                logits = logits[:, :self._total_classes]
                topk = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[PathRawNLLEval] total=%.2f", result["top1"])
        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[PathRawNLLEval] %s=%.2f", key, val)
        return result

    def evaluate_candidate_path_raw_nll_slot_student(self, data_loader) -> Dict:
        """
        For each candidate slot, force consistent path and sum raw NLL
        across L9/L10/L11. Pick slot with lowest total path NLL.
        This is a slow diagnostic, not the default inference.
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        backbone.eval()
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)

        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                B = inputs.shape[0]

                # Determine available slots (common across atlas layers)
                available = None
                for lid in self.atlas_layers:
                    blk = backbone.get_block(lid)
                    s = set(blk.get_available_slot_ids(0))
                    available = s if available is None else available & s
                available = sorted(available) if available else []

                if len(available) <= 1:
                    logits = backbone.compute_key_slot_logits(inputs)
                else:
                    path_scores = torch.full([B, len(available)], float("inf"), device=self._device)

                    for idx, sid in enumerate(available):
                        # Force consistent path with slot sid
                        backbone._clear_path_slot_ids()
                        for lid in self.atlas_layers:
                            backbone.blocks[lid].path_slot_id = torch.full(
                                [B], sid, device=self._device, dtype=torch.long)
                        backbone.set_adapter_mode("path_key_slot_student")
                        try:
                            out = backbone.forward(inputs)
                        finally:
                            backbone.set_adapter_mode("task_train")

                        # Compute path NLL: sum across layers using last_h_chart
                        path_nll = torch.zeros(B, device=self._device)
                        for lid in self.atlas_layers:
                            blk = backbone.get_block(lid)
                            cs = blk.chart_states.get(0)
                            ss = blk.slot_states.get(f"0_{sid}")
                            if cs is not None and ss is not None and blk.last_h_chart is not None:
                                nll = router.compute_nll(blk.last_h_chart, cs, ss)
                                path_nll += nll
                        path_scores[:, idx] = path_nll

                    best_idx = path_scores.argmin(dim=1)
                    best_slot = torch.tensor([available[i.item()] for i in best_idx],
                                             device=self._device, dtype=torch.long)

                    # Final forward with chosen path
                    backbone._clear_path_slot_ids()
                    for lid in self.atlas_layers:
                        backbone.blocks[lid].path_slot_id = best_slot
                    backbone.set_adapter_mode("path_key_slot_student")
                    try:
                        out = backbone.forward(inputs)
                        logits = out["logits"]
                    finally:
                        backbone.set_adapter_mode("task_train")

                logits = logits[:, :self._total_classes]
                topk = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[CandidatePathRawNLLEval] total=%.2f", result["top1"])
        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[CandidatePathRawNLLEval] %s=%.2f", key, val)
        return result

    def evaluate_hybrid_path_raw_nll_slot_student(self, data_loader) -> Dict:
        """Confidence-gated hybrid: L9 raw-NLL margin > tau → PathRawNLL, else PerLayerRawNLL."""
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        first_atlas = min(self.atlas_layers)
        taus = [-1e9, 0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 20.0, 50.0, 100.0, 1e9]
        backbone.eval()

        # Phase 1: collect all data + compute L9 raw NLL margin per sample
        all_inputs_list, all_targets_list = [], []
        all_best, all_margin = [], []
        with torch.no_grad():
            for _, inputs_in, targets in data_loader:
                inputs_in = inputs_in.to(self._device)
                B = inputs_in.shape[0]
                l9_blk = backbone.get_block(first_atlas)
                available = l9_blk.get_available_slot_ids(0)
                cs_l9 = l9_blk.chart_states.get(0)
                slot_states = {}
                for sid in available:
                    ss = l9_blk.slot_states.get(f"0_{sid}")
                    if ss is not None:
                        slot_states[sid] = ss
                if len(slot_states) < 2 or cs_l9 is None:
                    best_slot = torch.zeros(B, dtype=torch.long)
                    margin_vals = torch.zeros(B)
                else:
                    _, h9 = backbone.extract_layer_chart_feature_and_teacher(inputs_in, first_atlas)
                    nll_list, sid_list = [], sorted(slot_states.keys())
                    for sid in sid_list:
                        nll_list.append(router.compute_nll(h9, cs_l9, slot_states[sid]))
                    nll_stack = torch.stack(nll_list, dim=1)  # [B, S]

                    if len(all_best) == 0:
                        logging.info("[HybridMarginDebug] nll_scores_l9.shape=%s num_slots=%d",
                                     list(nll_stack.shape), nll_stack.shape[1])
                        logging.info("[HybridMarginDebug] best_slot[:20]=%s",
                                     nll_stack.argmin(dim=1)[:20].tolist())
                        if nll_stack.shape[1] >= 2:
                            top2_dbg = nll_stack.topk(k=2, dim=1, largest=False).values
                            best_nll_dbg = top2_dbg[:, 0]
                            second_nll_dbg = top2_dbg[:, 1]
                            margin_dbg = second_nll_dbg - best_nll_dbg
                            logging.info("[HybridMarginDebug] best_nll.shape=%s second_nll.shape=%s margin.shape=%s",
                                         list(best_nll_dbg.shape), list(second_nll_dbg.shape), list(margin_dbg.shape))
                            logging.info("[HybridMarginDebug] margin[:10]=%s best_nll[:10]=%s second_nll[:10]=%s",
                                         margin_dbg[:10].tolist(), best_nll_dbg[:10].tolist(),
                                         second_nll_dbg[:10].tolist())

                    top2 = nll_stack.topk(k=min(2, nll_stack.shape[1]), dim=1, largest=False).values
                    best_idx = nll_stack.argmin(dim=1)
                    margin_vals = top2[:, 1] - top2[:, 0] if nll_stack.shape[1] >= 2 else torch.full([B], float("inf"))
                    best_slot = torch.tensor([sid_list[i.item()] for i in best_idx], dtype=torch.long)
                all_best.append(best_slot)
                all_margin.append(margin_vals.cpu())
                all_inputs_list.append(inputs_in.cpu())
                all_targets_list.append(targets)

        all_best_cat = torch.cat(all_best)       # [N]
        all_margin_cat = torch.cat(all_margin)    # [N]
        all_inputs_cat = torch.cat(all_inputs_list)  # [N, C, H, W]
        all_targets_cat = torch.cat(all_targets_list)  # [N]
        bs = 16  # batch size for re-forward

        # Log margin stats
        logging.info("[HybridMarginStats] task=%d mean=%.2f std=%.2f min=%.2f q01=%.2f q05=%.2f q25=%.2f q50=%.2f q75=%.2f q90=%.2f q95=%.2f q99=%.2f max=%.2f",
                     self._cur_task, float(all_margin_cat.float().mean()), float(all_margin_cat.float().std()),
                     float(all_margin_cat.min()), float(torch.quantile(all_margin_cat.float(), 0.01)),
                     float(torch.quantile(all_margin_cat.float(), 0.05)), float(torch.quantile(all_margin_cat.float(), 0.25)),
                     float(torch.quantile(all_margin_cat.float(), 0.50)), float(torch.quantile(all_margin_cat.float(), 0.75)),
                     float(torch.quantile(all_margin_cat.float(), 0.90)), float(torch.quantile(all_margin_cat.float(), 0.95)),
                     float(torch.quantile(all_margin_cat.float(), 0.99)), float(all_margin_cat.max()))

        # CRITICAL: set nll_router so compute_key_slot_logits uses raw NLL, not shared-Q fallback
        backbone.set_nll_router(router)

        best_result, best_total, best_tau = None, 0.0, 0.0
        tau_totals: Dict[float, float] = {}

        try:
            for tau in taus:
                path_mask = all_margin_cat > tau
                per_layer_mask = ~path_mask
                path_ratio = float(path_mask.float().mean())
                all_preds, all_labels = [], []

                # Path branch: EXACT PathRawNLL forward (path_key_slot_student with forced slot)
                if path_mask.any():
                    idx_path = path_mask.nonzero(as_tuple=True)[0]
                    for start in range(0, len(idx_path), bs):
                        batch_idx = idx_path[start:start+bs]
                        x_sub = all_inputs_cat[batch_idx].to(self._device)
                        s_sub = all_best_cat[batch_idx].to(self._device)
                        backbone._clear_path_slot_ids()
                        for lid in self.atlas_layers:
                            backbone.blocks[lid].path_slot_id = s_sub
                        backbone.set_adapter_mode("path_key_slot_student")
                        try:
                            logits = backbone.forward(x_sub)["logits"][:, :self._total_classes]
                        finally:
                            backbone.set_adapter_mode("task_train")
                        topk = torch.topk(logits, k=self.topk, dim=1)[1]
                        all_preds.append((batch_idx, topk.cpu().numpy()))
                        all_labels.append((batch_idx, all_targets_cat[batch_idx].cpu().numpy()))

                # Per-layer branch: EXACT PerLayerRawNLL forward (key_slot_student, nll_router is set)
                if per_layer_mask.any():
                    idx_pl = per_layer_mask.nonzero(as_tuple=True)[0]
                    for start in range(0, len(idx_pl), bs):
                        batch_idx = idx_pl[start:start+bs]
                        x_sub = all_inputs_cat[batch_idx].to(self._device)
                        logits = backbone.compute_key_slot_logits(x_sub)[:, :self._total_classes]
                        topk = torch.topk(logits, k=self.topk, dim=1)[1]
                        all_preds.append((batch_idx, topk.cpu().numpy()))
                        all_labels.append((batch_idx, all_targets_cat[batch_idx].cpu().numpy()))

                if all_preds:
                    idx_concat = np.concatenate([p[0].cpu().numpy() for p in all_preds])
                    pred_concat = np.concatenate([p[1] for p in all_preds])
                    label_concat = np.concatenate([l[1] for l in all_labels])
                    sort_idx = np.argsort(idx_concat)
                    y_pred = pred_concat[sort_idx]
                    y_true = label_concat[sort_idx]
                    result = self._evaluate(y_pred, y_true)
                else:
                    result = {"top1": 0.0, "grouped": {}}

                total = result.get("top1", 0)
                tau_totals[tau] = total
                logging.info("[HybridPathRawNLLEval][tau=%.1f] total=%.2f path_ratio=%.2f",
                             tau, total, path_ratio)
                for key, val in sorted(result.get("grouped", {}).items()):
                    if "-" in key:
                        logging.info("[HybridPathRawNLLEval][tau=%.1f] %s=%.2f", tau, key, val)

                if total > best_total:
                    best_total = total
                    best_result = result
                    best_tau = tau
        finally:
            backbone.set_nll_router(None)

        # Equivalence check: verify Hybrid(-inf)==PathRawNLL, Hybrid(+inf)==PerLayerRawNLL at LOGIT level
        self._verify_hybrid_equivalence(all_inputs_cat, all_best_cat, bs)

        # Log tau sweep summary
        tau_neg, tau_pos = taus[0], taus[-1]
        logging.info("[HybridEquivSummary] task=%d tau=-inf(%.2f) tau=+inf(%.2f) "
                     "sweep=%s",
                     self._cur_task, tau_totals.get(tau_neg, 0), tau_totals.get(tau_pos, 0),
                     {f"{t:.0f}": v for t, v in sorted(tau_totals.items()) if abs(t) < 1e8})

        if best_result:
            logging.info("[HybridPathRawNLLBest] task=%d best_tau=%.1f total=%.2f",
                         self._cur_task, best_tau, best_total)
        return best_result or {"top1": 0.0, "grouped": {}}

    def _verify_hybrid_equivalence(self, all_inputs_cat, all_best_cat, bs: int) -> None:
        """Verify Hybrid(-inf)==PathRawNLL and Hybrid(+inf)==PerLayerRawNLL with max_abs_diff of logits."""
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        N_check = min(64, all_inputs_cat.shape[0])
        x = all_inputs_cat[:N_check].to(self._device)
        s = all_best_cat[:N_check].to(self._device)
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)

        # 1. True PathRawNLL: force-consistent path with L9-selected slot
        backbone._clear_path_slot_ids()
        for lid in self.atlas_layers:
            backbone.blocks[lid].path_slot_id = s
        backbone.set_adapter_mode("path_key_slot_student")
        try:
            logits_path = backbone.forward(x)["logits"][:, :self._total_classes].clone()
        finally:
            backbone.set_adapter_mode("task_train")

        # 2. True PerLayerRawNLL: per-layer routing with raw NLL router
        backbone.set_nll_router(router)
        try:
            logits_perlayer = backbone.compute_key_slot_logits(x)[:, :self._total_classes].clone()
        finally:
            backbone.set_nll_router(None)

        # 3. Hybrid(-inf) = all samples go to path branch (same as 1)
        backbone._clear_path_slot_ids()
        for lid in self.atlas_layers:
            backbone.blocks[lid].path_slot_id = s
        backbone.set_adapter_mode("path_key_slot_student")
        try:
            logits_hybrid_path = backbone.forward(x)["logits"][:, :self._total_classes].clone()
        finally:
            backbone.set_adapter_mode("task_train")

        # 4. Hybrid(+inf) = all samples go to per-layer branch (same as 2)
        backbone.set_nll_router(router)
        try:
            logits_hybrid_perlayer = backbone.compute_key_slot_logits(x)[:, :self._total_classes].clone()
        finally:
            backbone.set_nll_router(None)

        diff_path = (logits_hybrid_path - logits_path).abs().max().item()
        diff_perlayer = (logits_hybrid_perlayer - logits_perlayer).abs().max().item()

        ok = diff_path < 1e-4 and diff_perlayer < 1e-4
        logging.info("[HybridEquivCheck] task=%d path_max_abs_diff=%.6f perlayer_max_abs_diff=%.6f %s",
                     self._cur_task, diff_path, diff_perlayer,
                     "OK" if ok else "FAILED")
        if not ok:
            logging.warning("[HybridEquivCheck] EQUIVALENCE FAILED! "
                          "Hybrid(-inf) should == PathRawNLL, Hybrid(+inf) should == PerLayerRawNLL")

    def _eval_with_router(self, data_loader, router_name: str, router) -> Dict:
        """Evaluate PerLayerKey with a specific router, using custom log prefix."""
        backbone = self._network.backbone
        backbone.set_nll_router(router)

        # Override log prefix by patching evaluate_key_slot_student
        # We do a manual eval to get correct naming
        backbone.eval()
        all_preds, all_labels_list = [], []
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                logits = backbone.compute_key_slot_logits(inputs)
                logits = logits[:, :self._total_classes]
                topk = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        name_map = {
            "shared_q_dist": "PerLayerSharedQDist",
            "raw_nll": "PerLayerRawNLL",
            "calibrated_nll": "PerLayerCalibNLL",
            "calib_nll_prior": "PerLayerCalibPrior",
            "calib_nll_s0penalty": "PerLayerSlot0Penalty",
        }
        tag = name_map.get(router_name, router_name)
        logging.info("[%s] total=%.2f", tag, result["top1"])
        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[%s] %s=%.2f", tag, key, val)

        backbone.set_nll_router(None)
        return result

    def _eval_router_variants(self) -> None:
        """Compare multiple router variants with taskwise breakdown."""
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        logging.info("[RouterCompare] task=%d", self._cur_task)

        # Baseline: shared-Q distance (already computed by evaluate_key_slot_student)
        r_dist = self.phase6_key_eval.get("top1", 0) if self.phase6_key_eval else 0
        r_dist_old = self.phase6_key_eval.get("grouped", {}).get("00-09", 0) if self.phase6_key_eval else 0

        # Raw NLL (no calibration)
        r_nll = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        nll_result = self._eval_with_router(self.test_loader, "raw_nll", r_nll)

        # Calibrated NLL
        r_calib = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=True, use_logdet=True)
        calib_result = self._eval_with_router(self.test_loader, "calibrated_nll", r_calib)

        # Calibrated NLL + prior
        r_prior = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=True, use_logdet=True,
                                           prior_mode="uniform", prior_weight=0.5)
        prior_result = self._eval_with_router(self.test_loader, "calib_nll_prior", r_prior)

        # slot0 penalty
        r_s0p = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=True, use_logdet=True,
                                         slot0_penalty=1.0)
        s0p_result = self._eval_with_router(self.test_loader, "calib_nll_s0penalty", r_s0p)

        def _t(taskwise, key):
            return taskwise.get(key, 0) if taskwise else 0

        tw_dist = self.phase6_key_eval.get("grouped", {}) if self.phase6_key_eval else {}
        tw_nll = nll_result.get("grouped", {}) if nll_result else {}
        tw_calib = calib_result.get("grouped", {}) if calib_result else {}
        tw_prior = prior_result.get("grouped", {}) if prior_result else {}
        tw_s0p = s0p_result.get("grouped", {}) if s0p_result else {}

        for key in sorted(tw_dist):
            if "-" in key:
                logging.info("[RouterCompare] %s: dist=%s raw_nll=%s calib=%s prior=%s s0p=%s", key,
                             tw_dist.get(key, "-"), _t(tw_nll, key), _t(tw_calib, key),
                             _t(tw_prior, key), _t(tw_s0p, key))

        # Add path/hybrid variants
        path_nll_total = self.phase6_path_nll_eval.get("top1", 0) if self.phase6_path_nll_eval else 0
        cand_path_total = self.phase6_cand_path_eval.get("top1", 0) if self.phase6_cand_path_eval else 0
        hybrid_total = self.phase6_hybrid_eval.get("top1", 0) if self.phase6_hybrid_eval else 0

        logging.info("[RouterCompare] total: dist=%.2f raw_nll=%.2f path_nll=%.2f "
                     "cand_path=%.2f hybrid=%.2f calib=%.2f prior=%.2f s0p=%.2f",
                     r_dist, nll_result.get("top1", 0), path_nll_total, cand_path_total,
                     hybrid_total, calib_result.get("top1", 0), prior_result.get("top1", 0),
                     s0p_result.get("top1", 0))

        scores = {
            "shared_q_dist": r_dist, "raw_nll": nll_result.get("top1", 0),
            "calib_nll": calib_result.get("top1", 0), "calib_prior": prior_result.get("top1", 0),
            "slot0_penalty": s0p_result.get("top1", 0),
            "path_raw_nll": path_nll_total,
            "candidate_path_raw_nll": cand_path_total,
            "hybrid_path_raw_nll": hybrid_total,
        }

        best = max(scores, key=scores.get)
        logging.info("[RouterCompare] best=%s (%.2f) default=%s (%.2f)", best, scores[best],
                     self.default_router, scores.get(self.default_router, 0))

    def evaluate_aggregated_path_key_slot_student(self, data_loader) -> Dict:
        """
        Slow diagnostic: for each candidate slot, compute consistent-path
        distance aggregated across L9/L10/L11, then choose best path.

        This is an ablation, not the default inference.
        """
        backbone = self._network.backbone
        backbone.eval()
        all_preds, all_labels_list = [], []
        increment = self.args.get("increment", 10)

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                # Determine available slots from L9
                l9_blk = backbone.get_block(self.atlas_layers[0])
                available_slots = l9_blk.get_available_slot_ids(0)
                if len(available_slots) <= 1:
                    logits = backbone.compute_path_key_slot_logits(inputs)
                else:
                    B = inputs.shape[0]
                    all_slot_scores = torch.zeros(B, len(available_slots), device=self._device)
                    for idx, sid in enumerate(available_slots):
                        backbone._clear_path_slot_ids()
                        for lid in self.atlas_layers:
                            blk = backbone.blocks[lid]
                            blk.path_slot_id = torch.full([B], sid, device=self._device, dtype=torch.long)
                        backbone.set_adapter_mode("path_key_slot_student")
                        try:
                            _ = backbone.forward(inputs)
                        finally:
                            backbone.set_adapter_mode("task_train")
                        for lid in self.atlas_layers:
                            blk = backbone.blocks[lid]
                            if hasattr(blk, "last_routing_info") and blk.last_routing_info is not None:
                                s = blk.last_routing_info.get("scores")
                                if s is not None and idx < s.shape[1]:
                                    n = min(B, s.shape[0])
                                    all_slot_scores[:n, idx] += s[:n, idx]
                    best_slot_ids = all_slot_scores.argmax(dim=1)
                    backbone._clear_path_slot_ids()
                    for lid in self.atlas_layers:
                        blk = backbone.blocks[lid]
                        blk.path_slot_id = best_slot_ids
                    backbone.set_adapter_mode("path_key_slot_student")
                    try:
                        out = backbone.forward(inputs)
                        logits = out["logits"]
                    finally:
                        backbone.set_adapter_mode("task_train")

                logits = logits[:, :self._total_classes]
                topk_preds = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[AggregatedPathKeySlotEval] total=%.2f", result["top1"])
        for key, val in sorted(result["grouped"].items()):
            if "-" in key:
                logging.info("[AggregatedPathKeySlotEval] %s=%.2f", key, val)
        return result

    def evaluate_path_key_slot_student(self, data_loader) -> Dict:
        """
        Task-agnostic path-level consistent routing.
        L9 selects slot per sample, L10/L11 follow.
        """
        backbone = self._network.backbone
        backbone.eval()

        all_preds, all_labels_list = [], []
        routing_records: List[Dict] = []

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                logits = backbone.compute_path_key_slot_logits(inputs)
                routing_info = backbone.collect_last_routing_info()
                routing_records.append(routing_info)

                logits = logits[:, :self._total_classes]
                topk_preds = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[PathKeySlotEval] total=%.2f", result["top1"])
        for key, val in sorted(result["grouped"].items()):
            if "-" in key:
                logging.info("[PathKeySlotEval] %s=%.2f", key, val)

        # Routing diagnostics for path mode
        if routing_records and routing_records[0].get("path"):
            from gase.diagnostics.routing_diagnostics import summarize_routing_records
            summarize_routing_records(routing_records, torch.from_numpy(y_true),
                                      self.args.get("increment", 10), mode="path")

        return result

    # ------------------------------------------------------------------
    #  Phase-5 backward compat wrappers
    # ------------------------------------------------------------------

    def distill_l9_one_chart_one_slot(self, task_id: int) -> Dict:
        return self.distill_one_chart_one_slot_for_layer(task_id, layer_id=9)

    def evaluate_l9_student_vs_teacher(self, test_loader) -> Dict:
        return self.evaluate_sequential_student_vs_teacher(test_loader)

    def distill_sequential_one_chart_one_slot(self, task_id, train_loader) -> Dict:
        return self.distill_sequential_multi_slot_for_task(task_id, train_loader)

    def evaluate_sequential_student_vs_teacher(self, test_loader) -> Dict:
        """Evaluate using CURRENT_SLOT_STUDENT with slot=task_id."""
        backbone = self._network.backbone
        backbone.set_adapter_mode("task_train")
        teacher_acc = self._compute_accuracy(self._network, test_loader)

        backbone.set_adapter_mode("current_slot_student")
        backbone.set_active_slot_id(self.current_task_id)
        student_acc = self._compute_accuracy(self._network, test_loader)

        backbone.set_adapter_mode("task_train")
        gap = teacher_acc - student_acc
        logging.info(
            "[SequentialStudentEval] teacher_acc=%.2f student_acc=%.2f gap=%.2f",
            teacher_acc, student_acc, gap,
        )
        return {"teacher_acc": float(teacher_acc), "student_acc": float(student_acc),
                "gap": float(gap)}

    def train_task_adapter(self, task_id: int, train_loader: DataLoader) -> None:
        raise NotImplementedError("Phase-2 uses _train() directly.")

    def freeze_task_adapter(self) -> None:
        raise NotImplementedError("Phase-3+ will freeze after distillation.")

    def collect_chart_features_for_task(self, task_id: int, data_loader) -> None:
        """
        Phase-3 feature collection: collect authoritative L9 h_chart and delta_teacher.

        Creates a FeatureCollector, collects L9 features from the current
        task training data, and stores the result in self.distill_cache.

        Does NOT build charts or slots (Phase-4+).

        Args:
            task_id: current task index.
            data_loader: DataLoader for the current task's training data.
        """
        from gase.distill.feature_collector import FeatureCollector
        from gase.distill.cache import DistillCache

        logging.info("[GASE] Starting L9 feature collection for task %d...", task_id)

        collector = FeatureCollector(
            model=self._network,
            atlas_layers=[9],
            device=self._device,
            collect_mode="sequential",
        )

        l9_batch = collector.collect_l9_features(data_loader, task_id)

        if self.distill_cache is None:
            self.distill_cache = DistillCache()
        self.distill_cache.add_layer_batch(9, l9_batch)

        cache_summary = self.distill_cache.summary()
        logging.info("[GASE] DistillCache summary: %s", cache_summary)

        return l9_batch

    def build_or_update_atlas(self, task_id: int) -> None:
        raise NotImplementedError("Phase-3+ will implement chart construction.")

    def build_or_update_slots(self, task_id: int) -> None:
        raise NotImplementedError("Phase-3+ will implement slot construction.")

    def distill_chart_adapters(self, task_id: int) -> None:
        raise NotImplementedError("Phase-3+ will implement distillation.")

    def distill_free_adapters(self, task_id: int) -> None:
        raise NotImplementedError("Phase-3+ will implement free distillation.")

    def distill_slot_routers(self, task_id: int) -> None:
        raise NotImplementedError("Phase-3+ will implement router distillation.")

    def calibrate_classifier(self, task_id: int, train_loader) -> None:
        raise NotImplementedError("Phase-3+ will implement calibration.")

    def commit_current_task_adapters(self, task_id: int) -> None:
        raise NotImplementedError("Phase-3+ will implement adapter commit.")

    def remove_task_adapters(self) -> None:
        raise NotImplementedError("Phase-3+ will remove adapters after distillation.")
