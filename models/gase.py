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

        logging.info(
            "GASELearner Phase-2 initialized. atlas_layers=%s, adapter_dim=%d",
            self.atlas_layers, self.adapter_dim,
        )

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
        self._known_classes = self._total_classes
        logging.info("Task %d completed. known_classes=%d", self._cur_task, self._known_classes)

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
                self.evaluate_oracle_slot_student(self.test_loader)
                self.evaluate_key_slot_student(self.test_loader)
                self.evaluate_path_key_slot_student(self.test_loader)

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
        Evaluate using per-sample key slot routing (KEY_SLOT_STUDENT mode).

        Phase-7: each sample independently selects the nearest slot
        via Mahalanobis distance in P-space.
        """
        backbone = self._network.backbone
        backbone.eval()

        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                logits = backbone.compute_key_slot_logits(inputs)
                logits = logits[:, :self._total_classes]
                topk_preds = torch.topk(logits, k=self.topk, dim=1, largest=True, sorted=True)[1]
                all_preds.append(topk_preds.cpu().numpy())
                all_labels_list.append(targets.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels_list)
        result = self._evaluate(y_pred, y_true)

        logging.info("[PerSampleKeySlotEval] total=%.2f", result["top1"])
        for key, val in sorted(result["grouped"].items()):
            if "-" in key:
                logging.info("[PerSampleKeySlotEval] %s=%.2f", key, val)

        self.phase6_key_eval = result
        return result

    # ------------------------------------------------------------------
    #  Phase-7.5: Path-level consistent slot routing eval
    # ------------------------------------------------------------------

    def evaluate_path_key_slot_student(self, data_loader) -> Dict:
        """
        Task-agnostic path-level consistent routing.
        L9 selects slot per sample via shared Q-space router_key,
        L10/L11 follow the same slot.
        """
        backbone = self._network.backbone
        backbone.eval()

        all_preds, all_labels_list = [], []

        with torch.no_grad():
            for _, inputs, targets in data_loader:
                inputs = inputs.to(self._device)
                logits = backbone.compute_path_key_slot_logits(inputs)
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
