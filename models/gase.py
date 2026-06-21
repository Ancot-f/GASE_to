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

        # Phase-9.6: path gate config
        self.path_gate_eval: bool = args.get("routing", {}).get("path_gate_eval", False)
        pg_cfg = args.get("routing", {}).get("path_gate", {})
        self.path_gate_tau_list: List[float] = pg_cfg.get("tau_margin_list", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
        self.path_gate_agreement_list: List[int] = pg_cfg.get("agreement_min_list", [1, 2, 3])
        self.path_gate_gate_types: List[str] = pg_cfg.get("gate_types", ["candidate_margin", "layer_agreement", "candidate_agreement"])
        self.phase96_cand_margin_eval: Optional[Dict] = None
        self.phase96_layer_agreement_eval: Optional[Dict] = None
        self.phase96_cand_agreement_eval: Optional[Dict] = None

        # Phase-9.7: path score variant config
        self.path_score_eval: bool = args.get("routing", {}).get("path_score_eval", False)
        ps_cfg = args.get("routing", {}).get("path_score", {})
        self.balanced_z_gamma_list: List[float] = ps_cfg.get("balanced_z_gamma0_list", [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
        self.percentile_s0p_gamma_list: List[float] = ps_cfg.get("percentile_s0p_gamma0_list", [0.0, 0.05, 0.10, 0.20, 0.30, 0.50, 1.00])
        self.phase97_balanced_z_eval: Optional[Dict] = None
        self.phase97_percentile_eval: Optional[Dict] = None
        self.phase97_percentile_s0p_eval: Optional[Dict] = None

        # Phase-9.8: chart OOD dry-run
        self.chart_ood_dryrun: bool = args.get("routing", {}).get("chart_ood_dryrun", False)
        self.chart_ood_config: dict = args.get("routing", {}).get("chart_ood", {})
        self.phase98_dryrun_results: Optional[Dict] = None

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

                # Phase-9.6: path gate evals
                cand_margin_best = None
                layer_agreement_best = None
                cand_agreement_best = None
                if self.path_gate_eval:
                    logging.info("[PathGate] Phase-9.6: running path gate evaluations...")
                    if "candidate_margin" in self.path_gate_gate_types:
                        cm = self._run_candidate_margin_sweep(
                            self.test_loader, self.path_gate_tau_list)
                        self.phase96_cand_margin_eval = cm
                        cand_margin_best = cm.get("best_result", {})
                    if "layer_agreement" in self.path_gate_gate_types:
                        la = self._run_layer_agreement_sweep(
                            self.test_loader, self.path_gate_agreement_list)
                        self.phase96_layer_agreement_eval = la
                        layer_agreement_best = la.get("best_result", {})
                    if "candidate_agreement" in self.path_gate_gate_types:
                        ca = self._run_candidate_agreement_sweep(
                            self.test_loader, self.path_gate_tau_list,
                            self.path_gate_agreement_list)
                        self.phase96_cand_agreement_eval = ca
                        cand_agreement_best = ca.get("best_result", {})
                    # Diagnostics
                    self._log_path_gate_diagnostics(self.test_loader)

                # Phase-9.7: path score variants
                balanced_z_best = None
                percentile_result = None
                percentile_s0p_best = None
                if self.path_score_eval:
                    logging.info("[PathScore] Phase-9.7: running path score variants...")
                    self._log_slot_bias_diagnostics()
                    # Balanced z-score sweep
                    bz = self._run_balanced_z_sweep(self.test_loader, self.balanced_z_gamma_list)
                    self.phase97_balanced_z_eval = bz
                    balanced_z_best = bz.get("best_result", {})
                    # Percentile path
                    percentile_result = self.evaluate_path_score_variant_slot_student(
                        self.test_loader, score_type="percentile")
                    self.phase97_percentile_eval = percentile_result
                    # Percentile + slot0 penalty sweep
                    ps0p = self._run_percentile_s0p_sweep(
                        self.test_loader, self.percentile_s0p_gamma_list)
                    self.phase97_percentile_s0p_eval = ps0p
                    percentile_s0p_best = ps0p.get("best_result", {})

                # Phase-9.8: chart OOD dry-run
                dry_chart_result = None
                dry_chart_oracle_result = None
                if self.chart_ood_dryrun:
                    from gase.routing.nll_router import CalibratedNLLSlotRouter
                    r_nll = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
                    raw_nll_pre = self._eval_with_router(self.test_loader, "raw_nll_pre", r_nll)
                    raw_nll_pre_total = raw_nll_pre.get("top1", 0) if raw_nll_pre else 0
                    path_nll_pre_total = path_nll_result.get("top1", 0) if path_nll_result else 0

                    self.phase98_dryrun_results = self._run_chart_ood_dryrun(
                        raw_nll_total=raw_nll_pre_total,
                        path_nll_total=path_nll_pre_total)
                    # Extract best results if available
                    if self.phase98_dryrun_results:
                        # Find best oracle across methods/K
                        best_oracle_total = 0
                        for layer_results in self.phase98_dryrun_results.values():
                            for method_results in layer_results.values():
                                for k, res in method_results.items():
                                    if isinstance(res, dict) and "oracle_eval" in res:
                                        oe = res["oracle_eval"]
                                        if isinstance(oe, dict) and oe.get("total", 0) > best_oracle_total:
                                            best_oracle_total = oe["total"]
                                            dry_chart_oracle_result = oe
                        if best_oracle_total > 0:
                            dry_chart_result = {"top1": best_oracle_total}

                if oracle_result and key_result and path_result:
                    from gase.routing.nll_router import CalibratedNLLSlotRouter
                    r_nll = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
                    raw_nll_result = self._eval_with_router(self.test_loader, "raw_nll", r_nll)
                    raw_nll_total = raw_nll_result.get("top1", 0) if raw_nll_result else 0

                    # Build EvalCompare dynamically (skip disabled evals)
                    eval_parts = [
                        f"Oracle={oracle_result.get('top1', 0):.2f}",
                        f"PerLayerKey={key_result.get('top1', 0):.2f}",
                        f"PerLayerRawNLL={raw_nll_total:.2f}",
                        f"PathRawNLL={path_nll_result.get('top1', 0) if path_nll_result else 0:.2f}",
                        f"CandidatePathRawNLL={cand_path_result.get('top1', 0) if cand_path_result else 0:.2f}",
                        f"L9HybridBest={hybrid_result.get('top1', 0) if hybrid_result else 0:.2f}",
                    ]
                    if self.path_gate_eval:
                        cm_total = cand_margin_best.get("top1", 0) if cand_margin_best else 0
                        la_total = layer_agreement_best.get("top1", 0) if layer_agreement_best else 0
                        ca_total = cand_agreement_best.get("top1", 0) if cand_agreement_best else 0
                        eval_parts += [
                            f"CandidateMarginHybridBest={cm_total:.2f}",
                            f"LayerAgreementHybridBest={la_total:.2f}",
                            f"CandidateAgreementHybridBest={ca_total:.2f}",
                        ]
                    if self.path_score_eval:
                        bz_total = balanced_z_best.get("top1", 0) if balanced_z_best else 0
                        perc_total = percentile_result.get("top1", 0) if percentile_result else 0
                        ps0p_total = percentile_s0p_best.get("top1", 0) if percentile_s0p_best else 0
                        eval_parts += [
                            f"BalancedZBest={bz_total:.2f}",
                            f"PercentilePath={perc_total:.2f}",
                            f"PercentileSlot0PenaltyBest={ps0p_total:.2f}",
                        ]
                    if self.chart_ood_dryrun:
                        dry_total = dry_chart_oracle_result.get("total", 0) if dry_chart_oracle_result else None
                        dry_gain = (dry_total - path_nll_result.get("top1", 0)) if dry_total is not None and path_nll_result else None
                        eval_parts += [
                            f"DryChartOracleBest={dry_total:.2f}" if dry_total is not None else "DryChartOracleBest=NA",
                            f"DryChartOracleGainOverPath={dry_gain:+.2f}" if dry_gain is not None else "DryChartOracleGainOverPath=NA",
                        ]
                    best_so_far = max(v for v in [path_nll_result.get("top1", 0) if path_nll_result else 0,
                                                    raw_nll_total] if v > 0)
                    eval_parts += [f"Oracle-gap={oracle_result.get('top1', 0) - raw_nll_total:.2f}"]
                    eval_parts += [f"DefaultRouter={self.default_router}"]
                    logging.info("[EvalCompare] %s", " ".join(eval_parts))
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

            # Phase-9.6: path gate results
            if self.path_gate_eval:
                data["path_gate"] = {}
                if self.phase96_cand_margin_eval:
                    sweep = self.phase96_cand_margin_eval.get("sweep", {})
                    data["path_gate"]["candidate_margin"] = {
                        k: {"top1": v.get("top1", 0), "path_ratio": v.get("path_ratio", 0)}
                        for k, v in sweep.items()
                    }
                    data["path_gate"]["candidate_margin"]["best_tau"] = self.phase96_cand_margin_eval.get("best_tau")
                if self.phase96_layer_agreement_eval:
                    sweep = self.phase96_layer_agreement_eval.get("sweep", {})
                    data["path_gate"]["layer_agreement"] = {
                        k: {"top1": v.get("top1", 0), "path_ratio": v.get("path_ratio", 0)}
                        for k, v in sweep.items()
                    }
                    data["path_gate"]["layer_agreement"]["best_agreement_min"] = self.phase96_layer_agreement_eval.get("best_agreement_min")
                if self.phase96_cand_agreement_eval:
                    sweep = self.phase96_cand_agreement_eval.get("sweep", {})
                    data["path_gate"]["candidate_agreement"] = {
                        k: {"top1": v.get("top1", 0), "path_ratio": v.get("path_ratio", 0)}
                        for k, v in sweep.items()
                    }
                    data["path_gate"]["candidate_agreement"]["best_tau"] = self.phase96_cand_agreement_eval.get("best_tau")
                    data["path_gate"]["candidate_agreement"]["best_agreement_min"] = self.phase96_cand_agreement_eval.get("best_agreement_min")

            # Phase-9.7: path score variants
            if self.path_score_eval:
                data["path_score_variants"] = {}
                if self.phase97_balanced_z_eval:
                    sweep = self.phase97_balanced_z_eval.get("sweep", {})
                    data["path_score_variants"]["balanced_z"] = {
                        k: {"top1": v.get("top1", 0)} for k, v in sweep.items()
                    }
                    data["path_score_variants"]["balanced_z"]["best_gamma0"] = self.phase97_balanced_z_eval.get("best_gamma0")
                if self.phase97_percentile_eval:
                    data["path_score_variants"]["percentile"] = {
                        "top1": self.phase97_percentile_eval.get("top1", 0)
                    }
                if self.phase97_percentile_s0p_eval:
                    sweep = self.phase97_percentile_s0p_eval.get("sweep", {})
                    data["path_score_variants"]["percentile_slot0_penalty"] = {
                        k: {"top1": v.get("top1", 0)} for k, v in sweep.items()
                    }
                    data["path_score_variants"]["percentile_slot0_penalty"]["best_gamma0"] = self.phase97_percentile_s0p_eval.get("best_gamma0")

            # Phase-9.8: chart OOD dry-run results
            if self.chart_ood_dryrun and self.phase98_dryrun_results:
                data["chart_ood_dryrun"] = {}
                for layer_id, layer_results in self.phase98_dryrun_results.items():
                    data["chart_ood_dryrun"][str(layer_id)] = {}
                    for method, method_results in layer_results.items():
                        data["chart_ood_dryrun"][str(layer_id)][method] = {}
                        for k, res in method_results.items():
                            if isinstance(res, dict):
                                data["chart_ood_dryrun"][str(layer_id)][method][str(k)] = {
                                    "num_charts": res.get("num_charts", 0),
                                    "quality_gain": res.get("quality_gain", {}),
                                    "overlap": res.get("overlap", {}),
                                    "purity": res.get("purity", {}),
                                    "oracle_eval": res.get("oracle_eval", {}),
                                    "proposal": res.get("proposal", {}),
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

    # ==================================================================
    #  Phase-9.6: Normalized candidate-path margin + layer-agreement gate
    # ==================================================================

    def _compute_normalized_candidate_path_data(self, data_loader):
        """Phase 1: collect per-layer h_chart + compute normalized path scores.

        Returns dict with keys:
            all_inputs_cat  [N, C, H, W]
            all_targets_cat [N]
            all_best_path   [N]          best path slot per sample
            all_margin      [N]          candidate_path_margin per sample
            all_best_score  [N]          best path score per sample
            all_second_score[N]          second-best path score per sample
            all_per_layer_slots [N, 3]   per-layer independent best slots (L9, L10, L11)
            all_agreement   [N]          layer_agreement count (0-3)
            available_slots list[int]    candidate slot IDs
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        backbone.set_nll_router(router)
        backbone.eval()

        all_inputs_list, all_targets_list = [], []
        all_best_path_list, all_margin_list, all_best_score_list, all_second_score_list = [], [], [], []
        all_per_layer_list, all_agreement_list = [], []
        available_slots = None
        eps = 1e-6

        with torch.no_grad():
            for bi, (_, inputs_in, targets) in enumerate(data_loader):
                inputs_in = inputs_in.to(self._device)
                B = inputs_in.shape[0]

                # Determine available slots (intersection across all atlas layers)
                avail = None
                for lid in self.atlas_layers:
                    blk = backbone.get_block(lid)
                    s = set(blk.get_available_slot_ids(0))
                    avail = s if avail is None else avail & s
                avail = sorted(avail) if avail else []
                if available_slots is None:
                    available_slots = avail

                if len(avail) < 2:
                    zeros = torch.zeros(B, dtype=torch.long, device=self._device)
                    inf = torch.full([B], float("inf"), device=self._device)
                    all_best_path_list.append(zeros.cpu())
                    all_margin_list.append(inf.cpu())
                    all_best_score_list.append(inf.cpu())
                    all_second_score_list.append(inf.cpu())
                    all_per_layer_list.append(torch.zeros(B, 3, dtype=torch.long).cpu())
                    all_agreement_list.append(torch.zeros(B, dtype=torch.long).cpu())
                    all_inputs_list.append(inputs_in.cpu())
                    all_targets_list.append(targets)
                    continue

                # Collect h_chart at each atlas layer via KEY_SLOT_STUDENT forward
                _ = backbone.compute_key_slot_logits(inputs_in)
                h_by_layer: Dict[int, torch.Tensor] = {}
                for lid in self.atlas_layers:
                    blk = backbone.get_block(lid)
                    if blk.last_h_chart is not None:
                        h_by_layer[lid] = blk.last_h_chart.detach()
                    else:
                        h_by_layer[lid] = backbone._extract_h_chart_at_layer(inputs_in, lid)

                # Compute z-normalized NLL for each candidate slot at each layer
                num_slots = len(avail)
                path_scores = torch.zeros(B, num_slots, device=self._device)  # lower=better
                per_layer_best = torch.zeros(B, len(self.atlas_layers), dtype=torch.long, device=self._device)

                for slot_idx, sid in enumerate(avail):
                    sid_nll_sum = torch.zeros(B, device=self._device)
                    for layer_idx, lid in enumerate(self.atlas_layers):
                        blk = backbone.get_block(lid)
                        cs = blk.chart_states.get(0)
                        ss = blk.slot_states.get(f"0_{sid}")
                        h = h_by_layer[lid]
                        if cs is not None and ss is not None and h is not None:
                            raw_nll = router.compute_nll(h, cs, ss)
                            # z-normalize
                            self_mean = getattr(ss, "router_nll_mean", None)
                            self_std = getattr(ss, "router_nll_std", None)
                            if self_mean is not None and self_std is not None and self_std > 0.1:
                                z_nll = (raw_nll - self_mean) / (self_std + eps)
                            else:
                                z_nll = raw_nll
                            sid_nll_sum += z_nll

                        # Per-layer independent best
                        per_layer_nll = []
                        for sid2 in avail:
                            ss2 = blk.slot_states.get(f"0_{sid2}")
                            if cs is not None and ss2 is not None and h is not None:
                                nll2 = router.compute_nll(h, cs, ss2)
                                self_mean2 = getattr(ss2, "router_nll_mean", None)
                                self_std2 = getattr(ss2, "router_nll_std", None)
                                if self_mean2 is not None and self_std2 is not None and self_std2 > 0.1:
                                    nll2 = (nll2 - self_mean2) / (self_std2 + eps)
                                per_layer_nll.append(nll2)
                            else:
                                per_layer_nll.append(torch.full([B], float("inf"), device=self._device))
                        per_layer_stack = torch.stack(per_layer_nll, dim=1)  # [B, S]
                        per_layer_best[:, layer_idx] = per_layer_stack.argmin(dim=1)

                    path_scores[:, slot_idx] = sid_nll_sum

                # Best path slot and margin
                best_idx = path_scores.argmin(dim=1)  # [B]
                best_path_slot = torch.tensor([avail[i.item()] for i in best_idx], dtype=torch.long)
                best_score = path_scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # [B]
                path_scores_filled = path_scores.clone()
                path_scores_filled.scatter_(1, best_idx.unsqueeze(1), float("inf"))
                second_score = path_scores_filled.min(dim=1).values  # [B]
                margin = second_score - best_score  # [B], positive

                # Layer agreement
                best_path_idx = best_idx.unsqueeze(1).expand(-1, len(self.atlas_layers))  # [B, 3]
                agreement = (per_layer_best == best_path_idx).sum(dim=1)  # [B], 0-3

                all_best_path_list.append(best_path_slot.cpu())
                all_margin_list.append(margin.cpu())
                all_best_score_list.append(best_score.cpu())
                all_second_score_list.append(second_score.cpu())
                all_per_layer_list.append(per_layer_best.cpu())
                all_agreement_list.append(agreement.cpu())
                all_inputs_list.append(inputs_in.cpu())
                all_targets_list.append(targets)

                if bi == 0:
                    logging.info("[NormPathDebug] num_slots=%d avail=%s path_scores.shape=%s "
                                 "margin[:10]=%s agreement[:10]=%s best_path[:20]=%s",
                                 num_slots, avail, list(path_scores.shape),
                                 margin[:10].tolist(), agreement[:10].tolist(),
                                 best_path_slot[:20].tolist())

        backbone.set_nll_router(None)

        return {
            "all_inputs_cat": torch.cat(all_inputs_list),
            "all_targets_cat": torch.cat(all_targets_list),
            "all_best_path": torch.cat(all_best_path_list),
            "all_margin": torch.cat(all_margin_list),
            "all_best_score": torch.cat(all_best_score_list),
            "all_second_score": torch.cat(all_second_score_list),
            "all_per_layer_slots": torch.cat(all_per_layer_list),
            "all_agreement": torch.cat(all_agreement_list),
            "available_slots": available_slots or [],
        }

    def evaluate_path_gate_hybrid_slot_student(self, data_loader, gate_type: str,
                                                tau_margin: float = None,
                                                agreement_min: int = None) -> Dict:
        """Unified path-gate hybrid eval.

        gate_type: "candidate_margin" | "layer_agreement" | "candidate_agreement"
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        backbone.eval()

        data = self._compute_normalized_candidate_path_data(data_loader)
        all_inputs = data["all_inputs_cat"]
        all_targets = data["all_targets_cat"]
        all_best_path = data["all_best_path"]
        all_margin = data["all_margin"]
        all_agreement = data["all_agreement"]
        N = all_targets.shape[0]
        bs = 16

        # Gate decision
        if gate_type == "candidate_margin":
            path_mask = all_margin > tau_margin if tau_margin is not None else torch.ones(N, dtype=torch.bool)
        elif gate_type == "layer_agreement":
            path_mask = all_agreement >= agreement_min if agreement_min is not None else torch.ones(N, dtype=torch.bool)
        elif gate_type == "candidate_agreement":
            margin_ok = all_margin > tau_margin if tau_margin is not None else torch.ones(N, dtype=torch.bool)
            agree_ok = all_agreement >= agreement_min if agreement_min is not None else torch.ones(N, dtype=torch.bool)
            path_mask = margin_ok & agree_ok
        else:
            raise ValueError(f"Unknown gate_type: {gate_type}")

        per_layer_mask = ~path_mask
        path_ratio = float(path_mask.float().mean())
        all_preds, all_labels = [], []

        backbone.set_nll_router(router)
        try:
            # Path branch
            if path_mask.any():
                idx_path = path_mask.nonzero(as_tuple=True)[0]
                for start in range(0, len(idx_path), bs):
                    batch_idx = idx_path[start:start+bs]
                    x_sub = all_inputs[batch_idx].to(self._device)
                    s_sub = all_best_path[batch_idx].to(self._device)
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
                    all_labels.append((batch_idx, all_targets[batch_idx].cpu().numpy()))

            # Per-layer branch
            if per_layer_mask.any():
                idx_pl = per_layer_mask.nonzero(as_tuple=True)[0]
                for start in range(0, len(idx_pl), bs):
                    batch_idx = idx_pl[start:start+bs]
                    x_sub = all_inputs[batch_idx].to(self._device)
                    logits = backbone.compute_key_slot_logits(x_sub)[:, :self._total_classes]
                    topk = torch.topk(logits, k=self.topk, dim=1)[1]
                    all_preds.append((batch_idx, topk.cpu().numpy()))
                    all_labels.append((batch_idx, all_targets[batch_idx].cpu().numpy()))
        finally:
            backbone.set_nll_router(None)

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

        # Logging
        if gate_type == "candidate_margin":
            logging.info("[CandidateMarginHybridEval][tau=%.1f] total=%.2f path_ratio=%.2f",
                         tau_margin, total, path_ratio)
        elif gate_type == "layer_agreement":
            logging.info("[LayerAgreementHybridEval][agreement_min=%d] total=%.2f path_ratio=%.2f",
                         agreement_min, total, path_ratio)
        elif gate_type == "candidate_agreement":
            logging.info("[CandidateAgreementHybridEval][tau=%.1f][agreement_min=%d] total=%.2f path_ratio=%.2f",
                         tau_margin, agreement_min, total, path_ratio)

        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[%s] %s=%.2f", "PathGateHybrid", key, val)

        result["path_ratio"] = path_ratio
        return result

    # --- Sweep wrappers ---

    def _run_candidate_margin_sweep(self, data_loader, tau_list, path_data=None):
        results = {}
        best_result, best_total, best_tau = None, 0.0, 0.0
        for tau in tau_list:
            r = self.evaluate_path_gate_hybrid_slot_student(
                data_loader, gate_type="candidate_margin", tau_margin=tau)
            results[f"tau_{tau:.1f}"] = r
            total = r.get("top1", 0)
            if total > best_total:
                best_total, best_result, best_tau = total, r, tau
        if best_result:
            logging.info("[CandidateMarginHybridBest] task=%d best_tau=%.1f total=%.2f",
                         self._cur_task, best_tau, best_total)
        return {"sweep": results, "best_tau": best_tau, "best_result": best_result}

    def _run_layer_agreement_sweep(self, data_loader, agreement_list):
        results = {}
        best_result, best_total, best_agreement = None, 0.0, 0
        for agree_min in agreement_list:
            r = self.evaluate_path_gate_hybrid_slot_student(
                data_loader, gate_type="layer_agreement", agreement_min=agree_min)
            results[f"agreement_{agree_min}"] = r
            total = r.get("top1", 0)
            if total > best_total:
                best_total, best_result, best_agreement = total, r, agree_min
        if best_result:
            logging.info("[LayerAgreementHybridBest] task=%d best_agreement_min=%d total=%.2f",
                         self._cur_task, best_agreement, best_total)
        return {"sweep": results, "best_agreement_min": best_agreement, "best_result": best_result}

    def _run_candidate_agreement_sweep(self, data_loader, tau_list, agreement_list):
        results = {}
        best_result, best_total, best_tau, best_agreement = None, 0.0, 0.0, 0
        for tau in tau_list:
            for agree_min in agreement_list:
                r = self.evaluate_path_gate_hybrid_slot_student(
                    data_loader, gate_type="candidate_agreement",
                    tau_margin=tau, agreement_min=agree_min)
                results[f"tau_{tau:.1f}_agreement_{agree_min}"] = r
                total = r.get("top1", 0)
                if total > best_total:
                    best_total, best_result, best_tau, best_agreement = total, r, tau, agree_min
        if best_result:
            logging.info("[CandidateAgreementHybridBest] task=%d best_tau=%.1f best_agreement_min=%d total=%.2f",
                         self._cur_task, best_tau, best_agreement, best_total)
        return {"sweep": results, "best_tau": best_tau,
                "best_agreement_min": best_agreement, "best_result": best_result}

    # --- Diagnostics ---

    def _log_path_gate_diagnostics(self, data_loader) -> Dict:
        """Log normalized path score stats, best_slot hist, and layer agreement distrib."""
        data = self._compute_normalized_candidate_path_data(data_loader)
        margin = data["all_margin"].float()
        best_score = data["all_best_score"].float()
        second_score = data["all_second_score"].float()
        best_path = data["all_best_path"]
        agreement = data["all_agreement"]
        avail = data["available_slots"]
        num_slots = len(avail)
        margin_skipped = num_slots < 2

        # Overall stats
        if margin_skipped:
            logging.info("[NormPathScoreStats] task=%d num_slots=%d margin_skipped=True",
                         self._cur_task, num_slots)
        else:
            logging.info("[NormPathScoreStats] task=%d "
                         "best_score_mean=%.2f best_score_std=%.2f "
                         "second_score_mean=%.2f second_score_std=%.2f "
                         "margin_mean=%.2f margin_std=%.2f "
                         "margin_q05=%.2f margin_q25=%.2f margin_q50=%.2f margin_q75=%.2f margin_q95=%.2f",
                         self._cur_task,
                         float(best_score.mean()), float(best_score.std()),
                         float(second_score.mean()), float(second_score.std()),
                         float(margin.mean()), float(margin.std()),
                         float(torch.quantile(margin, 0.05)), float(torch.quantile(margin, 0.25)),
                         float(torch.quantile(margin, 0.50)), float(torch.quantile(margin, 0.75)),
                         float(torch.quantile(margin, 0.95)))

        # Best slot histogram
        slot_hist = {int(s): int((best_path == s).sum()) for s in avail}
        logging.info("[NormPathBestSlot] task=%d hist=%s", self._cur_task, slot_hist)

        # Layer agreement histogram
        agree_hist = {int(k): int((agreement == k).sum()) for k in range(4)}
        logging.info("[LayerAgreementStats] task=%d hist=%s", self._cur_task, agree_hist)

        # Per-source breakdown
        if not hasattr(self, "_increment"):
            increment = self.args.get("increment", 10)
        else:
            increment = self._increment if self._increment else self.args.get("increment", 10)

        all_targets = data["all_targets_cat"]
        task_ids = all_targets // increment

        for src in sorted(task_ids.unique().tolist()):
            smask = task_ids == src
            n_src = int(smask.sum())
            if n_src == 0:
                continue
            m_src = margin[smask]
            bp_src = best_path[smask]
            ag_src = agreement[smask]

            src_slot_hist = {int(s): int((bp_src == s).sum()) for s in avail}
            src_agree_hist = {int(k): int((ag_src == k).sum()) for k in range(4)}

            if margin_skipped:
                logging.info("[NormPathScoreStatsByTask] task=%d source=%d n=%d best_slot_hist=%s",
                             self._cur_task, src, n_src, src_slot_hist)
            else:
                logging.info("[NormPathScoreStatsByTask] task=%d source=%d n=%d "
                             "margin_mean=%.2f margin_q50=%.2f best_slot_hist=%s",
                             self._cur_task, src, n_src,
                             float(m_src.mean()), float(torch.quantile(m_src, 0.50)),
                             src_slot_hist)
            logging.info("[LayerAgreementStatsByTask] task=%d source=%d hist=%s",
                         self._cur_task, src, src_agree_hist)

        return {"margin_stats": {"mean": float(margin.mean()), "std": float(margin.std())} if not margin_skipped else {},
                "best_slot_hist": slot_hist, "layer_agreement_hist": agree_hist}

    # ==================================================================
    #  Phase-9.7: Slot0 Bias Correction + Path Score Normalization
    # ==================================================================

    def _log_slot_bias_diagnostics(self) -> None:
        """Log self-NLL stats and test-set NLL stats per slot for bias analysis."""
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        backbone.eval()

        for lid in self.atlas_layers:
            blk = backbone.get_block(lid)
            cs = blk.chart_states.get(0)
            if cs is None:
                continue
            available = blk.get_available_slot_ids(0)

            # Self NLL stats (from slot build time)
            for sid in available:
                ss = blk.slot_states.get(f"0_{sid}")
                if ss is not None:
                    logging.info("[SlotBiasDiag] task=%d layer=%d slot=%d "
                                 "self_mean=%.2f self_std=%.2f "
                                 "q05=%.2f q25=%.2f q50=%.2f q75=%.2f q90=%.2f q95=%.2f logdet=%.2f",
                                 self._cur_task, lid, sid,
                                 getattr(ss, "router_nll_mean", float("nan")),
                                 getattr(ss, "router_nll_std", float("nan")),
                                 getattr(ss, "router_nll_q05", float("nan")),
                                 getattr(ss, "router_nll_q25", float("nan")),
                                 getattr(ss, "router_nll_q50", float("nan")),
                                 getattr(ss, "router_nll_q75", float("nan")),
                                 getattr(ss, "router_nll_q90", float("nan")),
                                 getattr(ss, "router_nll_q95", float("nan")),
                                 getattr(ss, "router_logdet", float("nan")))

            # Test NLL stats: compute raw NLL on test set via single forward
            backbone.set_nll_router(router)
            slot_test_nlls = {sid: [] for sid in available}
            with torch.no_grad():
                for _, inputs, _ in self.test_loader:
                    inputs = inputs.to(self._device)
                    _ = backbone.compute_key_slot_logits(inputs)
                    h = blk.last_h_chart
                    if h is None:
                        continue
                    for sid in available:
                        ss = blk.slot_states.get(f"0_{sid}")
                        if ss is not None and cs is not None:
                            nll = router.compute_nll(h, cs, ss)
                            slot_test_nlls[sid].append(nll.cpu())
            backbone.set_nll_router(None)

            for sid in available:
                if slot_test_nlls[sid]:
                    all_nll = torch.cat(slot_test_nlls[sid])
                    logging.info("[SlotTestNLLStats] task=%d layer=%d slot=%d "
                                 "mean=%.2f std=%.2f q50=%.2f q90=%.2f",
                                 self._cur_task, lid, sid,
                                 float(all_nll.mean()), float(all_nll.std()),
                                 float(torch.quantile(all_nll, 0.50)),
                                 float(torch.quantile(all_nll, 0.90)))

            # Score comparison: aggregate per-slot raw, z, percentile means
            slot_raw_mean = {}
            slot_z_mean = {}
            with torch.no_grad():
                for _, inputs, _ in self.test_loader:
                    inputs = inputs.to(self._device)
                    _ = backbone.compute_key_slot_logits(inputs)
                    h = blk.last_h_chart
                    if h is None:
                        continue
                    for sid in available:
                        ss = blk.slot_states.get(f"0_{sid}")
                        if ss is not None and cs is not None:
                            raw = router.compute_nll(h, cs, ss)
                            self_mean = getattr(ss, "router_nll_mean", 0)
                            self_std = getattr(ss, "router_nll_std", 1)
                            z = (raw - self_mean) / (self_std + 1e-6) if self_std > 0.1 else raw
                            slot_raw_mean.setdefault(sid, []).append(raw.cpu())
                            slot_z_mean.setdefault(sid, []).append(z.cpu())

            for sid in available:
                if sid in slot_raw_mean and slot_raw_mean[sid]:
                    raw_all = torch.cat(slot_raw_mean[sid])
                    z_all = torch.cat(slot_z_mean[sid])
                    logging.info("[SlotScoreBias] task=%d layer=%d "
                                 "slot=%d raw_mean=%.2f z_mean=%.2f "
                                 "z_q10=%.2f z_q50=%.2f z_q90=%.2f",
                                 self._cur_task, lid, sid,
                                 float(raw_all.mean()), float(z_all.mean()),
                                 float(torch.quantile(z_all, 0.10)),
                                 float(torch.quantile(z_all, 0.50)),
                                 float(torch.quantile(z_all, 0.90)))

    @staticmethod
    def _approx_percentile(value: torch.Tensor, slot) -> torch.Tensor:
        """Approximate percentile using stored quantiles (piecewise linear)."""
        qs = torch.tensor([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], device=value.device)
        qvals_list = []
        for attr in ["router_nll_q05", "router_nll_q10", "router_nll_q25",
                     "router_nll_q50", "router_nll_q75", "router_nll_q90", "router_nll_q95"]:
            v = getattr(slot, attr, None)
            if v is not None:
                qvals_list.append(float(v))
            else:
                qvals_list.append(0.0)
        qvals = torch.tensor(qvals_list, device=value.device)

        perc = torch.zeros_like(value)
        for i in range(len(qs) - 1):
            mask = (value >= qvals[i]) & (value < qvals[i + 1])
            alpha = (value - qvals[i]) / (qvals[i + 1] - qvals[i] + 1e-10)
            perc = torch.where(mask, qs[i] + alpha * (qs[i + 1] - qs[i]), perc)
        perc = torch.where(value < qvals[0], torch.tensor(0.01, device=value.device), perc)
        perc = torch.where(value >= qvals[-1], torch.tensor(0.99, device=value.device), perc)
        return perc

    def _compute_variant_path_data(self, data_loader, score_type: str, gamma0: float = 0.0):
        """Compute path scores for a given variant and return per-sample tensors.

        score_type: "balanced_z" | "percentile" | "percentile_s0p"
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)
        backbone.set_nll_router(router)
        backbone.eval()

        all_inputs_list, all_targets_list = [], []
        all_best_path_list, all_margin_list, all_best_score_list = [], [], []
        available_slots = None
        eps = 1e-6

        with torch.no_grad():
            for bi, (_, inputs_in, targets) in enumerate(data_loader):
                inputs_in = inputs_in.to(self._device)
                B = inputs_in.shape[0]

                avail = None
                for lid in self.atlas_layers:
                    blk = backbone.get_block(lid)
                    s = set(blk.get_available_slot_ids(0))
                    avail = s if avail is None else avail & s
                avail = sorted(avail) if avail else []
                if available_slots is None:
                    available_slots = avail

                if len(avail) < 2:
                    zeros = torch.zeros(B, dtype=torch.long, device=self._device)
                    all_best_path_list.append(zeros.cpu())
                    all_margin_list.append(torch.zeros(B).cpu())
                    all_best_score_list.append(torch.zeros(B).cpu())
                    all_inputs_list.append(inputs_in.cpu())
                    all_targets_list.append(targets)
                    continue

                num_slots = len(avail)
                _ = backbone.compute_key_slot_logits(inputs_in)
                h_by_layer = {}
                for lid in self.atlas_layers:
                    blk = backbone.get_block(lid)
                    if blk.last_h_chart is not None:
                        h_by_layer[lid] = blk.last_h_chart.detach()
                    else:
                        h_by_layer[lid] = backbone._extract_h_chart_at_layer(inputs_in, lid)

                path_scores = torch.zeros(B, num_slots, device=self._device)

                for slot_idx, sid in enumerate(avail):
                    sid_sum = torch.zeros(B, device=self._device)
                    for lid in self.atlas_layers:
                        blk = backbone.get_block(lid)
                        cs = blk.chart_states.get(0)
                        ss = blk.slot_states.get(f"0_{sid}")
                        h = h_by_layer[lid]
                        if cs is not None and ss is not None and h is not None:
                            raw_nll = router.compute_nll(h, cs, ss)
                            self_mean = getattr(ss, "router_nll_mean", None)
                            self_std = getattr(ss, "router_nll_std", None)

                            if score_type == "balanced_z":
                                if self_mean is not None and self_std is not None and self_std > 0.1:
                                    score = (raw_nll - self_mean) / (self_std + eps)
                                else:
                                    score = raw_nll
                                if sid == 0:
                                    score = score + gamma0
                            elif score_type in ("percentile", "percentile_s0p"):
                                score = GASELearner._approx_percentile(raw_nll, ss)
                                if score_type == "percentile_s0p" and sid == 0:
                                    score = score + gamma0
                            else:
                                score = raw_nll
                            sid_sum += score
                    path_scores[:, slot_idx] = sid_sum

                best_idx = path_scores.argmin(dim=1)
                best_path_slot = torch.tensor([avail[i.item()] for i in best_idx], dtype=torch.long)
                best_score = path_scores.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                path_scores_filled = path_scores.clone()
                path_scores_filled.scatter_(1, best_idx.unsqueeze(1), float("inf"))
                second_score = path_scores_filled.min(dim=1).values
                margin = second_score - best_score

                all_best_path_list.append(best_path_slot.cpu())
                all_margin_list.append(margin.cpu())
                all_best_score_list.append(best_score.cpu())
                all_inputs_list.append(inputs_in.cpu())
                all_targets_list.append(targets)

        backbone.set_nll_router(None)

        return {
            "all_inputs_cat": torch.cat(all_inputs_list),
            "all_targets_cat": torch.cat(all_targets_list),
            "all_best_path": torch.cat(all_best_path_list),
            "all_margin": torch.cat(all_margin_list),
            "all_best_score": torch.cat(all_best_score_list),
            "available_slots": available_slots or [],
        }

    def evaluate_path_score_variant_slot_student(self, data_loader, score_type: str,
                                                  gamma0: float = 0.0) -> Dict:
        """Evaluate path-consistent forward using a specific path score variant.

        score_type: "balanced_z" | "percentile" | "percentile_s0p"
        """
        from gase.routing.nll_router import CalibratedNLLSlotRouter
        backbone = self._network.backbone
        router = CalibratedNLLSlotRouter(temperature=1.0, calibrate_nll=False, use_logdet=True)

        data = self._compute_variant_path_data(data_loader, score_type, gamma0)
        all_inputs = data["all_inputs_cat"]
        all_targets = data["all_targets_cat"]
        all_best_path = data["all_best_path"]
        avail = data["available_slots"]
        N = all_targets.shape[0]
        bs = 16

        backbone.eval()
        backbone.set_nll_router(router)
        all_preds, all_labels = [], []

        try:
            for start in range(0, N, bs):
                batch_idx = torch.arange(start, min(start + bs, N))
                x_sub = all_inputs[batch_idx].to(self._device)
                s_sub = all_best_path[batch_idx].to(self._device)
                backbone._clear_path_slot_ids()
                for lid in self.atlas_layers:
                    backbone.blocks[lid].path_slot_id = s_sub
                backbone.set_adapter_mode("path_key_slot_student")
                try:
                    logits = backbone.forward(x_sub)["logits"][:, :self._total_classes]
                finally:
                    backbone.set_adapter_mode("task_train")
                topk = torch.topk(logits, k=self.topk, dim=1)[1]
                all_preds.append(topk.cpu().numpy())
                all_labels.append(all_targets[batch_idx].cpu().numpy())
        finally:
            backbone.set_nll_router(None)

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        result = self._evaluate(y_pred, y_true)

        # Logging
        if score_type == "balanced_z":
            logging.info("[BalancedPathEval][gamma0=%.1f] total=%.2f", gamma0, result["top1"])
        elif score_type == "percentile":
            logging.info("[PercentilePathEval] total=%.2f", result["top1"])
        elif score_type == "percentile_s0p":
            logging.info("[PercentileSlot0PenaltyPathEval][gamma0=%.2f] total=%.2f",
                        gamma0, result["top1"])

        for key, val in sorted(result.get("grouped", {}).items()):
            if "-" in key:
                logging.info("[%s][gamma0=%.2f] %s=%.2f", score_type, gamma0, key, val)

        # Best slot histogram
        slot_hist = {int(s): int((all_best_path == s).sum()) for s in avail}
        if score_type == "balanced_z":
            logging.info("[BalancedPathBestSlot][gamma0=%.1f] task=%d hist=%s",
                        gamma0, self._cur_task, slot_hist)
        elif score_type == "percentile":
            logging.info("[PercentilePathBestSlot] task=%d hist=%s", self._cur_task, slot_hist)
        elif score_type == "percentile_s0p":
            logging.info("[PercentileSlot0PenaltyBestSlot][gamma0=%.2f] task=%d hist=%s",
                        gamma0, self._cur_task, slot_hist)

        return result

    # --- Sweep wrappers for Phase-9.7 ---

    def _run_balanced_z_sweep(self, data_loader, gamma0_list):
        results = {}
        best_result, best_total, best_gamma = None, 0.0, 0.0
        for g in gamma0_list:
            r = self.evaluate_path_score_variant_slot_student(
                data_loader, score_type="balanced_z", gamma0=g)
            results[f"gamma0_{g:.1f}"] = r
            total = r.get("top1", 0)
            if total > best_total:
                best_total, best_result, best_gamma = total, r, g
        if best_result:
            logging.info("[BalancedPathBest] task=%d best_gamma0=%.1f total=%.2f",
                        self._cur_task, best_gamma, best_total)
        return {"sweep": results, "best_gamma0": best_gamma, "best_result": best_result}

    def _run_percentile_s0p_sweep(self, data_loader, gamma0_list):
        results = {}
        best_result, best_total, best_gamma = None, 0.0, 0.0
        for g in gamma0_list:
            r = self.evaluate_path_score_variant_slot_student(
                data_loader, score_type="percentile_s0p", gamma0=g)
            results[f"gamma0_{g:.2f}"] = r
            total = r.get("top1", 0)
            if total > best_total:
                best_total, best_result, best_gamma = total, r, g
        if best_result:
            logging.info("[PercentileSlot0PenaltyBest] task=%d best_gamma0=%.2f total=%.2f",
                        self._cur_task, best_gamma, best_total)
        return {"sweep": results, "best_gamma0": best_gamma, "best_result": best_result}

    # ==================================================================
    #  Phase-9.8: Chart OOD Dry-run
    # ==================================================================

    def _run_chart_ood_dryrun(self, raw_nll_total: float, path_nll_total: float) -> Optional[Dict]:
        """Run chart OOD dry-run diagnostics without modifying the model."""
        from gase.diagnostics.chart_ood_dryrun import ChartOODDryRunner
        logging.info("[ChartOODDryRun] Phase-9.8: starting dry-run diagnostics...")

        ood_cfg = self.chart_ood_config
        logging.info("[ChartOODConfig] enabled=%s", ood_cfg.get("enabled", True))
        logging.info("[ChartOODConfig] dryrun=True")
        logging.info("[ChartOODConfig] use_labels_for_build=%s", ood_cfg.get("use_labels_for_build", False))
        logging.info("[ChartOODConfig] use_delta_for_build=%s", ood_cfg.get("use_delta_for_build", False))
        logging.info("[ChartOODConfig] diagnostics_use_labels=%s", ood_cfg.get("diagnostics_use_labels", True))
        logging.info("[ChartOODConfig] methods=%s", ood_cfg.get("methods", ["kmeans_pca"]))
        logging.info("[ChartOODConfig] k_list=%s", ood_cfg.get("k_list", [1, 2, 3]))

        increment = self.args.get("increment", 10)
        runner = ChartOODDryRunner(
            backbone=self._network.backbone,
            atlas_layers=self.atlas_layers,
            config={
                **ood_cfg,
                "increment": increment,
                "rank": self.args.get("chart", {}).get("rank", 8),
            },
        )

        try:
            results = runner.run_all_diagnostics(
                data_loader=self.test_loader,
                total_classes=self._total_classes,
                raw_nll_total=raw_nll_total,
                path_nll_total=path_nll_total,
                increment=increment,
            )
            logging.info("[ChartOODDryRun] diagnostics complete")
            return results
        except Exception as e:
            logging.warning("[ChartOODDryRun] failed: %s", e)
            import traceback
            logging.warning(traceback.format_exc())
            return None

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

        # Phase-9.6: path gate variants
        cm_total = self.phase96_cand_margin_eval.get("best_result", {}).get("top1", 0) if self.phase96_cand_margin_eval else 0
        la_total = self.phase96_layer_agreement_eval.get("best_result", {}).get("top1", 0) if self.phase96_layer_agreement_eval else 0
        ca_total = self.phase96_cand_agreement_eval.get("best_result", {}).get("top1", 0) if self.phase96_cand_agreement_eval else 0

        # Phase-9.7: path score variants (only if enabled)
        bz_total = None
        perc_total = None
        ps0p_total = None
        if self.path_score_eval:
            bz_total = self.phase97_balanced_z_eval.get("best_result", {}).get("top1", 0) if self.phase97_balanced_z_eval else None
            perc_total = self.phase97_percentile_eval.get("top1", 0) if self.phase97_percentile_eval else None
            ps0p_total = self.phase97_percentile_s0p_eval.get("best_result", {}).get("top1", 0) if self.phase97_percentile_s0p_eval else None

        # Phase-9.8: dry chart oracle
        dry_oracle_total = None
        if self.phase98_dryrun_results:
            for layer_results in self.phase98_dryrun_results.values():
                for method_results in layer_results.values():
                    for k, res in method_results.items():
                        if isinstance(res, dict) and "oracle_eval" in res:
                            oe = res["oracle_eval"]
                            if isinstance(oe, dict):
                                v = oe.get("total", 0)
                                if 0 <= v <= 100 and (dry_oracle_total is None or v > dry_oracle_total):
                                    dry_oracle_total = v

        # Build total line dynamically
        parts = [f"dist={r_dist:.2f}", f"raw_nll={nll_result.get('top1', 0):.2f}",
                 f"path_nll={path_nll_total:.2f}", f"cand_path={cand_path_total:.2f}"]
        if self.path_score_eval and bz_total is not None:
            parts += [f"balanced_z={bz_total:.2f}", f"percentile_path={perc_total:.2f}",
                      f"percentile_s0p={ps0p_total:.2f}"]
        if self.chart_ood_dryrun and dry_oracle_total is not None:
            parts += [f"dry_chart_oracle={dry_oracle_total:.2f}"]
        parts += [f"calib={calib_result.get('top1', 0):.2f}",
                  f"prior={prior_result.get('top1', 0):.2f}",
                  f"s0p={s0p_result.get('top1', 0):.2f}"]
        logging.info("[RouterCompare] total: %s", " ".join(parts))

        scores = {
            "shared_q_dist": r_dist, "raw_nll": nll_result.get("top1", 0),
            "calib_nll": calib_result.get("top1", 0), "calib_prior": prior_result.get("top1", 0),
            "slot0_penalty": s0p_result.get("top1", 0),
            "path_raw_nll": path_nll_total,
            "candidate_path_raw_nll": cand_path_total,
            "hybrid_path_raw_nll": hybrid_total,
        }
        if self.path_gate_eval:
            scores.update({
                "candidate_margin_hybrid": cm_total,
                "layer_agreement_hybrid": la_total,
                "candidate_agreement_hybrid": ca_total,
            })
        if self.path_score_eval and bz_total is not None:
            scores.update({
                "balanced_z": bz_total, "percentile_path": perc_total,
                "percentile_s0p": ps0p_total,
            })
        if self.chart_ood_dryrun and dry_oracle_total is not None:
            scores["dry_chart_oracle"] = dry_oracle_total

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
