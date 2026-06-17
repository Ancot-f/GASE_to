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

        if self._cur_task == 0:
            self._network.backbone.head = nn.Linear(
                self._network.backbone.embed_dim, data_manager.nb_classes
            )
            nn.init.kaiming_uniform_(self._network.backbone.head.weight, a=math.sqrt(5))
            nn.init.zeros_(self._network.backbone.head.bias)

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
        """Create fresh TaskAdapters for each GASE atlas layer."""
        dim = self._network.backbone.embed_dim
        for blk in self._network.backbone.get_atlas_blocks():
            blk.task_adapter = build_task_adapter(self.args, dim)
            blk.task_adapter.to(self._device)
        logging.info(
            "Created TaskAdapters at layers %s", self.atlas_layers
        )

    # ==================================================================
    #  Parameter freezing
    # ==================================================================

    def _freeze_backbone_except_task_adapters_and_classifier(self) -> None:
        """
        Freeze entire model, then unfreeze task adapters and classifier head.

        Strategy:
          1. Freeze all parameters.
          2. Unfreeze task_adapter parameters in GASE blocks.
          3. Unfreeze classifier head.
        """
        backbone = self._network.backbone

        # 1. Freeze all
        for p in backbone.parameters():
            p.requires_grad = False

        # 2. Unfreeze task adapters
        for blk in backbone.get_atlas_blocks():
            if blk.task_adapter is not None:
                for p in blk.task_adapter.parameters():
                    p.requires_grad = True

        # 3. Unfreeze head
        for p in backbone.head.parameters():
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

    def before_task(self, task_id: int) -> None:
        """Prepare for a new task (Phase-2: no-op)."""
        pass

    def after_task(self):
        """Post-task cleanup. Phase-2: keep task adapters for eval."""
        self._known_classes = self._total_classes
        logging.info("Task %d completed. known_classes=%d", self._cur_task, self._known_classes)

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

    def train_task_adapter(self, task_id: int, train_loader: DataLoader) -> None:
        raise NotImplementedError("Phase-2 uses _train() directly.")

    def freeze_task_adapter(self) -> None:
        raise NotImplementedError("Phase-3+ will freeze after distillation.")

    def collect_chart_features_for_task(self, task_id: int, data_loader) -> None:
        raise NotImplementedError("Phase-3+ will implement feature collection.")

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
