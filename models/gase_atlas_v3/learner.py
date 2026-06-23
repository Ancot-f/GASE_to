"""GASE-Atlas v3 learner.

V3 keeps v2's useful fitting components, but rebuilds the lifecycle around the
chart/slot/free separation:
  chart = feature geometry
  slot = residual transformation
  free = shared fallback chosen by the same policy in train and inference
"""

import gc
import logging

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from backbone.linears import CosineLinear, ProtoCalibratedCosine
from models.base import BaseLearner
from models.gase_atlas_v3.adapters import ChartMLPAdapter
from models.gase_atlas_v3.chart_adapter_builder import RidgeChartAdapterBuilder
from models.gase_atlas_v3.chart_builder import PPCAChartBuilder
from models.gase_atlas_v3.classifier import AtlasClassifier
from models.gase_atlas_v3.rollout import record_teacher_outputs, rollout_forward
from models.gase_atlas_v3.teacher_flow import TeacherFlowCache
from utils.inc_net import AtlasVitNet
from utils.toolkit import tensor2numpy


num_workers = 8


def _is_atlas_module(module):
    return type(module).__name__ == "GASEAtlasBlockModuleV3"


class GASEAtlasV3Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._normalize_args(args)
        self._network = AtlasVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get("min_lr", 1e-8)
        self._acc_history = []
        self._free_trained = {}

    def _normalize_args(self, args):
        defaults = {
            "gase_atlas_layers": [9, 10, 11],
            "gase_atlas_adapt_start_layer": 9,
            "gase_atlas_adapt_end_layer": 11,
            "gase_atlas_task_bottleneck": 16,
            "gase_atlas_task_reset_each_task": True,
            "gase_atlas_free_bottleneck": 16,
            "gase_atlas_freeze_early_after_task0": True,
            "gase_atlas_routing_temperature": 0.5,
            "gase_atlas_free_temperature": 1.0,
            "gase_atlas_top_k": 2,
            "gase_atlas_l11_disable_chart": False,
            "gase_atlas_l11_use_free": True,
            "chart_infer_radius_scale": 0.75,
            "chart_min_full_R2_for_routing": 0.25,
            "routing_beta_r2": 3.0,
            "routing_beta_conflict": 0.5,
            "routing_min_adapter_cos": 0.20,
            "routing_max_adapter_norm_ratio": 2.0,
            "routing_uncertainty_margin": 0.5,
            "routing_uncertainty_entropy": 1.2,
            "chart_max_per_layer": 4,
            "chart_growth_mode": "continual",
            "chart_max_new_per_task": args.get("chart_max_new_per_task", args.get("chart_max_per_layer", 4)),
            "chart_max_total_per_layer": 0,
            "chart_build_budget": 0,
            "chart_min_samples": 16,
            "chart_seed_sample_size": 512,
            "chart_fit_sample_size": 64,
            "chart_knn_size": 48,
            "chart_pca_energy": 0.80,
            "chart_dim_min": 2,
            "chart_dim_max": 8,
            "chart_radius_quantile": 0.70,
            "chart_radius_scale": 0.75,
            "chart_max_support_ratio": 0.50,
            "chart_quality_active": 0.45,
            "chart_quality_candidate": 0.30,
            "chart_quality_mode": "weighted_sum",
            "chart_feature_standardize": False,
            "chart_overlap_max": 0.30,
            "chart_rec_error_scale": 1.0,
            "chart_grassmann_tau": 0.5,
            "chart_adapter_type": "projected_linear",
            "chart_adapter_type_l10": "projected_mlp",
            "chart_adapter_type_l11": "projected_mlp",
            "chart_residual_energy": 0.90,
            "chart_residual_dim_min": 1,
            "chart_residual_dim_max": 8,
            "chart_adapter_ridge_lambda": 1e-3,
            "free_adapter_lr": 1e-4,
            "free_adapter_epochs": 3,
            "free_adapter_l2_to_prev": 1e-2,
            "gase_atlas_classifier": "cosine_imprint",
            "gase_atlas_cosine_scale": 24.0,
            "imprint_alpha": 0.7,
            "classifier_calibration_epochs": 3,
            "classifier_calibration_lr": args.get("init_lr", 0.005),
            "classifier_calibration_weight_decay": args.get("weight_decay", 0.002),
            "func_epoch": args.get("func_epoch", args.get("epochs", 20)),
            "optimizer": args.get("optimizer", "sgd"),
            "init_lr": args.get("init_lr", 0.005),
            "weight_decay": args.get("weight_decay", 0.002),
            "min_lr": args.get("min_lr", 0.0),
            "ffn_adapt": True,
            "ffn_option": "parallel",
            "ffn_adapter_scalar": "1.0",
            "ffn_num": 16,
            "ffn_adapter_type": "adaptmlp",
            "d_model": 768,
        }
        for key, value in defaults.items():
            args.setdefault(key, value)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = self._build_classifier(768, data_manager.nb_classes)

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("GASE-Atlas-v3 learning on %d-%d", self._known_classes, self._total_classes)

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test",
            mode="test",
        )
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        self.data_manager = data_manager

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        self._before_task()
        teacher_train_acc, teacher_test_acc = self._train_task_adapters(train_loader, test_loader)
        logging.info("[V3Teacher] task=%d train=%.2f test=%.2f",
                     self._cur_task, teacher_train_acc, teacher_test_acc)
        self._distill_task(train_loader)
        self._imprint_classifier(train_loader)
        self._calibrate_classifier(train_loader)
        self._log_atlas_diagnostics()
        test_acc = self._compute_accuracy(self._network, test_loader)
        self._acc_history.append(float(test_acc))
        logging.info("[V3Eval] task=%d all_seen_acc=%.2f history=%s",
                     self._cur_task, test_acc, [f"{x:.1f}" for x in self._acc_history])

    def _before_task(self):
        for module in self._iter_atlas_modules():
            layer = module.atlas_layer
            layer._current_task = self._cur_task
            if self.args.get("gase_atlas_task_reset_each_task", True):
                layer.reset_task_adapter()
            module.set_train_phase(current_task=self._cur_task)

    def _train_task_adapters(self, train_loader, test_loader):
        atlas_layers = self.args["gase_atlas_layers"]
        self._teacher_flow = TeacherFlowCache(layers=atlas_layers)
        for module in self._iter_atlas_modules():
            if module.layer_id in atlas_layers:
                module.atlas_layer.init_teacher_flow_cache(self._teacher_flow)
                module.atlas_layer.set_collect_enabled(False)

        self.update_optimizer_and_scheduler(self.args["func_epoch"], self.init_lr)
        final_train_acc = 0.0
        for epoch in range(self.args["func_epoch"]):
            collect = epoch == self.args["func_epoch"] - 1
            for module in self._iter_atlas_modules():
                if module.layer_id in atlas_layers:
                    module.atlas_layer.set_collect_enabled(collect)

            self._network.train()
            correct, total, loss_sum = 0, 0, 0.0
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                if collect:
                    self._teacher_flow.record_labels(targets.detach().cpu())

                logits = self._network(inputs)["logits"][:, :self._total_classes]
                if self._cur_task > 0:
                    logits[:, :self._known_classes] = -float("inf")
                loss = F.cross_entropy(logits, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(targets).cpu().sum()
                total += targets.numel()

            self.scheduler.step()
            final_train_acc = np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)
            logging.info("[V3TaskTrain] task=%d epoch=%d/%d loss=%.4f acc=%.2f",
                         self._cur_task, epoch + 1, self.args["func_epoch"],
                         loss_sum / max(len(train_loader), 1), final_train_acc)

        for module in self._iter_atlas_modules():
            module.atlas_layer.set_collect_enabled(False)
        test_acc = self._compute_accuracy_current_task(self._network, test_loader)
        del self.optimizer
        del self.scheduler
        gc.collect()
        torch.cuda.empty_cache()
        return final_train_acc, test_acc

    def _distill_task(self, train_loader):
        distill_loader = self._make_distill_loader(train_loader)
        atlas_layers = self.args["gase_atlas_layers"]

        # Cache augmented batches ONCE so teacher/rollout see identical inputs
        # while still benefiting from random augmentation.
        cached_inputs, cached_targets = [], []
        for _, inputs, targets in distill_loader:
            cached_inputs.append(inputs.clone())
            cached_targets.append(targets.clone())
        logging.info("[V3DistillCache] %d batches / %d samples",
                     len(cached_inputs), sum(t.shape[0] for t in cached_inputs))

        self._refresh_teacher_flow_cached(cached_inputs, cached_targets)
        teacher_outputs = self._record_teacher_cached(cached_inputs)

        chart_builder = self._make_chart_builder()
        adapter_builder = self._make_adapter_builder()
        modules = self._iter_atlas_modules_dict()
        active_layers = set()
        self._free_trained = {}

        for step, layer_id in enumerate(atlas_layers):
            module = modules.get(layer_id)
            if module is None:
                continue
            layer = module.atlas_layer
            if getattr(layer, "l11_identity", False) and int(layer_id) == 11:
                logging.info("[V3Distill:L%d] l11_identity=True, skipping", layer_id)
                active_layers.add(layer_id)
                continue
            device = next(layer.free_adapter.parameters()).device

            flow = self._teacher_flow.stack(layer_id)
            target_source = "teacher_delta"
            if step == 0:
                features_cpu = flow.h_pre
                residuals_cpu = flow.delta_task.float()
            else:
                rolled = self._rollout_cached(cached_inputs, active_layers)
                features_cpu = rolled["pre"].get(layer_id, flow.h_pre)
                teacher_out = teacher_outputs.get(layer_id)
                rollout_out = rolled["block"].get(layer_id)
                if teacher_out is None or rollout_out is None:
                    residuals_cpu = flow.delta_task.float()
                    target_source = "teacher_delta_fallback"
                    logging.info(
                        "[V3Distill:L%d] rollout target missing, fallback to teacher delta",
                        layer_id,
                    )
                else:
                    residuals_cpu = (teacher_out - rollout_out).float()
                    target_source = "teacher_block_minus_rollout_block"
            logging.info(
                "[V3DistillTarget:L%d] source=%s feat_norm=%.4f target_norm=%.4f",
                layer_id,
                target_source,
                features_cpu.norm(dim=-1).mean().item(),
                residuals_cpu.norm(dim=-1).mean().item(),
            )

            indices = self._balanced_budget_indices(flow.labels, features_cpu.shape[0])
            features_fit = features_cpu[indices].to(device)
            residuals_fit = residuals_cpu[indices].to(device)
            existing = [c for c in layer.atlas.charts if getattr(c, "birth_task", -1) != self._cur_task]
            growth_mode, chart_budget, existing_for_build = self._chart_growth_plan(layer, existing)
            chart_builder.max_charts = chart_budget

            charts, non_chart, reuse_pairs, update_pairs = chart_builder.build_layer_charts(
                features_fit,
                residuals_fit,
                existing_charts=existing_for_build,
                next_chart_id=max((c.chart_id for c in layer.atlas.charts), default=-1) + 1,
                birth_task=self._cur_task,
            )
            for chart in charts:
                chart.cpu()
            layer.register_charts(charts)

            features_all = features_cpu.to(device)
            residuals_all = residuals_cpu.to(device)
            new_slots = self._build_slots_for_new_charts(
                layer, charts, adapter_builder, features_all, residuals_all, layer_id)
            slots_add = self._build_slots_for_existing_charts(
                layer, update_pairs, adapter_builder, features_fit, residuals_fit, layer_id)
            self._deactivate_empty_charts(layer)

            free_mask = layer.compute_free_mask(features_all)
            n_free = int(free_mask.sum().item())
            self._free_trained[layer_id] = n_free
            if n_free > 0:
                self._train_free_adapter_for_layer(
                    layer, features_all[free_mask], residuals_all[free_mask])

            layer.set_mode("inference")
            layer.remove_task_adapter_grads()
            active_layers.add(layer_id)
            logging.info(
                "[V3Distill:L%d] charts_new=%d slots_new=%d slots_add=%d "
                "reuse=%d growth=%s budget=%d free_gate=%d/%d "
                "total_charts=%d active=%d",
                layer_id, len(charts), new_slots, slots_add, len(reuse_pairs),
                growth_mode, chart_budget,
                self._free_trained[layer_id], features_all.shape[0],
                len(layer.atlas.charts), layer.atlas.num_active,
            )
            gc.collect()
            torch.cuda.empty_cache()

    def _make_distill_loader(self, train_loader):
        # Keep train augmentation so chart/free adapters see augmented data,
        # but downstream code MUST cache batches so teacher/rollout share
        # identical inputs (see _distill_task).
        return torch.utils.data.DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=False,
            num_workers=getattr(train_loader, "num_workers", num_workers),
        )

    def _refresh_teacher_flow(self, loader):
        atlas_layers = self.args["gase_atlas_layers"]
        self._teacher_flow = TeacherFlowCache(layers=atlas_layers)
        for module in self._iter_atlas_modules():
            if module.layer_id in atlas_layers:
                module.atlas_layer.init_teacher_flow_cache(self._teacher_flow)
                module.atlas_layer.set_collect_enabled(True)
                module.atlas_layer.set_mode("task_train", self._cur_task)

        was_training = self._network.training
        self._network.eval()
        with torch.no_grad():
            for _, inputs, targets in loader:
                self._teacher_flow.record_labels(targets.detach().cpu())
                _ = self._network(inputs.to(self._device))

        for module in self._iter_atlas_modules():
            if module.layer_id in atlas_layers:
                module.atlas_layer.set_collect_enabled(False)
        self._network.train(was_training)
        logging.info("[V3TeacherFlowRefresh] task=%d samples=%d layers=%s ordered=True",
                     self._cur_task, self._teacher_flow.total_samples, atlas_layers)

    def _refresh_teacher_flow_cached(self, inputs_list, targets_list):
        """Same as _refresh_teacher_flow but uses pre-cached augmented batches."""
        atlas_layers = self.args["gase_atlas_layers"]
        self._teacher_flow = TeacherFlowCache(layers=atlas_layers)
        for module in self._iter_atlas_modules():
            if module.layer_id in atlas_layers:
                module.atlas_layer.init_teacher_flow_cache(self._teacher_flow)
                module.atlas_layer.set_collect_enabled(True)
                module.atlas_layer.set_mode("task_train", self._cur_task)

        was_training = self._network.training
        self._network.eval()
        with torch.no_grad():
            for inputs, targets in zip(inputs_list, targets_list):
                self._teacher_flow.record_labels(targets.detach().cpu())
                _ = self._network(inputs.to(self._device))
                del inputs, targets

        for module in self._iter_atlas_modules():
            if module.layer_id in atlas_layers:
                module.atlas_layer.set_collect_enabled(False)
        self._network.train(was_training)
        logging.info("[V3TeacherFlowRefresh] task=%d samples=%d layers=%s (cached)",
                     self._cur_task, self._teacher_flow.total_samples, atlas_layers)

    def _record_teacher_cached(self, inputs_list):
        """Record teacher block-output CLS using cached augmented batches."""
        atlas_layers = self.args["gase_atlas_layers"]
        was_training = self._network.training
        self._network.eval()
        outputs = {int(lid): [] for lid in atlas_layers}

        with torch.no_grad():
            for inputs in inputs_list:
                x = inputs.to(self._device)
                x = self._network.backbone.patch_embed(x)
                cls_token = self._network.backbone.cls_token.expand(x.shape[0], -1, -1)
                x = torch.cat((cls_token, x), dim=1)
                if hasattr(self._network.backbone, "pos_drop"):
                    x = self._network.backbone.pos_drop(x + self._network.backbone.pos_embed)
                else:
                    x = x + self._network.backbone.pos_embed
                for i, block in enumerate(self._network.backbone.blocks):
                    blk_out = block(x)
                    x = blk_out["blk_out"] if isinstance(blk_out, dict) else blk_out
                    if i in outputs:
                        outputs[i].append(x[:, 0].detach().cpu())
                del inputs, x

        self._network.train(was_training)
        return {lid: torch.cat(buf, dim=0) for lid, buf in outputs.items() if buf}

    def _rollout_cached(self, inputs_list, active_layers):
        """Rollout using cached augmented batches with already-distilled layers active."""
        from models.gase_atlas_v3.rollout import set_rollout_active_layers, restore_rollout_flags

        was_training = self._network.training
        self._network.eval()
        previous = set_rollout_active_layers(self._network, set(active_layers))
        pre_features = {}
        block_outputs = {}

        try:
            with torch.no_grad():
                for inputs in inputs_list:
                    x = inputs.to(self._device)
                    x = self._network.backbone.patch_embed(x)
                    cls_token = self._network.backbone.cls_token.expand(x.shape[0], -1, -1)
                    x = torch.cat((cls_token, x), dim=1)
                    if hasattr(self._network.backbone, "pos_drop"):
                        x = self._network.backbone.pos_drop(x + self._network.backbone.pos_embed)
                    else:
                        x = x + self._network.backbone.pos_embed
                    for i, block in enumerate(self._network.backbone.blocks):
                        x = x + block.drop_path(block.attn(block.norm1(x)))
                        if i in (9, 10, 11):
                            pre_features.setdefault(i, []).append(x[:, 0].detach().cpu())

                        adapter_out = block.adapter_module(x)
                        adapt_x = adapter_out["func_out"]
                        residual = x
                        mlp_out = block.mlp_drop(block.act(block.fc1(block.norm2(x))))
                        mlp_out = block.drop_path(block.mlp_drop(block.fc2(mlp_out)))

                        if block.config.ffn_option == "parallel":
                            x = residual + mlp_out + adapt_x
                        elif block.config.ffn_option == "sequential":
                            x = residual + block.adapter_module(mlp_out)["func_out"]
                        else:
                            x = residual + mlp_out

                        if i in (9, 10, 11):
                            block_outputs.setdefault(i, []).append(x[:, 0].detach().cpu())
                    del inputs, x
        finally:
            restore_rollout_flags(self._network, previous)
            self._network.train(was_training)

        return {
            "pre": {lid: torch.cat(buf, dim=0) for lid, buf in pre_features.items() if buf},
            "block": {lid: torch.cat(buf, dim=0) for lid, buf in block_outputs.items() if buf},
        }

    def _build_slots_for_new_charts(self, layer, charts, adapter_builder, features, residuals, layer_id):
        count = 0
        for chart in charts:
            if chart.status != "active":
                continue
            chart.to(features.device)
            assigned = chart.within_radius(features)
            idx = torch.where(assigned)[0]
            if idx.numel() < 2:
                chart.mark_inactive()
                chart.cpu()
                continue
            adapter = adapter_builder.build(chart, features[idx], residuals[idx])
            if adapter is None:
                chart.cpu()
                chart.mark_inactive()
                continue
            adapter = self._prepare_chart_adapter(chart, adapter, features[idx], residuals[idx], layer_id)
            chart.cpu()
            chart.attach_adapter(adapter, task_id=self._cur_task)
            layer.chart_adapters.append(adapter)
            chart.promote_to_adapter_initialized()
            chart.promote_to_distilled()
            chart.mark_used(self._cur_task)
            self._log_chart_adapter(layer_id, chart, adapter)
            count += 1
        return count

    def _build_slots_for_existing_charts(self, layer, update_pairs, adapter_builder,
                                         features_fit, residuals_fit, layer_id):
        """Add current-task adapter slots on reused charts; old slots are not overwritten."""
        count = 0
        for chart, mask in update_pairs:
            idx = torch.where(mask)[0]
            if idx.numel() < 2:
                continue
            chart.to(features_fit.device)
            adapter = adapter_builder.build(chart, features_fit[idx], residuals_fit[idx])
            if adapter is None:
                chart.cpu()
                continue
            adapter = self._prepare_chart_adapter(chart, adapter, features_fit[idx], residuals_fit[idx], layer_id)
            chart.cpu()
            chart.attach_adapter(adapter, task_id=self._cur_task)
            layer.chart_adapters.append(adapter)
            chart.promote_to_adapter_initialized()
            chart.promote_to_distilled()
            chart.mark_used(self._cur_task)
            self._log_chart_adapter(layer_id, chart, adapter)
            count += 1
        return count

    def _log_atlas_diagnostics(self):
        """Log lightweight V3 diagnostics without changing the training path.

        TeacherFit measures imitation of the temporary task adapter. ResidualScale,
        AtlasRoute, and AtlasAdapt are the atlas-facing diagnostics we care about
        when V3 is used as a modularized V2.
        """
        if not hasattr(self, "_teacher_flow"):
            return
        modules = self._iter_atlas_modules_dict()
        task_id = self._cur_task
        for layer_id in self.args["gase_atlas_layers"]:
            module = modules.get(layer_id)
            if module is None:
                continue
            layer = module.atlas_layer
            charts = layer.atlas.active_charts()
            total_slots = sum(c.num_adapters for c in charts)
            slot_layout = ",".join(
                f"c{int(c.chart_id)}:{sorted(int(k) for k in c._adapters.keys())}"
                for c in charts
            )

            if not self._teacher_flow.has_records(layer_id):
                reason = "l11_identity" if getattr(layer, "l11_identity", False) and int(layer_id) == 11 else "no_teacher_flow"
                logging.info("[V3AtlasDiag:T%d:L%d] skipped: %s", task_id, layer_id, reason)
                continue

            try:
                flow = self._teacher_flow.stack(layer_id)
            except Exception as exc:
                logging.info("[V3AtlasDiag:T%d:L%d] ERROR: %s", task_id, layer_id, str(exc))
                continue
            n_metric = min(int(flow.h_pre.shape[0]), 256)
            if n_metric <= 0:
                continue

            device = next(layer.free_adapter.parameters()).device
            feats = flow.h_pre[:n_metric].to(device)
            residuals = flow.delta_task[:n_metric].float().to(device)

            with torch.no_grad():
                chart_delta = layer.compute_chart_delta_features(feats)
                free_delta = layer.free_adapter(feats)
                combined = layer.compute_inference_delta_features(feats)

                task_norm = residuals.norm(dim=-1).mean().item()
                chart_norm = chart_delta.norm(dim=-1).mean().item()
                free_norm = free_delta.norm(dim=-1).mean().item()
                combined_norm = combined.norm(dim=-1).mean().item()
                input_norm = feats.norm(dim=-1).mean().item()

                chart_mse = (chart_delta - residuals).pow(2).mean().item()
                combined_mse = (combined - residuals).pow(2).mean().item()
                chart_cos = torch.cosine_similarity(chart_delta, residuals, dim=-1, eps=1e-8).mean().item()
                combined_cos = torch.cosine_similarity(combined, residuals, dim=-1, eps=1e-8).mean().item()

            logging.info(
                "[V3AtlasState:T%d:L%d] charts=%d slots=%d slot_tasks=[%s] free_gate=%d/%d",
                task_id, layer_id, len(charts), total_slots, slot_layout,
                int(self._free_trained.get(layer_id, 0)), int(flow.h_pre.shape[0]),
            )
            logging.info(
                "[V3TeacherFit:T%d:L%d] chart_mse=%.4f infer_mse=%.4f "
                "chart_cos=%.3f infer_cos=%.3f",
                task_id, layer_id, chart_mse, combined_mse, chart_cos, combined_cos,
            )
            logging.info(
                "[V3ResidualScale:T%d:L%d] input=%.2f task=%.2f chart=%.2f free=%.2f "
                "infer=%.2f ratio=[chart/task=%.2f infer/task=%.2f infer/input=%.3f]",
                task_id, layer_id, input_norm, task_norm, chart_norm, free_norm, combined_norm,
                chart_norm / max(task_norm, 1e-6),
                combined_norm / max(task_norm, 1e-6),
                combined_norm / max(input_norm, 1e-6),
            )
            self._log_route_diagnostics(task_id, layer_id, layer, feats)
            self._log_adapt_diagnostics(task_id, layer_id, feats, combined, flow.labels[:n_metric])

            del feats, residuals, chart_delta, free_delta, combined
            gc.collect()
            torch.cuda.empty_cache()

    def _log_route_diagnostics(self, task_id, layer_id, layer, feats):
        charts = layer.atlas.active_charts()
        if not charts:
            logging.info(
                "[V3AtlasRoute:T%d:L%d] entropy=0.000 margin=inf uncertain=0.000 "
                "free=1.000 routed=0/%d reasons=[no_charts=%d] top1=[]",
                task_id, layer_id, feats.shape[0], feats.shape[0],
            )
            return

        reasons = {}
        top_hist = {}
        slot_hist = {}  # (chart_id, slot_task_id) → count
        entropies = []
        margins = []
        uncertain = 0
        free = 0
        free_mix = 0
        with torch.no_grad():
            for i in range(feats.shape[0]):
                decision = layer.policy.decide(
                    feats[i:i + 1],
                    charts,
                    layer_id,
                    top_k=layer.top_k,
                )
                reasons[decision.reason] = reasons.get(decision.reason, 0) + 1
                entropies.append(float(decision.entropy))
                if decision.margin < float("inf"):
                    margins.append(float(decision.margin))
                if decision.reason == "mixed_uncertain":
                    uncertain += 1
                if decision.use_free:
                    free += 1
                elif decision.use_chart and decision.candidates:
                    weights = decision.weights
                    if weights is None or weights.numel() != len(decision.candidates):
                        weights = feats.new_ones(len(decision.candidates)) / len(decision.candidates)
                    quality = 0.0
                    radius_ratio = 0.0
                    for weight, cand in zip(weights, decision.candidates):
                        w = float(weight.item())
                        quality += w * max(cand.full_r2, cand.subspace_r2)
                        radius_ratio += w * cand.radius_ratio
                    if layer._free_gamma(quality, radius_ratio) > 0.0:
                        free_mix += 1
                if decision.candidate is not None:
                    cid = int(decision.candidate.chart_id)
                    top_hist[cid] = top_hist.get(cid, 0) + 1
                    sid = int(decision.candidate.slot_id)
                    slot_hist[(cid, sid)] = slot_hist.get((cid, sid), 0) + 1

        n = max(int(feats.shape[0]), 1)
        reason_str = " ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        top_str = " ".join(f"c{k}={v}" for k, v in sorted(top_hist.items()))
        slot_str = " ".join(f"c{cid}_s{sid}={v}" for (cid, sid), v in sorted(slot_hist.items()))
        logging.info(
            "[V3AtlasRoute:T%d:L%d] entropy=%.3f margin=%.3f uncertain=%.3f "
            "free=%.3f free_mix=%.3f routed=%d/%d reasons=[%s] top1=[%s] slots=[%s]",
            task_id, layer_id,
            sum(entropies) / n,
            sum(margins) / max(len(margins), 1) if margins else float("inf"),
            uncertain / n,
            free / n,
            free_mix / n,
            n - free,
            n,
            reason_str,
            top_str,
            slot_str,
        )

    def _log_adapt_diagnostics(self, task_id, layer_id, feats, combined_delta, labels_cpu):
        if self._total_classes < 2:
            return
        labels = labels_cpu.to(feats.device)
        if labels.numel() != feats.shape[0]:
            labels = None
        with torch.no_grad():
            adapted = feats + combined_delta
            fc_out = self._network.fc(adapted)
            logits = fc_out["logits"] if isinstance(fc_out, dict) else fc_out
            logits = logits[:, :self._total_classes]
            top2 = logits.topk(2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
            margin_mean = margins.mean().item()
            k = max(1, int(0.05 * margins.numel()))
            margin_p05 = margins.kthvalue(k).values.item()
            old_margin = 0.0
            new_margin = 0.0
            if labels is not None:
                old_mask = labels < self._known_classes
                new_mask = labels >= self._known_classes
                if old_mask.any():
                    old_margin = margins[old_mask].mean().item()
                if new_mask.any():
                    new_margin = margins[new_mask].mean().item()

        logging.info(
            "[V3AtlasAdapt:T%d:L%d] margin=%.3f p05=%.3f old_margin=%.3f new_margin=%.3f",
            task_id, layer_id, margin_mean, margin_p05, old_margin, new_margin,
        )

    def _log_chart_adapter(self, layer_id, chart, adapter):
        adapter_type = "mlp" if isinstance(adapter, ChartMLPAdapter) else "linear"
        logging.info(
            "[V3ChartAdapter:L%d#%d] type=%s r_p=%d s=%d params=%d "
            "fR2=%.3f subR2=%.3f cos=%.3f norm=%.2f",
            layer_id,
            chart.chart_id,
            adapter_type,
            chart.tangent_dim,
            chart.residual_dim,
            adapter.trainable_param_count,
            float(getattr(adapter, "full_r2", getattr(chart, "full_r2", 0.0))),
            float(getattr(adapter, "subspace_r2", getattr(chart, "subspace_r2", 0.0))),
            float(getattr(adapter, "adapter_cos", 0.0)),
            float(getattr(adapter, "norm_ratio", 0.0)),
        )

    def _layer_adapter_safety(self, layer_id):
        gains = {9: 1.0, 10: 0.2, 11: 0.05}
        max_ratios = {9: 1.5, 10: 0.75, 11: 0.3}
        return gains.get(int(layer_id), 1.0), max_ratios.get(int(layer_id), 1.5)

    def _set_layer_adapter_safety(self, adapter, layer_id):
        gain, max_ratio = self._layer_adapter_safety(layer_id)
        if hasattr(adapter, "gain"):
            adapter.gain.fill_(gain)
        if hasattr(adapter, "max_delta_ratio"):
            adapter.max_delta_ratio.fill_(max_ratio)

    def _use_mlp_adapter(self, layer_id):
        base_type = self.args.get("chart_adapter_type", "projected_linear")
        layer_type = self.args.get(f"chart_adapter_type_l{int(layer_id)}", base_type)
        return "mlp" in str(layer_type).lower()

    def _prepare_chart_adapter(self, chart, adapter, features, residuals, layer_id):
        self._set_layer_adapter_safety(adapter, layer_id)
        linear_fR2 = float(getattr(adapter, "full_r2", getattr(chart, "full_r2", 0.0)))
        if not self._use_mlp_adapter(layer_id):
            return adapter

        gain, max_ratio = self._layer_adapter_safety(layer_id)
        init_scale = {9: 1.0, 10: 0.5, 11: 0.5}.get(int(layer_id), 1.0)
        p_basis = getattr(chart, "P_adapter", None)
        mlp = ChartMLPAdapter(
            chart,
            P=p_basis,
            hidden=self.args.get("chart_adapter_hidden", 16),
            init_scale=init_scale,
            gain=gain,
            max_delta_ratio=max_ratio,
        )
        if all(hasattr(adapter, name) for name in ("key_geom", "key_adapt", "key_resid")):
            mlp.set_keys(adapter.key_geom, adapter.key_adapt, adapter.key_resid)
        for name in ("support", "full_r2", "subspace_r2", "adapter_cos", "norm_ratio", "masked"):
            if hasattr(adapter, name):
                setattr(mlp, name, getattr(adapter, name))

        tune_epochs = int(self.args.get("chart_adapter_tune_epochs", 0))
        if tune_epochs > 0:
            device = features.device
            chart.to(device)
            mlp = mlp.to(device)
            mlp.train()
            opt = torch.optim.Adam(
                mlp.parameters(),
                lr=float(self.args.get("chart_adapter_tune_lr", 1e-4)),
            )
            for _ in range(tune_epochs):
                pred = mlp(features, chart)
                loss = F.mse_loss(pred, residuals)
                opt.zero_grad()
                loss.backward()
                opt.step()
            mlp.eval()
            mlp = mlp.cpu()
            chart.cpu()
            del opt
        device = features.device
        chart.to(device)
        mlp = mlp.to(device)
        mlp.eval()
        with torch.no_grad():
            pred = mlp(features, chart)
            ss_res = (residuals - pred).pow(2).sum()
            ss_tot = residuals.pow(2).sum().clamp_min(1e-8)
            cos = torch.cosine_similarity(pred, residuals, dim=-1, eps=1e-8).mean().item()
            pred_norm = pred.norm(dim=-1).mean().item()
            resid_norm = residuals.norm(dim=-1).mean().item()
        mlp = mlp.cpu()
        chart.cpu()
        logging.info(
            "[V3ChartAdapter:L%d#%d] linear_fR2=%.3f mlp_fR2_check=%.3f gain=%.2f",
            int(layer_id), int(chart.chart_id), linear_fR2, float(mlp.full_r2),
            float(gain),
        )
        return mlp

    def _compute_chart_delta_for_layer(self, layer, features):
        with torch.no_grad():
            return layer.compute_chart_delta_features(features)

    def _deactivate_empty_charts(self, layer):
        for chart in layer.atlas.active_charts():
            if chart.num_adapters == 0:
                chart.mark_inactive()

    def _chart_growth_plan(self, layer, existing):
        """Choose whether V3 grows new charts continually or caps total charts."""
        mode = str(self.args.get("chart_growth_mode", "continual")).lower()
        max_new = int(self.args.get(
            "chart_max_new_per_task",
            self.args.get("chart_max_per_layer", 4),
        ))
        total_cap = int(self.args.get("chart_max_total_per_layer", 0))

        if mode in ("bounded", "legacy", "slot"):
            total_cap = int(self.args.get("chart_max_per_layer", total_cap or 4))
            budget = max(0, total_cap - len(layer.atlas.charts))
            existing_for_build = existing if existing else None
            return mode, budget, existing_for_build

        budget = max(0, max_new)
        if total_cap > 0:
            budget = min(budget, max(0, total_cap - len(layer.atlas.charts)))

        if mode in ("hybrid", "continual_with_slots", "continual"):
            existing_for_build = existing if existing else None
        else:
            existing_for_build = None
        return mode, budget, existing_for_build

    def _current_task_local_charts(self, new_charts, update_pairs):
        """Charts whose current-task slots should be excluded from free target."""
        charts = []
        seen = set()
        for chart in list(new_charts) + [pair[0] for pair in update_pairs]:
            key = id(chart)
            if key in seen:
                continue
            seen.add(key)
            charts.append(chart)
        return charts

    def _train_free_adapter_for_layer(self, layer, features, residuals):
        if features.shape[0] < 2:
            return
        free_adapter = layer.free_adapter
        prev_state = {k: v.detach().clone() for k, v in free_adapter.state_dict().items()}
        for p in free_adapter.parameters():
            p.requires_grad = True
        free_adapter.train()
        opt = torch.optim.Adam(free_adapter.parameters(), lr=self.args["free_adapter_lr"])
        batch_size = min(256, features.shape[0])
        l2_weight = float(self.args.get("free_adapter_l2_to_prev", 1e-3))

        for epoch in range(self.args["free_adapter_epochs"]):
            perm = torch.randperm(features.shape[0], device=features.device)
            total = 0.0
            steps = 0
            for start in range(0, features.shape[0], batch_size):
                idx = perm[start:start + batch_size]
                pred = free_adapter(features[idx])
                mse = F.mse_loss(pred, residuals[idx])
                l2 = torch.zeros((), device=features.device)
                if l2_weight > 0:
                    denom = 0
                    for name, param in free_adapter.named_parameters():
                        if name in prev_state:
                            l2 = l2 + (param - prev_state[name]).pow(2).sum()
                            denom += param.numel()
                    l2 = l2 / max(denom, 1)
                loss = mse + l2_weight * l2
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += mse.item()
                steps += 1
            logging.info("[V3Free:L%d] epoch=%d/%d mse=%.4f samples=%d",
                         layer.layer_id, epoch + 1, self.args["free_adapter_epochs"],
                         total / max(steps, 1), features.shape[0])
        for p in free_adapter.parameters():
            p.requires_grad = False
        free_adapter.eval()
        del opt

    def _imprint_classifier(self, train_loader):
        ctype = self.args.get("gase_atlas_classifier", "").lower()
        if ctype == "prototype":
            if isinstance(self._network.fc, ProtoCalibratedCosine):
                self._update_class_prototypes(train_loader)
            return
        if ctype != "cosine_imprint":
            return
        if not isinstance(self._network.fc, CosineLinear):
            return

        alpha = float(self.args.get("imprint_alpha", 0.7))
        lo, hi = self._known_classes, self._total_classes
        sums = {}
        counts = {}
        was_training = self._network.training
        self._network.eval()
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                out = self._network(inputs.to(self._device))
                feats = F.normalize(out["features"], dim=1)
                for cls in range(lo, hi):
                    mask = targets.to(self._device) == cls
                    if mask.any():
                        sums[cls] = sums.get(cls, torch.zeros_like(feats[0])) + feats[mask].sum(dim=0)
                        counts[cls] = counts.get(cls, 0) + int(mask.sum().item())
            for cls, vec in sums.items():
                mean = F.normalize(vec / max(counts[cls], 1), dim=0)
                old = F.normalize(self._network.fc.weight.data[cls], dim=0)
                self._network.fc.weight.data[cls] = F.normalize((1 - alpha) * old + alpha * mean, dim=0)
        self._network.train(was_training)
        logging.info("[V3Imprint] updated cosine weights for classes %d-%d", lo, hi)

    def _calibrate_classifier(self, train_loader):
        """Finetune only the classifier on post-distillation atlas features."""
        epochs = int(self.args.get("classifier_calibration_epochs", 0))
        if epochs <= 0 or self._network.fc is None:
            return

        lo, hi = self._known_classes, self._total_classes
        if hi <= lo:
            return

        was_training = self._network.training
        previous_requires_grad = {
            name: param.requires_grad
            for name, param in self._network.named_parameters()
        }

        for param in self._network.parameters():
            param.requires_grad = False
        for param in self._network.fc.parameters():
            param.requires_grad = True

        params = [p for p in self._network.fc.parameters() if p.requires_grad]
        if not params:
            for name, param in self._network.named_parameters():
                param.requires_grad = previous_requires_grad.get(name, param.requires_grad)
            return

        lr = float(self.args.get("classifier_calibration_lr", self.args["init_lr"]))
        weight_decay = float(self.args.get(
            "classifier_calibration_weight_decay",
            self.args.get("weight_decay", 0.0),
        ))
        opt_name = str(self.args.get("classifier_calibration_optimizer", self.args["optimizer"])).lower()
        if opt_name == "adam":
            optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif opt_name == "sgd":
            optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(opt_name)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(self.args.get("classifier_calibration_min_lr", 0.0)),
        )

        self._network.eval()
        self._network.fc.train()
        for epoch in range(epochs):
            loss_sum, correct, total = 0.0, 0, 0
            for _, inputs, targets in train_loader:
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)
                mask = (targets >= lo) & (targets < hi)
                if not mask.any():
                    continue

                logits = self._network(inputs)["logits"][:, lo:hi]
                logits = logits[mask]
                local_targets = targets[mask] - lo
                loss = F.cross_entropy(logits, local_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.item()
                preds = logits.argmax(dim=1)
                correct += preds.eq(local_targets).sum().item()
                total += int(local_targets.numel())

            scheduler.step()
            logging.info(
                "[V3ClassifierCalib] task=%d epoch=%d/%d loss=%.4f acc=%.2f classes=%d-%d",
                self._cur_task,
                epoch + 1,
                epochs,
                loss_sum / max(len(train_loader), 1),
                100.0 * correct / max(total, 1),
                lo,
                hi,
            )

        for name, param in self._network.named_parameters():
            param.requires_grad = previous_requires_grad.get(name, param.requires_grad)
        self._network.train(was_training)
        del optimizer
        del scheduler

    def _balanced_budget_indices(self, labels, n):
        budget = int(self.args.get("chart_build_budget", 512))
        if budget <= 0 or n <= budget:
            return torch.arange(n)
        if labels is None or labels.numel() != n:
            return torch.randperm(n)[:budget]
        selected = []
        unique = labels.unique()
        per_class = max(1, budget // max(int(unique.numel()), 1))
        for cls in unique:
            idx = torch.where(labels == cls)[0]
            if idx.numel() == 0:
                continue
            perm = idx[torch.randperm(idx.numel())[:per_class]]
            selected.append(perm)
        result = torch.cat(selected) if selected else torch.randperm(n)[:budget]
        if result.numel() < budget:
            rest = torch.tensor(
                list(set(range(n)) - set(result.tolist())),
                dtype=torch.long,
            )
            if rest.numel() > 0:
                extra = rest[torch.randperm(rest.numel())[:budget - result.numel()]]
                result = torch.cat([result, extra])
        return result[:budget]

    def _make_chart_builder(self):
        use_v2 = self.args.get("chart_builder_v2", True)
        if use_v2:
            from models.gase_atlas_v3.chart_builder_v2 import PPCAChartBuilderV2
            return PPCAChartBuilderV2(
                max_charts=self.args["chart_max_per_layer"],
                min_samples=self.args["chart_min_samples"],
                seed_sample_size=self.args["chart_seed_sample_size"],
                fit_sample_size=self.args["chart_fit_sample_size"],
                knn_size=self.args["chart_knn_size"],
                pca_energy=self.args["chart_pca_energy"],
                dim_min=self.args["chart_dim_min"],
                dim_max=self.args["chart_dim_max"],
                radius_quantile=self.args["chart_radius_quantile"],
                radius_scale=self.args["chart_radius_scale"],
                max_support_ratio=self.args["chart_max_support_ratio"],
                quality_active=self.args["chart_quality_active"],
                quality_candidate=self.args["chart_quality_candidate"],
                quality_mode=self.args["chart_quality_mode"],
                overlap_max=self.args["chart_overlap_max"],
                rec_error_scale=self.args["chart_rec_error_scale"],
                grassmann_tau=self.args["chart_grassmann_tau"],
                force_one_debug=self.args.get("chart_force_one_debug", False),
                standardize_features=self.args.get("chart_feature_standardize", False),
            )
        return PPCAChartBuilder(
            max_charts=self.args["chart_max_per_layer"],
            min_samples=self.args["chart_min_samples"],
            seed_sample_size=self.args["chart_seed_sample_size"],
            fit_sample_size=self.args["chart_fit_sample_size"],
            knn_size=self.args["chart_knn_size"],
            pca_energy=self.args["chart_pca_energy"],
            dim_min=self.args["chart_dim_min"],
            dim_max=self.args["chart_dim_max"],
            radius_quantile=self.args["chart_radius_quantile"],
            radius_scale=self.args["chart_radius_scale"],
            max_support_ratio=self.args["chart_max_support_ratio"],
            quality_active=self.args["chart_quality_active"],
            quality_candidate=self.args["chart_quality_candidate"],
            quality_mode=self.args["chart_quality_mode"],
            overlap_max=self.args["chart_overlap_max"],
            rec_error_scale=self.args["chart_rec_error_scale"],
            grassmann_tau=self.args["chart_grassmann_tau"],
            force_one_debug=self.args.get("chart_force_one_debug", False),
            standardize_features=self.args.get("chart_feature_standardize", True),
        )

    def _make_adapter_builder(self):
        return RidgeChartAdapterBuilder(
            residual_energy=self.args["chart_residual_energy"],
            residual_dim_min=self.args["chart_residual_dim_min"],
            residual_dim_max=self.args["chart_residual_dim_max"],
            ridge_lambda=self.args["chart_adapter_ridge_lambda"],
            tune_epochs=self.args.get("chart_adapter_tune_epochs", 0),
            cos_weight=self.args.get("chart_adapter_cos_weight", 0.1),
        )

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        params = [
            p for name, p in self._network.named_parameters()
            if ("task_adapter" in name or "free_adapter" in name or "fc." in name)
            and p.requires_grad
        ]
        if self.args["optimizer"] == "adam":
            self.optimizer = optim.AdamW(params, lr=lr, weight_decay=self.args["weight_decay"])
        elif self.args["optimizer"] == "sgd":
            self.optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay=self.args["weight_decay"])
        else:
            raise ValueError(self.args["optimizer"])
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=num_epoch,
            eta_min=self.args.get("min_lr", 0.0),
        )

    def _build_classifier(self, in_dim, out_dim):
        classifier = AtlasClassifier(
            in_dim,
            out_dim,
            classifier_type=self.args.get("gase_atlas_classifier", "cosine_imprint").lower(),
            cosine_scale=float(self.args.get("gase_atlas_cosine_scale", 24.0)),
            prototype_alpha=float(self.args.get("gase_atlas_prototype_alpha", 0.8)),
            prototype_mode=self.args.get("gase_atlas_prototype_mode", "add"),
        )
        return classifier.fc

    def _compute_accuracy_current_task(self, model, loader):
        model.eval()
        correct, total = 0, 0
        lo, hi = self._known_classes, self._total_classes
        for _, inputs, targets in loader:
            with torch.no_grad():
                outputs = model(inputs.to(self._device))["logits"][:, lo:hi]
            preds = outputs.argmax(dim=1).cpu()
            mask = (targets >= lo) & (targets < hi)
            if mask.any():
                correct += preds[mask].eq(targets[mask] - lo).sum().item()
                total += int(mask.sum().item())
        return np.around(correct * 100 / max(total, 1), decimals=2)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for _, inputs, targets in loader:
            with torch.no_grad():
                outputs = model(inputs.to(self._device))["logits"][:, :self._total_classes]
            preds = outputs.argmax(dim=1).cpu()
            correct += preds.eq(targets).sum()
            total += targets.numel()
        return np.around(tensor2numpy(correct) * 100 / max(total, 1), decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, inputs, targets in loader:
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _update_class_prototypes(self, train_loader):
        if not isinstance(self._network.fc, ProtoCalibratedCosine):
            return
        was_training = self._network.training
        self._network.eval()
        with torch.no_grad():
            for _, inputs, targets in train_loader:
                out = self._network(inputs.to(self._device))
                self._network.fc.update_prototypes(out["features"], targets.to(self._device))
        self._network.train(was_training)

    def _iter_atlas_modules(self):
        for module in self._network.modules():
            if _is_atlas_module(module):
                yield module

    def _iter_atlas_modules_dict(self):
        return {module.layer_id: module for module in self._iter_atlas_modules()}

    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {k: v for k, v in state_dict.items() if "task_adapter" not in k}
        torch.save(save_dict, f"{filename}.pth")

    def load_checkpoint(self, filename):
        self._network.load_state_dict(torch.load(filename), strict=False)

    def after_task(self):
        self._known_classes = self._total_classes
