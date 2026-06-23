"""Sequential rollout helpers for GASE-Atlas v3."""

import logging

import torch


def _is_v3_block(module):
    return type(module).__name__ == "GASEAtlasBlockModuleV3"


def iter_atlas_modules(model):
    for module in model.modules():
        if _is_v3_block(module):
            yield module


def set_rollout_active_layers(model, active_layers):
    modules = list(iter_atlas_modules(model))
    previous = {m.layer_id: getattr(m, "_no_adapter", False) for m in modules}
    for module in modules:
        module._no_adapter = module.layer_id not in active_layers
    return previous


def restore_rollout_flags(model, previous):
    for module in iter_atlas_modules(model):
        if module.layer_id in previous:
            module._no_adapter = previous[module.layer_id]


def record_teacher_outputs(model, loader, device, atlas_layers):
    """Record teacher block-output CLS tokens at chart layers.

    During teacher training the task adapters are active, so these outputs are
    the sequential targets V3 should chase after each layer is distilled.
    """
    was_training = model.training
    model.eval()
    outputs = {int(lid): [] for lid in atlas_layers}

    with torch.no_grad():
        for _, inputs, _ in loader:
            x = _embed_inputs(model, inputs.to(device))
            for i, block in enumerate(model.backbone.blocks):
                blk_out = block(x)
                x = blk_out["blk_out"] if isinstance(blk_out, dict) else blk_out
                if i in outputs:
                    outputs[i].append(x[:, 0].detach().cpu())

    model.train(was_training)
    result = {lid: torch.cat(buf, dim=0) for lid, buf in outputs.items() if buf}
    logging.info("[V3TeacherRecord] layers=%s outputs=%s",
                 list(atlas_layers), {k: list(v.shape) for k, v in result.items()})
    return result


def rollout_forward(model, loader, device, active_layers):
    """Roll out with already-distilled layers active and later layers identity.

    ``active_layers`` controls which already-distilled GASE layers are applied.
    Other GASE layers are forced to identity during rollout.

    Returns:
      pre_features[lid]: CLS immediately before the adapter branch.
      block_outputs[lid]: CLS after the whole block.
    """
    was_training = model.training
    model.eval()
    previous = set_rollout_active_layers(model, set(active_layers))
    pre_features = {}
    block_outputs = {}

    try:
        with torch.no_grad():
            for _, inputs, _ in loader:
                x = _embed_inputs(model, inputs.to(device))
                for i, block in enumerate(model.backbone.blocks):
                    x = x + block.drop_path(block.attn(block.norm1(x)))
                    if i in (9, 10, 11):
                        pre_features.setdefault(i, []).append(x[:, 0].detach().cpu())

                    adapter_out = block.adapter_module(x)
                    adapt_x = adapter_out["func_out"]
                    residual = x
                    mlp_out = block.mlp_drop(block.act(block.fc1(block.norm2(x))))
                    mlp_out = block.drop_path(block.mlp_drop(block.fc2(mlp_out)))

                    if block.config.ffn_option == "sequential":
                        seq_out = block.adapter_module(mlp_out)
                        x = residual + seq_out["func_out"]
                    elif block.config.ffn_option == "parallel":
                        x = residual + mlp_out + adapt_x
                    else:
                        x = residual + mlp_out

                    if i in (9, 10, 11):
                        block_outputs.setdefault(i, []).append(x[:, 0].detach().cpu())
    finally:
        restore_rollout_flags(model, previous)
        model.train(was_training)

    pre_result = {lid: torch.cat(buf, dim=0) for lid, buf in pre_features.items() if buf}
    out_result = {lid: torch.cat(buf, dim=0) for lid, buf in block_outputs.items() if buf}
    logging.info("[V3Rollout] active=%s pre=%s block=%s",
                 sorted(active_layers),
                 {k: list(v.shape) for k, v in pre_result.items()},
                 {k: list(v.shape) for k, v in out_result.items()})
    return {"pre": pre_result, "block": out_result}


def collect_pre_adapter_features(model, loader, device, target_layers, active_layers):
    """Compatibility wrapper returning only pre-adapter features."""
    rolled = rollout_forward(model, loader, device, active_layers)
    pre = rolled["pre"]
    target_set = {int(lid) for lid in target_layers}
    return {lid: value for lid, value in pre.items() if lid in target_set}


def _embed_inputs(model, inputs):
    backbone = model.backbone
    x = backbone.patch_embed(inputs)
    cls_token = backbone.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    if hasattr(backbone, "pos_drop"):
        return backbone.pos_drop(x + backbone.pos_embed)
    return x + backbone.pos_embed
