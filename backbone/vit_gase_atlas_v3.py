"""GASE-Atlas v3 ViT builder -zero V1 dependency.

Uses GASEAtlasBlockModuleV3 directly, no monkey-patching, no V1 layer creation.
"""

from backbone import vit_sema as _vit_sema
from backbone.gase_atlas_block_v3 import GASEAtlasBlockModuleV3


Attention = _vit_sema.Attention
Block = _vit_sema.Block
VisionTransformer = _vit_sema.VisionTransformer


def _build_with_gase_atlas_v3(builder, *args, **kwargs):
    old_modules = _vit_sema.SEMAModules
    _vit_sema.SEMAModules = GASEAtlasBlockModuleV3
    try:
        return builder(*args, **kwargs)
    finally:
        _vit_sema.SEMAModules = old_modules


def vit_base_patch16_224_gase_atlas_v3(pretrained=False, pretrained_path=None, **kwargs):
    return _build_with_gase_atlas_v3(
        _vit_sema.vit_base_patch16_224_sema,
        pretrained=pretrained,
        **kwargs,
    )


def vit_base_patch16_224_in21k_gase_atlas_v3(pretrained=False, pretrained_path=None, **kwargs):
    return _build_with_gase_atlas_v3(
        _vit_sema.vit_base_patch16_224_in21k_sema,
        pretrained=pretrained,
        **kwargs,
    )

