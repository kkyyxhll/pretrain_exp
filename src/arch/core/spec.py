"""Qwen3.5 architecture specifications and OLMo configuration translation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from olmo_core.config import DType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer.config import (
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.nn.transformer.init import InitMethod

from .registry import ArchError, get_config

__all__ = [
    "LayerSpec",
    "ArchSpec",
    "pattern_to_layers",
    "spec_to_olmo",
    "build_model",
    "spec_num_params",
]

_FIELD_KEYS = {"attn", "ffn", "norm", "res", "placement", "window"}
_KWARGS_GROUPS = {"attn", "ffn", "norm"}


@dataclass
class LayerSpec:
    """一层 = 一个变体组合：attn/ffn/norm 的名字 + 残差缩放 + placement + window。"""

    attn: str = "full"
    ffn: str = "swiglu"
    norm: str = "rms"
    res_alpha: float = 1.0  # 残差缩放（1.0 = 恒等）
    placement: str = "pre"  # v1 仅 pre；post/peri 需自定义 block 类型
    window: Optional[int] = None  # None = 全局因果
    attn_kwargs: Dict[str, Any] = field(default_factory=dict)
    ffn_kwargs: Dict[str, Any] = field(default_factory=dict)
    norm_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.placement not in ("pre", "post"):
            raise ArchError(f"placement 仅支持 pre/post: {self.placement!r}")
        if self.window is not None and not isinstance(self.window, int):
            raise ArchError(f"window 必须为 None 或 int: {self.window!r}")
        if not isinstance(self.res_alpha, (int, float)) or self.res_alpha <= 0:
            raise ArchError(f"res 必须为正数: {self.res_alpha!r}")


@dataclass
class ArchSpec:
    """一份完整架构描述。layers 长度必须等于 n_layers。"""

    vocab_size: int
    d_model: int
    n_layers: int
    rope_theta: float = 10000.0
    dtype: str = "bfloat16"
    tie_word_embeddings: bool = True
    init_method: str = "normal"
    init_std: float = 0.02
    final_norm: str = "rms"
    final_norm_kwargs: Dict[str, Any] = field(default_factory=dict)
    layers: List[LayerSpec] = field(default_factory=list)

    def __post_init__(self):
        for k, v in (
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
        ):
            if not isinstance(v, int) or v <= 0:
                raise ArchError(f"{k} 必须为正整数: {v!r}")
        if self.layers and len(self.layers) != self.n_layers:
            raise ArchError(f"layers 数 {len(self.layers)} != n_layers {self.n_layers}")

    # -- 序列化 ------------------------------------------------------------
    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["layers"] = [dataclasses.asdict(layer) for layer in self.layers]
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict) -> "ArchSpec":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ArchError(f"ArchSpec 未知字段: {sorted(unknown)}")
        lknown = {f.name for f in dataclasses.fields(LayerSpec)}
        layers = []
        for i, ld in enumerate(d.get("layers", [])):
            lunknown = set(ld) - lknown
            if lunknown:
                raise ArchError(f"layers[{i}] 未知字段: {sorted(lunknown)}")
            layers.append(LayerSpec(**ld))
        return cls(**{k: v for k, v in d.items() if k != "layers"}, layers=layers)

    def hash(self) -> str:
        """spec 内容哈希（与 dict 顺序无关），用作实验 ID。"""
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# pattern DSL
# ---------------------------------------------------------------------------

_ASSIGN_RE = re.compile(r"^\s*(?P<assign>[^@*;]+?)\s*(?:\*\s*(?P<rep>\d+))?\s*$")
_RANGE_RE = re.compile(r"^\s*(?P<assign>[^@*;]+?)\s*@\s*(?P<targets>[\d\-,]+)\s*$")


def _parse_value(v: str) -> Any:
    v = v.strip()
    if v in ("none", "None", ""):
        return None
    if v in ("true", "True"):
        return True
    if v in ("false", "False"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _parse_assign(assign: str) -> Dict[str, Any]:
    """解析一条赋值 → {字段名: 值} ∪ {kwargs 组: {k: v}}。"""
    assign = assign.strip()
    if assign in ("", "default"):
        return {}
    out: Dict[str, Any] = {}
    for part in assign.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ArchError(f"赋值项缺少 '=': {part!r}（完整语句: {assign!r}）")
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        if "." in k:
            group, kk = k.split(".", 1)
            if group not in _KWARGS_GROUPS:
                raise ArchError(f"点号键仅支持 {sorted(_KWARGS_GROUPS)}: {k!r}")
            out.setdefault(f"{group}_kwargs", {})[kk] = _parse_value(v)
            continue
        if k not in _FIELD_KEYS:
            raise ArchError(
                f"未知赋值键 {k!r}，可用: {sorted(_FIELD_KEYS)} 或 {'/'.join(sorted(_KWARGS_GROUPS))}.<超参>"
            )
        if k == "res":
            out["res_alpha"] = float(v)
        elif k == "window":
            val = _parse_value(v)
            out[k] = None if val is None else int(val)
        else:
            out[k] = _parse_value(v)
    return out


def _merge(base: LayerSpec, over: Dict[str, Any]) -> LayerSpec:
    d = dataclasses.asdict(base)
    for k, v in over.items():
        if k.endswith("_kwargs"):
            d[k].update(v)
        else:
            d[k] = v
    return LayerSpec(**d)


def pattern_to_layers(pattern: str, n_layers: int) -> List[LayerSpec]:
    """把紧凑 pattern 展开为逐层 LayerSpec 列表，长度必须恰为 n_layers。"""
    base = LayerSpec()
    layers: List[Optional[LayerSpec]] = [None] * n_layers
    pos = 0
    for stmt in pattern.split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        m = _RANGE_RE.match(stmt)
        if m:
            idxs = []
            for part in m.group("targets").split(","):
                if "-" in part:
                    lo_s, hi_s = part.split("-", 1)
                    idxs.extend(range(int(lo_s), int(hi_s) + 1))
                else:
                    idxs.append(int(part))
            for i in idxs:
                if not 0 <= i < n_layers:
                    raise ArchError(f"索引 {i} 越界（n_layers={n_layers}）: {stmt!r}")
            over = _parse_assign(m.group("assign"))
            for i in idxs:
                prev = layers[i] if layers[i] is not None else base
                layers[i] = _merge(prev, over)
            continue
        m = _ASSIGN_RE.match(stmt)
        if not m:
            raise ArchError(f"pattern 语句无法解析: {stmt!r}")
        rep = int(m.group("rep")) if m.group("rep") else 1
        over = _parse_assign(m.group("assign"))
        for _ in range(rep):
            if pos >= n_layers:
                raise ArchError(f"pattern 展开超过 n_layers={n_layers}: {stmt!r}")
            layers[pos] = _merge(base, over)
            pos += 1
    if pos != n_layers and any(layer is None for layer in layers):
        raise ArchError(f"pattern 只覆盖 {pos}/{n_layers} 层")
    return [layer for layer in layers if layer is not None]


# ---------------------------------------------------------------------------
# → OLMo 翻译与构建
# ---------------------------------------------------------------------------


def _dtype(name: str) -> DType:
    try:
        return getattr(DType, name)
    except AttributeError:
        raise ArchError(f"未知 dtype: {name!r}，可选: float32/bfloat16")


def _init(name: str) -> InitMethod:
    try:
        return getattr(InitMethod, name)
    except AttributeError:
        raise ArchError(
            f"未知 init_method: {name!r}，可选: normal/llama/llama_depth/normalized/fan_in"
        )


def _cfg(group: str, name: str, kwargs: Dict[str, Any]):
    cls = get_config(group, name)
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise ArchError(f"{group}.{name} 配置构造失败（未知字段？）: {e}") from e


def _inject(kwargs: Dict[str, Any], group: str, name: str, key: str, value: Any) -> Dict[str, Any]:
    """把 spec 级默认值（dtype/rope_theta/window）注入变体 config——**仅当变体有该字段**。

    OLMo 各模块 config 的 dtype 默认 float32，不注入会与 embedding/lm_head 混型；
    rope_theta/window 同理只对声明了该字段的变体（如 full）注入。
    """
    if key in kwargs:
        return kwargs
    fields = getattr(get_config(group, name), "__dataclass_fields__", {})
    if key in fields:
        kwargs[key] = value
    return kwargs


def spec_to_olmo(spec: ArchSpec) -> TransformerConfig:
    block_by_key: Dict[str, str] = {}
    block_cfgs: List[TransformerBlockConfig] = []
    pattern: List[str] = []
    for i, layer in enumerate(spec.layers):
        if layer.placement != "pre":
            raise ArchError(
                f"placement={layer.placement!r} 尚未映射到 OLMo block 类型（v1 仅支持 pre）；"
                "自定义 placement 需写 TransformerConfig 子类 + 自定义 block，见 attn/GUIDE.md"
            )
        key = json.dumps(dataclasses.asdict(layer), sort_keys=True)
        if key not in block_by_key:
            attn_kwargs = _inject(
                dict(layer.attn_kwargs), "attn", layer.attn, "rope_theta", spec.rope_theta
            )
            if layer.window is not None and "window" in getattr(
                get_config("attn", layer.attn), "__dataclass_fields__", {}
            ):
                attn_kwargs.setdefault("window", layer.window)
            attn_cfg = _cfg(
                "attn", layer.attn, _inject(attn_kwargs, "attn", layer.attn, "dtype", spec.dtype)
            )
            ffn_cfg = _cfg(
                "ffn",
                layer.ffn,
                _inject(dict(layer.ffn_kwargs), "ffn", layer.ffn, "dtype", spec.dtype),
            )
            norm_cfg = _cfg(
                "norm",
                layer.norm,
                _inject(dict(layer.norm_kwargs), "norm", layer.norm, "dtype", spec.dtype),
            )
            name = f"b{len(block_cfgs)}"
            is_moe = bool(getattr(ffn_cfg, "is_moe", False))
            if is_moe and layer.res_alpha != 1.0:
                raise ArchError(
                    f"layers[{i}]: MoE block 不支持残差缩放（OLMo MoETransformerBlock 无 alpha 参数），"
                    "res_alpha 需为 1.0"
                )
            block_cfgs.append(
                TransformerBlockConfig(
                    sequence_mixer=attn_cfg,
                    layer_norm=norm_cfg,
                    feed_forward=None if is_moe else ffn_cfg,
                    feed_forward_moe=ffn_cfg if is_moe else None,
                    name=TransformerBlockType.moe if is_moe else TransformerBlockType.default,
                    **(
                        {}
                        if is_moe
                        else dict(
                            attention_residual_alpha=layer.res_alpha,
                            feed_forward_residual_alpha=layer.res_alpha,
                        )
                    ),
                )
            )
            block_by_key[key] = name
        pattern.append(block_by_key[key])

    final_norm = _cfg("norm", spec.final_norm, dict(spec.final_norm_kwargs))
    return TransformerConfig(
        d_model=spec.d_model,
        vocab_size=spec.vocab_size,
        n_layers=spec.n_layers,
        block={
            name: cfg for name, cfg in zip((f"b{i}" for i in range(len(block_cfgs))), block_cfgs)
        },
        block_pattern=pattern,
        lm_head=LMHeadConfig(
            layer_norm=final_norm,
            bias=False,
        ),
        dtype=_dtype(spec.dtype),
        init_method=_init(spec.init_method),
        init_std=spec.init_std,
        tie_word_embeddings=spec.tie_word_embeddings,
    )


def build_model(spec: ArchSpec, *, init_device: str = "cpu", checkpoint_blocks: bool = False):
    """构建真实模型（init_device="meta" 用于仅算 FLOPs）。"""
    model = spec_to_olmo(spec).build(init_device=init_device)
    if checkpoint_blocks:
        model.apply_activation_checkpointing("full")
    return model


def spec_num_params(spec: ArchSpec) -> int:
    tc = spec_to_olmo(spec)
    n = tc.num_params
    return n() if callable(n) else n
