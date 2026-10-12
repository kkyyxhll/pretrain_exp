"""Qwen3.5 specifications and concrete OLMo configuration translation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from olmo_core.config import DType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import Transformer
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig
from olmo_core.nn.transformer.init import InitMethod

from src.arch.attn.full import FullAttentionConfig
from src.arch.attn.qwen35_gdn import Qwen35GDNConfig
from src.arch.ffn.swiglu import SwigluConfig
from src.arch.norm.rms import RmsNormConfig

ConfigScalar = bool | int | float | str | None
ConfigOptions = dict[str, ConfigScalar]


LayerPayload = dict[str, ConfigScalar | ConfigOptions]
ArchPayload = dict[str, ConfigScalar | ConfigOptions | list[LayerPayload]]


class ArchError(Exception):
    """Invalid architecture fields or unsupported components."""


def config_options(payload: object) -> ConfigOptions:
    """Validate scalar configuration values at a JSON boundary."""
    if not isinstance(payload, dict):
        raise ArchError("component options must be a mapping")
    else:
        options: ConfigOptions = {}
        for option_name, option_value in payload.items():
            if not isinstance(option_name, str):
                raise ArchError("component option names must be strings")
            elif option_value is None or isinstance(option_value, (bool, int, float, str)):
                options[option_name] = option_value
            else:
                raise ArchError(f"invalid scalar component option: {option_name}")
        return options


@dataclass(kw_only=True)
class LayerSpec:
    """One layer's components and residual scaling, with stable serialized keys."""

    attn: Literal["full", "qwen35_gdn"] = "full"
    ffn: Literal["swiglu"] = "swiglu"
    norm: Literal["rms"] = "rms"
    res_alpha: float = 1.0
    placement: Literal["pre", "post"] = "pre"
    window: int | None = None
    attn_kwargs: ConfigOptions = field(default_factory=dict)
    ffn_kwargs: ConfigOptions = field(default_factory=dict)
    norm_kwargs: ConfigOptions = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.placement not in ("pre", "post"):
            raise ArchError(f"unsupported placement: {self.placement!r}")
        else:
            pass
        if self.window is not None and not isinstance(self.window, int):
            raise ArchError(f"window must be None or int: {self.window!r}")
        else:
            pass
        if not isinstance(self.res_alpha, (int, float)) or self.res_alpha <= 0:
            raise ArchError(f"res must be positive: {self.res_alpha!r}")
        else:
            pass

    @classmethod
    def from_dict(cls, serialized_layer: Mapping[str, object]) -> LayerSpec:
        unknown_fields = set(serialized_layer) - {entry.name for entry in dataclasses.fields(cls)}
        if unknown_fields:
            raise ArchError(f"unknown layer fields: {sorted(unknown_fields)}")
        else:
            scalar_values = config_options(
                {
                    name: value
                    for name, value in serialized_layer.items()
                    if not name.endswith("_kwargs")
                }
            )
            return cls(
                **scalar_values,
                attn_kwargs=config_options(serialized_layer.get("attn_kwargs", {})),
                ffn_kwargs=config_options(serialized_layer.get("ffn_kwargs", {})),
                norm_kwargs=config_options(serialized_layer.get("norm_kwargs", {})),
            )


@dataclass(kw_only=True)
class ArchSpec:
    """Complete architecture with stable serialized dimension and layer keys."""

    vocab_size: int
    d_model: int
    n_layers: int
    rope_theta: float = 10000.0
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    tie_word_embeddings: bool = True
    init_method: Literal["normal", "llama", "llama_depth", "normalized", "fan_in"] = "normal"
    init_std: float = 0.02
    final_norm: Literal["rms"] = "rms"
    final_norm_kwargs: ConfigOptions = field(default_factory=dict)
    layers: list[LayerSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        for dimension_name, dimension_count in (
            ("vocab_size", self.vocab_size),
            ("d_model", self.d_model),
            ("n_layers", self.n_layers),
        ):
            if not isinstance(dimension_count, int) or dimension_count <= 0:
                raise ArchError(f"{dimension_name} must be a positive integer: {dimension_count!r}")
            else:
                pass
        if self.layers and len(self.layers) != self.n_layers:
            raise ArchError(f"layer count {len(self.layers)} != n_layers {self.n_layers}")
        else:
            pass

    def to_dict(self) -> ArchPayload:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, serialized_spec: Mapping[str, object]) -> ArchSpec:
        unknown_fields = set(serialized_spec) - {entry.name for entry in dataclasses.fields(cls)}
        if unknown_fields:
            raise ArchError(f"unknown architecture fields: {sorted(unknown_fields)}")
        else:
            pass
        serialized_layers = serialized_spec.get("layers", [])
        if not isinstance(serialized_layers, list):
            raise ArchError("layers must be a list")
        else:
            layers: list[LayerSpec] = []
            for layer_index, serialized_layer in enumerate(serialized_layers):
                if not isinstance(serialized_layer, dict):
                    raise ArchError(f"layers[{layer_index}] must be a mapping")
                else:
                    layers.append(LayerSpec.from_dict(serialized_layer))
        scalar_values = config_options(
            {
                name: value
                for name, value in serialized_spec.items()
                if name not in ("layers", "final_norm_kwargs")
            }
        )
        return cls(
            **scalar_values,
            final_norm_kwargs=config_options(serialized_spec.get("final_norm_kwargs", {})),
            layers=layers,
        )

    def hash(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()[:16]


ASSIGNMENT_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?P<assign>[^@*;]+?)\s*(?:\*\s*(?P<rep>\d+))?\s*$"
)
LAYER_RANGE_PATTERN: re.Pattern[str] = re.compile(
    r"^\s*(?P<assign>[^@*;]+?)\s*@\s*(?P<targets>[\d\-,]+)\s*$"
)
LAYER_ASSIGNMENT_FIELDS: set[str] = {"attn", "ffn", "norm", "res", "placement", "window"}
COMPONENT_GROUPS: set[str] = {"attn", "ffn", "norm"}


def parse_value(value_text: str) -> ConfigScalar:
    value_text = value_text.strip()
    if value_text in ("none", "None", ""):
        return None
    elif value_text in ("true", "True"):
        return True
    elif value_text in ("false", "False"):
        return False
    else:
        try:
            return int(value_text)
        except ValueError:
            try:
                return float(value_text)
            except ValueError:
                return value_text


def assign_layer(base_layer: LayerSpec, assignment_text: str) -> LayerSpec:
    layer_fields: LayerPayload = dataclasses.asdict(base_layer)
    assignments = [] if assignment_text.strip() in ("", "default") else assignment_text.split(",")
    for assignment in assignments:
        assignment = assignment.strip()
        if not assignment:
            continue
        elif "=" not in assignment:
            raise ArchError(f"assignment is missing '=': {assignment!r}")
        else:
            field_name, value_text = (part.strip() for part in assignment.split("=", 1))
        if "." in field_name:
            component_group, option_name = field_name.split(".", 1)
            if component_group not in COMPONENT_GROUPS:
                raise ArchError(f"unsupported component group: {component_group!r}")
            else:
                layer_fields[f"{component_group}_kwargs"][option_name] = parse_value(value_text)
        elif field_name not in LAYER_ASSIGNMENT_FIELDS:
            raise ArchError(f"unknown assignment field: {field_name!r}")
        elif field_name == "res":
            layer_fields["res_alpha"] = float(value_text)
        elif field_name == "window":
            parsed_value = parse_value(value_text)
            layer_fields["window"] = None if parsed_value is None else int(parsed_value)
        else:
            layer_fields[field_name] = parse_value(value_text)
    return LayerSpec.from_dict(layer_fields)


def pattern_to_layers(pattern: str, n_layers: int) -> list[LayerSpec]:
    """Expand repeated component assignments and layer-index overrides."""
    base_layer = LayerSpec()
    layers: list[LayerSpec | None] = [None] * n_layers
    next_layer_index = 0
    for statement in pattern.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        else:
            pass
        range_match = LAYER_RANGE_PATTERN.match(statement)
        if range_match:
            layer_indices: list[int] = []
            for target in range_match.group("targets").split(","):
                if "-" in target:
                    first_layer, last_layer = target.split("-", 1)
                    layer_indices.extend(range(int(first_layer), int(last_layer) + 1))
                else:
                    layer_indices.append(int(target))
            for layer_index in layer_indices:
                if not 0 <= layer_index < n_layers:
                    raise ArchError(f"layer index {layer_index} exceeds n_layers={n_layers}")
                else:
                    previous_layer = layers[layer_index]
                    layers[layer_index] = assign_layer(
                        previous_layer if previous_layer is not None else base_layer,
                        range_match.group("assign"),
                    )
        else:
            assignment_match = ASSIGNMENT_PATTERN.match(statement)
            if assignment_match is None:
                raise ArchError(f"invalid architecture pattern statement: {statement!r}")
            else:
                repetitions = int(assignment_match.group("rep") or "1")
                for _ in range(repetitions):
                    if next_layer_index >= n_layers:
                        raise ArchError(f"pattern exceeds n_layers={n_layers}: {statement!r}")
                    else:
                        layers[next_layer_index] = assign_layer(
                            base_layer, assignment_match.group("assign")
                        )
                        next_layer_index += 1
    if next_layer_index != n_layers and any(layer is None for layer in layers):
        raise ArchError(f"pattern covers only {next_layer_index}/{n_layers} layers")
    else:
        return [layer for layer in layers if layer is not None]


def spec_to_olmo(spec: ArchSpec) -> TransformerConfig:
    block_names_by_layer: dict[str, str] = {}
    block_configs: dict[str, TransformerBlockConfig] = {}
    block_pattern: list[str] = []
    for layer in spec.layers:
        if layer.placement != "pre":
            raise ArchError(f"unsupported OLMo block placement: {layer.placement!r}")
        else:
            pass
        layer_identity = json.dumps(dataclasses.asdict(layer), sort_keys=True)
        if layer_identity in block_names_by_layer:
            block_pattern.append(block_names_by_layer[layer_identity])
        else:
            attention_options = dict(layer.attn_kwargs)
            attention_options.setdefault("dtype", spec.dtype)
            if layer.attn == "full":
                attention_options.setdefault("rope_theta", spec.rope_theta)
                if layer.window is not None:
                    attention_options.setdefault("window", layer.window)
                else:
                    pass
                try:
                    attention_config = FullAttentionConfig(**attention_options)
                except TypeError as error:
                    raise ArchError(f"full attention configuration failed: {error}") from error
            elif layer.attn == "qwen35_gdn":
                try:
                    attention_config = Qwen35GDNConfig(**attention_options)
                except TypeError as error:
                    raise ArchError(f"GDN configuration failed: {error}") from error
            else:
                raise ArchError(f"unsupported attention: {layer.attn!r}")
            if layer.ffn != "swiglu" or layer.norm != "rms":
                raise ArchError(f"unsupported feed-forward or norm: {layer.ffn!r}, {layer.norm!r}")
            else:
                feed_forward_options = dict(layer.ffn_kwargs)
                feed_forward_options.setdefault("dtype", spec.dtype)
                norm_options = dict(layer.norm_kwargs)
                norm_options.setdefault("dtype", spec.dtype)
                block_name = f"b{len(block_configs)}"
                try:
                    layer_norm = RmsNormConfig(**norm_options)
                    feed_forward = SwigluConfig(**feed_forward_options)
                except TypeError as error:
                    raise ArchError(
                        f"norm or feed-forward configuration failed: {error}"
                    ) from error
                block_configs[block_name] = TransformerBlockConfig(
                    sequence_mixer=attention_config,
                    layer_norm=layer_norm,
                    feed_forward=feed_forward,
                    attention_residual_alpha=layer.res_alpha,
                    feed_forward_residual_alpha=layer.res_alpha,
                )
                block_names_by_layer[layer_identity] = block_name
                block_pattern.append(block_name)
    if spec.final_norm != "rms":
        raise ArchError(f"unsupported final norm: {spec.final_norm!r}")
    else:
        try:
            final_norm = RmsNormConfig(**spec.final_norm_kwargs)
        except TypeError as error:
            raise ArchError(f"final norm configuration failed: {error}") from error
    return TransformerConfig(
        d_model=spec.d_model,
        vocab_size=spec.vocab_size,
        n_layers=spec.n_layers,
        block=block_configs,
        block_pattern=block_pattern,
        lm_head=LMHeadConfig(layer_norm=final_norm, bias=False),
        dtype=DType(spec.dtype),
        init_method=InitMethod(spec.init_method),
        init_std=spec.init_std,
        tie_word_embeddings=spec.tie_word_embeddings,
    )


def build_model(
    spec: ArchSpec, *, init_device: str = "cpu", checkpoint_blocks: bool = False
) -> Transformer:
    model = spec_to_olmo(spec).build(init_device=init_device)
    if checkpoint_blocks:
        model.apply_activation_checkpointing("full")
    else:
        pass
    return model


def spec_num_params(spec: ArchSpec) -> int:
    return spec_to_olmo(spec).num_params
