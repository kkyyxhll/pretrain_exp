"""Explicit registry for the four Qwen3.5 building blocks."""

from __future__ import annotations

from typing import Dict, Type

from olmo_core.nn.config import ModuleConfig

__all__ = ["REGISTRY", "register", "get_config", "ArchError"]


class ArchError(Exception):
    """arch 包所有契约违约/未知字段/未注册变体的统一异常。"""


REGISTRY: Dict[str, Dict[str, Type[ModuleConfig]]] = {}


def register(group: str, name: str):
    """注册装饰器：``@register("attn", "nsa")`` 挂到变体 Config 类上。"""

    def deco(cls):
        if not (isinstance(cls, type) and issubclass(cls, ModuleConfig)):
            raise ArchError(f"register 只能用于 ModuleConfig 子类: {cls}")
        g = REGISTRY.setdefault(group, {})
        if name in g:
            raise ArchError(f"重复注册 {group}.{name}: 已有 {g[name]}")
        g[name] = cls
        return cls

    return deco


def get_config(group: str, name: str) -> Type[ModuleConfig]:
    """按组+名字取配置类；未注册抛 ArchError（列出组内已注册名）。"""
    g = REGISTRY.get(group)
    if g is None or name not in g:
        raise ArchError(f"未注册: {group}.{name}；组内已有: {sorted(g) if g else []}")
    return g[name]
