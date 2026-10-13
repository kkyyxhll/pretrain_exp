"""Reuse Qwen3.5 streaming training with the explicit experimental Engram recipe."""

from __future__ import annotations

import third_party as third_party  # isort: skip

import importlib
import sys
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> None:
    training_arguments = list(sys.argv[1:] if argv is None else argv)
    if not any(
        argument == "--engram-config" or argument.startswith("--engram-config=")
        for argument in training_arguments
    ):
        training_arguments.extend(["--engram-config", str(Path(__file__).with_name("config.json"))])
    training = importlib.import_module("recipes.qwen35-08b.train")
    training.main(training_arguments)


if __name__ == "__main__":
    main()
