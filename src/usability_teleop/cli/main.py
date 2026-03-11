"""CLI entrypoint."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from usability_teleop.cli.parser import build_parser
from usability_teleop.utils.logging import get_logger

logger = get_logger("cli")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handler = args.func
    if not callable(handler):
        raise RuntimeError("Invalid CLI handler")
    return _run_handler(handler, args)


def _run_handler(handler: Callable[[argparse.Namespace, object], int], args: argparse.Namespace) -> int:
    return handler(args, logger)


__all__ = ["build_parser", "main"]
