"""Centralized project path and runtime settings helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem paths used by pipeline stages."""

    root: Path
    configs: Path
    src: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    outputs: Path
    outputs_figures: Path
    outputs_tables: Path
    outputs_runs: Path


def discover_project_paths(start: Path | None = None) -> ProjectPaths:
    """Resolve project paths relative to current working directory."""
    root = (start or Path.cwd()).resolve()
    return ProjectPaths(
        root=root,
        configs=root / "configs",
        src=root / "src",
        data_raw=root / "data" / "raw",
        data_interim=root / "data" / "interim",
        data_processed=root / "data" / "processed",
        outputs=root / "outputs",
        outputs_figures=root / "outputs" / "figures",
        outputs_tables=root / "outputs" / "tables",
        outputs_runs=root / "outputs" / "runs",
    )
