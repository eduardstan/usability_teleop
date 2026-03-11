from pathlib import Path

from usability_teleop.config.settings import discover_project_paths


def test_project_layout_paths_exist() -> None:
    paths = discover_project_paths(Path.cwd())
    assert paths.configs.exists()
    assert paths.src.exists()
    assert paths.data_raw.exists()
    assert paths.outputs.exists()
