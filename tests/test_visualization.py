from dataclasses import replace
from pathlib import Path

from abm.config import load_stage1_config
from abm.harness import run_stage1
from abm.visualization import create_stage1_visualizations

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_stage1_visualizations_are_exported_as_png_and_svg(tmp_path) -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=30,
        trading_days=60,
        announcements=(),
    )
    run_dir = tmp_path / "run"
    run_stage1(config).write(run_dir, config)

    expected = (
        run_dir / "stage1_overview.png",
        run_dir / "stage1_overview.svg",
        run_dir / "stage1_learning.png",
        run_dir / "stage1_learning.svg",
    )
    assert all(path.is_file() and path.stat().st_size > 1000 for path in expected)
    assert expected[0].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in expected[1].read_text(encoding="utf-8")[:1000]

    regenerated = create_stage1_visualizations(run_dir)
    assert regenerated == expected

