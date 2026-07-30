from dataclasses import replace
from pathlib import Path

import numpy as np

from abm.config import load_stage1_config
from abm.harness import run_stage1
from abm.mechanism_validation import (
    MechanismProtocol,
    build_scenarios,
    result_metrics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1.json"


def test_protocol_separates_development_and_formal_seeds(tmp_path) -> None:
    path = tmp_path / "protocol.json"
    path.write_text(
        """
        {
          "version": "test",
          "purpose": "test",
          "seeds": [1, 2],
          "excluded_development_seeds": [3, 4],
          "criteria": {"one": 1.0}
        }
        """,
        encoding="utf-8",
    )

    protocol = MechanismProtocol.from_json(path)

    assert protocol.seeds == (1, 2)
    assert protocol.excluded_development_seeds == (3, 4)


def test_scenarios_make_complex_shocks_explicit_controls() -> None:
    config = load_stage1_config(CONFIG_PATH)
    scenarios = build_scenarios(config)

    assert scenarios["full"].fundamental_process == "gaussian"
    assert scenarios["full"].public_news_price_pass_through == 0.0
    assert scenarios["garch_t_control"].fundamental_process == "garch_t"
    assert scenarios["independent_signals"].common_signal_correlation == 0.0
    assert not scenarios["no_logit"].learning_enabled


def test_result_metrics_are_finite_and_capture_strategy_turnover() -> None:
    config = replace(
        load_stage1_config(CONFIG_PATH),
        population_size=120,
        trading_days=60,
        announcements=(),
    )
    result = run_stage1(config)
    metrics = result_metrics(result)

    assert set(metrics)
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["price_cap_hits"] == 0.0
    assert metrics["strategy_turnover"] > 0.0
