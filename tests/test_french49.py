import io
import zipfile
from datetime import date, timedelta

import numpy as np

from abm.french49 import (
    French49Dataset,
    build_benchmark_report,
    deterministic_split,
    parse_french49_zip,
)


def test_french49_parser_reads_value_weighted_daily_table(tmp_path) -> None:
    industries = [f"Ind{index:02d}" for index in range(49)]
    rows = [
        "Synthetic French fixture",
        "Average Value Weighted Returns -- Daily",
        "," + ",".join(industries),
    ]
    start = date(2020, 1, 1)
    for offset in range(25):
        current = start + timedelta(days=offset)
        values = ["-99.99" if offset == 0 and index == 0 else "1.00" for index in range(49)]
        rows.append(current.strftime("%Y%m%d") + "," + ",".join(values))
    rows.extend(["", "Average Equal Weighted Returns -- Daily"])
    archive_path = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("49_Industry_Portfolios_Daily.csv", "\n".join(rows))

    dataset = parse_french49_zip(archive_path)

    assert dataset.returns.shape == (25, 49)
    assert np.isnan(dataset.returns[0, 0])
    assert dataset.returns[1, 0] == 0.01
    assert dataset.industries == tuple(industries)


def test_deterministic_split_and_report_keep_test_metrics_sealed() -> None:
    industries = tuple(f"Ind{index:02d}" for index in range(49))
    split = deterministic_split(industries)
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0004, 0.015, size=(100, 49))
    dataset = French49Dataset(
        dates=np.arange(
            np.datetime64("2020-01-01"),
            np.datetime64("2020-04-10"),
            dtype="datetime64[D]",
        )[:100],
        industries=industries,
        returns=returns,
    )
    simulation_returns = rng.normal(0.0004, 0.015, size=100)
    prices = 100 * np.cumprod(np.concatenate(([1.0], 1 + simulation_returns)))

    report = build_benchmark_report(dataset, split, prices)

    assert len(split.train) == 29
    assert len(split.validation) == 10
    assert len(split.test) == 10
    assert report["sealed_test"]["metrics_read"] is False
    assert "per_industry" not in report["sealed_test"]
    assert len(report["training"]["per_industry"]) == 29
    assert len(report["validation"]["per_industry"]) == 10

