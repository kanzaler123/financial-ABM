"""Frozen Kenneth French 49-industry daily-return data and benchmarks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FRENCH49_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "ftp/49_Industry_Portfolios_daily_CSV.zip"
)
FRENCH49_DETAILS_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"
    "Data_Library/det_49_ind_port.html"
)

FloatMatrix = NDArray[np.float64]
DateArray = NDArray[np.datetime64]


@dataclass(frozen=True, slots=True)
class French49Dataset:
    dates: DateArray
    industries: tuple[str, ...]
    returns: FloatMatrix

    def __post_init__(self) -> None:
        if len(self.industries) != 49:
            raise ValueError("French 49 dataset must contain exactly 49 industries")
        if self.returns.shape != (self.dates.size, len(self.industries)):
            raise ValueError("French 49 date and return shapes do not match")
        if self.dates.size == 0 or np.any(self.dates[1:] <= self.dates[:-1]):
            raise ValueError("French 49 dates must be nonempty, unique, and ordered")

    def between(self, start: date, end: date) -> "French49Dataset":
        start_value = np.datetime64(start.isoformat())
        end_value = np.datetime64(end.isoformat())
        mask = (self.dates >= start_value) & (self.dates <= end_value)
        if not np.any(mask):
            raise ValueError(f"no French 49 observations between {start} and {end}")
        return French49Dataset(
            dates=self.dates[mask].copy(),
            industries=self.industries,
            returns=self.returns[mask].copy(),
        )


@dataclass(frozen=True, slots=True)
class IndustrySplit:
    version: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        if (len(self.train), len(self.validation), len(self.test)) != (29, 10, 10):
            raise ValueError("industry split must contain 29/10/10 names")
        all_names = self.train + self.validation + self.test
        if len(set(all_names)) != 49:
            raise ValueError("industry split contains duplicates")

    @classmethod
    def from_json(cls, path: str | Path) -> "IndustrySplit":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            version=str(payload["version"]),
            train=tuple(payload["train"]),
            validation=tuple(payload["validation"]),
            test=tuple(payload["test"]),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_french49_snapshot(
    snapshot_dir: str | Path,
    *,
    source_url: str = FRENCH49_DAILY_URL,
) -> Path:
    """Download an immutable official ZIP plus provenance manifest."""
    destination = Path(snapshot_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="french49-", dir=destination.parent
    ) as temporary:
        temporary_dir = Path(temporary)
        archive_path = temporary_dir / "49_Industry_Portfolios_daily_CSV.zip"
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "auditable-market-abm/0.1"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            with archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        dataset = parse_french49_zip(archive_path)
        retrieved_at = datetime.now(UTC).isoformat()
        manifest = {
            "schema_version": "0.1.0",
            "source_url": source_url,
            "details_url": FRENCH49_DETAILS_URL,
            "retrieved_at": retrieved_at,
            "archive_name": archive_path.name,
            "sha256": sha256_file(archive_path),
            "bytes": archive_path.stat().st_size,
            "date_start": str(dataset.dates[0]),
            "date_end": str(dataset.dates[-1]),
            "observations": int(dataset.dates.size),
            "industries": list(dataset.industries),
            "missing_value_codes": [-99.99, -999.0],
            "units_in_source": "percent",
            "stored_units": "decimal_return",
            "copyright": "Eugene F. Fama and Kenneth R. French",
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_dir.replace(destination)
    return destination


def parse_french49_zip(path: str | Path) -> French49Dataset:
    with zipfile.ZipFile(path) as archive:
        candidates = [
            name for name in archive.namelist() if name.lower().endswith(".csv")
        ]
        if len(candidates) != 1:
            raise ValueError(f"expected one CSV in French ZIP, found {candidates}")
        text = archive.read(candidates[0]).decode("latin-1")
    lines = text.splitlines()
    marker_index = next(
        index
        for index, line in enumerate(lines)
        if "Average Value Weighted Returns -- Daily" in line
    )
    reader = csv.reader(lines[marker_index + 1 :])
    header: list[str] | None = None
    date_values: list[np.datetime64] = []
    return_rows: list[list[float]] = []
    for row in reader:
        if not row or not any(cell.strip() for cell in row):
            if return_rows:
                break
            continue
        first = row[0].strip()
        if header is None:
            candidate = tuple(cell.strip() for cell in row[1:])
            if len(candidate) == 49:
                header = list(candidate)
            continue
        if len(first) != 8 or not first.isdigit():
            break
        if len(row) != 50:
            raise ValueError(f"unexpected French 49 row width for {first}: {len(row)}")
        date_values.append(
            np.datetime64(
                f"{first[:4]}-{first[4:6]}-{first[6:8]}",
                "D",
            )
        )
        values = []
        for raw_value in row[1:]:
            value = float(raw_value.strip())
            values.append(np.nan if value in (-99.99, -999.0) else value / 100.0)
        return_rows.append(values)
    if header is None or not return_rows:
        raise ValueError("could not locate the value-weighted daily French 49 table")
    return French49Dataset(
        dates=np.array(date_values, dtype="datetime64[D]"),
        industries=tuple(header),
        returns=np.array(return_rows, dtype=np.float64),
    )


def deterministic_split(industries: Sequence[str]) -> IndustrySplit:
    """Create a data-independent stable split; persist it before analysis."""
    if len(industries) != 49 or len(set(industries)) != 49:
        raise ValueError("expected 49 unique industry names")
    ranked = sorted(
        industries,
        key=lambda name: hashlib.sha256(
            f"french49-split-v1:{name}".encode("utf-8")
        ).hexdigest(),
    )
    return IndustrySplit(
        version="french49-split-v1",
        train=tuple(ranked[:29]),
        validation=tuple(ranked[29:39]),
        test=tuple(ranked[39:]),
    )


def _autocorrelation(values: NDArray[np.float64], lag: int = 1) -> float:
    if lag < 1:
        raise ValueError("autocorrelation lag must be positive")
    # Preserve trading-day spacing. Removing missing days first would join
    # nonadjacent observations and turn a multi-day gap into a one-day lag.
    first, second = values[:-lag], values[lag:]
    valid_pairs = np.isfinite(first) & np.isfinite(second)
    first, second = first[valid_pairs], second[valid_pairs]
    if first.size < 2:
        raise ValueError("autocorrelation requires at least two finite lagged pairs")
    if np.std(first) == 0 or np.std(second) == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def return_statistics(values: NDArray[np.float64]) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size < 20:
        raise ValueError("at least 20 finite returns are required")
    mean = float(np.mean(finite))
    standard_deviation = float(np.std(finite, ddof=1))
    centered = finite - mean
    population_std = float(np.std(finite, ddof=0))
    if population_std == 0:
        skewness = 0.0
        excess_kurtosis = 0.0
    else:
        standardized = centered / population_std
        skewness = float(np.mean(standardized**3))
        excess_kurtosis = float(np.mean(standardized**4) - 3.0)
    return {
        "observations": int(finite.size),
        "mean_daily_return": mean,
        "daily_volatility": standard_deviation,
        "return_autocorrelation_lag1": _autocorrelation(values),
        "absolute_return_autocorrelation_lag1": _autocorrelation(np.abs(values)),
        "skewness": skewness,
        "excess_kurtosis": excess_kurtosis,
        "quantile_01": float(np.quantile(finite, 0.01)),
        "quantile_99": float(np.quantile(finite, 0.99)),
    }


def benchmark_group(
    dataset: French49Dataset,
    industries: Sequence[str],
) -> dict[str, object]:
    index = {name: position for position, name in enumerate(dataset.industries)}
    missing = set(industries) - set(index)
    if missing:
        raise ValueError(f"split references unknown industries: {sorted(missing)}")
    per_industry = {
        industry: return_statistics(dataset.returns[:, index[industry]])
        for industry in industries
    }
    metric_names = tuple(next(iter(per_industry.values())).keys())
    summary: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.array(
            [float(metrics[metric]) for metrics in per_industry.values()],
            dtype=np.float64,
        )
        summary[metric] = {
            "median": float(np.nanmedian(values)),
            "q25": float(np.nanquantile(values, 0.25)),
            "q75": float(np.nanquantile(values, 0.75)),
        }
    selected = dataset.returns[
        :, [index[industry] for industry in industries]
    ]
    daily_cross_sectional_std = np.nanstd(selected, axis=1, ddof=1)
    return {
        "industries": list(industries),
        "per_industry": per_industry,
        "summary": summary,
        "mean_daily_cross_sectional_volatility": float(
            np.nanmean(daily_cross_sectional_std)
        ),
    }


def build_benchmark_report(
    dataset: French49Dataset,
    split: IndustrySplit,
    simulation_prices: NDArray[np.float64],
) -> dict[str, object]:
    if set(dataset.industries) != set(
        split.train + split.validation + split.test
    ):
        raise ValueError("frozen split does not match the dataset industries")
    simulation_returns = simulation_prices[1:] / simulation_prices[:-1] - 1.0
    return {
        "schema_version": "0.1.0",
        "period_start": str(dataset.dates[0]),
        "period_end": str(dataset.dates[-1]),
        "split_version": split.version,
        "training": benchmark_group(dataset, split.train),
        "validation": benchmark_group(dataset, split.validation),
        "sealed_test": {
            "industries": list(split.test),
            "metrics_read": False,
        },
        "simulation": return_statistics(simulation_returns),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2000, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2025, 12, 31))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = parse_french49_zip(args.archive).between(args.start, args.end)
    split = IndustrySplit.from_json(args.split)
    with np.load(args.run_dir / "state_arrays.npz") as arrays:
        prices = arrays["prices"]
    report = build_benchmark_report(dataset, split, prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

