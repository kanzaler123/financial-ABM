"""Public, synthetic regressions for the November 2026 inference corrections."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from abm import research_inference as inference

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("research_corrections", ROOT / "scripts/recompute_research_corrections.py")
corrections = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corrections)


def test_profile_minimum_uses_full_covariance_not_original_estimation_weights():
    # A transparent overidentified linear example: diagonal-weight OLS gives
    # theta=0.2, whereas the full-covariance distance is minimized at -0.75.
    covariance = 0.01 * np.array([[1.0, 0.0, 1.9], [0.0, 1.0, 0.0], [1.9, 0.0, 4.0]])
    result = inference.distance_profile_intervals(
        lambda t: np.array([t[0], t[1], t[0]]), [0.2, 0.0], [0.0, 0.0, 1.0], covariance,
        bounds=[(-2.0, 2.0), (-1.0, 1.0)], names=["persistence"],
    )
    assert result["reference_parameters"][0] == pytest.approx(-0.75, abs=1e-6)
    assert result["minimum"] == pytest.approx(100 / 1.2)
    assert result["distance_at_original_estimate"] == pytest.approx(361.025641025641)
    # Analytic profile: J*(x)-min J*=(x+0.75)^2/0.00325.
    grid = np.linspace(-2, 2, 61)
    expected = grid[(grid + 0.75) ** 2 / 0.00325 <= inference.CHI2_95_ONE_DF]
    assert result["intervals"]["persistence"] == pytest.approx([expected.min(), expected.max()])
    assert not next(row for row in result["profiles"]["persistence"]
                    if np.isclose(row["value"], 0.2))["accepted"]
    # The historical bug would accept 0.2 because it makes that point the floor.


def test_boundary_minimum_and_fixed_nuisance_parameter_are_handled():
    result = inference.distance_profile_intervals(
        lambda t: np.asarray(t), [0.7, 0.0], [-1.0, 0.0], np.eye(2) * 0.01,
        bounds=[(0.0, 2.0), (0.0, 0.0)], names=["rate"],
    )
    assert result["reference_parameters"] == pytest.approx([0.0, 0.0])
    assert result["minimum"] == pytest.approx(100.0)
    assert result["intervals"]["rate"] == [0.0, 0.0]
    assert result["touches_parameter_bounds"]["rate"] == {"lower": True, "upper": False}
    assert len(result["profiles"]["rate"]) == 61


def test_failed_optimizations_cannot_be_reported_as_confidence_intervals(monkeypatch):
    def failure(fun, x, **kwargs):
        return SimpleNamespace(x=np.asarray(x), fun=fun(x), success=False)

    monkeypatch.setattr(inference, "minimize", failure)
    with pytest.raises(inference.OptimizationFailure, match="all bounded"):
        inference.distance_profile_intervals(
            lambda t: t, [0.5], [0.0], np.eye(1), bounds=[(0.0, 1.0)], names=["rate"], grid_points=3,
        )


@pytest.mark.parametrize("covariance", [np.array([[0.0]]), np.array([[-1.0]]), np.array([[np.nan]])])
def test_invalid_covariance_is_rejected(covariance):
    with pytest.raises(ValueError, match="covariance"):
        inference.distance_profile_intervals(
            lambda t: t, [0.5], [0.0], covariance, bounds=[(0.0, 1.0)], names=["rate"],
        )


def test_nonfinite_model_moments_are_rejected():
    with pytest.raises(ValueError, match="model moments"):
        inference.distance_profile_intervals(
            lambda t: np.array([np.nan]), [0.5], [0.0], np.eye(1), bounds=[(0.0, 1.0)], names=["rate"],
        )


def test_public_h3_correction_uses_baseline_without_inventing_paired_intervals():
    path = ROOT / "reports/disclosure_stage_h3_20261027/h3_v1_analysis.json"
    before = path.read_bytes()
    result = corrections.h3_correction(path)
    v, j = (result["equal_budget_contrasts"][name] for name in ("V_minus_B", "J_minus_B"))
    assert v["mean"] == pytest.approx(-0.020926597892002977)
    assert j["mean"] == pytest.approx(-0.009767268907335078)
    assert result["different_budget_comparisons_not_equal_budget"]["J_minus_A2"] == pytest.approx(-0.019756313393374393)
    for row in (v, j):
        assert row["ci_low"] is None and row["ci_high"] is None
        assert row["interval_status"] == "not_computable_from_scenario_means"
        assert row["n"] == 30
    assert path.read_bytes() == before


def make_synthetic_h3(tmp_path):
    # Strong common seed variation; paired differences are constant. An unpaired
    # calculation or accidental pairing in record order would give a wide interval.
    seeds = [31, 11, 21]
    values = {"B": [10.0, 20.0, 30.0], "V": [8.0, 18.0, 28.0],
              "J": [9.0, 19.0, 29.0], "A2": [14.0, 24.0, 34.0]}
    analysis = {"experiment_id": "synthetic", "planned_trials": 12, "complete_trials": 12,
                "describe": {name: {"complete": 3, "true_change": np.mean(v)} for name, v in values.items()}}
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps(analysis))
    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "experiment.json").write_text(json.dumps({"experiment_id": "synthetic", "spec": {"seeds": seeds}}))
    records = [{"scenario": name, "seed": seed, "status": "complete",
                "metrics": {"true_change": {"equal_weight_sd": value}}}
               for name in ("B", "V", "J") for seed, value in zip(seeds, values[name])]
    (archive / "summary.json").write_text(json.dumps({"experiment_id": "synthetic", "records": records[::-1]}))
    return path, archive


def test_archived_h3_intervals_pair_by_seed_and_read_only(tmp_path):
    path, archive = make_synthetic_h3(tmp_path)
    before = (archive / "summary.json").read_bytes()
    result = corrections.h3_correction(path, archive)
    v = result["equal_budget_contrasts"]["V_minus_B"]
    assert v["mean"] == -2.0 and v["ci_low"] == -2.0 and v["ci_high"] == -2.0
    assert v["seeds"] == [31, 11, 21]
    assert result["status"] == "paired_intervals_recomputed_from_archive"
    assert (archive / "summary.json").read_bytes() == before


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_id", "wrong_mean"])
def test_h3_archive_mismatch_fails_instead_of_dropping_evidence(tmp_path, mutation):
    path, archive = make_synthetic_h3(tmp_path)
    summary_path = archive / "summary.json"
    summary = json.loads(summary_path.read_text())
    if mutation == "missing":
        summary["records"].pop()
    elif mutation == "duplicate":
        summary["records"].append(summary["records"][0])
    elif mutation == "wrong_id":
        summary["experiment_id"] = "different"
    else:
        summary["records"][0]["metrics"]["true_change"]["equal_weight_sd"] += 1
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        corrections.h3_correction(path, archive)


def test_output_cannot_replace_evidence_or_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(corrections, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="historical reports"):
        corrections.write_new_output({}, tmp_path / "reports" / "new.json")
    with pytest.raises(ValueError, match="input archives"):
        corrections.write_new_output({}, tmp_path / "private_archive" / "new.json", (tmp_path / "private_archive",))
    output = tmp_path / "correction.json"
    corrections.write_new_output({"status": "new"}, output)
    with pytest.raises(FileExistsError):
        corrections.write_new_output({"status": "changed"}, output)
    assert json.loads(output.read_text()) == {"status": "new"}


def test_profiles_refuse_a_mismatched_archive_before_loading_signals(tmp_path):
    (tmp_path / "experiment.json").write_text(json.dumps({"experiment_id": "different"}))
    with pytest.raises(ValueError, match="does not match"):
        corrections.recompute_profiles("h2", tmp_path, ["Q"])
