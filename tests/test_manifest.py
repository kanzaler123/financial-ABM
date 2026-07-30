import json

from abm.manifest import build_run_manifest
from abm.schemas import RunManifest


def test_manifest_is_deterministic_across_mapping_order(tmp_path) -> None:
    first_data = tmp_path / "first.json"
    second_data = tmp_path / "second.json"
    first_data.write_text('{"value": 1}\n', encoding="utf-8")
    second_data.write_text('{"value": 2}\n', encoding="utf-8")

    first = build_run_manifest(
        config={"market": {"enabled": True}, "seed": 7},
        data_files={"second": second_data, "first": first_data},
        random_seed=7,
        code_revision="test-revision",
    )
    second = build_run_manifest(
        config={"seed": 7, "market": {"enabled": True}},
        data_files={"first": first_data, "second": second_data},
        random_seed=7,
        code_revision="test-revision",
    )

    assert first == second
    assert RunManifest.from_dict(first.to_dict()) == first


def test_manifest_identity_changes_when_data_changes(tmp_path) -> None:
    data_path = tmp_path / "fixture.json"
    data_path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    before = build_run_manifest(
        config={"seed": 7},
        data_files={"fixture": data_path},
        random_seed=7,
        code_revision="test-revision",
    )

    data_path.write_text(json.dumps({"value": 2}), encoding="utf-8")
    after = build_run_manifest(
        config={"seed": 7},
        data_files={"fixture": data_path},
        random_seed=7,
        code_revision="test-revision",
    )

    assert before.run_id != after.run_id
    assert before.data_sha256 != after.data_sha256

