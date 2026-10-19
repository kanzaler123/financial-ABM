import json
from pathlib import Path

import numpy as np
import pytest

from abm.disclosure.domain import DomainSpec, load_domain


DOMAIN_PATH = Path("configs/domains/carbon_claims.json")


def test_carbon_claims_domain_loads_with_explicit_exploratory_provenance():
    payload = json.loads(DOMAIN_PATH.read_text(encoding="utf-8"))
    domain = load_domain(DOMAIN_PATH)
    assert domain.name == "carbon_claims"
    assert domain.goodness_bounds == pytest.approx((-3.3333333333333335, 5.0))
    assert domain.to_goodness(np.array([40.0]))[0] == pytest.approx(0.0)
    assert domain.size_cost_multipliers == (1.0, 1.0, 1.0)
    assert domain.label("wash") == "碳中和漂绿"
    assert DomainSpec.from_dict(domain.to_dict()) == domain
    assert "exploratory" in payload["_note"].lower()
    assert "uncalibrated" in payload["_note"].lower()
    assert payload["_sources"] == [
        "https://www.ipcc.ch/report/ar6/wg3/",
        "https://sciencebasedtargets.org/resources/files/Net-Zero-Standard.pdf",
    ]
    assert "modeling choice" in payload["_note"].lower() or "exploratory assumption" in payload["_note"].lower()
