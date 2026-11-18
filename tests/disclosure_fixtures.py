"""Public, entirely synthetic inputs for disclosure mechanism tests.

These inputs exercise the quantile link, saturation and size correlation.
They contain no vendor observations and are not an empirical calibration.
"""

from dataclasses import replace

import numpy as np

from abm.disclosure.domain import DomainSpec


def synthetic_quantile_domain(**changes) -> DomainSpec:
    support = ((20.0, 3), (40.0, 8), (60.0, 12), (80.0, 6), (100.0, 1))
    values, counts = np.asarray(support).T
    mean = float(np.average(values, weights=counts))
    sd = float(np.sqrt(np.average((values - mean) ** 2, weights=counts)))
    domain = DomainSpec(
        name="synthetic_quantile", attribute_label="Synthetic quality", unit="points",
        direction=1, initial_mean=mean, initial_sd=sd, raw_lower=0.0, raw_upper=100.0,
        natural_decay=0.2, real_cost_scale=0.05, real_cost_log_sd=0.5,
        size_cost_multipliers=(0.6, 1.0, 1.6), cost_level_elasticity=1.5,
        max_change_per_period=0.25, wash_cost_ratio=0.2,
        implementation_delay_days=120, disclosure_delay_days=20,
        baseline_audit_probability=0.01, baseline_penalty=0.0,
        baseline_transparency=0.1, baseline_subsidy=0.0, baseline_fixed_fine=0.00005,
        initial_empirical_distribution=support, observation_link="empirical_quantile",
        initial_size_rank_correlation=0.3, signal_saturation=True,
    )
    return replace(domain, **changes)
