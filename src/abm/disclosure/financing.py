"""Expected financing-fee cash flows for a fixed external exposure.

The horizon limits prospective fee savings. A false report's eventual fine
remains a liability beyond that horizon. Quotes observed on day t can first
affect a contract repriced on day t + 1; audits follow that day's trading.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .config import TRADING_DAYS_PER_YEAR


def financing_discount_factors(
    discount_rate: NDArray[np.float64], *, decision_day: int, horizon_days: int,
    repricing_interval: int, period_days: int, implementation_delay: int,
    disclosure_delay: int, audit_probability: float | NDArray[np.float64],
    first_audit_probability: float | NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return real/wash annuities and the expected discounted fine factor.

Rates and future quote responses are locally held fixed. These are firms'
conditional forecasts, not an assertion of a permanent equilibrium premium.
Under targeted audits the probability may differ by firm, and the first
audit after disclosure may have its own probability ``first_audit_probability``.
    """
    times = np.arange(1, horizon_days + 1)
    calendar = decision_day + times
    last_reprice = calendar // repricing_interval * repricing_interval
    quoted_day = last_reprice - 1
    real_due = decision_day + max(1, implementation_delay)
    wash_due = decision_day + max(1, disclosure_delay)
    real_visible = quoted_day >= real_due
    wash_visible = quoted_day >= wash_due
    first_audit = ((wash_due + period_days - 1) // period_days) * period_days
    # A quote precedes the audit on its own day, so only earlier audits
    # remove the signal used at a contract's most recent repricing.
    audits = np.maximum(0, (quoted_day - 1 - first_audit) // period_days + 1)
    if first_audit_probability is not None or np.ndim(audit_probability):
        return _firm_rate_factors(discount_rate, times=times, decision_day=decision_day, period_days=period_days,
                                  real_visible=real_visible, wash_visible=wash_visible, first_audit=first_audit,
                                  audits=audits, later=audit_probability, first=first_audit_probability)
    survival = np.power(1.0 - audit_probability, audits) * wash_visible
    discount = np.exp(-times[:, None] * np.asarray(discount_rate)[None, :] / TRADING_DAYS_PER_YEAR)
    real = real_visible @ discount / TRADING_DAYS_PER_YEAR
    wash = survival @ discount / TRADING_DAYS_PER_YEAR
    if audit_probability == 0:
        fine = np.zeros_like(discount_rate)
    else:
        first_discount = np.exp(-discount_rate * (first_audit - decision_day) / TRADING_DAYS_PER_YEAR)
        repeat_discount = np.exp(-discount_rate * period_days / TRADING_DAYS_PER_YEAR)
        fine = audit_probability * first_discount / (1.0 - (1.0 - audit_probability) * repeat_discount)
    return real, wash, fine


def _firm_rate_factors(
    discount_rate: NDArray[np.float64], *, times: NDArray[np.int64], decision_day: int, period_days: int,
    real_visible: NDArray[np.bool_], wash_visible: NDArray[np.bool_], first_audit: int, audits: NDArray[np.int64],
    later: float | NDArray[np.float64], first: float | NDArray[np.float64] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The factors of ``financing_discount_factors`` with audit probabilities that differ by firm."""
    rate = np.asarray(discount_rate, dtype=np.float64)
    later = np.broadcast_to(np.asarray(later, dtype=np.float64), rate.shape)
    discount = np.exp(-times[:, None] * rate[None, :] / TRADING_DAYS_PER_YEAR)
    first_discount = np.exp(-rate * (first_audit - decision_day) / TRADING_DAYS_PER_YEAR)
    repeat_discount = np.exp(-rate * period_days / TRADING_DAYS_PER_YEAR)
    if first is None:
        survival = np.power(1.0 - later[None, :], audits[:, None])
        fine = later * first_discount / (1.0 - (1.0 - later) * repeat_discount)
    else:
        first = np.broadcast_to(np.asarray(first, dtype=np.float64), rate.shape)
        survival = np.where(audits[:, None] > 0,
                            (1.0 - first)[None, :] * np.power(1.0 - later[None, :], np.maximum(audits - 1, 0)[:, None]),
                            1.0)
        fine = first_discount * (first + (1.0 - first) * repeat_discount * later
                                 / (1.0 - (1.0 - later) * repeat_discount))
    real = real_visible @ discount / TRADING_DAYS_PER_YEAR
    wash = np.sum(survival * wash_visible[:, None] * discount, axis=0) / TRADING_DAYS_PER_YEAR
    return real, wash, fine


def financing_signal_factors(
    discount_rate: NDArray[np.float64], *, decision_day: int, horizon_days: int,
    repricing_interval: int, period_days: int, implementation_delay: int,
    disclosure_delay: int, audit_probability: float | NDArray[np.float64], transparency: float,
    publish_audit_results: bool = False, published_now: NDArray[np.bool_] | None = None,
    reporting_interval_days: int = 1, real_persistence: NDArray[np.float64] | None = None,
    first_audit_probability: float | NDArray[np.float64] | None = None,
    assurance_probability: float | NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return low/high real-signal annuities, wash survival and the fine factor.

    Remaining report capacity is locally fixed. Audits remove the old
    inflation displaced by a real action. Avoided fines on that old stock
    are excluded for both publication regimes. A published snapshot lasts
    one decision period and contains an action only if it was implemented
    by the snapshot's observation date. ``published_now`` is the firm's
    known, just-completed audit; future checks remain random.

    ``real_persistence[d]`` is the share of a real change still present d
    days after it is implemented (reverting true values); displaced
    inflation does not come back. It is supported with daily reports and
    unpublished audits only.

    Under targeted audits ``audit_probability`` may differ by firm, and the
    new report's first audit may have ``first_audit_probability``; the old
    stock displaced by a real action faces ``audit_probability``. Both are
    supported with daily reports and unpublished audits only.

    ``assurance_probability`` is the chance, possibly per firm, that a check
    on every disclosure day (decision day + disclosure delay + k periods)
    finds the inflation in the report and has it corrected before
    publication, without a fine (see ``assured_signal_factors``).
    """
    if assurance_probability is not None:
        if (publish_audit_results or reporting_interval_days != 1 or real_persistence is not None
                or first_audit_probability is not None or np.ndim(audit_probability) > 0):
            raise ValueError("assurance checks are modelled only with daily reports, unpublished audits, "
                             "random audits and non-reverting true values")
        return assured_signal_factors(
            discount_rate, decision_day=decision_day, horizon_days=horizon_days,
            repricing_interval=repricing_interval, period_days=period_days,
            implementation_delay=implementation_delay, disclosure_delay=disclosure_delay,
            audit_probability=float(audit_probability), transparency=transparency,
            assurance_probability=assurance_probability,
        )
    if real_persistence is not None and (publish_audit_results or reporting_interval_days != 1):
        raise ValueError("reverting true values are modelled only with daily reports and unpublished audits")
    firm_rates = first_audit_probability is not None or np.ndim(audit_probability) > 0
    if firm_rates and (publish_audit_results or reporting_interval_days != 1 or real_persistence is not None):
        raise ValueError("firm-level audit probabilities are modelled only with daily reports, unpublished "
                         "audits and non-reverting true values")
    real, wash, fine = financing_discount_factors(
        discount_rate, decision_day=decision_day, horizon_days=horizon_days,
        repricing_interval=repricing_interval, period_days=period_days,
        implementation_delay=implementation_delay, disclosure_delay=disclosure_delay,
        audit_probability=audit_probability, first_audit_probability=first_audit_probability,
    )
    if reporting_interval_days != 1:
        signals = _periodic_report_signals(
            decision_day=decision_day, horizon_days=horizon_days, period_days=period_days,
            implementation_delay=implementation_delay, disclosure_delay=disclosure_delay,
            audit_probability=audit_probability, transparency=transparency,
            publish_audit_results=publish_audit_results, reporting_interval_days=reporting_interval_days,
        )
        times = np.arange(1, horizon_days + 1)
        quoted_day = (decision_day + times) // repricing_interval * repricing_interval - 1
        quote_index = np.maximum(quoted_day - decision_day, 0)
        known = np.zeros(np.asarray(discount_rate).size, dtype=bool) if published_now is None else np.asarray(published_now, dtype=bool)
        discount = np.exp(-times[:, None] * np.asarray(discount_rate)[None, :] / TRADING_DAYS_PER_YEAR)
        factors = tuple(np.sum(signals[quote_index[:, None], known.astype(np.int64)[None, :], index] * discount,
                               axis=0) / TRADING_DAYS_PER_YEAR for index in range(3))
        return *factors, fine
    times = np.arange(1, horizon_days + 1)
    quoted_day = (decision_day + times) // repricing_interval * repricing_interval - 1
    real_due = decision_day + max(1, implementation_delay)
    wash_due = decision_day + max(1, disclosure_delay)
    real_visible = quoted_day >= real_due
    checks = np.maximum(0, (quoted_day - 1) // period_days - decision_day // period_days)
    discount = np.exp(-times[:, None] * np.asarray(discount_rate)[None, :] / TRADING_DAYS_PER_YEAR)
    if np.ndim(audit_probability):
        old_survival = np.power(1.0 - np.asarray(audit_probability, dtype=np.float64)[None, :], checks[:, None])
        displaced = real - (1.0 - transparency) * (
            np.sum(real_visible[:, None] * old_survival * discount, axis=0) / TRADING_DAYS_PER_YEAR)
        return real, displaced, wash, fine
    old_survival = np.power(1.0 - audit_probability, checks)
    if real_persistence is not None:
        persistence = np.asarray(real_persistence, dtype=np.float64)
        if persistence.ndim != 1 or persistence.size < horizon_days + 1 or not np.isfinite(persistence).all():
            raise ValueError("real_persistence needs one finite value per day of the horizon")
        real = (real_visible * persistence[np.clip(quoted_day - real_due, 0, None)]) @ discount / TRADING_DAYS_PER_YEAR
    if not publish_audit_results:
        displaced = real - (1.0 - transparency) * ((real_visible * old_survival) @ discount / TRADING_DAYS_PER_YEAR)
        return real, displaced, wash, fine

    known = np.zeros(np.asarray(discount_rate).size, dtype=bool) if published_now is None else np.asarray(published_now, dtype=bool)
    latest_audit = (quoted_day - 1) // period_days * period_days
    current = latest_audit == decision_day
    certificate = np.where((latest_audit > decision_day)[:, None], audit_probability,
                           np.where(current[:, None], known[None, :], 0.0))
    visible = real_visible[:, None]
    real_signal = visible * (1.0 - certificate) + certificate * (latest_audit >= real_due)[:, None]
    # Surviving old inflation and a future published check are mutually
    # exclusive. A known pre-action certificate can coexist with it.
    unverified_old = old_survival[:, None] * np.where(current[:, None], ~known[None, :], 1.0)
    displaced_signal = real_signal - (1.0 - transparency) * visible * unverified_old
    first_audit = ((wash_due + period_days - 1) // period_days) * period_days
    wash_checks = np.maximum(0, (quoted_day - 1 - first_audit) // period_days + 1)
    wash_survival = np.power(1.0 - audit_probability, wash_checks)[:, None]
    # A pre-disclosure certificate can hide a surviving new report. After
    # disclosure, the selected check already removes that report.
    wash_signal = (quoted_day >= wash_due)[:, None] * (
        wash_survival - certificate * (latest_audit < wash_due)[:, None]
    )
    return (np.sum(real_signal * discount, axis=0) / TRADING_DAYS_PER_YEAR,
            np.sum(displaced_signal * discount, axis=0) / TRADING_DAYS_PER_YEAR,
            np.sum(wash_signal * discount, axis=0) / TRADING_DAYS_PER_YEAR, fine)


def assured_signal_factors(
    discount_rate: NDArray[np.float64], *, decision_day: int, horizon_days: int, repricing_interval: int,
    period_days: int, implementation_delay: int, disclosure_delay: int, audit_probability: float,
    transparency: float, assurance_probability: float | NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """The four factors of ``financing_signal_factors`` with pre-publication assurance checks.

    Checks fall on every disclosure day, decision day + disclosure delay + k periods, after that
    day's actions land and before trading, so a quote on a check day sees the checked report. A
    check finds the whole inflation in the report with probability ``a``; it is corrected without a
    fine, so the detection discount counts only audits that find a report every earlier check missed.
    New inflation meets its first check on its own disclosure day. The old stock displaced by a real
    action meets the same checks. Daily reports and unpublished audits only; the decision day must
    be an audit day and the disclosure delay shorter than a period, so no check falls between the
    decision and the disclosure.
    """
    if decision_day % period_days or not 0 < disclosure_delay < period_days:
        raise ValueError("assurance checks need decisions on audit days and a disclosure delay shorter than a period")
    rate = np.asarray(discount_rate, dtype=np.float64)
    keep = 1.0 - np.broadcast_to(np.asarray(assurance_probability, dtype=np.float64), rate.shape)
    if np.any(keep < 0) or np.any(keep > 1):
        raise ValueError("assurance_probability must be in [0, 1]")
    times = np.arange(1, horizon_days + 1)
    quoted_day = (decision_day + times) // repricing_interval * repricing_interval - 1
    real_due = decision_day + max(1, implementation_delay)
    wash_due = decision_day + disclosure_delay
    real_visible = quoted_day >= real_due
    wash_visible = quoted_day >= wash_due
    first_audit = ((wash_due + period_days - 1) // period_days) * period_days
    audits = np.maximum(0, (quoted_day - 1 - first_audit) // period_days + 1)
    old_audits = np.maximum(0, (quoted_day - 1) // period_days - decision_day // period_days)
    # Checks on wash_due + k periods up to the quote day; the old stock meets the same ones.
    checks = np.where(wash_visible, (quoted_day - wash_due) // period_days + 1, 0)
    q = audit_probability
    discount = np.exp(-times[:, None] * rate[None, :] / TRADING_DAYS_PER_YEAR)
    checked = keep[None, :] ** checks[:, None]
    real = real_visible @ discount / TRADING_DAYS_PER_YEAR
    wash = np.sum((wash_visible * (1.0 - q) ** audits)[:, None] * checked * discount, axis=0) / TRADING_DAYS_PER_YEAR
    old = (real_visible * (1.0 - q) ** old_audits)[:, None] * checked
    displaced = real - (1.0 - transparency) * np.sum(old * discount, axis=0) / TRADING_DAYS_PER_YEAR
    # Audit j (day first_audit + j periods) follows j audits and checks_before + j checks.
    checks_before = (first_audit - 1 - wash_due) // period_days + 1
    first_discount = np.exp(-rate * (first_audit - decision_day) / TRADING_DAYS_PER_YEAR)
    repeat_discount = np.exp(-rate * period_days / TRADING_DAYS_PER_YEAR)
    if q == 0:
        fine = np.zeros_like(rate)
    else:
        fine = q * keep**checks_before * first_discount / (1.0 - (1.0 - q) * keep * repeat_discount)
    return real, displaced, wash, fine


def _periodic_report_signals(
    *, decision_day: int, horizon_days: int, period_days: int, implementation_delay: int,
    disclosure_delay: int, audit_probability: float, transparency: float,
    publish_audit_results: bool, reporting_interval_days: int,
) -> NDArray[np.float64]:
    """Expected incremental signals by day, initial certificate and action segment.

    Completed records are auditable before their scheduled publication. Every
    selected audit refreshes the ordinary report after trading. Each probability
    cell tracks surviving records and the current certificate jointly; cached
    report moments are weighted by that same cell's probability.
    """
    real_due = decision_day + max(1, implementation_delay)
    wash_due = decision_day + max(1, disclosure_delay)
    probability = np.zeros((2, 2, 2))
    probability[0, 1, 0] = 1.0
    probability[1, 1, int(publish_audit_results and decision_day % period_days == 0)] = 1.0
    wash_probability = probability.copy()
    low = np.zeros_like(probability)
    high = np.zeros_like(probability)
    wash = np.zeros_like(probability)
    alive = np.arange(2)[None, :, None]
    signals = np.zeros((horizon_days + 1, 2, 3))
    p = audit_probability
    certificate = int(publish_audit_results)
    for offset in range(1, horizon_days + 1):
        day = decision_day + offset
        implemented = float(day >= real_due)
        if day % reporting_interval_days == 0:
            low = implemented * probability
            high = implemented * (1.0 - (1.0 - transparency) * alive) * probability
            wash = float(day >= wash_due) * alive * wash_probability
        latest_audit = (day - 1) // period_days * period_days
        certified_real = float(latest_audit >= real_due) * probability[:, :, 1].sum(axis=1)
        signals[offset, :, 0] = low[:, :, 0].sum(axis=1) + certified_real
        signals[offset, :, 1] = high[:, :, 0].sum(axis=1) + certified_real
        signals[offset, :, 2] = wash[:, :, 0].sum(axis=1)
        if day % period_days == 0:
            selected = p * probability.sum(axis=(1, 2))
            selected_wash = p * wash_probability.sum(axis=(1, 2))
            probability = np.stack(((1.0 - p) * probability.sum(axis=2), np.zeros((2, 2))), axis=2)
            low = np.stack(((1.0 - p) * low.sum(axis=2), np.zeros((2, 2))), axis=2)
            high = np.stack(((1.0 - p) * high.sum(axis=2), np.zeros((2, 2))), axis=2)
            wash_probability = np.stack(((1.0 - p) * wash_probability.sum(axis=2), np.zeros((2, 2))), axis=2)
            wash = np.stack(((1.0 - p) * wash.sum(axis=2), np.zeros((2, 2))), axis=2)
            probability[:, 0, certificate] += selected
            low[:, 0, certificate] += implemented * selected
            high[:, 0, certificate] += implemented * selected
            wash_probability[:, int(day < wash_due), certificate] += selected_wash
    return signals
