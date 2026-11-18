import json, sys, numpy as np
from _no_wmi import prefill_uname
prefill_uname()
sys.path.insert(0, "../../src")
from abm.disclosure.experiments import freeze_spec, _inputs
from abm.disclosure.game import DisclosureMarket
fr = freeze_spec("../../configs/disclosure_stage_i_check_v1.json")
sc = {r["name"]: r for r in fr["scenarios"]}["B"]
cfg, dom, pol = _inputs(sc, 38101)
m = DisclosureMarket(cfg, dom, pol)
P0 = m.prices.copy()
try:
    m.run(); print("completed")
except ValueError as e:
    print("error:", e)
V, P, E, r = m.firms.intrinsic_value, m.prices, m.financing_exposure, m.contract_rate
bad = np.argsort(V / P0)[:5]
for i in bad:
    print(f"firm {i}: V/P0 {V[i]/P0[i]:.4f}  P/P0 {P[i]/P0[i]:.4f}  E/P0 {E[i]/P0[i]:.3f}  E/V {E[i]/max(V[i],1e-12):.1f}  rate {r[i]:.4f}  r0 {m.initial_contract_rate[i]:.4f}")
print("share of firms with P/P0 < 0.3:", float(np.mean(P / P0 < 0.3)), " median P/P0", float(np.median(P / P0)))
print("annual financing drag at P->0 for E=0.5P0, s=%.4f: %.3f of P0" % (cfg.financing_spread, 0.5 * cfg.financing_spread * (1 - 1/3)))
