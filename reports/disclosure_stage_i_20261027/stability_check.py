import os, sys, json
for n in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"): os.environ[n] = "1"
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from _no_wmi import prefill_uname
prefill_uname()
sys.path.insert(0, "../../src")
from abm.platform_tuning import disable_power_throttling

def one(job):
    s, seed = job
    from abm.disclosure.experiments import freeze_spec, _inputs
    from abm.disclosure.game import DisclosureMarket
    sc = {r["name"]: r for r in freeze_spec("../../configs/disclosure_stage_i_check_v1.json")["scenarios"]}["s0_B"]
    cfg, dom, pol = _inputs(sc, seed)
    try:
        res = DisclosureMarket(replace(cfg, financing_spread=s), dom, pol).run()
        return {"s": s, "seed": seed, "status": "complete", "cap_hits": int(res.market.price_cap_hits.sum())}
    except Exception as e:
        return {"s": s, "seed": seed, "status": "failed", "error": str(e)}

if __name__ == "__main__":
    disable_power_throttling()
    jobs = [(s, seed) for s in (0.1, 0.15, 0.2, 0.3, 0.4) for seed in (38101, 38102)]
    with ProcessPoolExecutor(10, initializer=disable_power_throttling) as ex:
        out = list(ex.map(one, jobs))
    for r in out: print(json.dumps(r))
