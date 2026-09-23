#!/usr/bin/env python
"""
Referee-2 point 6 --- FAITHFUL pool-baseline comparison for C2H2.

All pools are evaluated with the exact variational energy over the HF product
state (pool_eval_exact.pool_energy_exact): the lowest generalized eigenvalue of
H_eff[a,b]=<HF|P_a^dag H P_b|HF>, N_eff[a,b]=<HF|P_a^dag P_b|HF>, with canonical
(Loewdin) orthogonalization. Because |HF> is a product state these matrix
elements are exact Hamiltonian matrix elements between basis states -- no MPS/MPO
truncation and no low-rank bridge approximation -- so every strategy is compared
on the same, strictly variational footing.

Three pool-construction strategies (answering the Referee's "at least one random
baseline and one chemistry-motivated or adaptive baseline"):
  ours       : MPS-sampled + curated pool (paper workflow).
  random     : random Pauli pool of matched size and weight distribution.
  chemistry  : qubit-excitation singles+doubles pool, frontier-ordered
               (see r26_chemistry_pool_C2H2.py).

Run (repo root, qusim env):
  python scripts/referee_baselines/r26_baselines_exact_C2H2.py
"""
from __future__ import annotations
import os
import json
import time
import numpy as np

from r26_baselines_C2H2 import (load_notebook_state, build_ours_pool,
                                build_random_pool, REPO)
from r26_chemistry_pool_C2H2 import build_chemistry_pool
from pool_eval_exact import pool_energy_exact

OUT = os.path.join(REPO, "scripts", "referee_baselines",
                   "r26_baselines_exact_C2H2.json")


def main():
    ns = load_notebook_state()                       # full setup incl. chi=8 DMRG
    Ha = ns["Ha"]
    hf_bits = ns["hf_bits_msb_left_ones_then_zeros_from_out"](ns["out"])
    psi_ref, E_dmrg = ns["psi"], float(ns["E_dmrg"])
    print(f"[exact] HF product state bits={hf_bits}")
    print(f"[exact] E_dmrg(chi=8) = {E_dmrg:.6f}")

    def E_of(pool):
        E, rank = pool_energy_exact(list(pool.keys()), Ha, hf_bits)
        return E, len(pool), rank

    runs = []
    if os.path.exists(OUT):
        runs = json.load(open(OUT)).get("runs", [])
    done = {(r["strategy"], r.get("n_samples"), r.get("seed"), r.get("target"))
            for r in runs}

    def save():
        json.dump({"molecule": "C2H2_sto3g_frozen", "chi": 8, "E_dmrg": E_dmrg,
                   "runs": runs}, open(OUT, "w"), indent=2)

    # ours + random: matched size/weight, 5 seeds per sample count
    N_GRID = [100, 200, 300, 500, 750, 1000]
    SEEDS = [0, 1, 2, 3, 4]
    for n_smp in N_GRID:
        for seed in SEEDS:
            ours = build_ours_pool(ns, psi_ref, n_smp, seed)
            rand = build_random_pool(ns, ours, seed)
            for strat, pool in (("ours", ours), ("random", rand)):
                key = (strat, n_smp, seed, None)
                if key in done:
                    continue
                t0 = time.perf_counter()
                E, npool, rank = E_of(pool)
                runs.append({"strategy": strat, "n_samples": n_smp, "seed": seed,
                             "target": None, "energy": E, "n_pool": npool,
                             "rank": rank, "dE": E - E_dmrg,
                             "wall_s": round(time.perf_counter() - t0, 2)})
                done.add(key)
                save()
                print(f"{strat:9s} n={n_smp:5d} seed={seed} |pool|={npool:4d} "
                      f"rank={rank:4d} E={E:.5f} dE={E-E_dmrg:+.4f}")

    # chemistry: qubit-excitation singles+doubles at increasing target sizes
    for tgt in [30, 50, 80, 110, 150, 200, 250, 300, 400]:
        key = ("chemistry", None, None, tgt)
        if key in done:
            continue
        pool = build_chemistry_pool(ns, tgt)
        t0 = time.perf_counter()
        E, npool, rank = E_of(pool)
        runs.append({"strategy": "chemistry", "n_samples": None, "seed": None,
                     "target": tgt, "energy": E, "n_pool": npool, "rank": rank,
                     "dE": E - E_dmrg, "wall_s": round(time.perf_counter()-t0, 2)})
        done.add(key)
        save()
        print(f"chemistry target={tgt:4d} |pool|={npool:4d} rank={rank:4d} "
              f"E={E:.5f} dE={E-E_dmrg:+.4f}")

    print("done ->", OUT)


if __name__ == "__main__":
    main()
