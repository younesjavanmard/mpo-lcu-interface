#!/usr/bin/env python
"""
Referee-2 (highest-priority) point 6 --- infrastructure for the C2H2 pool
baseline comparison.

This module provides the shared machinery imported by the AUTHORITATIVE driver
r26_baselines_exact_C2H2.py:
  * load_notebook_state() -- rebuild the paper's in-notebook state (build_case ->
    Hamiltonian -> MPO -> HF MPS -> DMRG reference) by executing the setup cells
    of jupyter-notebooks/tensor_networks/coefficient_decoupled_mpo_clean.ipynb
    in-process (no code re-implemented here), so results stay consistent with the
    paper;
  * build_ours_pool()   -- MPS-sampled + curated pool (paper workflow);
  * build_random_pool() -- random pool of matched size and weight distribution.

NOTE on evaluation. This file also exposes bridge_energy(), which optimizes a
pool through the low-rank Option-C bridge, and its __main__ runs a bridge sweep.
We found the low-rank bridge is not variationally exact for arbitrary hand-built
pools (it does not even reproduce the Hartree--Fock energy for the identity
pool), so it is unreliable for a fair pool COMPARISON. The manuscript's baseline
figure is therefore produced by r26_baselines_exact_C2H2.py, which evaluates
every pool with the exact variational energy over the HF product state
(pool_eval_exact.py). Keep the bridge path only for reference/reproduction.

Run the authoritative comparison:
  python scripts/referee_baselines/r26_baselines_exact_C2H2.py
"""
from __future__ import annotations

import os
# CPU / NumPy everywhere: the bridge optimizer is CPU-bound and NumPy-only.
os.environ["DMRG_BACKEND"] = "numpy"

import json
import time
import argparse
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NB = os.path.join(REPO, "jupyter-notebooks", "tensor_networks",
                  "coefficient_decoupled_mpo_clean.ipynb")


# ----------------------------------------------------------------------------
# 1. Rebuild the paper's in-notebook state by executing its setup cells.
# ----------------------------------------------------------------------------
def load_notebook_state(defs_only=frozenset({35}), skip=frozenset()):
    """Exec the notebook setup cells in a shared namespace and return it.

    We run only the cells needed to define: config, imports, build_case, the
    Hamiltonian `Ha`, the compressed Hamiltonian MPO `wmpo`, the HF product MPS
    `hf_psi`, the Pauli op stack `ops`, `filter_pauli_excitations`, the DMRG
    helper `_dmrg_ground_state_from_mpo`, and `optimize_bridge_optionC_adaptive`.

    `defs_only` : cells for which only the top-level definitions are kept (their
    trailing driver statements are dropped). Cell 35 (bridge optimizer, ends with
    notebook-only driver code) is always included. Add cell 24 to skip the ~4 min
    DMRG reference when only the HF-based bridge is needed (e.g. the chemistry
    pool, which does not use the sampled reference state).
    """
    defs_only = set(defs_only) | {35}
    nb = json.load(open(NB))
    cells = nb["cells"]              # full cell list; indices below are full indices

    ns: dict = {"__name__": "__nb__"}
    # Full-notebook cell indices to execute (setup only; skip plotting/sweep).
    #   2  config vars      5  imports          6  helpers
    #   8  build_case defs  9  out=build_case  11  HF bits
    #  12  hf_psi          14  Ha              22  wmpo (get_mpo)
    #  24  DMRG helper + chi8 reference        26  ops + res_samples
    #  30  filter_pauli_excitations            35  bridge optimizer
    setup = [2, 5, 6, 8, 9, 11, 12, 14, 22, 24, 26, 30, 35]

    # RESULTS_DIR/FIGS_DIR are referenced by some cells; point them at scratch.
    scratch = os.path.join(REPO, "scripts", "referee_baselines", "_nb_scratch")
    os.makedirs(scratch, exist_ok=True)
    ns["RESULTS_DIR"] = scratch
    ns["FIGS_DIR"] = scratch

    # Make repo importable.
    src = os.path.join(REPO, "src")
    import sys
    if src not in sys.path:
        sys.path.insert(0, src)

    import ast
    for idx in setup:
        if idx in skip:
            continue
        assert cells[idx]["cell_type"] == "code", f"cell {idx} is not code"
        code = "".join(cells[idx]["source"])
        # Force CPU regardless of the notebook's DEVICE="auto".
        code = code.replace('DEVICE = "auto"', 'DEVICE = "cpu"')
        if idx in defs_only:
            # Keep ONLY the definitions (functions/classes/imports and the
            # `.last_info` sentinel assignments), dropping any trailing driver
            # statements (e.g. cell 35's `Gs` block, or cell 24's DMRG call).
            tree = ast.parse(code)
            defkinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            keepkinds = defkinds + (ast.Import, ast.ImportFrom,
                                    ast.Assign, ast.AnnAssign)
            last_def = max(i for i, n in enumerate(tree.body)
                           if isinstance(n, defkinds))
            # Everything up to the last def (imports, module constants such as
            # chi_dmrg = CHI_DMRG, the .last_info sentinels, and all defs); the
            # trailing driver block (best = ..., prints, DMRG call) is dropped.
            keep = [n for i, n in enumerate(tree.body)
                    if i <= last_def and isinstance(n, keepkinds)]
            keep += [n for i, n in enumerate(tree.body)
                     if i > last_def and isinstance(n, (ast.Import, ast.ImportFrom))]
            tree.body = keep
            code = ast.unparse(ast.fix_missing_locations(tree))
        t0 = time.perf_counter()
        exec(compile(code, f"<nb-cell-{idx}>", "exec"), ns)
        print(f"[setup] cell {idx:2d} ok ({time.perf_counter()-t0:.1f}s)")
    return ns


# ----------------------------------------------------------------------------
# 2. Pool builders.
# ----------------------------------------------------------------------------
def build_ours_pool(ns, psi_ref, n_smp, smp_seed):
    """MPS-sampled + curated pool (paper Sec. VI workflow)."""
    mps_helper = ns["mps_helper"]
    ops = ns["ops"]
    res = mps_helper.estimate_magic_perfect_from_mps(
        psi_ref.Bs, ops, n_samples=n_smp, seed=smp_seed)
    pool = ns["filter_pauli_excitations"](res["samples"],
                                          top_k_diagonal=ns["TOP_K_DIAGONAL"])
    pool["I" * ns["n_qubits"]] = 0.9
    return pool


def build_random_pool(ns, ref_pool, seed):
    """Random Pauli pool matched to `ref_pool` in size and weight distribution.

    For every string in the reference pool we emit a random Pauli string with the
    SAME weight (number of non-identity sites) placed at random positions, with
    random X/Y/Z at those positions. This matches |pool| and the histogram of
    Pauli weights exactly, but discards the MPS-derived support information --
    which is precisely the ablation the Referee asks for.
    """
    n = ns["n_qubits"]
    rng = np.random.default_rng(1000 + seed)
    ident = "I" * n
    out = {}
    for p, c in ref_pool.items():
        if p == ident:
            out[ident] = c            # keep the identity term as-is
            continue
        w = sum(ch != "I" for ch in p)          # weight of this reference string
        # draw a fresh random string of the same weight until it is new
        for _ in range(64):
            pos = rng.choice(n, size=w, replace=False)
            chars = ["I"] * n
            for j in pos:
                chars[j] = rng.choice(["X", "Y", "Z"])
            s = "".join(chars)
            if s not in out and s != ident:
                out[s] = c
                break
    return out


# ----------------------------------------------------------------------------
# 3. Run one pool through the bridge optimizer (identical settings to Fig. 5).
# ----------------------------------------------------------------------------
def bridge_energy(ns, pool, outer_rounds):
    ham = ns["ham"]
    gb = ns["gb"]
    n_qubits = ns["n_qubits"]

    Gp = ham.Hamiltonian(n_qubits=n_qubits)
    Gp.terms = dict(pool)

    _, _, _, Sg_seed_p, _ = gb.build_mpo_tensors_in_details(
        list(Gp.terms), n_qubits, Gp.as_dict, Gp.term_indices)
    (_, wgs_L_p, wgs_R_p, _, _, A_left_p, A_right_p, _
     ) = gb.build_mpo_tensors_in_details_truncated(
        list(Gp.terms), n_qubits, Gp.as_dict, Gp.term_indices,
        chi_max=360, epsilon=1e-16, normalize_halves=True, verbose=False)

    best = ns["optimize_bridge_optionC_adaptive"](
        n_qubits=n_qubits, qmpo=ns["qmpo"], mps_hf=ns["hf_psi"],
        wgs_L=wgs_L_p, wgs_R=wgs_R_p,
        A_left=A_left_p, A_right=A_right_p,
        Sg_old_seed=Sg_seed_p, wmpo_H=ns["wmpo"],
        rH=256, regN=1e-10, tol=1e-3, maxiter=200, block=6,
        outer_rounds=outer_rounds, add_per_round=1, max_pairs=50000,
        seed=ns["BRIDGE_SEED"], verbose=False,
        dense_threshold=1000, inner_method="lobpcg",
        plot_convergence=False,
    )
    return float(best["E0"]), len(pool)


# ----------------------------------------------------------------------------
# 4. Sweep.
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi", type=int, default=8)
    ap.add_argument("--samples", type=int, nargs="+",
                    default=[100, 300, 500, 1000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--adaptive-max-n", type=int, default=10**9,
                    help="skip the (expensive) adaptive pool for n_samples above "
                         "this; the residual-driven adaptive baseline is most "
                         "informative at low sample counts.")
    ap.add_argument("--out", default=os.path.join(
        REPO, "scripts", "referee_baselines", "r26_baselines_C2H2.json"))
    ap.add_argument("--time-one", action="store_true",
                    help="run a single (ours) point to measure wall time and exit")
    args = ap.parse_args()

    ns = load_notebook_state()
    dmrg_ref = ns["_dmrg_ground_state_from_mpo"]

    # Cell 24 already ran DMRG at CHI_DMRG (=8); reuse it when chi matches.
    if args.chi == ns.get("CHI_DMRG") and "psi" in ns and "E_dmrg" in ns:
        psi_ref, E_dmrg = ns["psi"], float(ns["E_dmrg"])
        print(f"[dmrg] reusing notebook chi={args.chi} reference")
    else:
        print(f"\n[dmrg] building chi={args.chi} reference ...")
        psi_ref, E_dmrg = dmrg_ref(ns["wmpo"], chi_max=args.chi, verbose=False)
    print(f"[dmrg] E_dmrg(chi={args.chi}) = {E_dmrg:.6f}")

    if args.time_one:
        pool = build_ours_pool(ns, psi_ref, 300, 0)
        t0 = time.perf_counter()
        E, npool = bridge_energy(ns, pool, outer_rounds=1)
        print(f"[time-one] ours n=300 seed=0: E={E:.6f} |pool|={npool} "
              f"wall={time.perf_counter()-t0:.1f}s")
        return

    # resume-friendly raw records
    raw = []
    if os.path.exists(args.out):
        raw = json.load(open(args.out)).get("runs", [])
    done = {(r["strategy"], r["n_samples"], r["seed"]) for r in raw}

    def save():
        json.dump({"molecule": "C2H2_sto3g_frozen", "chi": args.chi,
                   "E_dmrg": E_dmrg, "runs": raw,
                   "agg": aggregate(raw, E_dmrg)},
                  open(args.out, "w"), indent=2)

    total = 0
    for n_smp in args.samples:
        for seed in args.seeds:
            ours_pool = build_ours_pool(ns, psi_ref, n_smp, seed)
            rand_pool = build_random_pool(ns, ours_pool, seed)
            jobs = [
                ("ours",     ours_pool, 1),
                ("random",   rand_pool, 1),
            ]
            if n_smp <= args.adaptive_max_n:
                jobs.append(("adaptive", ours_pool, 8))
            for strat, pool, rounds in jobs:
                if (strat, n_smp, seed) in done:
                    continue
                t0 = time.perf_counter()
                E, npool = bridge_energy(ns, pool, outer_rounds=rounds)
                rec = {"strategy": strat, "n_samples": n_smp, "seed": seed,
                       "energy": E, "n_pool": npool,
                       "wall_s": round(time.perf_counter() - t0, 1)}
                raw.append(rec)
                done.add((strat, n_smp, seed))
                save()
                total += 1
                print(f"[{total:3d}] {strat:8s} n={n_smp:5d} seed={seed} "
                      f"E={E:.6f} |pool|={npool} ({rec['wall_s']}s)")
    print("done ->", args.out)


def aggregate(raw, E_dmrg):
    groups = {}
    for r in raw:
        groups.setdefault((r["strategy"], r["n_samples"]), []).append(r)
    agg = {}
    for (strat, n_smp), rs in sorted(groups.items()):
        es = np.array([r["energy"] for r in rs], float)
        ns_ = np.array([r["n_pool"] for r in rs], float)
        agg.setdefault(strat, []).append({
            "n_samples": n_smp, "n_seeds": len(rs),
            "E_mean": float(es.mean()),
            "E_std": float(es.std(ddof=1)) if len(es) > 1 else 0.0,
            "dE_mean": float(es.mean() - E_dmrg),
            "n_pool_mean": float(ns_.mean()),
        })
    return agg


if __name__ == "__main__":
    main()
