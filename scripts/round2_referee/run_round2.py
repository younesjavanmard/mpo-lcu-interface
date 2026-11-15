#!/usr/bin/env python
"""
Round-2 referee numerics (QST-105164), one task = (molecule, source bond dim chi).

For the source state |psi_chi> (chi=1 means the HF product state itself; chi>1 a
DMRG MPS) and each (N_samples, seed): sample Pauli strings from
Pi(P)=<psi|P|psi>^2/2^N (paper Alg. 2), curate (paper rule), and evaluate the pool
EXACTLY over |HF> (selected-CI in the reached determinant set, detspace.py).

Recorded per point: pool energy, error vs FCI, fidelity with FCI, number of
reached determinants, Lambda_min = ||c||_1 (optimal LCU one-norm for the
normalized output), p_succ = 1/Lambda_min^2, fragment/edge statistics.
Recorded per task: E_HF, E_FCI, E_src (MPS energy), source fidelity with FCI.

Usage: python run_round2.py --mol C2H2_sto3g_frozen --chi 8 --nsamples 250,500 --seeds 0,1,2,3,4
"""
from __future__ import annotations
import os, sys, json, time, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
from nbstate import load, dmrg, dmrg_penalty, sample_pool
from detspace import (QubitHamiltonian, pool_energy, fci_energy, bits_to_int,
                      fragment_stats, pool_dets)


def mps_amplitudes(psi, dets, n):
    """<det|psi> for all dets (MSB-left: site j <-> bit n-1-j)."""
    Bs = [np.asarray(B) for B in psi.Bs]
    v = np.ones((len(dets), 1), dtype=complex)
    for j, B in enumerate(Bs):
        s = (dets >> (n - 1 - j)) & 1
        v = np.einsum("ka,kab->kb", v, B[:, s, :].transpose(1, 0, 2))
    return v[:, 0]


def overlap_sq(dets_a, c_a, dets_b, c_b):
    pos = np.searchsorted(dets_b, dets_a)
    pos[pos >= len(dets_b)] = 0
    hit = dets_b[pos] == dets_a
    return float(abs(np.vdot(c_b[pos[hit]], c_a[hit])) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", required=True)
    ap.add_argument("--chi", type=int, required=True)
    ap.add_argument("--nsamples", default="250,500,1000,2000,4000,8000")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"r2_{a.mol}_chi{a.chi}.json")
    rec = json.load(open(fn)) if os.path.exists(fn) else {"molecule": a.mol, "chi": a.chi, "points": []}
    done = {(p["n_samples"], p["seed"]) for p in rec["points"]}

    ns = load(a.mol)
    n = ns["n_qubits"]; hf_bits = ns["hf_bits"]; h = bits_to_int(hf_bits)
    H = QubitHamiltonian(ns["Ha"].terms, n)
    nel = hf_bits.count("1")
    t0 = time.time()
    E_hf, _, _ = pool_energy(H, ["I" * n], h)
    E_fci, d_fci, c_fci = fci_energy(H, nel // 2, nel // 2)
    rec.update(n_qubits=n, n_terms=len(ns["Ha"].terms), E_HF=E_hf, E_FCI=E_fci,
               fci_dim=len(d_fci), t_fci=round(time.time() - t0, 1))
    print(f"[{a.mol}] n={n} E_HF={E_hf:.8f} E_FCI={E_fci:.8f} dim={len(d_fci)}", flush=True)

    if a.chi == 1:
        psi, E_src = ns["hf_psi"], E_hf
        rec["dmrg_mode"] = "hf"
        amp = mps_amplitudes(psi, d_fci, n)
    else:
        # paper protocol (random init); fall back to the symmetry-penalty DMRG if the
        # variational state left the HF (N_alpha, N_beta) sector (spin-polarized minimum)
        t0 = time.time()
        psi, E_src = dmrg(ns, a.chi)
        amp = mps_amplitudes(psi, d_fci, n)
        rec["dmrg_rand"] = dict(E=float(E_src), sector_norm=float(np.vdot(amp, amp).real))
        rec["dmrg_mode"] = "random"
        if rec["dmrg_rand"]["sector_norm"] < 0.99:
            print(f"[{a.mol} chi={a.chi}] random-init DMRG left the HF sector "
                  f"(norm {rec['dmrg_rand']['sector_norm']:.4f}) -> penalty DMRG", flush=True)
            psi, E_src = dmrg_penalty(ns, a.chi, mu=1.0)
            amp = mps_amplitudes(psi, d_fci, n)
            rec["dmrg_mode"] = "penalty"
        rec["t_dmrg"] = round(time.time() - t0, 1)
    rec.update(E_src=float(E_src), src_norm_in_sector=float(np.vdot(amp, amp).real),
               F_src=float(abs(np.vdot(c_fci, amp)) ** 2))
    print(f"[{a.mol} chi={a.chi}] E_src={E_src:.8f} dE={1e3*(E_src-E_fci):.3f} mHa "
          f"F_src={rec['F_src']:.5f} norm_sector={rec['src_norm_in_sector']:.6f}", flush=True)
    json.dump(rec, open(fn, "w"), indent=1)

    for N in [int(x) for x in a.nsamples.split(",")]:
        for seed in [int(x) for x in a.seeds.split(",")]:
            if (N, seed) in done:
                continue
            t0 = time.time()
            pool, _ = sample_pool(ns, psi, N, seed)
            strings = list(pool)
            E, dets, c = pool_energy(H, strings, h)
            c = c / np.linalg.norm(c)
            xy = [p for p in strings if ("X" in p or "Y" in p)]
            exc = [sum(ch in "XY" for ch in p) // 2 for p in xy]
            pt = dict(n_samples=N, seed=seed, pool_size=len(strings), n_xy=len(xy),
                      max_exc=max(exc) if exc else 0,
                      exc_hist={str(k): exc.count(k) for k in sorted(set(exc))},
                      n_dets=len(dets), energy=E, dE_fci=E - E_fci, dE_src=E - E_src,
                      F_fci=overlap_sq(dets, c, d_fci, c_fci),
                      Lambda_min=float(np.abs(c).sum()),
                      p_succ=float(1.0 / np.abs(c).sum() ** 2),
                      **fragment_stats(strings), wall_s=round(time.time() - t0, 2))
            rec["points"].append(pt); done.add((N, seed))
            json.dump(rec, open(fn, "w"), indent=1)
            print(f"[{a.mol} chi={a.chi}] N={N} seed={seed} |U|={len(strings)} dets={len(dets)} "
                  f"dE_fci={1e3*pt['dE_fci']:.3f} dE_src={1e3*pt['dE_src']:+.3f} mHa "
                  f"F={pt['F_fci']:.4f} Lam={pt['Lambda_min']:.3f} p={pt['p_succ']:.3f} "
                  f"({pt['wall_s']}s)", flush=True)
    print("done", fn, flush=True)


if __name__ == "__main__":
    main()
