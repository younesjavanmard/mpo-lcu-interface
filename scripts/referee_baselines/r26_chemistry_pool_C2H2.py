#!/usr/bin/env python
"""
Referee-2 point 6 --- CHEMISTRY-MOTIVATED pool baseline for C2H2.

This adds the (optional) chemistry-motivated baseline to complement the random
and adaptive baselines of r26_baselines_C2H2.py. The pool is the standard
quantum-chemistry excitation pool: single and double excitations out of the
Hartree--Fock reference. We use the *qubit-excitation* encoding (Yordanov et al.,
Commun. Phys. 4, 228 (2021)) rather than the bare Jordan--Wigner mapping: each
excitation becomes a small set of X/Y-only Pauli strings supported on just the
2 (singles) or 4 (doubles) orbitals involved, with NO Jordan--Wigner Z-string
tail. This keeps the operators local across the central cut (so the bridge needs
no extra long-range connections) and free of the many Z/I sites that a bare-JW
pool would carry. Excitations are ordered by their orbital-energy gap from the
Fermi level (a Moller--Plesset / frontier-orbital screening: small-denominator
excitations first) and accumulated until a target pool size is reached, so the
chemistry pool is compared with the sampled pool at matched pool sizes.

The bridge is solved once on the fixed pool (outer_rounds=1): no residual-driven
connections are added, so the reported energy reflects the chemistry pool itself.

Each pool is fed through the IDENTICAL curation + bridge-optimization pipeline
used for the other baselines (functions imported from r26_baselines_C2H2.py); the
bridge optimizer uses the HF product state, so no DMRG reference is required here.

Run (repo root, qusim env):
  python scripts/referee_baselines/r26_chemistry_pool_C2H2.py
"""
from __future__ import annotations

import os
import json
import time
import numpy as np

from r26_baselines_C2H2 import load_notebook_state, REPO


def _spin_orbital_energies(ns):
    """(occ_so, virt_so, e_so) for the active space, spin-orbital interleaved."""
    out = ns["out"]
    mol = out["mol"]
    act = list(out["active_indices"])
    occ_frozen = list(out["occupied_indices"])
    n_qubits = int(out["n_qubits"])
    n_active_elec = int(mol.n_electrons) - 2 * len(occ_frozen)
    e_spatial = np.asarray(mol.orbital_energies)[act]        # local order 0..9
    # openfermion interleaves spin-orbitals: 2p = spatial p up, 2p+1 = down.
    e_so = np.empty(n_qubits)
    for p in range(len(act)):
        e_so[2 * p] = e_so[2 * p + 1] = e_spatial[p]
    occ_so = list(range(n_active_elec))          # HF fills lowest n_elec spin-orbs
    virt_so = list(range(n_active_elec, n_qubits))
    return occ_so, virt_so, e_so


def _excitations_by_gap(ns):
    """List of (gap, sites) for singles+doubles, sorted by orbital-energy gap.

    `sites` is the tuple of spin-orbitals involved (2 for a single, 4 for a
    double). Spin (S_z) is conserved. `gap` is the sum of virtual minus occupied
    orbital energies (the MP2 denominator), so the most important
    (small-denominator) excitations come first."""
    occ, virt, e = _spin_orbital_energies(ns)
    items = []
    # singles: i -> a, same spin
    for i in occ:
        for a in virt:
            if (i % 2) == (a % 2):
                items.append((float(e[a] - e[i]), (i, a)))
    # doubles: (i<j) -> (a<b), spin multiset conserved
    for ii in range(len(occ)):
        for jj in range(ii + 1, len(occ)):
            i, j = occ[ii], occ[jj]
            for aa in range(len(virt)):
                for bb in range(aa + 1, len(virt)):
                    a, b = virt[aa], virt[bb]
                    if sorted((i % 2, j % 2)) == sorted((a % 2, b % 2)):
                        items.append((float(e[a] + e[b] - e[i] - e[j]),
                                      (i, j, a, b)))
    items.sort(key=lambda t: t[0])
    return items


def _qubit_excitation_strings(sites, n):
    """Pauli strings of a qubit excitation on `sites` (Yordanov et al.).

    A qubit excitation on k spin-orbitals is the anti-Hermitian sum of all
    X/Y assignments to those sites with an ODD number of Y's (2^{k-1} strings):
    2 strings for a single (e.g. XY, YX) and 8 for a double. These carry no
    Jordan--Wigner Z-string, so their support is exactly `sites`."""
    from itertools import product
    out = []
    for combo in product("XY", repeat=len(sites)):
        if combo.count("Y") % 2 == 1:
            s = ["I"] * n
            for site, p in zip(sites, combo):
                s[site] = p
            out.append("".join(s))
    return out


def build_chemistry_pool(ns, target_size):
    """Accumulate qubit-excitation Pauli strings (gap-ordered) up to target size.

    No Z-string tails and no extra diagonal terms: the only I/Z-type term is the
    single identity added for the variational floor."""
    n = ns["n_qubits"]
    items = ns.setdefault("_chem_items", _excitations_by_gap(ns))
    pool = {}
    for gap, sites in items:
        for s in _qubit_excitation_strings(sites, n):
            pool.setdefault(s, 1.0)              # value only seeds the graph
        if len(pool) >= target_size:
            break
    pool["I" * n] = 0.9                          # single identity term
    return pool


def main():
    # This module is primarily a pool BUILDER (build_chemistry_pool), imported by
    # r26_baselines_exact_C2H2.py, which is the authoritative driver: it evaluates
    # every pool with the exact variational energy (pool_eval_exact) rather than
    # the low-rank bridge. Run that script to reproduce the manuscript figure.
    # Here we just build the pools and report their sizes and rank as a check.
    ns = load_notebook_state(defs_only={24}, skip={26, 30})
    for tgt in [30, 50, 80, 110, 150, 200, 300, 400]:
        pool = build_chemistry_pool(ns, tgt)
        n_diag = sum(all(c in "IZ" for c in s) for s in pool)
        print(f"[chem] target={tgt:4d} |pool|={len(pool):4d} (I/Z-only: {n_diag})")
    print("To evaluate energies, run r26_baselines_exact_C2H2.py")


if __name__ == "__main__":
    main()
