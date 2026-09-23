#!/usr/bin/env python
"""
MPO compression is lossy in the Pauli-string picture: discarding MPO bond
dimension means the Hamiltonian cannot be recovered exactly.

For a molecule (LiH, 12 qubits) and a random 12-qubit Pauli-sum Hamiltonian
(~1000 terms), we:
  1. build the exact Hamiltonian MPO two ways -- with our builder (get_mpo) and
     with the Renormalizer package -- and confirm the (lossless) bond dimensions
     agree; this common value D_exact is the minimal bond dimension that
     represents H exactly.
  2. compress our MPO to a capped bond dimension D <= D_exact (SVD truncation,
     i.e. discarding the smallest singular values / least information), convert
     it back to a Pauli-string operator H_D, and measure the relative
     reconstruction error ||H - H_D||_F / ||H||_F.

The Frobenius norms use MPO overlaps tr(A^dag B) contracted as transfer
matrices, so no 2^N dense matrix is ever formed. The error vanishes only at
D = D_exact: any discarded bond dimension leaves H unrecoverable.

Run (repo root, qusim env):
  python scripts/mpo_recovery/mpo_recovery.py
"""
from __future__ import annotations
import os
import json
import numpy as np

os.environ["DMRG_BACKEND"] = "numpy"
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NB = os.path.join(REPO, "jupyter-notebooks", "tensor_networks",
                  "coefficient_decoupled_mpo_clean.ipynb")
OUT = os.path.join(REPO, "scripts", "mpo_recovery", "mpo_recovery.json")


def load_builders():
    """Exec notebook cells [2,5,6,8] to get build_case + helpers + get_mpo."""
    import sys
    sys.path.insert(0, os.path.join(REPO, "src"))
    nb = json.load(open(NB))
    cells = nb["cells"]
    ns = {"__name__": "__nb__", "RESULTS_DIR": "/tmp", "FIGS_DIR": "/tmp"}
    for idx in [2, 5, 6, 8]:            # config, imports, helpers, build_case defs
        code = "".join(cells[idx]["source"]).replace('DEVICE = "auto"',
                                                      'DEVICE = "cpu"')
        exec(compile(code, f"<c{idx}>", "exec"), ns)
    from qu_alg_qu_sim_tensor_networks.hamiltonians import (
        renormalizer_model_from_hamiltonian)
    ns["renormalizer_model_from_hamiltonian"] = renormalizer_model_from_hamiltonian
    return ns


def mpo_overlap(A, B):
    """tr(A^dag B) for two MPOs given as lists of (Dl,o,s,Dr) tensors."""
    E = np.ones((1, 1), dtype=complex)
    for a, b in zip(A, B):
        assert a.ndim == 4 and b.ndim == 4
        E = np.einsum("AB,osAC,osBD->CD", E, np.conj(a), b, optimize=True)
    return complex(E[0, 0])


def bonds(Ms):
    return [1] + [int(m.shape[-1]) for m in Ms]


def study(ns, name, H, n, d_list=None):
    ham = ns["ham"]
    get_mpo = ns["get_mpo"]
    BIG = 100000

    # exact MPO (our builder), lossless
    wmpo, _, _ = get_mpo(N_qubit=n, sys_Ham=H, epsilon=1e-13, chi_max=BIG,
                         show_info=False)
    Ms_exact = [np.asarray(m) for m in wmpo.Ms]
    D_ours = max(bonds(Ms_exact))

    # Renormalizer MPO bond dimension (its own exact construction)
    try:
        _, rmpo, _ = ns["renormalizer_model_from_hamiltonian"](H, build_mpo=True)
        D_renorm = int(max(rmpo.bond_dims))
    except Exception as e:
        D_renorm = None
        print(f"  [warn] renormalizer failed: {e}")

    # sanity: <H,H> should equal 2^n * sum|c|^2 (Pauli orthogonality)
    normH2 = mpo_overlap(Ms_exact, Ms_exact).real
    sum_c2 = sum(abs(c) ** 2 for c in H.terms.values())
    print(f"[{name}] n={n} terms={len(H.terms)} D_ours={D_ours} "
          f"D_renorm={D_renorm}  <H,H>={normH2:.6g}  2^n*S|c|^2={2**n*sum_c2:.6g}")

    if d_list is None:
        cand = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 40, 48, 64, 96,
                128, 160, 200, 256]
        d_list = sorted({d for d in cand if d < D_ours} | {D_ours})

    pts = []
    for D in d_list:
        wD, _, _ = get_mpo(N_qubit=n, sys_Ham=H, epsilon=1e-13, chi_max=int(D),
                           show_info=False)
        Ms_D = [np.asarray(m) for m in wD.Ms]
        cross = mpo_overlap(Ms_exact, Ms_D).real
        normD2 = mpo_overlap(Ms_D, Ms_D).real
        err2 = normH2 + normD2 - 2.0 * cross
        rel = float(np.sqrt(max(err2, 0.0)) / np.sqrt(normH2))
        pts.append({"D": int(D), "D_actual": max(bonds(Ms_D)), "rel_err": rel})
        print(f"    D={D:4d} (actual {pts[-1]['D_actual']:4d})  "
              f"rel_err={rel:.3e}")
    return {"name": name, "n": n, "n_terms": len(H.terms),
            "D_ours": D_ours, "D_renorm": D_renorm, "points": pts}


def main():
    ns = load_builders()
    ham = ns["ham"]
    results = {}

    # (a) LiH molecule, 12 qubits
    out = ns["build_case"]("LiH_sto3g_full", transform="jw", rotate=False)
    n = int(out["n_qubits"])
    H_lih = ham.from_openfermion(out["qubit_h"], n)
    results["LiH_sto3g_full"] = study(ns, "LiH (12q)", H_lih, n)

    # (b) random Pauli-sum Hamiltonian, 12 qubits, ~1000 terms
    rng = np.random.default_rng(1234)
    H_rand = ns["build_random_like_renormalizer"](12, 1000, rng)
    results["random_12q_1000"] = study(ns, "random (12q, 1000 terms)",
                                        H_rand, 12)

    json.dump(results, open(OUT, "w"), indent=2)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
