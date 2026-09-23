#!/usr/bin/env python
"""
Exact variational evaluation of a Pauli-string pool over the HF product state.

The ansatz is |psi(a)> = G(a)|HF> with G(a) = sum_P a_P P and |HF> a single
computational-basis (product) state. The optimal energy is the lowest generalized
eigenvalue of

    H_eff[a,b] = <HF| P_a^dag H P_b |HF>,   N_eff[a,b] = <HF| P_a^dag P_b |HF>.

Because |HF> is a product state, P_b|HF> is again a signed basis state, so every
matrix element is an exact Hamiltonian matrix element between two basis states --
no MPS/MPO truncation and no low-rank bridge approximation. This gives a faithful,
strictly variational energy for ANY pool (used to benchmark pool-construction
strategies fairly). Canonical (Loewdin) orthogonalization removes the N_eff null
space before diagonalizing, exactly as in the manuscript's numerical safeguard.
"""
from __future__ import annotations
import numpy as np

# Pauli action on a single qubit basis value b in {0,1}: returns (new_b, phase).
_PAULI = {
    "I": lambda b: (b, 1.0 + 0j),
    "Z": lambda b: (b, 1.0 + 0j if b == 0 else -1.0 + 0j),
    "X": lambda b: (1 - b, 1.0 + 0j),
    "Y": lambda b: (1 - b, (1j if b == 0 else -1j)),
}


def apply_pauli(pstr, bits):
    """Apply full-length Pauli string to a bit tuple -> (new_bits, phase)."""
    out = list(bits)
    ph = 1.0 + 0j
    for i, P in enumerate(pstr):
        nb, p = _PAULI[P](bits[i])
        out[i] = nb
        ph *= p
    return tuple(out), ph


def _H_terms(Ha):
    """Hamiltonian as list of (pauli_string, complex_coeff)."""
    return [(t, complex(c)) for t, c in Ha.terms.items()]


def build_effective_matrices(pool_strings, Ha, hf_bits):
    """Return dense (H_eff, N_eff) for the pool over |HF>=hf_bits."""
    K = len(pool_strings)
    hf = tuple(int(x) for x in hf_bits)
    # phi_k = P_k|HF> = (bitstring, phase)
    phis = [apply_pauli(P, hf) for P in pool_strings]

    # index basis states appearing among the phi_k
    idx = {}
    for b, _ in phis:
        idx.setdefault(b, len(idx))
    # N_eff
    N = np.zeros((K, K), dtype=complex)
    for a in range(K):
        ba, pa = phis[a]
        for b in range(K):
            bb, pb = phis[b]
            if ba == bb:
                N[a, b] = np.conj(pa) * pb

    # H|phi_b> as dict bit->amp, then H_eff[a,b] = conj(pa) * (H|phi_b>)[ba]
    Hterms = _H_terms(Ha)
    Hphi = []
    for b in range(K):
        bb, pb = phis[b]
        acc = {}
        for q, h in Hterms:
            nb, ph = apply_pauli(q, bb)
            acc[nb] = acc.get(nb, 0j) + h * ph * pb
        Hphi.append(acc)
    H = np.zeros((K, K), dtype=complex)
    for a in range(K):
        ba, pa = phis[a]
        cpa = np.conj(pa)
        for b in range(K):
            amp = Hphi[b].get(ba)
            if amp is not None:
                H[a, b] = cpa * amp
    return H, N


def solve_gep_canonical(H, N, rcond=1e-10):
    """Lowest generalized eigenvalue via canonical (Loewdin) orthogonalization."""
    N = 0.5 * (N + N.conj().T)
    H = 0.5 * (H + H.conj().T)
    s, U = np.linalg.eigh(N)
    smax = s.max()
    keep = s > rcond * smax
    X = U[:, keep] / np.sqrt(s[keep])
    Hs = X.conj().T @ H @ X
    w = np.linalg.eigvalsh(0.5 * (Hs + Hs.conj().T))
    return float(w[0]), int(keep.sum())


def pool_energy_exact(pool_strings, Ha, hf_bits, rcond=1e-10):
    H, N = build_effective_matrices(list(pool_strings), Ha, hf_bits)
    E, rank = solve_gep_canonical(H, N, rcond=rcond)
    return E, rank
