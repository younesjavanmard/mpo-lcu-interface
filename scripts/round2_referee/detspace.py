#!/usr/bin/env python
"""
Round-2 referee numerics: fast exact evaluation of Pauli pools over the HF determinant.

For a computational-basis reference |HF>=|h>, every Pauli string P with X-mask x_P
maps |h> to a phase times the determinant |h XOR x_P>.  Hence
    span{P_j|HF>} = span{|b> : b in D(U)},   D(U) = {h XOR x_P : P in U},
and the exact variational energy of the pool ansatz G(a)|HF> is the lowest
eigenvalue of H restricted to D(U) (a selected-CI problem).  This is identical to
pool_eval_exact.pool_energy_exact (the N_eff null space is exactly the set of
strings that hit the same determinant) but vectorized and much faster.

The same machinery gives FCI in the fixed (N_alpha, N_beta) sector.

Bit convention: string index i (MSB-left, as in Ha.terms keys and hf_bits) is
bit (n-1-i) of the integer.
"""
from __future__ import annotations
import itertools
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _parity(v):
    v = v.copy()
    for s in (32, 16, 8, 4, 2, 1):
        v ^= v >> s
    return v & 1


def pauli_masks(pstr):
    n = len(pstr)
    x = z = ny = 0
    for i, c in enumerate(pstr):
        bit = 1 << (n - 1 - i)
        if c in "XY":
            x |= bit
        if c in "ZY":
            z |= bit
        if c == "Y":
            ny += 1
    return x, z, ny


class QubitHamiltonian:
    """Vectorized Pauli-sum Hamiltonian acting on computational-basis integers."""

    def __init__(self, terms: dict, n: int):
        self.n = n
        keys = list(terms)
        m = np.array([pauli_masks(k) for k in keys], dtype=np.int64)
        self.x, self.z, ny = m[:, 0], m[:, 1], m[:, 2]
        coef = np.array([complex(terms[k]) for k in keys])
        self.c = coef * (1j ** (ny % 4))       # P|b> = i^ny (-1)^{b.z} |b^x>

    def row(self, b):
        """H|b> = sum_t amp_t |b ^ x_t>  -> (targets, amps)."""
        b = np.int64(b)
        sign = 1 - 2 * _parity(b & self.z)
        return b ^ self.x, self.c * sign

    def restricted(self, dets):
        """Sparse H restricted to the determinant list `dets` (sorted unique ints).
        Duplicates are summed per column; values must be real (molecular JW H)."""
        dets = np.asarray(dets, dtype=np.int64)
        K = len(dets)
        indptr = [0]; indices = []; data = []
        imag_max = 0.0
        for j, b in enumerate(dets):
            tgt, amp = self.row(b)
            pos = np.searchsorted(dets, tgt)
            pos[pos >= K] = 0
            hit = dets[pos] == tgt
            p, a = pos[hit], amp[hit]
            u, inv = np.unique(p, return_inverse=True)
            acc = np.zeros(len(u), dtype=complex)
            np.add.at(acc, inv, a)
            keep = np.abs(acc) > 1e-14
            u, acc = u[keep], acc[keep]
            if len(acc):
                imag_max = max(imag_max, float(np.abs(acc.imag).max()))
            indices.append(u.astype(np.int32)); data.append(acc.real)
            indptr.append(indptr[-1] + len(u))
        assert imag_max < 1e-10, f"non-real matrix elements ({imag_max})"
        # column-wise construction (CSC) of a symmetric matrix
        M = sp.csc_matrix((np.concatenate(data), np.concatenate(indices), np.array(indptr)),
                          shape=(K, K))
        return M.tocsr()


def ground_state(M, dense_max=3000):
    M = 0.5 * (M + M.T)
    K = M.shape[0]
    if K <= dense_max:
        w, V = np.linalg.eigh(M.toarray())
        return float(w[0]), V[:, 0]
    w, V = spla.eigsh(M, k=1, which="SA", tol=1e-12)
    return float(w[0]), V[:, 0]


def bits_to_int(bits: str) -> int:
    return int(bits, 2)


def pool_dets(pool_strings, hf_int):
    xs = np.array([pauli_masks(p)[0] for p in pool_strings], dtype=np.int64)
    return np.unique(np.int64(hf_int) ^ xs)


def pool_energy(H: QubitHamiltonian, pool_strings, hf_int):
    """Exact variational energy of G(a)|HF> over the pool, plus the CI vector."""
    dets = pool_dets(pool_strings, hf_int)
    E, c = ground_state(H.restricted(dets))
    return E, dets, c


def sector_dets(n, n_alpha, n_beta):
    """All determinants with given alpha/beta counts. JW order 2p=alpha, 2p+1=beta
    in the fermionic index; in MSB-left strings fermionic index k <-> string pos k."""
    n_sp = n // 2
    out = []
    for occ_a in itertools.combinations(range(n_sp), n_alpha):
        for occ_b in itertools.combinations(range(n_sp), n_beta):
            v = 0
            for p in occ_a:
                v |= 1 << (n - 1 - 2 * p)
            for p in occ_b:
                v |= 1 << (n - 1 - (2 * p + 1))
            out.append(v)
    return np.unique(np.array(out, dtype=np.int64))


def fci_energy(H: QubitHamiltonian, n_alpha, n_beta):
    dets = sector_dets(H.n, n_alpha, n_beta)
    E, c = ground_state(H.restricted(dets))
    return E, dets, c


def fragment_stats(pool_strings, cut=None):
    """Left/right fragment dictionaries, active edges, Select Clifford count."""
    pool_strings = list(pool_strings)
    n = len(pool_strings[0])
    cut = n // 2 if cut is None else cut
    L = {p[:cut] for p in pool_strings}
    R = {p[cut:] for p in pool_strings}
    E = {(p[:cut], p[cut:]) for p in pool_strings}
    weight = sum(sum(ch != "I" for ch in p) for p in pool_strings)
    return dict(n_left=len(L), n_right=len(R), n_edges=len(E),
                sum_weight=int(weight),
                sum_weight_LR=int(sum(sum(ch != "I" for ch in f) for f in L | R)))
