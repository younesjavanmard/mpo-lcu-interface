#!/usr/bin/env python
"""Rebuild the paper notebook state (Hamiltonian, MPO, HF MPS, DMRG helper,
curation rule) for an arbitrary MOLECULE_KEY, without running DMRG/bridge cells.
Mirrors scripts/referee_baselines/r26_baselines_C2H2.load_notebook_state."""
from __future__ import annotations
import os, sys, json, ast, time, contextlib, io
os.environ["DMRG_BACKEND"] = "numpy"
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
NB = os.path.join(REPO, "jupyter-notebooks", "tensor_networks",
                  "coefficient_decoupled_mpo_clean.ipynb")
SETUP = [2, 5, 6, 8, 9, 11, 12, 14, 22, 24, 30]
DEFS_ONLY = {24, 30}


def _defs_only(code):
    tree = ast.parse(code)
    defkinds = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    keepkinds = defkinds + (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)
    last = max(i for i, n in enumerate(tree.body) if isinstance(n, defkinds))
    tree.body = [n for i, n in enumerate(tree.body) if i <= last and isinstance(n, keepkinds)]
    return ast.unparse(ast.fix_missing_locations(tree))


def load(molecule_key: str, quiet=True):
    cells = json.load(open(NB))["cells"]
    ns = {"__name__": "__nb__"}
    scratch = os.path.join(os.path.dirname(__file__), "_scratch")
    os.makedirs(scratch, exist_ok=True)
    ns["RESULTS_DIR"] = ns["FIGS_DIR"] = scratch
    src = os.path.join(REPO, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    for idx in SETUP:
        code = "".join(cells[idx]["source"])
        code = code.replace('DEVICE = "auto"', 'DEVICE = "cpu"')
        if idx == 2:
            code = code.replace('MOLECULE_KEY   = "C2H2_sto3g_frozen"',
                                f'MOLECULE_KEY   = "{molecule_key}"')
            assert molecule_key in code
        if idx in DEFS_ONLY:
            code = _defs_only(code)
        if idx == 9:   # inject uniform H-chain series (round-2 size scaling)
            for nat in (4, 6, 8, 10, 12):
                for r in (1.0, 1.5):
                    ns["MOLECULES"][f"H{nat}_chain_r{r}"] = {
                        "kind": "linear_chain", "atoms": ("H",), "n_atoms": nat,
                        "r_default": r, "basis": "sto-3g", "charge": 0,
                        "multiplicity": 1, "frozen_core": 0, "n_active": nat}
        t0 = time.perf_counter()
        buf = io.StringIO()
        with (contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()):
            exec(compile(code, f"<nb-cell-{idx}>", "exec"), ns)
        print(f"[setup {molecule_key}] cell {idx:2d} ok ({time.perf_counter()-t0:.1f}s)", flush=True)
    return ns


def dmrg(ns, chi, quiet=True):
    buf = io.StringIO()
    with (contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext()):
        psi, E = ns["_dmrg_ground_state_from_mpo"](ns["wmpo"], chi_max=chi)
    return psi, E


OPS = np.stack([np.eye(2, dtype=complex),
                np.array([[0, 1], [1, 0]], dtype=complex),
                np.array([[0, -1j], [1j, 0]], dtype=complex),
                np.diag([1, -1]).astype(complex)])


def sample_pool(ns, psi, n_samples, seed, top_k_diag=5):
    """Paper workflow: perfect sampling -> curation -> + identity."""
    from qu_alg_qu_sim_tensor_networks import mps_helper
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = mps_helper.estimate_magic_perfect_from_mps(psi.Bs, OPS, n_samples=n_samples, seed=seed)
        pool = ns["filter_pauli_excitations"](res["samples"], top_k_diagonal=top_k_diag)
    pool["I" * ns["n_qubits"]] = 0.9
    return pool, res


def dmrg_from_hf(ns, chi, sweeps=None, eps=None, quiet=True):
    """Two-site DMRG started from the HF product MPS. H conserves (N_alpha, N_beta),
    so the variational state stays in the HF symmetry sector (no spin contamination
    from random initialization)."""
    import copy
    from qu_alg_qu_sim_tensor_networks import mps_helper, dmrg as _dmrg
    sweeps = ns["DMRG_SWEEPS"] if sweeps is None else sweeps
    eps = ns["DMRG_EPS"] if eps is None else eps
    psi = copy.deepcopy(ns["hf_psi"])
    model = mps_helper.vvdd_ddvv_mpo_swap(ns["wmpo"])
    eng = _dmrg.DMRGEngine(psi, model, chi_max=chi, eps=eps)
    E_prev = None
    for it in range(sweeps):
        eng.sweep()
        E = float(np.real(mps_helper.expectation_value(ns["wmpo"], psi)))
        if not quiet:
            print(f"[DMRG-HF chi={chi}] sweep {it}: E={E:.10f} chi={psi.get_chi()}", flush=True)
        if E_prev is not None and abs((E - E_prev) / E) < 1e-11:
            break
        E_prev = E
    return psi, E


def penalty_terms(n, n_alpha, n_beta, mu):
    """mu * [(N_a - n_a)^2 + (N_b - n_b)^2] as Pauli strings (alpha = even modes,
    beta = odd modes; string position k = fermionic mode k)."""
    terms = {}
    def add(s, v):
        terms[s] = terms.get(s, 0.0) + v
    for parity, target in ((0, n_alpha), (1, n_beta)):
        modes = [k for k in range(n) if k % 2 == parity]
        m = len(modes); c0 = m / 2 - target
        add("I" * n, mu * (c0 ** 2 + m / 4))
        for i in modes:
            s = ["I"] * n; s[i] = "Z"; add("".join(s), -mu * c0)
        for a_, i in enumerate(modes):
            for j in modes[a_ + 1:]:
                s = ["I"] * n; s[i] = "Z"; s[j] = "Z"; add("".join(s), 0.5 * mu)
    return terms


def dmrg_penalty(ns, chi, mu=1.0, quiet=True):
    """Randomly initialized two-site DMRG (paper settings) on H + symmetry penalty,
    which removes spin-polarized / wrong-particle-number local minima. Returns the
    MPS and its energy with respect to the bare Hamiltonian."""
    import copy
    from qu_alg_qu_sim_tensor_networks import mps_helper
    n = ns["n_qubits"]; bits = ns["hf_bits"]
    na, nb = bits[0::2].count("1"), bits[1::2].count("1")
    Hp = copy.deepcopy(ns["Ha"])
    for s, v in penalty_terms(n, na, nb, mu).items():
        Hp.terms[s] = Hp.terms.get(s, 0.0) + v
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        wmpo_p, _, _ = ns["get_mpo"](N_qubit=n, sys_Ham=Hp, epsilon=ns["MPO_EPS"],
                                     chi_max=ns["MPO_CHI_MAX"], show_info=False)
        psi, _ = ns["_dmrg_ground_state_from_mpo"](wmpo_p, chi_max=chi, verbose=False)
    E = float(np.real(mps_helper.expectation_value(ns["wmpo"], psi)))
    return psi, E
