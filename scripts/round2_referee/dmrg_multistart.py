"""Multi-start symmetry-penalty DMRG for strongly correlated sources (stretched H chains).

The single random start used by the paper protocol (seed 1234) can converge to a
spin-polarized sector or, with the (N_alpha, N_beta) penalty, to a poor local
minimum in the correct sector. Here we run the penalized two-site DMRG from
several random initial MPSs (more sweeps) and keep the lowest bare-H energy.

Usage: python dmrg_multistart.py --mol H8_chain_r1.5 --chi 32 --seed 1234
Writes results_src/src_{mol}_chi{chi}_seed{seed}.pkl  (MPS + energies).
"""
import os, sys, io, json, time, pickle, argparse, contextlib, copy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from nbstate import load, penalty_terms  # noqa: E402


def penalized_dmrg(ns, chi, seed, mu=1.0, sweeps=40, init_chi=None):
    from qu_alg_qu_sim_tensor_networks import mps_helper
    n = ns["n_qubits"]; bits = ns["hf_bits"]
    na, nb = bits[0::2].count("1"), bits[1::2].count("1")
    Hp = copy.deepcopy(ns["Ha"])
    for s, v in penalty_terms(n, na, nb, mu).items():
        Hp.terms[s] = Hp.terms.get(s, 0.0) + v
    with contextlib.redirect_stdout(io.StringIO()):
        wmpo_p, _, _ = ns["get_mpo"](N_qubit=n, sys_Ham=Hp, epsilon=ns["MPO_EPS"],
                                     chi_max=ns["MPO_CHI_MAX"], show_info=False)
    kw = dict(chi_max=chi, sweeps=sweeps, seed=seed, verbose=False)
    if init_chi is not None:
        kw["init_chi"] = init_chi
    psi, _ = ns["_dmrg_ground_state_from_mpo"](wmpo_p, **kw)
    E = float(np.real(mps_helper.expectation_value(ns["wmpo"], psi)))
    Ep = float(np.real(mps_helper.expectation_value(wmpo_p, psi)))
    return psi, E, Ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mol", required=True)
    ap.add_argument("--chi", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--sweeps", type=int, default=40)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--out", default=os.path.join(HERE, "results_src"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ns = load(a.mol)
    t0 = time.time()
    psi, E, Ep = penalized_dmrg(ns, a.chi, a.seed, mu=a.mu, sweeps=a.sweeps)
    rec = dict(mol=a.mol, chi=a.chi, seed=a.seed, sweeps=a.sweeps, mu=a.mu,
               E=E, E_penalized=Ep, t=time.time() - t0)
    print(json.dumps(rec), flush=True)
    fn = os.path.join(a.out, f"src_{a.mol}_chi{a.chi}_seed{a.seed}.pkl")
    pickle.dump(dict(rec=rec, Bs=psi.Bs, Ss=psi.Ss, psi=psi), open(fn, "wb"))


if __name__ == "__main__":
    main()
