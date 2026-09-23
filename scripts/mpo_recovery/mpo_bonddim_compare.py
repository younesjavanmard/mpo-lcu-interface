#!/usr/bin/env python
"""Validation: our Hamiltonian-MPO builder vs the Renormalizer package.

For LiH (12q) and a random 12q Pauli-sum Hamiltonian, build the exact MPO with
both methods and compare the bond dimension at every cut. Agreement at all cuts
confirms both builders produce the same (minimal, lossless) MPO.
"""
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpo_recovery import load_builders, bonds

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "legend.fontsize": 9,
})
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESUB = os.environ.get("PAPER_OUT", os.path.join(REPO, "paper_outputs"))
for _sub in ("figs", "data", "tables"):
    os.makedirs(os.path.join(RESUB, _sub), exist_ok=True)
FIG = os.path.join(RESUB, "figs", "fig_mpo_vs_renormalizer.pdf")
DATA_OUT = os.path.join(RESUB, "data", "mpo_vs_renormalizer.json")


def profiles(ns, H, n):
    wmpo, _, _ = ns["get_mpo"](N_qubit=n, sys_Ham=H, epsilon=1e-13,
                               chi_max=100000, show_info=False)
    ours = bonds([np.asarray(m) for m in wmpo.Ms])
    _, rmpo, _ = ns["renormalizer_model_from_hamiltonian"](H, build_mpo=True)
    renorm = [int(b) for b in rmpo.bond_dims]
    return ours, renorm


def main():
    ns = load_builders()
    ham = ns["ham"]
    out = ns["build_case"]("LiH_sto3g_full", transform="jw", rotate=False)
    n = int(out["n_qubits"])
    H_lih = ham.from_openfermion(out["qubit_h"], n)
    rng = np.random.default_rng(1234)
    H_rand = ns["build_random_like_renormalizer"](12, 1000, rng)

    data = {}
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    panels = [("LiH (12q)", H_lih, n, "#1f77b4"),
              ("random (12q, 877 terms)", H_rand, 12, "#d62728")]
    for ax, (name, H, nn, color) in zip(axes, panels):
        ours, renorm = profiles(ns, H, nn)
        L = min(len(ours), len(renorm))
        x = list(range(L))
        ax.plot(x, renorm[:L], "-", color="0.35", lw=3.2, alpha=0.55,
                label="Renormalizer")
        ax.plot(x, ours[:L], "o", color=color, ms=5.5, label="ours")
        match = ours[:L] == renorm[:L]
        data[name] = {"ours": ours, "renorm": renorm, "match": bool(match),
                      "max_ours": max(ours), "max_renorm": max(renorm)}
        ax.set_title(f"{name}: max $D=${max(ours)} "
                     + (r"($\checkmark$ match)" if match else "(MISMATCH)"),
                     fontsize=10)
        ax.set_xlabel("cut index")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(frameon=False)
    axes[0].set_ylabel("MPO bond dimension")
    fig.tight_layout()
    fig.savefig(FIG)
    json.dump(data, open(DATA_OUT, "w"), indent=2)
    print("wrote", FIG)
    for k, v in data.items():
        print(f"  {k}: match={v['match']}  max ours={v['max_ours']} "
              f"renorm={v['max_renorm']}")


if __name__ == "__main__":
    main()
