#!/usr/bin/env python
"""Multi-molecule validation: our MPO bond dimension vs the Renormalizer package,
across nine molecular Hamiltonians (H2 ... C2H2). Reads the precomputed
data/mpo_bonddim_vs_renormalizer.json and renders a bar comparison of the maximal
bond dimension per system (serif fonts, to match the other manuscript figures)."""
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "legend.fontsize": 9,
})
RESUB = os.environ.get("PAPER_OUT",
    os.path.join(os.path.dirname(__file__),"..","..","paper_outputs"))
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# input data ships in the package under results/
DATA = os.path.join(REPO, "results", "mpo_bonddim_vs_renormalizer.json")
FIG = os.path.join(RESUB, "figs", "fig_mpo_bonddim_vs_renormalizer.pdf")
os.makedirs(os.path.join(RESUB,"figs"), exist_ok=True)

NICE = {"H2_sto3g":"H$_2$","LiH_sto3g_small":"LiH$^\\ast$","LiH_sto3g_medium":"LiH$^{\\ast\\ast}$",
        "BeH_sto3g_small":"BeH","LiH_sto3g_full":"LiH","HF_sto3g_full":"HF",
        "H2O_sto3g_full_active":"H$_2$O","N2_sto3g_full":"N$_2$","C2H2_sto3g_frozen":"C$_2$H$_2$"}

def main():
    d = json.load(open(DATA))
    items = [(NICE.get(k,k), v["our_max"], v["renorm_max"])
             for k,v in d.items() if v.get("our_max") is not None]
    # order by size
    items.sort(key=lambda t: t[2])
    labels=[i[0] for i in items]; ours=[i[1] for i in items]; ren=[i[2] for i in items]
    x=np.arange(len(labels)); w=0.4
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    ax.bar(x-w/2, ren, w, label="Renormalizer", color="0.6")
    ax.bar(x+w/2, ours, w, label="ours (SVD-compressed)", color="#1f77b4")
    for xi,o,r in zip(x,ours,ren):
        if o!=r:
            ax.annotate(f"{o} vs {r}", (xi, max(o,r)), textcoords="offset points",
                        xytext=(0,3), ha="center", fontsize=8, color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("max MPO bond dimension")
    ax.set_title("Hamiltonian-MPO bond dimension: ours vs Renormalizer")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(FIG)
    print("wrote", FIG)
    print("agreement:", sum(o==r for _,o,r in items), "/", len(items),
          "| mismatches:", [(l,o,r) for l,o,r in items if o!=r])

if __name__=="__main__":
    main()
