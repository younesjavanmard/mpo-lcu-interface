#!/usr/bin/env python
"""Figure: MPO compression is lossy -- reconstruction error vs retained bond dim.
Reads scripts/mpo_recovery/mpo_recovery.json (from mpo_recovery.py)."""
import os, json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# match the other manuscript figures (serif / STIX mathtext)
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 11, "legend.fontsize": 9,
})

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "scripts", "mpo_recovery", "mpo_recovery.json")
RESUB = os.environ.get("PAPER_OUT", os.path.join(REPO, "paper_outputs"))
for _sub in ("figs", "data", "tables"):
    os.makedirs(os.path.join(RESUB, _sub), exist_ok=True)
FIG = os.path.join(RESUB, "figs", "fig_mpo_recovery.pdf")
DATA_OUT = os.path.join(RESUB, "data", "mpo_recovery.json")

FLOOR = 1e-16
PANELS = [("LiH_sto3g_full", "(a) LiH  (12 qubits, 631 terms)", "#1f77b4"),
          ("random_12q_1000", "(b) random  (12 qubits, 877 terms)", "#d62728")]


def main():
    d = json.load(open(DATA))
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    for ax, (key, title, color) in zip(axes, PANELS):
        r = d[key]
        pts = r["points"]
        Dex = r["D_ours"]
        # curve for D < D_exact (error > 0)
        x = [p["D"] for p in pts if p["D"] < Dex]
        y = [max(p["rel_err"], FLOOR) for p in pts if p["D"] < Dex]
        ax.plot(x, y, "o-", color=color, ms=5, lw=1.6)
        # vertical line at the exact (lossless) bond dimension
        ax.axvline(Dex, ls="--", color="0.4", lw=1.2)
        ax.set_xscale("log"); ax.set_yscale("log")
        # compact rotated label just left of the line (drawn after scales set)
        ymid = np.sqrt(max(min(y), FLOOR) * max(y)) if y else 1e-2
        ax.text(Dex, ymid,
                rf"exact $D={Dex}$ (ours $=$ Renormalizer), error $\to 0$  ",
                rotation=90, ha="right", va="center", fontsize=8.5,
                color="0.3")
        ax.set_xlabel(r"retained MPO bond dimension $D$")
        ax.set_title(title, fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel(r"$\|H-H_D\|_F\,/\,\|H\|_F$")
    fig.tight_layout()
    fig.savefig(FIG)
    json.dump(d, open(DATA_OUT, "w"), indent=2)
    print("wrote", FIG)
    print("wrote", DATA_OUT)


if __name__ == "__main__":
    main()
