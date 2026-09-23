#!/usr/bin/env python
"""
Build the Referee-2 point-6 baseline-comparison figure and LaTeX table from
scripts/referee_baselines/r26_baselines_exact_C2H2.json (produced by
r26_baselines_exact_C2H2.py).

All strategies are evaluated with the exact variational energy over the HF
product state (faithful, strictly variational), and plotted against the pool size
(number of Pauli operators) -- the fair resource metric.

Outputs (into the resubmission tree):
  figs/fig_baseline_pools_C2H2.pdf
  tables/tab_baseline_pools.tex
  data/baseline_pools_C2H2.json
"""
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Match the font of the other manuscript figures (Fig. 5, polar map): serif body
# with STIX mathtext, TrueType-embedded, so Fig. 4 is typographically consistent.
mpl.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["STIXGeneral", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "font.size":        11,
    "legend.fontsize":  9,
})

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(REPO, "scripts", "referee_baselines",
                    "r26_baselines_exact_C2H2.json")
RESUB = os.environ.get("PAPER_OUT", os.path.join(REPO, "paper_outputs"))
for _sub in ("figs", "data", "tables"):
    os.makedirs(os.path.join(RESUB, _sub), exist_ok=True)
FIG = os.path.join(RESUB, "figs", "fig_baseline_pools_C2H2.pdf")
DATA_OUT = os.path.join(RESUB, "data", "baseline_pools_C2H2.json")
TAB = os.path.join(RESUB, "tables", "tab_baseline_pools.tex")

LABEL = {"ours": r"MPS-sampled (ours)",
         "chemistry": r"chemistry (singles+doubles)",
         "random": r"random (matched size/weight)"}
COLOR = {"ours": "#1f77b4", "random": "#d62728", "chemistry": "#2ca02c"}
MARK = {"ours": "s", "random": "o", "chemistry": "^"}


def agg_ours_random(runs, strat, E_dmrg):
    """Aggregate ours/random by sample count -> (pool_size, dE, dE_std)."""
    pts = []
    for n in sorted({r["n_samples"] for r in runs
                     if r["strategy"] == strat and r["n_samples"]}):
        rs = [r for r in runs if r["strategy"] == strat and r["n_samples"] == n]
        E = np.array([r["energy"] for r in rs])
        npool = np.array([r["n_pool"] for r in rs])
        sd = float(E.std(ddof=1)) if len(E) > 1 else 0.0
        pts.append({"n_samples": n, "n_pool": float(npool.mean()),
                    "n_seeds": len(E),
                    "E_mean": float(E.mean()),
                    "E_std": sd,
                    "E_sem": sd / np.sqrt(len(E)) if len(E) > 1 else 0.0,
                    "dE_mean": float(E.mean() - E_dmrg)})
    return pts


def agg_chemistry(runs, E_dmrg):
    pts = [{"n_pool": r["n_pool"], "E_mean": r["energy"], "E_std": 0.0,
            "E_sem": 0.0, "dE_mean": r["energy"] - E_dmrg}
           for r in runs if r["strategy"] == "chemistry"]
    return sorted(pts, key=lambda p: p["n_pool"])


def main():
    d = json.load(open(DATA))
    runs, E_dmrg = d["runs"], d["E_dmrg"]
    agg = {"ours": agg_ours_random(runs, "ours", E_dmrg),
           "random": agg_ours_random(runs, "random", E_dmrg),
           "chemistry": agg_chemistry(runs, E_dmrg)}
    E_hf = -75.85339

    YFLOOR = 5e-4
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    for strat in ["ours", "chemistry", "random"]:
        pts = agg[strat]
        x = [p["n_pool"] for p in pts]
        y = np.array([max(p["dE_mean"], YFLOOR) for p in pts])
        # error bar = +/-1 standard deviation over the sampling seeds (run-to-run
        # variability), consistent with Fig. 5
        sd = np.array([p["E_std"] for p in pts])
        # lower whisker clipped at the axis floor so a small mean does not
        # produce a whisker running off the bottom of the log axis
        lower = y - np.maximum(y - sd, YFLOOR)
        ax.errorbar(x, y, yerr=[lower, sd], fmt=MARK[strat] + "-",
                    color=COLOR[strat], capsize=2.5, ms=6, lw=1.6,
                    elinewidth=1.0, label=LABEL[strat])
    ax.axhline(E_hf - E_dmrg, ls="--", color="0.5", lw=1.0,
               label="Hartree–Fock")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=YFLOOR)
    ax.set_xlabel(r"pool size (number of Pauli operators)")
    ax.set_ylabel(r"$E - E_{\mathrm{DMRG}}$ (Ha)")
    ax.set_title(r"C$_2$H$_2$ pool baselines ($\chi=8$, exact evaluation)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG, bbox_inches="tight", pad_inches=0.02)
    print("wrote", FIG)

    json.dump({"molecule": d["molecule"], "chi": d["chi"], "E_dmrg": E_dmrg,
               "E_hf": E_hf, "agg": agg}, open(DATA_OUT, "w"), indent=2)
    print("wrote", DATA_OUT)

    # ---- LaTeX table: dE (mHa) vs pool size, one block per strategy ----
    def rows(pts):
        return "\n".join(
            f"{int(round(p['n_pool']))} & {p['dE_mean']*1e3:.1f}"
            + (f"({p['E_std']*1e3:.1f})" if p['E_std'] else "") + r" \\"
            for p in pts)
    tab = (r"\begin{tabular}{lrr}" + "\n\\hline\\hline\n"
           r"strategy & pool size & $E-E_{\rm DMRG}$ (mHa) \\" + "\n\\hline\n")
    for strat, name in (("ours", "MPS-sampled"),
                        ("chemistry", "singles+doubles")):
        pts = agg[strat]
        first = True
        for p in pts:
            lead = name if first else ""
            std = f"({p['E_std']*1e3:.1f})" if p["E_std"] else ""
            tab += (f"{lead} & {int(round(p['n_pool']))} & "
                    f"{p['dE_mean']*1e3:.1f}{std}" + r" \\" + "\n")
            first = False
        tab += "\\hline\n"
    # random is constant (stays exactly at HF): one summary row.
    rp = agg["random"]
    lo, hi = int(round(rp[0]["n_pool"])), int(round(rp[-1]["n_pool"]))
    tab += (f"random & {lo}--{hi} & {rp[0]['dE_mean']*1e3:.1f}" + r" \\" + "\n")
    tab += "\\hline\n"
    tab += "\\hline\n\\end{tabular}\n"
    os.makedirs(os.path.dirname(TAB), exist_ok=True)
    open(TAB, "w").write(tab)
    print("wrote", TAB)
    print("\n(random pool std=0: it stays exactly at HF for every seed)\n")
    print(tab)


if __name__ == "__main__":
    main()
