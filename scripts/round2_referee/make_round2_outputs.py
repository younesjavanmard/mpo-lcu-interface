#!/usr/bin/env python
"""Figures/tables for the round-2 referee numerics (reads results_colab/results/*.json)."""
import os, sys, glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results_colab")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "paper_outputs")
os.makedirs(OUT, exist_ok=True)
# Merge every r2_*.json below RES (main queue + per-seed N=16000 runs). Points of the
# same (molecule, chi) are merged only if they come from the same source state
# (identical E_src); a file with "src_tag" (multi-start DMRG reruns) supersedes the rest.
recs = {}
for f in sorted(glob.glob(os.path.join(RES, "**", "r2_*.json"), recursive=True)):
    r = json.load(open(f)); k = (r["molecule"], r["chi"])
    if k not in recs or (r.get("src_tag") and not recs[k].get("src_tag")):
        recs[k] = r; continue
    if bool(r.get("src_tag")) != bool(recs[k].get("src_tag")):
        continue
    assert abs(r["E_src"] - recs[k]["E_src"]) < 1e-6, (f, r["E_src"], recs[k]["E_src"])
    have = {(p["n_samples"], p["seed"]) for p in recs[k]["points"]}
    recs[k]["points"] += [p for p in r["points"] if (p["n_samples"], p["seed"]) not in have]

def agg(r):
    by = {}
    for p in r["points"]:
        by.setdefault(p["n_samples"], []).append(p)
    out = []
    for N in sorted(by):
        ps = by[N]; g = lambda k: np.array([p[k] for p in ps], float)
        out.append(dict(N=N, n=len(ps), dE=1e3*g("dE_fci").mean(), dE_sd=1e3*g("dE_fci").std(ddof=1) if len(ps)>1 else 0,
                        dEsrc=1e3*g("dE_src").mean(), F=g("F_fci").mean(), pool=g("pool_size").mean(),
                        dets=g("n_dets").mean(), lam=g("Lambda_min").mean(), p=g("p_succ").mean(),
                        L=g("n_left").mean(), R=g("n_right").mean(), E=g("n_edges").mean(),
                        maxexc=g("max_exc").mean()))
    return out

# ---------------- Fig. (point 2): error vs FCI for sources of increasing chi ----------
RAMP = {4: "#86b6ef", 8: "#5598e7", 16: "#2a78d6", 32: "#104281"}
MARK = {4: "v", 8: "s", 16: "o", 32: "D"}
mols = [("C2H2_sto3g_frozen", r"(a) C$_2$H$_2$"), ("C2H4_sto3g_frozen", r"(b) C$_2$H$_4$")]
# Match the other manuscript figures (Figs. 4, 5, 8, 9): serif body with STIX mathtext, TrueType-embedded.
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif", "Times"],
                     "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.size": 11, "legend.fontsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), sharey=True)
for ax, (mol, title) in zip(axes, mols):
    any_ = False
    for chi, col in RAMP.items():
        r = recs.get((mol, chi))
        if not r or not r["points"]:
            continue
        any_ = True
        a = agg(r); N = [x["N"] for x in a]
        ax.errorbar(N, [x["dE"] for x in a], yerr=[x["dE_sd"] for x in a], color=col, marker=MARK[chi],
                    ms=4.5, lw=1.5, capsize=2, label=rf"source $\chi={chi}$", zorder=3)
        ax.axhline(1e3*(r["E_src"]-r["E_FCI"]), color=col, ls="--", lw=1.0, zorder=2)
    r1 = next((recs[k] for k in recs if k[0] == mol), None)
    if r1:
        ax.axhline(1e3*(r1["E_HF"]-r1["E_FCI"]), color="#8a8a86", ls="-", lw=1.0)
        ax.text(260, 1e3*(r1["E_HF"]-r1["E_FCI"])*1.08, "Hartree–Fock", color="#55554f", fontsize=9)
    ax.axhline(1.6, color="#8a8a86", ls=":", lw=1.0)
    ax.text(260, 1.75, "chemical accuracy", color="#55554f", fontsize=9)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel(r"$\mathcal{N}_{\rm samples}$"); ax.set_title(title, loc="left", fontsize=11)
    ax.grid(True, which="major", color="#e6e6e3", lw=0.6); ax.set_axisbelow(True)
axes[0].set_ylabel(r"$E-E_{\rm FCI}$ (mHa)")
from matplotlib.lines import Line2D
h, l = axes[0].get_legend_handles_labels()
h.append(Line2D([], [], color="#55554f", ls="--", lw=1.0)); l.append("source energy")
fig.legend(h, l, title=None, fontsize=9, frameon=False, loc="lower center", ncol=len(h),
           bbox_to_anchor=(0.5, 0.0), handlelength=2.2)
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(OUT, "fig_source_chi_vs_fci.pdf")); fig.savefig(os.path.join(OUT, "fig_source_chi_vs_fci.png"), dpi=160)

# ---------------- Table (point 2): source vs pool, per chi --------------------------
rows = []
for mol, _ in mols:
    for chi in (1, 2, 4, 8, 16, 32):
        r = recs.get((mol, chi))
        if not r or not r["points"]:
            continue
        a = agg(r); best = a[-1]
        rows.append((mol, chi, r.get("dmrg_mode", ""), 1e3*(r["E_src"]-r["E_FCI"]), r["F_src"], best["N"],
                     best["pool"], best["dets"], best["dE"], best["dE_sd"], best["dEsrc"], best["F"]))
with open(os.path.join(OUT, "tab_source_chi.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrrrrr}\n\\toprule\n")
    f.write("mol. & $\\chi$ & $E_{\\rm MPS}-E_{\\rm FCI}$ & $F_{\\rm MPS}$ & $\\mathcal N_{\\rm s}$ & $|\\mathcal U|$ & $|\\mathcal D|$ & $E-E_{\\rm FCI}$ & $E-E_{\\rm MPS}$ & $F$ \\\\\n")
    f.write(" & & (mHa) & & & & & (mHa) & (mHa) & \\\\\n\\midrule\n")
    for (mol, chi, mode, dsrc, Fs, N, pool, dets, dE, sd, dEs, F) in rows:
        m = "C$_2$H$_2$" if "C2H2" in mol else "C$_2$H$_4$"
        lab = "HF" if chi == 1 else str(chi)
        f.write(f"{m} & {lab} & {dsrc:.1f} & {Fs:.3f} & {N} & {pool:.0f} & {dets:.0f} & {dE:.1f}({sd:.1f}) & {dEs:+.1f} & {F:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------- Table (point 4): resources vs system size --------------------------
# Size series (H chains, 1.0 A) + the two molecules of the paper. The other systems that were
# computed (H2, LiH, H2O, N2) stay in results_colab/ and in round2_summary.json.
SCAL = [("H4_chain_r1.0", "H$_4$"), ("H6_chain_r1.0", "H$_6$"), ("H8_chain_r1.0", "H$_8$"),
        ("H10_chain_r1.0", "H$_{10}$"), ("C2H2_sto3g_frozen", "C$_2$H$_2$"), ("C2H4_sto3g_frozen", "C$_2$H$_4$")]
srows = []
for mol, lab in SCAL:
    r = recs.get((mol, 32))
    if not r or not r["points"]:
        continue
    for x in agg(r):
        if x["N"] != 4000:
            continue
        toff = 2 * x["E"] + min(x["E"], x["L"] + x["R"])   # Eq. (40): cheaper of flat / factorized Select
        srows.append((lab, r["n_qubits"], x["N"], x["pool"], x["dets"], x["L"], x["R"], x["E"], toff,
                      x["dE"], x["F"], x["lam"], x["p"], 1e3*(r["E_src"]-r["E_FCI"]), mol))
with open(os.path.join(OUT, "tab_scaling.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\n")
    f.write("system & $N$ & $|\\mathcal U|$ & $|\\mathcal D|$ & $|\\mathcal I_{\\mathsf L}|{+}|\\mathcal I_{\\mathsf R}|$ & $|\\mathcal E|$ & $C_{\\rm Toff}$ & $\\|c\\|_1$ & $p_{\\rm succ}$ \\\\\n\\midrule\n")
    for (lab, n, N, pool, dets, L, R, E, toff, dE, F, lam, p, dsrc, mol) in srows:
        f.write(f"{lab} & {n} & {pool:.0f} & {dets:.0f} & {L+R:.0f} & {E:.0f} & {toff:.0f} & {lam:.2f} & {p:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
json.dump(dict(source=rows, scaling=srows), open(os.path.join(OUT, "round2_summary.json"), "w"), indent=1, default=float)
print("tasks loaded:", sorted(recs)); print("wrote", OUT)
for r in rows: print("SRC", r)
for r in srows: print("SCAL", r[:-1])
