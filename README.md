# Coefficient-Decoupled MPOs as an Interface to LCU Circuits — code

This repository is the official implementation accompanying the paper

> **Coefficient-Decoupled Matrix Product Operators as an Interface to
> Linear-Combination-of-Unitaries Circuits**
> Younes Javanmard
> arXiv: https://arxiv.org/abs/2603.24822

It is the reproducibility package for that work and contains **only** the code
used in the paper: the
coefficient-decoupled Hamiltonian-MPO construction, the symbolic left/right graph
+ coefficient-bridge machinery, MPS perfect sampling, the bridge optimizer, and
the scripts that generate the paper's figures.

## Layout
```
src/qu_alg_qu_sim_tensor_networks/   core library (paper subset)
  hamiltonians.py     Pauli-sum Hamiltonian, get_mpo (SVD-compressed MPO),
                      from_openfermion, Renormalizer cross-check helper
  graph_builder_new_version.py
                      symbolic left/right graphs + coefficient bridge at a cut
  mpo_builder.py      cut environments / effective operators for the bridge
  mps.py, mps_helper.py, dmrg.py, tdvp.py, lanczos.py, backend.py, lattice/
jupyter-notebooks/tensor_networks/
  coefficient_decoupled_mpo_clean.ipynb   full pipeline + all paper figures
results/                             raw data for the main-text figures
  fig5_sweep_C2H2_*.json, fig5_sweep_C2H4_*.json   (per-seed + aggregated)
scripts/
  referee_baselines/  pool-baseline comparison (Fig. 4 / Table I)
    pool_eval_exact.py            exact variational energy of a Pauli pool
    r26_baselines_exact_C2H2.py   ours vs random vs chemistry (authoritative)
    r26_chemistry_pool_C2H2.py    qubit singles+doubles pool builder
    make_r26_figure.py            builds the figure + table
  mpo_recovery/       MPO validation + lossy-compression study (Figs. 7-8)
    mpo_bonddim_compare.py        our MPO builder vs Renormalizer (bond dims)
    mpo_recovery.py               reconstruction error vs retained bond dim
    make_mpo_recovery_figure.py   builds the figures
```

## Install
```bash
conda env create -f environment.yml      # or use an existing env
pip install -e .                          # installs qu_alg_qu_sim_tensor_networks
# extras used by the scripts: pyscf, openfermion, openfermionpyscf, renormalizer
```

## Reproduce the figures
Outputs go to `./paper_outputs/{figs,data,tables}` (override with `PAPER_OUT`).
```bash
# Fig. 4 / Table I  (pool baselines: MPS-sampled vs random vs chemistry)
python scripts/referee_baselines/r26_baselines_exact_C2H2.py
python scripts/referee_baselines/make_r26_figure.py

# Fig. 7  (our MPO builder vs Renormalizer)  and  Fig. 8  (lossy compression)
python scripts/mpo_recovery/mpo_bonddim_compare.py
python scripts/mpo_recovery/mpo_recovery.py
python scripts/mpo_recovery/make_mpo_recovery_figure.py
```
The Jupyter notebook reproduces the full pipeline (DMRG reference, perfect
sampling, curation, bridge optimization, and the main-text figures) end to end;
it runs locally or on Google Colab.

## License
MIT — see `LICENSE`.
