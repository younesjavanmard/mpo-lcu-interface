import sys, time, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
sys.path.insert(0, __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "referee_baselines"))
from nbstate import load, dmrg, sample_pool
from detspace import QubitHamiltonian, pool_energy, fci_energy, bits_to_int, fragment_stats
from pool_eval_exact import pool_energy_exact
ns = load("C2H2_sto3g_frozen")
Ha, n = ns["Ha"], ns["n_qubits"]
hf_bits = ns["hf_bits"]; h = bits_to_int(hf_bits)
H = QubitHamiltonian(Ha.terms, n)
print("terms", len(Ha.terms), "HF bits", hf_bits)
E_hf, *_ = pool_energy(H, ["I"*n], h)
print("E_HF(det) =", E_hf, " pyscf HF =", ns["out"]["mol"].hf_energy)
nel = hf_bits.count("1")
t=time.time(); E_fci, dets, c = fci_energy(H, nel//2, nel//2); print("E_FCI =", E_fci, "dim", len(dets), f"{time.time()-t:.1f}s")
t=time.time(); psi, Ed = dmrg(ns, 8); print("E_DMRG(8) =", Ed, f"{time.time()-t:.1f}s")
pool, _ = sample_pool(ns, psi, 1000, 0)
t=time.time(); E1, d1, c1 = pool_energy(H, list(pool), h); t1=time.time()-t
t=time.time(); E2, r2 = pool_energy_exact(list(pool), Ha, hf_bits); t2=time.time()-t
print(f"pool |U|={len(pool)} dets={len(d1)} E_fast={E1:.10f} ({t1:.2f}s) E_ref={E2:.10f} rank={r2} ({t2:.1f}s)")
print("||c||_1 =", np.abs(c1).sum(), "p_succ_max =", 1/np.abs(c1).sum()**2, fragment_stats(pool))
