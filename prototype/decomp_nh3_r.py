"""
Decomposition of RI-JK RHF Hessian for NH3 (restricted)

The total Hessian de_ref is decomposed into:
  de_ref = de_hcore + de_ovlp
           + de_J_basis_2nd + de_J_basis_1st_aux_1st + de_J_aux_2nd
           - de_K_basis_2nd - de_K_basis_1st_aux_1st - de_K_aux_2nd
           + de_cphf + de_nuc

Where:
  de_hcore  : 2nd derivative of core Hamiltonian contracted with dm0
  de_ovlp   : 2nd derivative of overlap contracted with dme0 (negative)
  de_J_basis_2nd       : J contributions from 2 orbital derivs, 0 aux derivs
                          i.e. (20|0)(0|00), (11|0)(0|00), (10|0)(0|10)
  de_J_basis_1st_aux_1st : J contributions from 1 orbital + 1 aux deriv
                          i.e. (10|1)(0|00), (10|0)(0|1)(0|00), (10|0)(1|0)(0|00), (10|0)(1|00)
  de_J_aux_2nd         : J contributions from 0 orbital + 2 aux derivs
                          i.e. (00|2)(0|00), (00|0)(1|1)(0|00), etc.
  de_K_*  : same structure as de_J_* but for exchange
  de_cphf  : CPHF orbital response contribution
  de_nuc   : nuclear repulsion 2nd derivative

Strategy for separating J/K sub-contributions:
  - auxbasis_response=0 gives only basis_2nd (no aux response)
  - auxbasis_response=1 adds 1st-order aux response with factor 1.0 for J, 0.5 for K
  - auxbasis_response=2 adds full 1st-order aux (factor 2.0 for J, 1.0 for K) + 2nd-order aux

  Therefore:
    J_basis_1st_aux_1st = 2 * (ej_aux1 - ej_aux0)  # full 1st-order aux, factor 2.0
    J_aux_2nd           = ej_aux2 - 2*ej_aux1 + ej_aux0
    K_basis_1st_aux_1st = 2 * (ek_aux1 - ek_aux0)  # full 1st-order aux, factor 1.0
    K_aux_2nd           = ek_aux2 - 2*ek_aux1 + ek_aux0
"""

from pyscf import gto, scf, lib, df
import numpy as np
from pyscf.hessian import rhf as rhf_hess
from pyscf.df.hessian import rhf as df_rhf_hess
from pyscf.df.grad.rhf import _gen_metric_solver, LINEAR_DEP_THRESHOLD

lib.num_threads(16)
np.set_printoptions(5, suppress=True, linewidth=150)

# ============================================================
# Molecule and SCF
# ============================================================
xyz = """
N  0   0   0
H  1.0 0.1 0.2
H  0.3 1.1 0.2
H  0.1 0.1 1.2
"""

mol = gto.Mole(atom=xyz, basis="def2-TZVP", max_memory=32000).build()
mf = scf.RHF(mol).density_fit()
mf.with_df.build()
mf.run()

# ============================================================
# Reference Hessian
# ============================================================
mf_hess = mf.Hessian().run()
de_ref = mf_hess.de.copy()
print("de_ref shape:", de_ref.shape)
print("de_ref max abs:", np.max(np.abs(de_ref)))

# ============================================================
# Basic quantities
# ============================================================
mo_coeff = mf.mo_coeff
mo_occ = mf.mo_occ
mo_energy = mf.mo_energy
nao, nmo = mo_coeff.shape
mocc = mo_coeff[:, mo_occ > 0]
mocc_2 = np.einsum('pi,i->pi', mocc, mo_occ[mo_occ > 0]**.5)
nocc = mocc.shape[1]
dm0 = np.dot(mocc, mocc.T) * 2
dme0 = np.einsum('pi,qi,i->pq', mocc, mocc, mo_energy[mo_occ > 0]) * 2
natm = mol.natm
atmlst = range(natm)
aoslices = mol.aoslice_by_atom()

# ============================================================
# Nuclear repulsion
# ============================================================
de_nuc = rhf_hess.hess_nuc(mol)
print("de_nuc max abs:", np.max(np.abs(de_nuc)))

# ============================================================
# _partial_hess_ejk with different auxbasis_response levels
# ============================================================
# auxbasis_response = 0 (orbital-only derivatives, no aux response)
hessobj_aux0 = mf.Hessian()
hessobj_aux0.auxbasis_response = 0
e1_aux0, ej_aux0, ek_aux0 = df_rhf_hess._partial_hess_ejk(hessobj_aux0)
print("ej_aux0 max abs:", np.max(np.abs(ej_aux0)))
print("ek_aux0 max abs:", np.max(np.abs(ek_aux0)))

# auxbasis_response = 1 (1st-order aux response with half factor for K)
hessobj_aux1 = mf.Hessian()
hessobj_aux1.auxbasis_response = 1
e1_aux1, ej_aux1, ek_aux1 = df_rhf_hess._partial_hess_ejk(hessobj_aux1)
print("ej_aux1 max abs:", np.max(np.abs(ej_aux1)))
print("ek_aux1 max abs:", np.max(np.abs(ek_aux1)))

# auxbasis_response = 2 (full aux response, default)
hessobj_aux2 = mf.Hessian()
hessobj_aux2.auxbasis_response = 2
e1_aux2, ej_aux2, ek_aux2 = df_rhf_hess._partial_hess_ejk(hessobj_aux2)
print("ej_aux2 max abs:", np.max(np.abs(ej_aux2)))
print("ek_aux2 max abs:", np.max(np.abs(ek_aux2)))

# ============================================================
# Decompose e1 into hcore and overlap
# ============================================================
e1 = e1_aux2.copy()

hcore_deriv = mf_hess.hcore_generator(mol)
s1aa, s1ab, s1a_ovlp = rhf_hess.get_ovlp(mol)

de_hcore = np.zeros((natm, natm, 3, 3))
de_ovlp = np.zeros((natm, natm, 3, 3))

for i0, ia in enumerate(atmlst):
    shl0, shl1, p0, p1 = aoslices[ia]
    # overlap diagonal: s1aa contracted with dme0
    de_ovlp[i0, i0] -= np.einsum('xypq,pq->xy', s1aa[:, :, p0:p1], dme0[p0:p1]) * 2
    for j0, ja in enumerate(atmlst[:i0 + 1]):
        q0, q1 = aoslices[ja][2:]
        # hcore second derivative contracted with dm0
        h1ao_hc = hcore_deriv(ia, ja)
        de_hcore[i0, j0] += np.einsum('xypq,pq->xy', h1ao_hc, dm0)
        # overlap cross: s1ab contracted with dme0
        de_ovlp[i0, j0] -= np.einsum('xypq,pq->xy', s1ab[:, :, p0:p1, q0:q1], dme0[p0:p1, q0:q1]) * 2
    for j0 in range(i0):
        de_hcore[j0, i0] = de_hcore[i0, j0].T
        de_ovlp[j0, i0] = de_ovlp[i0, j0].T

print("e1 = hcore + ovlp:", np.allclose(e1, de_hcore + de_ovlp))
print("de_hcore max abs:", np.max(np.abs(de_hcore)))
print("de_ovlp max abs:", np.max(np.abs(de_ovlp)))

# ============================================================
# Decompose ej into J_basis_2nd, J_basis_1st_aux_1st, J_aux_2nd
# ============================================================
de_J_basis_2nd = ej_aux0.copy()

# Full 1st-order aux response: aux1 gives factor 1.0, aux2 gives factor 2.0
# The "correct" (full) contribution uses factor 2.0
de_J_basis_1st_aux_1st = 2.0 * (ej_aux1 - ej_aux0)

# Second-order aux response
de_J_aux_2nd = ej_aux2 - 2.0 * ej_aux1 + ej_aux0

# Verify ej decomposition
print("ej decomposition check:", np.allclose(ej_aux2, de_J_basis_2nd + de_J_basis_1st_aux_1st + de_J_aux_2nd))
print("de_J_basis_2nd max abs:", np.max(np.abs(de_J_basis_2nd)))
print("de_J_basis_1st_aux_1st max abs:", np.max(np.abs(de_J_basis_1st_aux_1st)))
print("de_J_aux_2nd max abs:", np.max(np.abs(de_J_aux_2nd)))

# ============================================================
# Decompose ek into K_basis_2nd, K_basis_1st_aux_1st, K_aux_2nd
# ============================================================
de_K_basis_2nd = ek_aux0.copy()

# Full 1st-order aux response: aux1 gives factor 0.5, aux2 gives factor 1.0
# The "correct" (full) contribution uses factor 1.0
de_K_basis_1st_aux_1st = 2.0 * (ek_aux1 - ek_aux0)

# Second-order aux response
de_K_aux_2nd = ek_aux2 - 2.0 * ek_aux1 + ek_aux0

# Verify ek decomposition
print("ek decomposition check:", np.allclose(ek_aux2, de_K_basis_2nd + de_K_basis_1st_aux_1st + de_K_aux_2nd))
print("de_K_basis_2nd max abs:", np.max(np.abs(de_K_basis_2nd)))
print("de_K_basis_1st_aux_1st max abs:", np.max(np.abs(de_K_basis_1st_aux_1st)))
print("de_K_aux_2nd max abs:", np.max(np.abs(de_K_aux_2nd)))

# ============================================================
# CPHF response
# ============================================================
# Compute full hess_elec
de_hess_elec = mf_hess.hess_elec()

# CPHF response = hess_elec - partial_hess_elec
de_partial = e1 + ej_aux2 - ek_aux2
de_cphf = de_hess_elec - de_partial

print("partial_hess_elec check:", np.allclose(de_partial, mf_hess.partial_hess_elec()))
print("hess_elec = partial + cphf:", np.allclose(de_hess_elec, de_partial + de_cphf))
print("de_cphf max abs:", np.max(np.abs(de_cphf)))

# ============================================================
# Final verification
# ============================================================
de_sum = de_hcore + de_ovlp \
         + de_J_basis_2nd + de_J_basis_1st_aux_1st + de_J_aux_2nd \
         - de_K_basis_2nd - de_K_basis_1st_aux_1st - de_K_aux_2nd \
         + de_cphf + de_nuc

print("\n========== Final Verification ==========")
print("de_ref == de_sum:", np.allclose(de_ref, de_sum))
print("max abs difference:", np.max(np.abs(de_ref - de_sum)))

# ============================================================
# Print summary of contributions
# ============================================================
print("\n========== Contribution Summary ==========")
contributions = {
    "hcore":               de_hcore,
    "ovlp":                de_ovlp,
    "J_basis_2nd":         de_J_basis_2nd,
    "J_basis_1st_aux_1st": de_J_basis_1st_aux_1st,
    "J_aux_2nd":           de_J_aux_2nd,
    "K_basis_2nd":         de_K_basis_2nd,
    "K_basis_1st_aux_1st": de_K_basis_1st_aux_1st,
    "K_aux_2nd":           de_K_aux_2nd,
    "cphf":                de_cphf,
    "nuc":                 de_nuc,
}

for name, arr in contributions.items():
    sign = "+" if name.startswith("J") or name in ["hcore", "cphf", "nuc"] else "-"
    if name == "ovlp":
        sign = "+"  # ovlp is subtracted in e1, but stored as negative
    print(f"  {sign} {name:30s}: max = {np.max(np.abs(arr)):12.6f}, norm = {np.linalg.norm(arr):12.6f}")

print(f"\n  {'de_ref':30s}: max = {np.max(np.abs(de_ref)):12.6f}, norm = {np.linalg.norm(de_ref):12.6f}")