"""Reference values for NH3 (restricted) Hessian decomposition with SVWN (LDA).

Mirrors the components produced in 06-1, 06-2, 06-3 notebooks (TPSS0) but uses
PySCF library functions directly to obtain reference quantities.

Run from project root or from the prototype/ directory; the npz reference data
file is expected to live next to this script.
"""

import os
from functools import partial

import numpy as np
from pyscf import dft, gto, lib
from pyscf.df.hessian import rhf as df_rhf_hess
from pyscf.hessian import rhf as rhf_hess
from pyscf.hessian import rks as rks_hess

lib.num_threads(16)
np.einsum = partial(np.einsum, optimize="greedy")

XC = "SVWN"
NPZ = "nh3_r_svwn.npz"

XYZ = """
N  0   0   0
H  1.0 0.1 0.2
H  0.3 1.1 0.2
H  0.1 0.1 1.2
"""


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    npz_path = os.path.join(here, NPZ)

    mol = gto.Mole(atom=XYZ, basis="def2-TZVP", max_memory=32000).build()
    mf = dft.RKS(mol, xc=XC).density_fit()
    dat0 = np.load(npz_path)
    mf.mo_coeff = dat0["mo_coeff"]
    mf.mo_occ = dat0["mo_occ"]
    mf.mo_energy = dat0["mo_energy"]
    mf.with_df.build()
    mf.converged = True

    ni = mf._numint
    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(mf.xc, spin=mol.spin)
    xctype = ni._xc_type(mf.xc)
    assert xctype == "LDA"
    assert hyb == 0.0
    assert omega == 0.0

    mo_coeff = mf.mo_coeff
    mo_occ = mf.mo_occ
    mocc = mo_coeff[:, mo_occ > 0]
    natm = mol.natm
    aoslices = mol.aoslice_by_atom()
    dm0 = mocc @ mocc.T * 2

    # === Full reference Hessian (taken from npz; SCF-derivative kernel is costly) ===
    de_ref = np.asarray(dat0["ref_de"])
    mf_hess = mf.Hessian()
    mf_hess.auxbasis_response = 2

    # === Nuclear repulsion ===
    de_nuc = rhf_hess.hess_nuc(mol)

    # === J/K skeleton (auxbasis_response 0/1/2 decomposition) ===
    hessobj_aux0 = mf.Hessian(); hessobj_aux0.auxbasis_response = 0
    de_1, ej_aux0, ek_aux0 = df_rhf_hess._partial_hess_ejk(hessobj_aux0, with_k=True)
    hessobj_aux1 = mf.Hessian(); hessobj_aux1.auxbasis_response = 1
    _, ej_aux1, ek_aux1 = df_rhf_hess._partial_hess_ejk(hessobj_aux1, with_k=True)
    hessobj_aux2 = mf.Hessian(); hessobj_aux2.auxbasis_response = 2
    _, ej_aux2, ek_aux2 = df_rhf_hess._partial_hess_ejk(hessobj_aux2, with_k=True)

    de_J20 = ej_aux0.copy()
    de_J11 = 2.0 * (ej_aux1 - ej_aux0)
    de_J02 = ej_aux2 - 2.0 * ej_aux1 + ej_aux0
    de_K20 = 2 * ek_aux0.copy()
    de_K11 = 2 * 2.0 * (ek_aux1 - ek_aux0)
    de_K02 = 2 * (ek_aux2 - 2.0 * ek_aux1 + ek_aux0)

    # === XC skeleton 2nd derivative (no grid response) ===
    max_memory = 4000
    veff_diag = rks_hess._get_vxc_diag(hessobj_aux0, mo_coeff, mo_occ, max_memory)
    vxc_list = rks_hess._get_vxc_deriv2(hessobj_aux0, mo_coeff, mo_occ, max_memory)
    de_xc = np.zeros((natm, natm, 3, 3))
    for ia in range(natm):
        p0, p1 = aoslices[ia][2:]
        de_xc[ia, ia] += np.einsum("xypq,pq->xy", veff_diag[:, :, p0:p1], dm0[p0:p1]) * 2
        for jb in range(natm):
            q0, q1 = aoslices[jb][2:]
            de_xc[ia, jb] += np.einsum("xypq,pq->xy", vxc_list[ia][:, :, q0:q1], dm0[q0:q1]) * 2
    de_xc = (de_xc + de_xc.transpose(1, 0, 3, 2)) / 2

    # === CP-KS contribution ===
    de_hess_elec = mf_hess.hess_elec()
    de_partial = de_1 + ej_aux2 - hyb * ek_aux2 + de_xc
    de_cphf = de_hess_elec - de_partial

    # === f1ao / f1mo and Vxc_deriv1 ===
    f1ao_ref = np.asarray(mf_hess.make_h1(mo_coeff, mo_occ))
    f1mo_ref = np.einsum("up, Atuv, vi -> Atpi", mo_coeff, f1ao_ref, mocc)
    vxc_deriv1 = rks_hess._get_vxc_deriv1(mf_hess, mo_coeff, mo_occ, max_memory)
    vxc_deriv1_mo = np.einsum("up, Atuv, vi -> Atpi", mo_coeff, vxc_deriv1, mocc)

    # === Sum check ===
    de_sum = (de_1
              + de_J20 + de_J11 + de_J02
              - 0.5 * hyb * (de_K20 + de_K11 + de_K02)
              + de_xc
              + de_cphf
              + de_nuc)
    assert np.allclose(de_ref, de_sum), \
        f"de decomposition mismatch: max diff = {np.max(np.abs(de_ref - de_sum))}"

    # === Print fingerprints and self-verify ===
    refs = {
        "de_ref":        (de_ref,         1.5282789158),
        "de_nuc":        (de_nuc,        10.9421515037),
        "de_1":          (de_1,         -16.9317030677),
        "de_J20":        (de_J20,        -1.4018770518),
        "de_J11":        (de_J11,        18.5616234449),
        "de_J02":        (de_J02,        -9.2730871331),
        "de_K20":        (de_K20,        -5.9374494995),
        "de_K11":        (de_K11,        19.2265642736),
        "de_K02":        (de_K02,        -9.6059419301),
        "de_vxc":        (de_xc,        -1.0550494670),
        "de_cphf":       (de_cphf,        0.6862206867),
        "f1ao_ref":      (f1ao_ref,      -3.1969847301),
        "f1mo_ref":      (f1mo_ref,       4.5969810787),
        "vxc_deriv1":    (vxc_deriv1,    -4.5790197174),
        "vxc_deriv1_mo": (vxc_deriv1_mo,  0.4744765645),
    }
    print(f"=== {XC} (RKS, NH3) ===")
    print(f"  hyb={hyb}, xctype={xctype}")
    print(f"  {'name':<16} {'shape':<24} {'fp':>16}    {'ref':>16}")
    bad = []
    for name, (arr, ref_fp) in refs.items():
        cur_fp = lib.fp(arr)
        ok = abs(cur_fp - ref_fp) < 1e-6
        marker = "" if ok else "  <-- DRIFT"
        print(f"  {name:<16} {str(arr.shape):<24} {cur_fp:>16.10f}    {ref_fp:>16.10f}{marker}")
        if not ok:
            bad.append((name, cur_fp, ref_fp))
    if bad:
        raise AssertionError(f"Fingerprint drift detected: {bad}")
    print("All fingerprints match.")

    # === Write decomposed values to nh3_r_<xc>_decomp.npz ===
    out_path = os.path.join(here, NPZ.replace(".npz", "_decomp.npz"))
    out = dict(dat0)
    out.update({
        "de_nuc": de_nuc,
        "de_1": de_1,
        "de_J20": de_J20, "de_J11": de_J11, "de_J02": de_J02,
        "de_K20": de_K20, "de_K11": de_K11, "de_K02": de_K02,
        "de_xc": de_xc,
        "de_cphf": de_cphf,
        "de_ref": de_ref,
        "hyb": np.asarray(hyb),
        "f1ao_ref": f1ao_ref,
        "f1mo_ref": f1mo_ref,
        "vxc_deriv1": vxc_deriv1,
        "vxc_deriv1_mo": vxc_deriv1_mo,
    })
    np.savez(out_path, **out)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
