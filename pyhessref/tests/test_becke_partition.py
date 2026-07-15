"""Tests for the reference Becke partitioning implementation.

Standalone: the partition weights are a pure function of the grid and atom coordinates, so the
tests load precomputed reference data from ``prototype/becke_deriv{1,2}_dict.npz`` (same NH3 /
def2-TZVP molecule and grid as the Rust port) and never run an SCF.

Validation strategy
-------------------
- ``w``  vs PySCF ``grids.weights`` (independent, machine precision).
- ``dw`` vs PySCF ``hessian.rks.get_dweight_dA`` (independent analytical reference).
- ``ddw`` vs the ``10-5`` analytical second derivative saved in ``becke_deriv2_dict.npz``
  (same algorithm as this module; checked to near machine precision, plus the ``(A,t)\\leftrightarrow
  (B,s)`` symmetry and the translation-invariance sums ``\\sum_A dw = \\sum_A ddw = 0``).
- An independent finite-difference of ``w`` (with the atom-centred grid carried along by the
  perturbed atom) re-derives ``dw`` end to end, validating the translation-invariance fix.
- ``_becke_s_derivs`` and the zeroth-order weights are checked for arbitrary ``hardness`` against
  independent finite-difference / per-pair-loop references.
"""

import unittest

import numpy as np

from pyhessref.nimatmul.becke_partition import _becke_s_derivs, becke_partition

PATH_PROTOTYPE = "prototype/"  # tests are run from the project root


def _becke_partition_weights_ref(grid_coords, atm_coords, atm_indices, wquad, radii_table, hardness):
    """Independent per-pair loop reference for the partition weights.

    Mirrors the ``10-1`` notebook's ``becke_partition_weights_scale``: an explicit pair loop with
    an unrolled switch iteration and no log-derivatives, so it shares no code with
    :func:`becke_partition`.  Used to validate the zeroth-order weights for arbitrary ``hardness``.
    """
    natm = atm_coords.shape[0]
    ngrids = grid_coords.shape[0]
    atom_dist = np.linalg.norm(atm_coords[:, None, :] - atm_coords[None, :, :], axis=-1)
    np.fill_diagonal(atom_dist, np.inf)
    grid_dist = np.linalg.norm(grid_coords[None, :, :] - atm_coords[:, None, :], axis=-1)  # (natm, ngrids)
    P = np.ones((natm, ngrids))
    for A in range(natm):
        for B in range(A):
            mu = (grid_dist[A] - grid_dist[B]) / atom_dist[A, B]
            af = radii_table[A, B]
            nu = mu + af * (1.0 - mu * mu)
            f = nu
            for _ in range(hardness):
                f = 1.5 * f - 0.5 * f**3
            P[A] *= 0.5 * (1.0 - f)
            P[B] *= 0.5 * (1.0 + f)
    Z = P.sum(axis=0)
    Pg = P[atm_indices, np.arange(ngrids)]
    return wquad * Pg / Z


class TestBeckePartition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d1 = np.load(PATH_PROTOTYPE + "becke_deriv1_dict.npz")
        d2 = np.load(PATH_PROTOTYPE + "becke_deriv2_dict.npz")
        cls.grid_coords = d1["grids"]
        cls.atm_coords = d1["atm_coords"]
        cls.atm_indices = d1["atm_indices"].astype(int)
        cls.wquad = d1["wquad"]
        cls.radii_table = d1["radii_table"]
        cls.w_ref = d1["weights"]
        cls.dw_ref = d1["dw_ref"]
        cls.ddw_ref = d2["ddw_ref"]
        cls.natm = cls.atm_coords.shape[0]
        cls.ngrids = cls.grid_coords.shape[0]

    def _run(self, deriv, nbatch=512, hardness=3):
        return becke_partition(
            self.grid_coords,
            self.atm_coords,
            self.atm_indices,
            self.wquad,
            self.radii_table,
            hardness,
            nbatch,
            deriv,
            None,
        )

    # ------------------------------------------------------------------ #
    #  analytical-vs-reference checks (hardness = 3, the npz data)
    # ------------------------------------------------------------------ #

    def test_deriv0_weights(self):
        res = self._run(0)
        self.assertIsNone(res["dw"])
        self.assertIsNone(res["ddw"])
        self.assertEqual(res["w"].shape, (self.ngrids,))
        np.testing.assert_allclose(res["w"], self.w_ref, atol=1e-9, rtol=1e-7)

    def test_deriv1(self):
        res = self._run(1)
        dw = res["dw"]
        self.assertIsNone(res["ddw"])
        self.assertEqual(dw.shape, (self.natm, 3, self.ngrids))
        np.testing.assert_allclose(dw, self.dw_ref, atol=1e-9, rtol=1e-7)
        # translation invariance: sum_A dw[:, :, g] = 0
        np.testing.assert_allclose(dw.sum(axis=0), 0.0, atol=1e-9)
        # raising deriv does not perturb the lower-order outputs
        np.testing.assert_allclose(res["w"], self._run(0)["w"], atol=1e-9, rtol=1e-7)

    def test_deriv2(self):
        res = self._run(2)
        ddw = res["ddw"]
        self.assertEqual(ddw.shape, (self.natm, 3, self.natm, 3, self.ngrids))
        np.testing.assert_allclose(ddw, self.ddw_ref, atol=1e-9, rtol=1e-7)
        # (A, t) <-> (B, s) symmetry
        np.testing.assert_allclose(ddw, ddw.transpose(2, 3, 0, 1, 4), atol=1e-9, rtol=1e-7)
        # translation invariance along both derivative axes
        np.testing.assert_allclose(ddw.sum(axis=0), 0.0, atol=1e-9)
        np.testing.assert_allclose(ddw.sum(axis=2), 0.0, atol=1e-9)
        # lower-order outputs unchanged vs deriv 0 / 1
        np.testing.assert_allclose(res["w"], self.w_ref, atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(res["dw"], self._run(1)["dw"], atol=1e-9, rtol=1e-7)

    # ------------------------------------------------------------------ #
    #  structural / numerical-robustness checks
    # ------------------------------------------------------------------ #

    def test_nbatch_independence(self):
        # the result is independent of nbatch, including non-multiple / single-grid batches
        r512 = self._run(2, nbatch=512)
        r7 = self._run(2, nbatch=7)
        rbig = self._run(2, nbatch=10 * self.ngrids)  # single batch (fully vectorised)
        np.testing.assert_allclose(r512["ddw"], r7["ddw"], atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(r512["ddw"], rbig["ddw"], atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(r512["dw"], r7["dw"], atol=1e-9, rtol=1e-7)
        np.testing.assert_allclose(r512["w"], r7["w"], atol=1e-9, rtol=1e-7)

    def test_dw_finite_difference(self):
        # Re-derive dw by central finite-difference of w, carrying the atom-centred grid along
        # the perturbed atom so the *total* derivative (incl. the translation-invariance fix) is
        # reproduced.  Independent of both the analytical algorithm and the PySCF reference.
        d = 1e-5
        dw = self._run(1)["dw"]
        dw_fd = np.zeros_like(dw)
        for A in range(self.natm):
            mask = self.atm_indices == A
            for t in range(3):
                ac_p = self.atm_coords.copy()
                ac_p[A, t] += d
                gc_p = self.grid_coords.copy()
                gc_p[mask, t] += d
                ac_m = self.atm_coords.copy()
                ac_m[A, t] -= d
                gc_m = self.grid_coords.copy()
                gc_m[mask, t] -= d
                wp = becke_partition(gc_p, ac_p, self.atm_indices, self.wquad, self.radii_table, 3, 512, 0, None)["w"]
                wm = becke_partition(gc_m, ac_m, self.atm_indices, self.wquad, self.radii_table, 3, 512, 0, None)["w"]
                dw_fd[A, t] = (wp - wm) / (2 * d)
        diff = np.abs(dw - dw_fd)
        # the bulk of grids match to machine precision; a handful of tiny-|s| grids suffer FD
        # truncation (steeper local curvature), which is bounded well below a real-error scale.
        self.assertLess(np.median(diff), 1e-11)
        self.assertLess(np.percentile(diff, 99.9), 1e-7)
        self.assertLess(np.max(diff), 1e-6)

    def test_switch_derivatives_fd(self):
        # _becke_s_derivs for arbitrary hardness via finite differences of s(mu).
        mu = np.linspace(-0.95, 0.95, 11)
        a = 0.3
        hp = 3e-4
        for h in [1, 2, 3, 4, 5]:
            s, ds, dds = _becke_s_derivs(mu, a, h)
            sp, _, _ = _becke_s_derivs(mu + hp, a, h)
            sm, _, _ = _becke_s_derivs(mu - hp, a, h)
            np.testing.assert_allclose(ds, (sp - sm) / (2 * hp), atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(dds, (sp - 2 * s + sm) / hp**2, atol=1e-4, rtol=1e-3)

    def test_general_hardness_weights(self):
        # zeroth-order weights for arbitrary hardness vs the independent per-pair loop reference.
        for h in [1, 2, 3, 4, 5]:
            w = self._run(0, hardness=h)["w"]
            ref = _becke_partition_weights_ref(
                self.grid_coords, self.atm_coords, self.atm_indices, self.wquad, self.radii_table, hardness=h
            )
            np.testing.assert_allclose(w, ref, atol=1e-12, rtol=1e-12)
        # the loop reference itself must reproduce PySCF at hardness = 3 (sanity of the reference).
        ref3 = _becke_partition_weights_ref(
            self.grid_coords, self.atm_coords, self.atm_indices, self.wquad, self.radii_table, hardness=3
        )
        np.testing.assert_allclose(ref3, self.w_ref, atol=1e-9, rtol=1e-7)


if __name__ == "__main__":
    unittest.main()
