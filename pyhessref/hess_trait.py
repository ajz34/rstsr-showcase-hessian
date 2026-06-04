import numpy as np
from abc import ABC, abstractmethod


class HessRscfAPI(ABC):
    """Abstract class for Hessian-related API for restricted SCF components."""

    @abstractmethod
    def make_hess(
        self,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        mo_energy: np.ndarray,
        dm0: np.ndarray = None,
    ) -> np.ndarray:
        """Generate the Hessian for current SCF component.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.
        mo_energy : np.ndarray
            Molecular orbital energies, shape [nmo].
        dm0 : np.ndarray, optional
            The density matrix for current SCF component, shape [nao, nao].
            If not provided, it will be generated from `mo_coeff` and `mo_occ` if necessary.

        Returns
        -------
        hess : np.ndarray
            The Hessian matrix for current SCF component, shape [natm, natm, 3, 3].
        """
        pass

    @staticmethod
    def get_dm0(mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        """Generate the density matrix for current SCF component.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.

        Returns
        -------
        dm0 : np.ndarray
            The density matrix for current SCF component, shape [nao, nao].
        """
        import pyhessref.util

        return pyhessref.util.get_dm0(mo_coeff, mo_occ)

    @staticmethod
    def get_dme0(
        mo_coeff: np.ndarray, mo_occ: np.ndarray, mo_energy: np.ndarray
    ) -> np.ndarray:
        """Generate the energy-weighted density matrix for current SCF component.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.
        mo_energy : np.ndarray
            Molecular orbital energies, shape [nmo].

        Returns
        -------
        dme0 : np.ndarray
            The energy-weighted density matrix for current SCF component, shape [nao, nao].
        """
        import pyhessref.util

        return pyhessref.util.get_dme0(mo_coeff, mo_occ, mo_energy)
