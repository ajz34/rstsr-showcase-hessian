import numpy as np
from abc import ABC, abstractmethod


class HessCoreAPI(ABC):
    """Abstract class for Hessian-related API for restricted SCF core components.

    Term Explanation
    ----------------

    **Core component** here actually means the term is of zero/one-order with right of (electron) density matrix.

    - Nuclear repulsion is zero-order (unrelated to density matrix).
    - Core Hamiltonian is one-order (linear to density matrix).
    - External field may have nuclear and electronic contributions.
      For dipole field, as an example, the electronic contribution is of one-order, and can be counted in core-hamiltonian in some frameworks.

    We have function `make_skeleton_hess` here to count the **skeleton** contribution of the Hessian.
    We do not handle derivative of density matrix here, which is the responsibility of CPHF solver.
    """

    @abstractmethod
    def make_skeleton_hess(
        self,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        dm0: np.ndarray = None,
    ) -> np.ndarray:
        """Generate the **skeleton** contribution of Hessian for current SCF component.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.
        dm0 : np.ndarray, optional
            The density matrix for current SCF component, shape [nao, nao].
            If not provided, it will be generated from `mo_coeff` and `mo_occ` if necessary.

        Returns
        -------
        hess : np.ndarray
            The Hessian matrix for current SCF component, shape [natm, natm, 3, 3].
        """
        pass

    @abstractmethod
    def generator_deriv1(self) -> callable:
        """Generate the function to compute the first-order derivative of core component.

        The returned function should take atom index as input, and returns the first-order derivative, of shape [3, nao, nao].

        This function only works for first-order density matrix contribution (like hcore).
        If this component does not contribute (like nuclear repulsion), return None.
        """
        pass


class HessElecInteractAPI(ABC):
    """Abstract class for Hessian-related API for restricted SCF electronic interaction components.

    Term Explanation
    ----------------

    **Electronic interaction** here actually means the term is of two-order (or higher-order) with right of (electron) density matrix.

    - J/K contribution from Hartree-Fock is exactly two-order.
    - DFT contribution is non-linear to density matrix, and should be counted as infinity-order.
    - Implicit-solvent/VV10 is probably categorized here.

    In SCF iteration, introducing two-order (or higher-order) contribution requires the program to make some modification to Fock matrix construction.
    This kind of terms is substentially different from zero/one-order core components, and should be handled separately.
    """

    @abstractmethod
    def make_skeleton_hess(
        self,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
        dm0: np.ndarray = None,
    ) -> np.ndarray:
        """Generate the **skeleton** contribution of Hessian for current SCF component.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.
        dm0 : np.ndarray, optional
            The density matrix for current SCF component, shape [nao, nao].
            If not provided, it will be generated from `mo_coeff` and `mo_occ` if necessary.

        Returns
        -------
        hess : np.ndarray
            The Hessian matrix for current SCF component, shape [natm, natm, 3, 3].
        """
        pass
