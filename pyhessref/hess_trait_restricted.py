import numpy as np
from abc import ABC, abstractmethod


class RHessCoreAPI(ABC):
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
        raise NotImplementedError

    @abstractmethod
    def generator_deriv1(self) -> callable:
        """Generate the function to compute the first-order derivative of core component.

        The returned function should take atom index as input, and returns the first-order derivative, of shape [3, nao, nao].

        This function only works for first-order density matrix contribution (like hcore).
        If this component does not contribute (like nuclear repulsion), return None.
        """
        raise NotImplementedError


class RHessElecInteractAPI(ABC):
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
    ) -> np.ndarray:
        """Generate the **skeleton** contribution of Hessian for current SCF component.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.

        Returns
        -------
        hess : np.ndarray
            The Hessian matrix for current SCF component, shape [natm, natm, 3, 3].
        """
        raise NotImplementedError

    @abstractmethod
    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        """First order skeleton derivative in AO basis.

        Parameters
        ----------
        mo_coeff : np.ndarray
            Molecular orbital coefficients, shape [nao, nmo].
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape [nmo].
            In usual cases, the occupied orbitals should have occupation 2, and virtual orbitals should have occupation 0.

        Returns
        -------
        deriv_ao : np.ndarray
            The first-order skeleton derivative in AO basis, shape [natm, 3, nao, nao].
        """
        raise NotImplementedError

    def get_deriv1_bra(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        """First order skeleton derivative in half-transformed MO basis.

        See also
        --------
        get_deriv1_ao

        Notes
        -----
        If `get_deriv1_ao` implemented, this function should behave like `deriv_bra = deriv_ao @ mocc`, where `mocc` is
        the occupied molecular coefficients (as ket).

        However, in some cases, it is probably better to skip the usage of `get_deriv1_ao` and directly use this function.
        By ket half-transformation, some RI-JK or DFT methods will benefit from boost by using low-rank occupied orbitals,
        instead of using full AO basis.

        Returns
        -------
        deriv_bra : np.ndarray
            The first-order skeleton derivative in half-transformed MO basis, shape [natm, 3, nao, nocc].
            Note that this function will handle the order of occupied orbitals. If occupation number is not sorted contiguously,
            you may be extra cautious to this function.
        """
        occidx = mo_occ > 1e-15
        mocc = mo_coeff[:, occidx]
        return self.get_deriv1_ao(mo_coeff, mo_occ) @ mocc

    @abstractmethod
    def make_response_preparation(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        """Prepare the data for response calculation.

        Response (related to second order of density matrix derivative to energy) will be called multiple-times in CP-HF solver and other places.
        Some methods (especially DFT) may be helpful to prepare some data for response calculation, and store them in the object.
        """
        pass

    @abstractmethod
    def get_response_bra(self, bra: np.ndarray) -> np.ndarray:
        r"""Get the response contribution for current SCF component.

        This function will be called multiple-times in CP-HF solver and other places.
        Call `make_response_preparation` before this function to make sure the data is ready.

        Also, this function will not pass in the MO coefficients and occupation numbers.
        If you need them, you should store them in the object by function `make_response_preparation`.

        Parameters
        ----------
        bra : np.ndarray
            The bra part. Shape [..., nao, nocc].
            This is usually the derivative of MO coefficients (like :math:`U_{\mu i}^\mathbb{A}` given by CP-HF).

        Returns
        -------
        resp_bra : np.ndarray
            The response potential (related to second order of density matrix derivative to energy).
            Shape [..., nao, nocc].

        Notes
        -----
        This function may not work for fractional occupation.
        We have not prepared to propose a good API for fractional occupation.
        """
        raise NotImplementedError
