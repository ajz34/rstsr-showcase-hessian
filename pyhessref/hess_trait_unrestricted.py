import numpy as np
from abc import ABC, abstractmethod


class UHessCoreAPI(ABC):
    """Abstract class for Hessian-related API for unrestricted SCF core components.

    Term Explanation
    ----------------

    See `RHessCoreAPI` (restricted version). The interpretation of *core* is identical.

    UHF Conventions
    ---------------

    For UHF, the molecular orbital descriptors are stored per spin:
    - ``mo_coeff``: shape ``[2, nao, nmo]``
    - ``mo_occ``:   shape ``[2, nmo]`` (occupied orbitals have occupation 1, not 2)
    - ``mo_energy``: shape ``[2, nmo]``

    Since core/skeleton hessian terms depend only on the *total* density matrix
    ``D = D^alpha + D^beta`` and the AO-basis derivative is spin-independent,
    the contract here looks identical to the restricted case from the caller's
    point of view -- the implementing class is responsible for assembling the total
    density internally if needed.
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
            Molecular orbital coefficients, shape ``[2, nao, nmo]``.
        mo_occ : np.ndarray
            Molecular orbital occupation numbers, shape ``[2, nmo]``.
        dm0 : np.ndarray, optional
            The **total** density matrix ``D^alpha + D^beta``, shape ``[nao, nao]``.

        Returns
        -------
        hess : np.ndarray
            The Hessian matrix for current SCF component, shape ``[natm, natm, 3, 3]``.
        """
        raise NotImplementedError

    @abstractmethod
    def generator_deriv1(self) -> callable:
        """Generate the function to compute the first-order derivative of core component.

        For UHF this is spin-independent (e.g. hcore), so the returned function is the
        same as in the restricted case: takes atom index, returns shape ``[3, nao, nao]``.

        If this component does not contribute (like nuclear repulsion), return None.
        """
        raise NotImplementedError


class UHessElecInteractAPI(ABC):
    """Abstract class for Hessian-related API for unrestricted SCF electronic interaction components.

    UHF Conventions
    ---------------

    Differences from the restricted version:
    - ``mo_coeff`` / ``mo_occ`` are passed as per-spin stacked arrays of shape
      ``[2, nao, nmo]`` and ``[2, nmo]`` respectively.
    - The first-order skeleton derivative in AO basis is **spin-resolved**:
      ``get_deriv1_ao`` returns ``[2, natm, 3, nao, nao]``.
      (UHF Fock matrix has different J/K coupling between alpha and beta channels, so
      the Fock skeleton derivative differs by spin even when the AO integrals don't.)
    - The half-transformation to bra uses spin-specific occupied orbitals, so
      ``get_deriv1_bra`` returns a *list* of two arrays whose last dimension differs
      (``[natm, 3, nao, nocc_alpha]`` and ``[natm, 3, nao, nocc_beta]``).
    - The response operator ``get_response_bra`` couples the two spin channels through
      the J part and decouples through the K part. It takes and returns a list of two
      arrays, of shape ``[..., nao, nocc_alpha]`` and ``[..., nao, nocc_beta]``.
    """

    @abstractmethod
    def make_skeleton_hess(
        self,
        mo_coeff: np.ndarray,
        mo_occ: np.ndarray,
    ) -> np.ndarray:
        """Generate the **skeleton** contribution of Hessian for current SCF component.

        Returns
        -------
        hess : np.ndarray
            The Hessian matrix for current SCF component, shape ``[natm, natm, 3, 3]``.
        """
        raise NotImplementedError

    @abstractmethod
    def get_deriv1_ao(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> np.ndarray:
        """First order skeleton derivative in AO basis, spin-resolved.

        Returns
        -------
        deriv_ao : np.ndarray
            Shape ``[2, natm, 3, nao, nao]``.
        """
        raise NotImplementedError

    def get_deriv1_bra(self, mo_coeff: np.ndarray, mo_occ: np.ndarray) -> list[np.ndarray]:
        """First order skeleton derivative in half-transformed (bra) MO basis.

        Default implementation calls ``get_deriv1_ao`` and right-multiplies each spin block
        with the corresponding occupied coefficients.

        Returns
        -------
        deriv_bra : list[np.ndarray]
            A list ``[bra_alpha, bra_beta]`` where each entry has shape
            ``[natm, 3, nao, nocc_sigma]``.
        """
        deriv_ao = self.get_deriv1_ao(mo_coeff, mo_occ)
        out = []
        for s in range(2):
            occidx = mo_occ[s] > 1e-15
            mocc = mo_coeff[s][:, occidx]
            out.append(deriv_ao[s] @ mocc)
        return out

    @abstractmethod
    def make_response_preparation(self, mo_coeff: np.ndarray, mo_occ: np.ndarray):
        """Prepare the data for response calculation."""
        pass

    @abstractmethod
    def get_response_bra(self, bra: list[np.ndarray]) -> list[np.ndarray]:
        r"""Get the response contribution for current SCF component.

        Parameters
        ----------
        bra : list[np.ndarray]
            ``[bra_alpha, bra_beta]``. Each entry has shape ``[..., nao, nocc_sigma]``.
            Leading dimensions must agree across the two spins.

        Returns
        -------
        resp_bra : list[np.ndarray]
            ``[resp_alpha, resp_beta]`` with the same shapes as the inputs.
        """
        raise NotImplementedError
