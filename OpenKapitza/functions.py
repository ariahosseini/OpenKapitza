"""Provide the primary functions."""

import jax.numpy as jnp
import functools
from copy import deepcopy
from typing import Any
import sys
import numpy as np


def matrix_decomposition(hsn_matrix: np.ndarray, block_size: int,
                         block_indices: list[int], rep: list[int], natm_per_unitcell: int) -> dict[Any, Any]:

    """
    A method to decompose the Hessian matrix in to sub-matrices (onsite and hopping matrices)

    Parameters
    ----------
    hsn_matrix: np.ndarray
        Hessian matrix is the "read_hessian" method return
    block_indices : list
        Pointer to the block position
    block_size: int
        Number of unit cells in the block
    rep: list
        This term shows how many times the unit cell is replicated in each lead
    natm_per_unitcell: int
        Number of atoms per unit cell

    Returns
    ----------
    Hsn : dict
        The keys are: 'H0', 'H1', 'H2', 'H3', 'H4', 'T1', 'T2', 'T3', 'T4' showing onsite and hopping matrices
    """

    nearest_neighbor_idx = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0],
                      [0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1]])

    elements_idx = (block_indices[2] + nearest_neighbor_idx[:, 2] - 1) + \
                   ((block_indices[1] + nearest_neighbor_idx[:, 1] - 1)* rep[0] +
                    block_indices[0] + nearest_neighbor_idx[:, 0] - 1) * 2 * rep[2]
    Hsn_keys = ['H0', 'H1', 'H2', 'H3', 'H4', 'T1', 'T2', 'T3', 'T4']
    Hsn = {}  # Return dict — decomposed Hessian matrix
    for i in range(9):
        Hsn_block = hsn_matrix[natm_per_unitcell * 3 * elements_idx[0]:
                               natm_per_unitcell * 3 * (elements_idx[0] + block_size),
                    natm_per_unitcell * 3 * elements_idx[i]:
                    natm_per_unitcell * 3 * (elements_idx[i] + block_size)]
        Hsn[Hsn_keys[i]] = Hsn_block

    return Hsn


def define_wavevectors(periodicity_length: float, num_kpoints: int) -> dict:

    """
    A function to define wave vectors in the surface perpendicular to the transport direction

    Parameters
    ----------
    periodicity_length: float
        The periodicity length along the transverse direction
    num_kpoints : int
        Number of kpoints

    Returns
    ----------
    dic_output : dict
        First key includes the kpoints, and the second one includes the periodicity length
    """

    kpoints_y = np.linspace(-np.sqrt(2) * np.pi / periodicity_length, np.sqrt(2) * np.pi / periodicity_length,
                            num_kpoints,
                            endpoint=True)
    kpoints_x = np.linspace(-np.sqrt(2) * np.pi / periodicity_length, np.sqrt(2) * np.pi / periodicity_length,
                            num_kpoints,
                            endpoint=True)

    kx_grid, ky_grid = np.meshgrid(kpoints_x, kpoints_y)
    kpoints = np.array([ky_grid.flatten(), kx_grid.flatten()])
    periodicity_len = periodicity_length
    dict_output = dict(kpoints=kpoints, periodicity_length=periodicity_len)

    return dict_output


def hessian_fourier_form(Hsn: dict, kpoints: dict) -> dict[Any, Any]:

    """
        A function to transform the Hessian matrix in the Fourier's space

        Parameters
        ----------
        Hsn: dict
            Hessian matrix is the "read_hessian" method return
        kpoints : dict
            Return object of the define_wavevectors function

        Returns
        ----------
        output-dict : dict
            First keys are index of the kpoints, the values are 'Onsite_fourier', 'Hopping_fourier', and 'wavevector'
        """

    wavevector = kpoints['kpoints']
    periodicity_length = kpoints['periodicity_length']
    distance_vector = periodicity_length * np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    unit_planewave = np.exp(-1j * (distance_vector @ wavevector).T)  # Construct a plane wave
    print(np.shape(distance_vector), np.shape(wavevector), np.shape(unit_planewave))

    def fourier_transform(planewave):

        Hsn_fourier = Hsn['H0'] * planewave[0] + Hsn['H1'] * planewave[1] \
                      + Hsn['H2'] * planewave[2] + Hsn['H3'] * planewave[3] \
                      + Hsn['H4'] * planewave[4]
        Hopping_fourier = Hsn['T1'] * planewave[1] + Hsn['T2'] * planewave[2] + \
                          Hsn['T3'] * planewave[3] + Hsn['T4'] * planewave[4]

        Hsn_matrix = {'Onsite_fourier': Hsn_fourier, 'Hopping_fourier': Hopping_fourier}
        return Hsn_matrix

    Hsn_matrix_fourier = map(fourier_transform, unit_planewave)
    Hsn_keys = np.arange(np.shape(wavevector)[1])
    output_dict = dict(zip(Hsn_keys, [*Hsn_matrix_fourier]))

    for _, __ in enumerate(Hsn_keys):

        output_dict[__]['wavevector'] = wavevector[:, _]

    return output_dict


def surface_green_func(left_hsn_bulk: dict, right_hsn_surface: dict,
                       omega_min: float, omega_max: float, omega_num: int, number_atom_unitcell: int,
                       block_size: int, delta_o: float = 1e-6) -> dict:

    """
    A function to compute surface Green's function

        Parameters
        ----------
        left_hsn_bulk: dict
            Return object of the hessian_fourier_form for the left lead bulk
        right_hsn_surface: dict
            Return object of the hessian_fourier_form for the right lead surface
        omega_min: float
            Minimum frequency
        omega_max: float
            Maximum frequency
        omega_num: int
            Sampling the frequency
        number_atom_unitcell: int
            Number of atoms per unit cell
        block_size: int
            Block size
        delta_o: float
            Infinitesimal positive number

        Returns
        ----------
        output-dict : dict
            First keys are frequencies, the values are left and right surface Green's function
        """

    omega = np.linspace(omega_min, omega_max, omega_num, endpoint=True)  # An array of frequencies

    def decimation_iteration(omega_val):

        def iteration_loop(e_surface, deepcopy_e_surface, e, alpha, beta):

            itr = 0
            while np.linalg.norm(abs(e_surface) - abs(deepcopy_e_surface)) > 1e-5:

                a_term = jnp.linalg.inv(e) @ alpha
                b_term = jnp.linalg.inv(e) @ beta
                e_surface += alpha @ b_term
                e += beta @ a_term + alpha @ b_term
                alpha = alpha @ a_term
                beta = beta @ b_term
                deepcopy_e_surface = deepcopy(e_surface)
                itr += 1
                if itr > 10000:
                    print("Error: Make sure code does not diverge:",
                          np.linalg.norm(abs(e_surface) - abs(deepcopy_e_surface)))
                    sys.exit()

            return e_surface

        def iter_func_right(Hsn_bulk):

            e_surface = Z - Hsn_bulk['Onsite_fourier']
            deepcopy_e_surface = deepcopy(e_surface)
            e = deepcopy(e_surface)
            alpha = Hsn_bulk['Hopping_fourier']
            beta = Hsn_bulk['Hopping_fourier'].conj().T
            e_surf = iteration_loop(e_surface=e_surface, deepcopy_e_surface=deepcopy_e_surface,
                                    e=e, alpha=alpha, beta=beta)
            g_surf = jnp.linalg.inv(e_surf)
            return g_surf

        def iter_func_left(Hsn_bulk):

            e_surface = Z - Hsn_bulk['Onsite_fourier']
            deepcopy_e_surface = deepcopy(e_surface)
            e = deepcopy(e_surface)
            alpha = Hsn_bulk['Hopping_fourier'].conj().T
            beta = Hsn_bulk['Hopping_fourier']
            e_surf = iteration_loop(e_surface=e_surface, deepcopy_e_surface=deepcopy_e_surface,
                                    e=e, alpha=alpha, beta=beta)
            g_surf = jnp.linalg.inv(e_surf)
            return g_surf

        Z = omega_val ** 2 * (1 + 1j * delta_o) * np.eye(3 * number_atom_unitcell * block_size, k=0)
        right_g_surf = dict(
            map(lambda _: (_[0], iter_func_right(_[1])), right_hsn_surface.items()))
        left_g_surf = dict(
            map(lambda _: (_[0], iter_func_left(_[1])), left_hsn_bulk.items()))

        g_surf = {'left_g_surface': left_g_surf, 'right_g_surface': right_g_surf}

        return g_surf

    surf_green_func = map(decimation_iteration, omega)
    output_dict = dict(zip(omega, surf_green_func))

    return output_dict


def device_green_func(left_hsn_surface: dict, hsn_device: dict, surface_green: dict, number_atom_unitcell, block_size):

    """
    A function to compute surface Green's function

        Parameters
        ----------
        left_hsn_surface: dict
            Return object of the hessian_fourier_form for the left surface
        hsn_device: dict
            Return object of the hessian_fourier_form for the device
        surface_green: dict
            Return object of the surface_green_func
        number_atom_unitcell: int
            Number of atoms per unit cell
        block_size: int
            Block size

        Returns
        ----------
        omega: np.ndarray
            Frequency
        green_dev: np.ndarray
            Device Green's function
        transmission: np.ndarray
            Frequency-dependent transmission coefficient
        modal_transmission: np.ndarray
            Frequency and wave-vector dependent transmission coefficient
        """

    def dev_green_unit(omega_val, surf_green):

        left_g_surface = surf_green['left_g_surface']
        right_g_surface = surf_green['right_g_surface']

        def gsurt_kpoint(left_sf_green, right_sf_green, left_hsn_surf, hsn_dev):

            self_energy_left = left_hsn_surf['Hopping_fourier'].conj().T \
                               @ left_sf_green \
                               @ left_hsn_surf['Hopping_fourier']

            self_energy_right = hsn_dev['Hopping_fourier'] \
                                @ right_sf_green \
                                @ hsn_dev['Hopping_fourier'].conj().T

            gamma_left = 1j * (self_energy_left - self_energy_left.conj().T)
            gamma_right = 1j * (self_energy_right - self_energy_right.conj().T)
            green_ret = omega_val ** 2 * np.eye(3 * number_atom_unitcell * block_size, k=0) - \
                        hsn_dev['Onsite_fourier'] - self_energy_left - self_energy_right
            green_adv = green_ret.conj().T
            Xi = np.trace(gamma_right @ green_ret @ gamma_left @ green_adv)

            return green_ret, Xi

        def sum_transmittance_kpoint(xi_k1, xi_k2):

            return xi_k1 + xi_k2

        output = list(map(lambda w, x, y, z: (gsurt_kpoint(w[1], x[1], y[1], z[1])), left_g_surface.items(),
                               right_g_surface.items(), left_hsn_surface.items(), hsn_device.items()))

        green_dev_ret = np.array(list(zip(*output))[0])
        trans_modal = np.array(list(zip(*output))[1]).real

        trans_omega = functools.reduce(sum_transmittance_kpoint, trans_modal) / len(hsn_device)

        return green_dev_ret, trans_omega, trans_modal

    transmission_func = list(map(lambda x: dev_green_unit(x[0], x[1]), surface_green.items()))

    omega = np.array(list(surface_green.keys()))
    green_dev = np.array(list(zip(*transmission_func))[0])
    transmission = np.array(list(zip(*transmission_func))[1])
    modal_transmission = np.array(list(zip(*transmission_func))[2])

    print('sizes:', np.shape(omega), np.shape(green_dev), np.shape(transmission), np.shape(modal_transmission))

    return omega, green_dev, transmission, modal_transmission


def modal_properties(left_hsn_bulk: dict, left_hsn_surf: dict,
                     device_hsn: dict, right_hsn_surf: dict,
                     surf_gfunc: dict, device_gfunc: np.ndarray,
                     frq: np.ndarray, lattice_parameter: list[float]) -> dict:

    """
        A method to compute mode-resolved transmission coefficients

        Parameters
        ----------
        left_hsn_bulk: dict
            Return object of the hessian_fourier_form for the left lead
        left_hsn_surf: dict
            Return object of the hessian_fourier_form for the left surface
        device_hsn: dict
            Return object of the hessian_fourier_form for the device
        right_hsn_surf: dict
            Return object of the hessian_fourier_form for the right surface
        surf_gfunc: dict
            Return object of the surface_green_func
        device_gfunc: np.ndarray
            Device Green's function
        frq: np.ndarray
            The frequency
        lattice_parameter: list[float]
            lattice_parameter for left and right leads, respectively

        Returns
        ----------
        transmission_matrix: np.ndarray
            The mode-resolved transmission coefficients
        """

    def iter_func(left_lead_hopping_hsn: np.ndarray, left_s_gfunc: np.ndarray,
                  left_surf_hopping_hsn: np.ndarray,
                  dev_hopping_hsn: np.ndarray, dev_gfunc: np.ndarray,
                  right_lead_hopping_hsn: np.ndarray, right_s_gfunc: np.ndarray,
                  omega: float):

        """
            An inner method to compute mode-resolved transmission coefficients

            Parameters
            ----------
            left_lead_hopping_hsn: np.ndarray
                The hopping matrix — left bulk lead
            left_s_gfunc: np.ndarray
                The surface Green's function — left surface
            left_surf_hopping_hsn: np.ndarray
                The hopping matrix — left surface
            dev_hopping_hsn: np.ndarray
                The hopping matrix — right lead
            dev_gfunc: np.ndarray
                The Green's function — device
            right_lead_hopping_hsn: np.ndarray
                The hopping matrix — right bulk lead
            right_s_gfunc: np.ndarray
                The surface Green's function — right lead
            omega: float
                The frequency

            Returns
            ----------
            output: dict
                The frequency (keys) and the mode-resolved transmission coefficients (values)
            """

        left_self_energy = left_lead_hopping_hsn.conj().T @ left_s_gfunc @ left_lead_hopping_hsn
        right_self_energy = right_lead_hopping_hsn @ right_s_gfunc @ right_lead_hopping_hsn.conj().T
        left_gamma = 1j * (left_self_energy - left_self_energy.conj().T)
        right_gamma = 1j * (right_self_energy - right_self_energy.conj().T)

        right_fwd_ret_bloch = right_s_gfunc @ right_lead_hopping_hsn.conj().T
        left_bkwd_adv_bloch = (left_lead_hopping_hsn.conj().T @ left_s_gfunc).conj().T

        _, Ur_ret = np.linalg.eigh(right_fwd_ret_bloch)
        _, Ul_adv = np.linalg.eigh(left_bkwd_adv_bloch)

        vel_left = lattice_parameter[0] / 2 / omega * Ul_adv.conj().T @ left_gamma @ Ul_adv
        vel_right = lattice_parameter[1] / 2 / omega * Ur_ret.conj().T @ right_gamma @ Ur_ret

        ret_dev_gfunc = right_s_gfunc @ dev_hopping_hsn @ dev_gfunc @ left_surf_hopping_hsn @ left_s_gfunc

        transmission_matrix = 2j * omega / np.sqrt(np.prod(lattice_parameter)) \
                              * np.sqrt(vel_right) @ jnp.linalg.inv(Ur_ret) @ ret_dev_gfunc \
                              @ jnp.linalg.inv(Ul_adv.conj().T) @ np.sqrt(vel_left)

        return transmission_matrix

    output = {}
    for iter_omg, omg in enumerate(frq):

        dev_gr_func = device_gfunc[iter_omg]
        surf_gr_func = surf_gfunc[omg]

        modal_transmission = map(lambda t, u, v, w, x, y, z:
                                 (iter_func(t[1]['Hopping_fourier'], u[1], v[1]['Hopping_fourier'],
                                            w[1]['Hopping_fourier'], x[1]['Hopping_fourier'], y, z[1], omg)),
                                 left_hsn_bulk.items(), surf_gr_func['left_g_surface'].items(), left_hsn_surf.items(),
                                 device_hsn.items(), right_hsn_surf.items(), dev_gr_func,
                                 surf_gr_func['right_g_surface'].items())

        output.update({omg: list(modal_transmission)})

    return output


if __name__ == "__main__":
    print('OpenKapitza')