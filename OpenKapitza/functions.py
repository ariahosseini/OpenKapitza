"""Provide the primary functions."""
import jax.numpy as jnp
import functools
from copy import deepcopy
from typing import Any

# from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import ChainMap
sns.set()
sns.set_style("white", {"xtick.major.size": 2, "ytick.major.size": 2})
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 4})


def read_hessian(file_name: str) -> np.ndarray:
    """
    A function to read Hessian matrix

    Parameters
    ----------
    file_name : str
        Lammps output file -- hessian-mass-weighted-hessian.d

    Returns
    ----------
    hessian : np.ndarray
        Phonon hessian matrix
    """

    hessian_data_file = np.loadtxt(file_name, delimiter=None, skiprows=0)  # Load data
    hessian_symmetric = (np.triu(hessian_data_file, k=0) + np.tril(hessian_data_file, k=0).T) / 2  # Remove noises
    hessian = np.triu(hessian_symmetric) + np.triu(hessian_symmetric, k=1).T  # Hessian is symmetric
    ang2m = 1e-10  # Convert angstrom to meter

    return hessian*ang2m*ang2m/1e3  # The Hessian has a funnny unit [J/Ang^2/gram]


def plot_2darray(array_to_plot: np.ndarray, pic_name: str, set_title: str,
                 x_label: str = None, y_label: str = None) -> None:
    """
    A function to plot two-dimensional numpy arrays


    Parameters
    ----------
    array_to_plot : np.ndarray
        Numpy two-dimensional arrays we seek to plot
    set_title : str
        Title of the plot
    x_label : str
        Heatmap x label
    y_label : str
        Heatmap y label

    Returns
    ----------
    None
    """

    plt.figure(figsize=(6.5, 6.5))
    ax = sns.heatmap(array_to_plot, cbar=False)
    ax.set_frame_on(False)
    ax.tick_params(axis='y', labelleft='off')
    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24, labelpad=15)
    ax.tick_params(axis="y", labelsize=24)
    ax.tick_params(axis="x", labelsize=24)
    ax.set_title(set_title)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    plt.tight_layout()
    plt.savefig(pic_name)


def read_crystal(natm_per_unitcell: int, rep: list, skip_rows: int = 9, file_crystal: str = 'data.unwrapped') -> dict:
    """
    A function to read unwrapped atoms position from lammps output and compute lattice points

    Parameters
    ----------
    natm_per_unitcell : int
        Number of atoms per unit cell
    rep: list
        This term shows how many times the unit cell is replicated in each lead
    skip_rows: int
        Number of lines to skip in data.unwrapped
    file_crystal: str
        Lammps output file — data.unwrapped

    Returns
    ----------
    crystal_info : dict
        First key includes the crystal points, the second key includes the lattice points
        and the third key is the lattice_constant in [m]
    """

    ang2m = 1e-10  # Convert angstrom to meter
    with open(file_crystal, 'r') as read_obj:
        for line_number, line in enumerate(read_obj):
            if "ITEM: BOX BOUNDS pp pp pp" in line:
                x_min, x_max = next(read_obj).split()
                y_min, y_max = next(read_obj).split()
                z_min, z_max = next(read_obj).split()
                break
        lattice_constant: np.array = \
            np.array([float(x_max) - float(x_min), float(y_max) - float(y_min), float(z_max) - float(z_min)]) / \
            np.array([rep[0], rep[1], 2 * rep[2]]) * ang2m  # Lattice constant

    crystal_points = np.loadtxt(file_crystal, delimiter=None, skiprows=skip_rows)  # Read data file
    lattice_points = crystal_points[::natm_per_unitcell, 3:] - crystal_points[0, 3:]  # Find lattice point
    crystal_info = {'crystal_points': crystal_points, 'lattice_points': lattice_points,
                    'lattice_constant': lattice_constant}  # return dict

    return crystal_info


# def mitigate_periodic_effect(
#         hessian: np.ndarray, crystal_info: dict, natm_per_unitcell: int,
#         rep: list, file_crystal: str = 'data.unwrapped') -> dict:
#     """
#         A function to remove the atoms on the edge and theirs force constant elements
#
#         Parameters
#         ----------
#         hessian : np.ndarray
#             Return object of the read_hessian function
#         crystal_info: dict
#             Return object of the crystal_info function
#         natm_per_unitcell: int
#             Number of atoms per unit cell
#         rep: list
#             This term shows how many times the unit cell is replicated in each lead
#         file_crystal: str
#             Name of the data file -- data.unwrapped
#
#         Returns
#         ----------
#         return_dic : dict
#             A dictionary that includes Hessian matrix, lattice points and lattice constant, respectively
#         """
#
#     ang2m = 1e-10  # convert angstrom to meter
#     with open(file_crystal, 'r') as read_obj:
#         for line_number, line in enumerate(read_obj):
#             if "ITEM: BOX BOUNDS pp pp pp" in line:
#                 x_min, x_max = next(read_obj).split()
#                 y_min, y_max = next(read_obj).split()
#                 z_min, z_max = next(read_obj).split()
#                 break
#         lattice_constant: np.array = \
#             np.array([float(x_max) - float(x_min), float(y_max) - float(y_min), float(z_max) - float(z_min)]) / \
#             np.array([rep[0], rep[1], 2 * rep[2]]) * ang2m  # Lattice constant
#
#     # Remove lattice points on the edge
#     lattice_points = crystal_info['lattice_points']
#     lattice_points[::2 * rep[2]] = np.inf
#     lattice_points[2 * rep[2] - 1::2 * rep[2]] = np.inf
#     lp = lattice_points[np.isfinite(lattice_points).all(1)]  # New lattice points
#
#     # Remove corresponding elements in Hessian
#     for io in range(natm_per_unitcell * 3):
#         hessian[io::natm_per_unitcell * 3 * rep[-1] * 2] = np.inf
#         hessian[:, io::natm_per_unitcell * 3 * rep[-1] * 2] = np.inf
#         hessian[io + natm_per_unitcell * 3 * (rep[-1] * 2 - 1)::natm_per_unitcell * 3 * rep[-1] * 2] = np.inf
#         hessian[:, io + natm_per_unitcell * 3 * (rep[-1] * 2 - 1)::natm_per_unitcell * 3 * rep[-1] * 2] = np.inf
#     hsn = hessian[~(np.isinf(hessian).all(axis=1))]
#     hsn_matrix = np.transpose(hsn.T[~(np.isinf(hsn.T).all(axis=1))])  # New Hessian matrix
#
#     return_dic = {'hsn_matrix': hsn_matrix, 'lattice_points': lp, 'lattice_constant': lattice_constant}
#
#     return return_dic


def matrix_decomposition(hsn_matrix: np.ndarray, block_size: int,
                         block_indices: list[int], rep: list[int], natm_per_unitcell: int) -> dict[Any, Any]:
    """
    A function to read unwrapped atoms position from lammps output and compute lattice points

    Parameters
    ----------
    hsn_matrix: NP.ndarray
        The ['hsn_matrix'] key of the "mitigate_periodic_effect" function return
    block_indices : list
        Pointer to the block position
    block_size: int
        Number of unit cells in the block
    rep: list
        This term shows how many times the unit cell is replicated in each lead after removinng the edge atoms
    natm_per_unitcell: int
        Number of atoms per unit cell

    Returns
    ----------
    Hsn : dict
        The keys are: 'H0', 'H1', 'H2', 'H3', 'H4', 'T1', 'T2', 'T3', 'T4' showing diagonal and off-diagonal matrices
    """

    f_idx = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0],
                      [0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1]])
    elements_idx = (block_indices[2] + f_idx[:, 2] - 1) + ((block_indices[1] + f_idx[:, 1] - 1)
                                                           * rep[0] + block_indices[0] + f_idx[:, 0] - 1) * 2 * rep[2]
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
    A function to read unwrapped atoms position from lammps output and compute lattice points

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
        A function to display Hessian matrix in the Fourier's space

        Parameters
        ----------
        Hsn: dict
            Return object of the mitigate_periodic_effect function
        kpoints : dict
            Return object of the define_wavevectors function

        Returns
        ----------
        output-dict : dict
            First keys are index of the kpoints, the values are 'Hsn_fourier', 'Hopping_fourier', and 'wavevector'
        """

    wavevector = kpoints['kpoints']
    periodicity_length = kpoints['periodicity_length']
    distance_vector = periodicity_length * np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    unit_planewave = np.exp(-1j * (np.matmul(distance_vector, wavevector)).T)  # Construct a plane wave

    def fourier_transform(planewave, Hsn_mat):
        Hsn_fourier = Hsn_mat['H0'] * planewave[0] + Hsn_mat['H1'] * planewave[1] \
                      + Hsn_mat['H2'] * planewave[2] + Hsn_mat['H3'] * planewave[3] \
                      + Hsn_mat['H4'] * planewave[4]
        Hopping_fourier = Hsn_mat['T1'] * planewave[1] + Hsn_mat['T2'] * planewave[2] + \
                          Hsn_mat['T3'] * planewave[3] + Hsn_mat['T4'] * planewave[4]
        Hsn_matrix = {'Hsn_fourier': Hsn_fourier, 'Hopping_fourier': Hopping_fourier}
        return Hsn_matrix

    f_transform = functools.partial(fourier_transform, Hsn_mat=Hsn)
    Hsn_matrix_fourier = map(f_transform, unit_planewave)
    Hsn_keys = np.arange(np.shape(wavevector)[1])
    output_dict = dict(zip(Hsn_keys, [*Hsn_matrix_fourier]))

    # Add 'wavevector' to the returned output
    for _, __ in enumerate(Hsn_keys):
        output_dict[__]['wavevector'] = wavevector[:, _]

    return output_dict


def surface_green_func(left_hsn_bulk: dict, left_hsn_surface: dict, right_hsn_surface: dict, right_hsn_bulk: dict,
                       omega_min: float, omega_max: float, omega_num: int,
                       number_atom_unitcell: int, block_size: int, delta: float = 1e-6):
    """
            A function to compute surface Green's function

            Parameters
            ----------
            left_hsn_bulk: dict
                Return object of the hessian_fourier_form for left lead bulk
            left_hsn_surface: dict
                Return object of the hessian_fourier_form for left lead surface
            right_hsn_surface: dict
                Return object of the hessian_fourier_form for right lead surface
            right_hsn_bulk: dict
                Return object of the hessian_fourier_form for right lead bulk
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
            delta: float
                Infinitesimal positive number

            Returns
            ----------
            output-dict : dict
                First keys are frequencies, the values are left and right surface Green's function
            """

    omega = np.linspace(omega_min, omega_max, omega_num, endpoint=True)  # An array of frequencies

    # A function to implement the decimation method
    def decimation_iteration(omega_val, left_Hsn_bulk, left_Hsn_surface, right_Hsn_surface, right_Hsn_bulk,
                             num_atom_unitcell=number_atom_unitcell, delta_o=delta):

        def iteration_loop(e_surface, deepcopy_e_surface, e, alpha, beta):
            io = 0
            while True:
                a_term = jnp.linalg.inv(e) @ alpha
                b_term = jnp.linalg.inv(e) @ beta
                e_surface += alpha @ b_term
                e += beta @ a_term + alpha @ b_term
                alpha = alpha @ a_term
                beta = beta @ b_term
                if np.linalg.norm(abs(e_surface) - abs(deepcopy_e_surface)) < 1e-1 or io > 500:
                    # print('Finally:')
                    # print('Max real part of e_surface: ', np.amax(e_surface.real))
                    # print('Max imag part of e_surface: ',np.amax(e_surface.imag))
                    # print('Error: ', np.max(np.linalg.norm(abs(e_surface) - abs(deepcopy_e_surface))))
                    # print('Iteration: ', io)
                    break
                deepcopy_e_surface = deepcopy(e_surface)
                io += 1
            return e_surface

        def iter_func_right(Z, Hsn_bulk):

            e_surface = Z - Hsn_bulk['Hsn_fourier']
            deepcopy_e_surface = deepcopy(e_surface)
            e = deepcopy(e_surface)
            alpha = Hsn_bulk['Hopping_fourier']
            beta = Hsn_bulk['Hopping_fourier'].conj().T
            e_surface = iteration_loop(e_surface, deepcopy_e_surface, e, alpha, beta)
            return e_surface

        def iter_func_left(Z, Hsn_bulk):

            e_surface = Z - Hsn_bulk['Hsn_fourier']
            deepcopy_e_surface = deepcopy(e_surface)
            e = deepcopy(e_surface)
            alpha = Hsn_bulk['Hopping_fourier'].conj().T
            beta = Hsn_bulk['Hopping_fourier']
            e_surface = iteration_loop(e_surface, deepcopy_e_surface, e, alpha, beta)
            return e_surface

        Z = omega_val ** 2 * (1 + 1j * delta_o) * np.eye(3 * num_atom_unitcell * block_size, k=0)
        right_e_bulk_surface = dict(
            map(lambda _: (_[0], iter_func_right(Z, _[1])), right_Hsn_bulk.items()))
        left_e_bulk_surface = dict(
            map(lambda _: (_[0], iter_func_left(Z, _[1])), left_Hsn_bulk.items()))

        def g_surf_right(omega_value, e_surface, Hsn_bulk, block_sze=block_size,
                         num_atom_unit_cell=number_atom_unitcell):
            e_surf = omega_value ** 2 * np.eye(3 * num_atom_unit_cell * block_sze, k=0) - \
                     Hsn_bulk['Hsn_fourier'] - (Hsn_bulk['Hopping_fourier']
                                                @ jnp.linalg.inv(e_surface)
                                                @ Hsn_bulk['Hopping_fourier'].conj().T)
            g_surface = jnp.linalg.inv(e_surf)
            return g_surface

        def g_surf_left(omega_value, e_surface, Hsn_bulk, block_sze=block_size,
                        num_atom_unit_cell=number_atom_unitcell):
            e_surf = omega_value ** 2 * np.eye(3 * num_atom_unit_cell * block_sze, k=0) - \
                     Hsn_bulk['Hsn_fourier'] - (Hsn_bulk['Hopping_fourier'].conj().T
                                                @ jnp.linalg.inv(e_surface)
                                                @ Hsn_bulk['Hopping_fourier'])
            g_surface = jnp.linalg.inv(e_surf)
            return g_surface

        right_g_surface = dict(
            map(lambda x, y: (x[0], g_surf_right(omega_val, x[1], y[1])),
                right_e_bulk_surface.items(), right_Hsn_bulk.items()))
        left_g_surface = dict(
            map(lambda x, y: (x[0], g_surf_left(omega_val, x[1], y[1])),
                left_e_bulk_surface.items(), left_Hsn_bulk.items()))

        g_surf = {'left_g_surface': left_g_surface, 'right_g_surface': right_g_surface}

        return g_surf

    decimate_iter = functools.partial(decimation_iteration, left_Hsn_bulk=left_hsn_bulk,
                                      left_Hsn_surface=left_hsn_surface,
                                      right_Hsn_surface=right_hsn_surface,
                                      right_Hsn_bulk=right_hsn_bulk,
                                      num_atom_unitcell=number_atom_unitcell, delta_o=delta)

    surf_green_func = map(decimate_iter, omega)
    output_dict = dict(zip(omega, surf_green_func))

    return output_dict


def device_green_func(left_hsn_surface: dict, hsn_device: dict, surface_green: dict, number_atom_unitcell, block_sze):

    number_kpoints = len(hsn_device)

    def dev_green_unit(omega_val, surf_green, num_atom_unitcell=number_atom_unitcell, block_size=block_sze):
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
            green_ret = omega_val ** 2 * np.eye(3 * num_atom_unitcell * block_size, k=0) - \
                        hsn_dev['Hsn_fourier'] - self_energy_left - self_energy_right
            green_adv = green_ret.conj().T
            Xi = np.trace(gamma_right @ green_ret @ gamma_left @ green_adv)
            return Xi

        def sum_transmittance_kpoint(xi_k1, xi_k2):
            return xi_k1 + xi_k2

        trans = list(map(lambda x, y, z, w: (gsurt_kpoint(x[1], y[1], z[1], w[1])), left_g_surface.items(),
                         right_g_surface.items(), left_hsn_surface.items(), hsn_device.items()))
        trans_omega = functools.reduce(sum_transmittance_kpoint, trans)/number_kpoints
        output = {omega_val: trans_omega}

        return output

    transmittance = list(map(lambda x: dev_green_unit(x[0], x[1]), surface_green.items()))

    return dict(ChainMap(*transmittance))


def single_mode_properties(surf_green_func: dict, left_hsn_bulk: dict, right_hsn_bulk: dict,
                           dev_green_func: dict, lattice_par: list[float]):

    def iter_func(surf_gfunc: dict, l_hsn_bulk: dict, r_hsn_bulk: np.array,
                  dev_gfunc: np.array, frq, lattice_parameter: list[float] = lattice_par):

        def inner_iter_func(l_hsn_b: np.array, r_hsn_b: np.array,
                            left_s_gfunc: dict, right_s_gfunc: dict, omega=frq, lp=lattice_parameter):

            left_bulk_self_energy = l_hsn_b.conj().T @ left_s_gfunc @ l_hsn_b
            right_bulk_self_energy = r_hsn_b @ right_s_gfunc @ r_hsn_b.conj().T
            left_bulk_gamma = 1j * (left_bulk_self_energy - left_bulk_self_energy.connj().T)
            right_bulk_gamma = 1j * (right_bulk_self_energy - right_bulk_self_energy.connj().T)

            right_fwd_ret_bloch = right_s_gfunc @ r_hsn_b['Hopping_fourier'].conj().T
            left_bkwd_adv_bloch_inv = (l_hsn_bulk['Hopping_fourier'].conj().T @ surf_gfunc['left_g_surface']).conj().T

            _, Ur_ret = np.linalg.eigh(right_fwd_ret_bloch)
            _, Ul_adv = np.linalg.eigh(left_bkwd_adv_bloch_inv)

            vel_left = lp[0] / 2 / omega * Ul_adv.conj().T @ left_bulk_gamma @ Ul_adv
            vel_right = lp[1] / 2 / omega * Ur_ret.conj().T @ right_bulk_gamma @ Ur_ret

            tr = 2j * omega / np.sqrrt(lp[0] * lp[1]) \
                 * np.sqrt(vel_right) \
                 * jnp.linalg.inv(Ur_ret) * dev_gfunc * jnp.linalg.inv(Ul_adv.conj().T) \
                 * np.sqrt(vel_left)

            return tr

        trans = list(map(lambda x, y, z, w: (inner_iter_func(x[1], y[1], z[1], w[1])), l_hsn_bulk.items(),
                         r_hsn_bulk.items(), left_hsn_surface.items(), dev_gfunc.items()))




        # left_bulk_self_energy = l_hsn_bulk['Hopping_fourier'].conj().T \
        #                         @ surf_gfunc['left_g_surface'] \
        #                         @ l_hsn_bulk['Hopping_fourier']
        #
        # right_bulk_self_energy = r_hsn_bulk['Hopping_fourier'] \
        #                          @ surf_gfunc['right_g_surface'] \
        #                          @ r_hsn_bulk['Hopping_fourier'].conj().T
        #
        # left_bulk_gamma = 1j * (left_bulk_self_energy - left_bulk_self_energy.connj().T)
        # right_bulk_gamma = 1j * (right_bulk_self_energy - right_bulk_self_energy.connj().T)
        #
        # right_fwd_ret_bloch = surf_gfunc['right_g_surface'] @ r_hsn_bulk['Hopping_fourier'].conj().T
        # left_bkwd_adv_bloch_inv = (l_hsn_bulk['Hopping_fourier'].conj().T @ surf_gfunc['left_g_surface']).conj().T
        #
        # _, Ur_ret = np.linalg.eigh(right_fwd_ret_bloch)
        # _, Ul_adv = np.linalg.eigh(left_bkwd_adv_bloch_inv)
        #
        # vel_left = lp[0] /2 / omega * Ul_adv.conj().T @ left_bulk_gamma @ Ul_adv
        # vel_right = lp[1] / 2 / omega * Ur_ret.conj().T @ left_bulk_gamma @ Ur_ret
        #
        # t = 2j * omega/np.sqrrt(lp[0]*lp[1]) \
        #     * np.sqrt(vel_right) \
        #     * jnp.linalg.inv(Ur_ret) * dev_gfunc * jnp.linalg.inv(Ul_adv.conj().T) \
        #     * np.sqrt(vel_left)
        return 0


    transmition_coeff = list(map(lambda x: iter_func(x[0], x[1]), surface_green.items()))













if __name__ == "__main__":
    print('OpenKapitza')
