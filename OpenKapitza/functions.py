"""Provide the primary functions."""
import jax.numpy as jnp
import functools
from copy import deepcopy
from typing import Any
import toolz


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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
    -------
    hessian : np.ndarray
        Phonon hessian matrix -- symmetric
    """

    hessian_data_file = np.loadtxt(file_name, delimiter=None, skiprows=0)
    hessian_symmetric = (np.triu(hessian_data_file, k=0) + np.tril(hessian_data_file, k=0).T) / 2
    hessian = np.triu(hessian_symmetric) + np.triu(hessian_symmetric, k=1).T

    return hessian


def plot_2darray(array_to_plot: np.ndarray, pic_name: str, set_title: str, x_label: str = None,
                 y_label: str = None) -> None:
    """
    A function to plot two-dimensional arrays

    Parameters
    ----------
    array_to_plot : np.ndarray
        Numpy two-dimensional  arrays we are plotting
    set_title : str
        Title of the plot
    x_label : str
        Heatmap x label
    y_label : str
        Heatmap y label

    Returns
    -------
    None
    """

    plt.figure(figsize=(6.5, 6.5))
    ax = sns.heatmap(array_to_plot, cbar=False)  # linewidth=0.1
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


def read_crystal(file_name: str, natm_per_unitcell: int, skip_rows: int = 9) -> dict:
    """
    A function to read unwrapped atoms position from lammps output and compute lattice points

    Parameters
    ----------
    file_name: str
        Lammps output file — data.wrapper
    natm_per_unitcell : int
        Number of atoms per unit cell
    skip_rows: int
        Number of lines to skip in data.unwrapped

    Returns
    -------
    output : dict
        First key includes the crystal points and the second key includes the lattice points
    """

    crystal_points = np.loadtxt(file_name, delimiter=None, skiprows=skip_rows)
    lattice_points = crystal_points[::natm_per_unitcell, 3:] - crystal_points[0, 3:]

    crystal_info = {'crystal_points': crystal_points, 'lattice_points': lattice_points}

    return crystal_info


def mitigate_periodic_effect(
        hessian: np.ndarray, crystal_info: dict, file_crystal: str, natm_per_unitcell: int, rep: list, skip_rows: int = 9) -> dict:
    ang2m = 1e-10
    with open(file_crystal, 'r') as read_obj:
        # Read all lines in the file one by one
        for line_number, line in enumerate(read_obj):
            # For each line, check if line contains the string
            if "ITEM: BOX BOUNDS pp pp pp" in line:
                x_min, x_max = next(read_obj).split()
                y_min, y_max = next(read_obj).split()
                z_min, z_max = next(read_obj).split()
                break
        lattice_constant: np.array = \
            np.array([float(x_max) - float(x_min), float(y_max) - float(y_min), float(z_max) - float(z_min)]) / \
            np.array([rep[0], rep[1], 2 * rep[2]])*ang2m
    lattice_points = crystal_info['lattice_points']
    lattice_points[::2 * rep[2]] = np.inf
    lattice_points[2 * rep[2] - 1::2 * rep[2]] = np.inf
    lp = lattice_points[np.isfinite(lattice_points).all(1)]

    for io in range(natm_per_unitcell * 3):
        hessian[io::natm_per_unitcell * 3 * rep[2] * 2] = np.inf
        hessian[:, io::natm_per_unitcell * 3 * rep[2] * 2] = np.inf
        hessian[io + natm_per_unitcell * 3 * (rep[2] * 2 - 1)::natm_per_unitcell * 3 * rep[2] * 2] = np.inf
        hessian[:, io + natm_per_unitcell * 3 * (rep[2] * 2 - 1)::natm_per_unitcell * 3 * rep[2] * 2] = np.inf
    hsn = hessian[~(np.isinf(hessian).all(axis=1))]
    hsn_matrix = np.transpose(hsn.T[~(np.isinf(hsn.T).all(axis=1))])

    return {'hsn_matrix': hsn_matrix, 'lattice_points': lp, 'lattice_constant': lattice_constant}


def matrix_decomposition(hsn_matrix: np.ndarray, indices: list,
                         block_size: int, rep: list, natm_per_unitcell: int) -> dict:
    """
    A function to read unwrapped atoms position from lammps output and compute lattice points

    Parameters
    ----------
    file_name: str
        Lammps output file — data.wrapper
    natm_per_unitcell : int
        Number of atoms per unit cell
    skip_rows: int
        Number of lines to skip in data.unwrapped

    Returns
    -------
    output : dict
        First key includes the crystal points and the second key includes the lattice points
    """

    f_idx = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0], [-1, 0, 0], [1, 0, 0],
                    [0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1]])
    elements_idx = (indices[2] + f_idx[:, 2] - 1) + ((indices[1]+f_idx[:, 1] - 1)
                                                     * rep[0] + indices[0] + f_idx[:, 0] - 1) * 2 * rep[2]

    Hsn_keys = ['H0', 'H1', 'H2', 'H3', 'H4', 'T1', 'T2', 'T3', 'T4']
    Hsn = {}
    for i in range(9):
        Hsn_block = hsn_matrix[natm_per_unitcell * 3 * elements_idx[i]: natm_per_unitcell * 3 * (elements_idx[i] + block_size),
         natm_per_unitcell * 3 * elements_idx[0]: natm_per_unitcell * 3 * (elements_idx[0] + block_size)]
        Hsn[Hsn_keys[i]] = Hsn_block

    return Hsn

def define_wavevectors(periodicity_lenght: float, num_kpoints: int) -> np.ndarray:
    """
    A function to read unwrapped atoms position from lammps output and compute lattice points

    Parameters
    ----------
    file_name: str
        Lammps output file — data.wrapper
    natm_per_unitcell : int
        Number of atoms per unit cell
    skip_rows: int
        Number of lines to skip in data.unwrapped

    Returns
    -------
    output : dict
        First key includes the crystal points and the second key includes the lattice points
    """

    ang2m = 1e-10

    kpoints_y = np.linspace(-np.sqrt(2)*np.pi/periodicity_lenght, np.sqrt(2)*np.pi/periodicity_lenght,num_kpoints,
                            endpoint=True)
    kpoints_x = np.linspace(-np.sqrt(2)*np.pi/periodicity_lenght, np.sqrt(2)*np.pi/periodicity_lenght,num_kpoints,
                            endpoint=True)

    kx_grid, ky_grid = np.meshgrid(kpoints_x, kpoints_y)

    kpoints = np.array([ky_grid.flatten(), kx_grid.flatten()])

    periodicity_len = periodicity_lenght

    return {'kpoints':kpoints, 'periodicity_lenght': periodicity_len}


def hessian_fourier_form(Hsn: dict, kpoints: dict) -> dict[Any, Any]:

    wavevector = kpoints['kpoints']
    periodicity_lenght = kpoints['periodicity_lenght']

    distance_vector = periodicity_lenght*np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    unit_planewave = np.exp(1j*(np.matmul(distance_vector, wavevector)).T)

    # Hsn_fourier = {}
    # Hopping_fourier = {}

    def fourier_transform(Hsn_mat, planewave):
        Hsn_fourier = Hsn_mat['H0'] * planewave[0] + Hsn_mat['H1'] * planewave[1]\
                                   + Hsn_mat['H2'] * planewave[2] + Hsn_mat['H3'] * planewave[3]\
                                   + Hsn_mat['H4'] * planewave[4]

        Hopping_fourier = Hsn_mat['T1'] * planewave[1] + Hsn_mat['T2'] * planewave[2] + \
                                       Hsn_mat['T3'] * planewave[3] + Hsn_mat['T4'] * planewave[4]
        Hsn_matrix = {'Hsn_fourier': Hsn_fourier, 'Hopping_fourier': Hopping_fourier}

        return Hsn_matrix

    f_transform = functools.partial(fourier_transform, Hsn)

    Hsn_matrix_fourier = map(f_transform, unit_planewave)
    Hsn_keys = np.arange(np.shape(wavevector)[1])

    return dict(zip(Hsn_keys, [*Hsn_matrix_fourier]))


# def surface_green_func(left_Hsn_bulk, left_Hsn_surface, Hsn_device, right_Hsn_surface, right_Hsn_bulk, omega_min,
#                        omega_max, omega_num, num_atom_unitcell, delta_o= 1e-6):

def surface_green_func(left_Hsn_bulk, left_Hsn_surface, right_Hsn_surface, right_Hsn_bulk, omega_min, omega_max, omega_num, num_atom_unitcell, block_size, delta_o= 1e-6):

    omega = np.linspace(omega_min, omega_min, omega_num, endpoint=True)

    def decimation_iteration(left_Hsn_bulk, left_Hsn_surface, right_Hsn_surface, right_Hsn_bulk, omega_val, num_atom_unitcell, delta_o):

        def iter_func(Z, Hsn_bulk, Hsn_surface):

            e_surface = Z - Hsn_bulk['Hsn_fourier']
            deepcopy_e_surface = deepcopy(e_surface)
            e = deepcopy(e_surface)
            alpha = Hsn_surface['Hopping_fourier']
            beta = Hsn_surface['Hopping_fourier'].conj().T

            io = 1
            while True:
                a_term = jnp.linalg.inv(e) @ alpha
                # print(a_term)
                b_term = jnp.linalg.inv(e) @ beta
                # print(b_term)
                e_surface += alpha @ b_term
                # print(e_surface)
                e += beta @ a_term + alpha @ b_term
                # print(e)
                alpha = alpha @ a_term
                # print(alpha)
                beta = beta @ b_term
                # print(beta)
                if np.linalg.norm(e_surface.real - deepcopy_e_surface.real) < 1e-6 or io > 5000:
                    break
                deepcopy_e_surface = deepcopy(e_surface)
            io += 1
            print(f'Number of interation: {io}')
            print(f'Error: {np.linalg.norm(e_surface.real - deepcopy_e_surface.real)}')
            return e_surface

        Z = omega_val**2*(1+1j*delta_o)*np.eye(3*num_atom_unitcell*block_size, k=0)
        right_e_surface = iter_func(Z, right_Hsn_bulk, right_Hsn_surface)
        left_e_surface = iter_func(Z, left_Hsn_bulk, left_Hsn_surface)

        omega_val ** 2 * np.eye(3 * num_atom_unitcell * block_size, k=0)

        left_g_surface = omega_val ** 2 * np.eye(3 * num_atom_unitcell * block_size, k=0) - left_Hsn_bulk['Hsn_fourier'] - (
                    left_Hsn_bulk['Hopping_fourier'] @ jnp.linalg.inv(left_e_surface) @ left_Hsn_bulk[
                'Hopping_fourier'].conj().T)

        right_g_surface = omega_val ** 2 * np.eye(3 * num_atom_unitcell * block_size, k=0) - right_Hsn_bulk['Hsn_fourier'] - (
                    right_Hsn_surface['Hopping_fourier'] @ jnp.linalg.inv(right_e_surface) @ right_Hsn_surface[
                'Hopping_fourier'].conj().T)

        return {'left_g_surface': left_g_surface, 'right_g_surface': right_g_surface}

    decimation_iterate = functools.partial(decimation_iteration, omega_val = omega[0], num_atom_unitcell = num_atom_unitcell, delta_o = delta_o)
    g_surface = dict(map(lambda w, x, y, z: (x[0], decimation_iterate(w[1], x[1], y[1], z[1])), left_Hsn_bulk.items(), left_Hsn_surface.items(), right_Hsn_surface.items(), right_Hsn_bulk.items()))
    return g_surface





        # Hsn_fourier = Hsn_mat['H0'] * planewave[0] + Hsn_mat['H1'] * planewave[1]\
        #                            + Hsn_mat['H2'] * planewave[2] + Hsn_mat['H3'] * planewave[3]\
        #                            + Hsn_mat['H4'] * planewave[4]
        #
        # Hopping_fourier = Hsn_mat['T1'] * planewave[1] + Hsn_mat['T2'] * planewave[2] + \
        #                                Hsn_mat['T3'] * planewave[3] + Hsn_mat['T4'] * planewave[4]
        # Hsn_matrix = {'Hsn_fourier': Hsn_fourier, 'Hopping_fourier': Hopping_fourier}

        # return Hsn_matrix


    # Hsn_matrix_fourier = map(f_transform, unit_planewave)
    # Hsn_keys = np.arange(np.shape(wavevector)[1])

    # return dict(zip(Hsn_keys, [*Hsn_matrix_fourier]))


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print('Done')
