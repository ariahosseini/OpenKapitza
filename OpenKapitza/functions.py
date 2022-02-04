"""Provide the primary functions."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import ndarray

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
        file_hessian: str, file_crystal: str, natm_per_unitcell: int, rep: list, skip_rows: int = 9) -> dict:
    hessian = read_hessian(file_hessian)

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
            np.array([rep[0], rep[1], 2 * rep[2]])
    crystal_info = read_crystal(file_crystal, natm_per_unitcell, skip_rows)
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


def matrix_decomposition(hsn_matrix: np.darray, indices: list,
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

    f_idx = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
                    [1, -1, 0], [1, 1, 0], [1, 0, 0], [1, 0, -1]])
    elements_idx = (indices[2] - 1) + ((indices[1]+f_idx[:, 0] - 1) * rep[0] + indices[0] + f_idx[:, 1]- 1) * 2 * rep[2]

    Hsn_keys = ['H0', 'H1', 'H2', 'H3', 'H4', 'T1', 'T2', 'T3', 'T4']
    Hsn = {}
    for i in range(7):
        Hsn_block = hsn_matrix[natm_per_unitcell * 3 * elements_idx[i]: natm_per_unitcell * 3 * (elements_idx[i] + block_size),
         natm_per_unitcell * 3 * elements_idx[0]: natm_per_unitcell * 3 * (elements_idx[0] + block_size)]
        Hsn[Hsn_keys[i]] = Hsn_block



    return Hsn


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print('Done')
