"""Provide the input/ output functions."""

import numpy as np


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
        Phonon hessian matrix [J/m^2/amu]
    """

    amu2kg = 1.66054e-27  # Convert atomic mass unit (AMU) to meter
    hessian_data_file = np.loadtxt(file_name, delimiter=None, skiprows=0)  # Load data
    hess_mat = hessian_data_file
    hessian_symmetric = (np.triu(hess_mat, k=0) + np.tril(hess_mat, k=0).T) / 2  # Remove noises
    hessian = np.triu(hessian_symmetric) + np.triu(hessian_symmetric, k=1).T  # Hessian is symmetric

    return hessian


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
        Number of lines to skip in "data.unwrapped"
    file_crystal: str
        Lammps output file — "data.unwrapped"

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
    lattice_points = crystal_points[::natm_per_unitcell, 2:] - crystal_points[0, 2:]  # Find lattice point
    crystal_info = {'atoms_position': crystal_points, 'lattice_points': lattice_points,
                    'lattice_constant': lattice_constant}  # Output

    return crystal_info
