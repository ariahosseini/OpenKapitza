"""compute Green's function in bulk materials."""

import jax.numpy as jnp
import functools
from copy import deepcopy
from typing import Any
import sys
import numpy as np


def poscar(vasp_poscar='POSCAR'):

    poscar_file = open(vasp_poscar, "r")
    _pos = poscar_file.readlines()
    # pos = _pos.strip().split()
    print(_pos)
    scaling_factor = float(_pos[1].strip().split()[0])
    _unit_vec = [_pos[io].strip().split() for io in range(2,5)]
    unit_vec = [[float(io) for io in jo] for jo in _unit_vec]
    atoms_types = _pos[5].strip().split()
    num_each_elm_uc = [int(io) for io in _pos[6].strip().split()]
    num_atoms_uc = sum(num_each_elm_uc)
    # num_elm_uc = zip(atoms_types, _pos[6].strip().split())


    print(num_atoms_uc)





    # pos = 0
    # for _line in poscar_file:
    #     line = _line.strip().split()
    #     print(line)
    #     if len(line) == 0:  # empty line
    #         continue
    #     else:
    #         if pos == 1:
    #             print('done')
    #
    #         if line == 'Direct':
    #             pos = 1
    #
    #         else:
    #             continue


def force_constant(forces="FORCE_CONSTANTS", rep=5):

    forces_file = open(forces, "r")
    forces = forces_file.readlines()
    number_atoms = int(forces.pop(0))
    del forces[::4]
    number_atoms_unitcell = int(number_atoms / rep**3)
    ifc = []
    for elm in forces:
        _ifc = elm.split()
        ifc.append(np.asarray(_ifc, dtype=float))
    ifc = np.array(ifc)

    hsn_matrix = ifc[:3 * number_atoms, :]

    for idx in np.arange(1, number_atoms):
        _hessian = ifc[3 * idx * number_atoms:(idx + 1) * 3 * number_atoms, :]
        hsn_matrix = np.hstack((hsn_matrix, _hessian))

    return hsn_matrix, number_atoms, number_atoms_unitcell


def phonopy_matrix_decomposition(hsn_matrix: np.ndarray, num_atoms_uc, rep: int = 5):

    """
    A method to decompose the Hessian matrix in to sub-matrices (onsite and hopping matrices)

    Parameters
    ----------
    hsn_matrix: np.ndarray
        Hessian matrix is the "read_hessian" method return
    rep: list
        This term shows how many times the unit cell is replicated in each lead
    num_atoms_uc: int
        Number of atoms per unit cell

    Returns
    ----------
        H0, H1, H2, H3, H4, T1, T2, T3, T4 showing onsite and hopping matrices
    """

    H0 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2)]

    H1 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-1+rep): 3*num_atoms_uc*(int(rep**3/2)+rep)]

    H2 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-1-rep): 3*num_atoms_uc*(int(rep**3/2)-rep)]

    H3 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-1+rep**2): 3*num_atoms_uc*(int(rep**3/2)+rep**2)]

    H4 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-1-rep**2): 3*num_atoms_uc*(int(rep**3/2)-rep**2)]

    T1 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)+rep): 3*num_atoms_uc*(int(rep**3/2)+1+rep)]

    T2 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-rep): 3*num_atoms_uc*(int(rep**3/2)+1-rep)]

    T3 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)+rep**2): 3*num_atoms_uc*(int(rep**3/2)+1+rep**2)]

    T4 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
                    3*num_atoms_uc*(int(rep**3/2)-rep**2): 3*num_atoms_uc*(int(rep**3/2)+1-rep**2)]

    Hsn = {'H0': H0, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4}

    return Hsn


def wave_vectors(wave_len: float, num_kpoints: int) -> dict:

    """
    A function to define wave vectors in the surface perpendicular to the transport direction

    Parameters
    ----------
    wave_len: float
        The periodicity length along the transverse direction
    num_kpoints : int
        Number of kpoints

    Returns
    ----------
    dic_output : dict
        First key includes the kpoints, and the second one includes the periodicity length
    """

    kpoints_i = np.linspace(-np.sqrt(2) * np.pi / wave_len, np.sqrt(2) * np.pi / wave_len,
                            num_kpoints,
                            endpoint=True)
    kpoints_j = np.linspace(-np.sqrt(2) * np.pi / wave_len, np.sqrt(2) * np.pi / wave_len,
                            num_kpoints,
                            endpoint=True)

    ki_grid, kj_grid = np.meshgrid(kpoints_i, kpoints_j)
    kpoints = np.array([kj_grid.flatten(), ki_grid.flatten()])

    dict_output = dict(kpoints=kpoints, wave_len=wave_len)

    return dict_output
