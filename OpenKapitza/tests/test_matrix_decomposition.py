"""
Unit and regression test for the matrix_decomposition module.
"""

import numpy as np
from unittest import TestCase
import OpenKapitza as OK


def test_matrix_decomposition():

    hessian = np.loadtxt('tests/test_hessian.out')
    num_atom_per_unitcell = 8

    H0 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2)]

    H1 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((1 - 1) * 3 + (2 - 1)) + (3 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((1 - 1) * 3 + (2 - 1)) + (3 - 1) + 2)]

    H2 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((3 - 1) * 3 + (2 - 1)) + (3 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((3 - 1) * 3 + (2 - 1)) + (3 - 1) + 2)]

    H3 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (1 - 1)) + (3 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (1 - 1)) + (3 - 1) + 2)]

    H4 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (3 - 1)) + (3 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (3 - 1)) + (3 - 1) + 2)]

    T1 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((1 - 1) * 3 + (2 - 1)) + (4 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((1 - 1) * 3 + (2 - 1)) + (4 - 1) + 2)]

    T2 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((3 - 1) * 3 + (2 - 1)) + (4 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((3 - 1) * 3 + (2 - 1)) + (4 - 1) + 2)]

    T3 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (1 - 1)) + (4 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (1 - 1)) + (4 - 1) + 2)]

    T4 = hessian[num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1)):
                 num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (2 - 1)) + (3 - 1) + 2),
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (3 - 1)) + (4 - 1)):
         num_atom_per_unitcell * 3 * (5 * 2 * ((2 - 1) * 3 + (3 - 1)) + (4 - 1) + 2)]

    expected_value = {'H0': H0, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4}

    kapitza_prediction = OK.matrix_decomposition(hsn_matrix=hessian, block_size=2, block_indices=[2, 2, 3],
                                                 rep=[3, 3, 5], natm_per_unitcell=num_atom_per_unitcell)

    assert kapitza_prediction['H0'].all() == expected_value['H0'].all() and \
           kapitza_prediction['H1'].all() == expected_value['H1'].all() and \
           kapitza_prediction['H2'].all() == expected_value['H2'].all() and \
           kapitza_prediction['H3'].all() == expected_value['H4'].all() and \
           kapitza_prediction['T1'].all() == expected_value['T1'].all() and \
           kapitza_prediction['T2'].all() == expected_value['T2'].all() and \
           kapitza_prediction['T3'].all() == expected_value['T3'].all() and \
           kapitza_prediction['T4'].all() == expected_value['T4'].all()