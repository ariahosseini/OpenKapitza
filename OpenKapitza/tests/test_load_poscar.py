import OpenKapitza as OK
import numpy as np
from pytest import approx
from unittest import TestCase


def test_load_poscar():

    base = OK.init()
    actual_poscar = OK.load_poscar(base, path='./tests/')
    expected_poscar = {'cell': np.array([[0.00000000e+00, 2.73355606e-10, 2.73355606e-10],
       [2.73355606e-10, 0.00000000e+00, 2.73355606e-10],
       [2.73355606e-10, 2.73355606e-10, 0.00000000e+00]]), 'numbers': np.array([14, 14]),
                       'base': np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [1.36677803e-20, 1.36677803e-20, 1.36677803e-20]]), 'spacegroup': 'Fd-3m (227)',
                       'a': 3.865832050566194e-10, 'elements': ['Si', 'Si'], 'na': 2, 'n_pol': 6,
                       'spglib_input': (np.array([[0., 2.73355606, 2.73355606],
       [2.73355606, 0., 2.73355606],
       [2.73355606, 2.73355606, 0.0]]), np.array([[0., 0., 0.],
       [0.25, 0.25, 0.25]]), np.array([14, 14]))}

    # print(TestCase().assertDictEqual(expected_poscar, actual_poscar))

    assert len(actual_poscar) == len(expected_poscar)

    for key in actual_poscar.keys():
        if key == 'spglib_input':
            for spg in np.arange(0, len(actual_poscar[key])):
                assert actual_poscar[key][spg].all() == expected_poscar[key][spg].all()
        elif key == 'spacegroup' or key == 'a' or key == 'na' or key == 'n_pol' or key == 'elements':
            assert actual_poscar[key] == expected_poscar[key]
        else:
            assert actual_poscar[key].all() == expected_poscar[key].all()


