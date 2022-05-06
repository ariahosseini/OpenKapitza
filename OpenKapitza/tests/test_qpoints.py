import OpenKapitza as OK
import numpy as np
from pytest import approx


def test_get_q_grid():

    base = OK.init()
    poscar = OK.load_poscar(base, path='./tests/')
    actual_qpoints = OK.get_q_grid(poscar, [2, 2, 1])
    expected_qpoints = np.array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                 [-5.74634759e+09, 5.74634759e+09, 5.74634759e+09],
                                 [5.74634759e+09, -5.74634759e+09, 5.74634759e+09],
                                 [0.00000000e+00, 0.00000000e+00, 1.14926952e+10]])

    assert approx(actual_qpoints, abs=1e2) == expected_qpoints


