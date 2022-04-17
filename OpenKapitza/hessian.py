import pathlib
import numpy as np
import gzip, os, pickle
import spglib


def init():

    hbar = 1.054571817e-34  # Plank constant [J.s]
    kb = 1.380649e-23  # Boltzmann constant [J/K]
    kb_ev = 8.617333262145e-5  # Boltzmann constant [eV/K]
    Kg = 6.0221408585491615e+26  # Kg to AMU conversion factor
    q = 1.602176634e-19  # Electric unit charge [C]
    J = 6.241509125883258e+18  # J to eV conversion factor
    L = 1e10  # m to A conversion factor
    hbar_ev = 6.582119569e-16  # Plank constant [eV.s]
    t = L*np.sqrt(Kg/J)  # s to au conversion factor

    pt = load(str(pathlib.Path(__file__).parent.absolute()) + '/periodic_table')

    return {'pt': pt, 'hbar': hbar, 'kb_ev': kb_ev, 'kb': kb, 'Kg': Kg, 'J': J,
            'q':q, 'hbar_ev': hbar_ev, 'L': L, 't': t}


def dynamical_matrix_q(poscar, forces, q):

    na = len(poscar['numbers'])
    D = np.zeros((na*3, na*3), dtype=np.complex128)
    for l in range(forces['nsc']):
        for a1 in range(na):
            for a2 in range(na):
                phase = np.exp(1j*np.dot(q, forces['distances'][a1,a2,l]))
                d = forces['forces'][0, l, a1*3:(a1*3+1)*3,a2*3:(a2*3+1)*3]*phase
                D[a1*3:(a1+1)*3, a2*3:(a2+1)*3] += d
    return D


def get_indices(displ, n):

    [n1, n2, n3] = displ

    s1 = n % n1
    s2 = (n//n1) % n2
    s3 = ((n//n1)//n2) % n3
    sa = ((n//n1)//n2)//n3
    l = s1 + n1 * s2 + n1 * n2 * s3

    index = s1 + n1*s2 + n1 * n2 * s3 + n1 * n2 * n3 * sa
    assert(index == n)

    return s1, s2, s3, l, sa


def load(name='state'):
    if os.path.isfile(name+ '.npz'):
        with gzip.open(name + '.npz', 'rb') as f:
            return pickle.load(f)

    return -1


def load_poscar(data, path='.'):

    """Load POSCAR"""

    f = open(path + '/POSCAR', 'r')
    f.readline()
    a = float(f.readline())
    cell = np.zeros((3, 3))
    for i in range(3):
        tmp = f.readline().split()
        for j in range(3):
            cell[i, j] = a * float(tmp[j])

    numbers = []
    el2 = f.readline().split()
    na = f.readline().split()
    el = []
    for (i, j) in zip(el2, na):
        for n in range(int(j)):
            numbers.append(data['pt'][i]['number'])
            el.append(i)
    numbers = np.array(numbers)
    na = len(numbers)

    base = np.zeros((na, 3))
    f.readline()
    for i in range(na):
        tmp = f.readline().split()
        for j in range(3):
            base[i, j] = float(tmp[j])
    f.close()

    spglib_input = (cell.copy(), base.copy(), numbers)
    spacegroup = spglib.get_spacegroup(spglib_input)

    cell /= data['L']  # Lattice constant in SI
    base = np.dot(base, cell)  # This is in regular coordinates
    a = np.linalg.norm(cell[0])

    return {'cell': cell, 'numbers': numbers, 'base': base / data['L'], 'spacegroup': spacegroup, 'a': a,
            'elements': el, 'na': len(numbers), 'n_pol': len(numbers) * 3, 'spglib_input': spglib_input}


def get_q_grid(poscar, mesh):

    reciprocal_grid = 2 * np.pi * np.linalg.inv(poscar['cell'].T)
    # Follow this tutorial: https://spglib.github.io/spglib/python-spglib.html#id27
    mapping, grid = spglib.get_ir_reciprocal_mesh(mesh, poscar['spglib_input'], is_shift=[0, 0, 0])
    grid = grid.astype(float) / mesh

    return np.dot(grid, reciprocal_grid)


def distance_between_two_atoms(poscar, forces, a1, a2, l1, l2):

    [n1, n2, n3] = forces['supercell']

    # atom 1
    s = np.zeros(3)
    s[0] = l1 % n1
    s[1] = (l1 // n1) % n2
    s[2] = ((l1 // n1) // n2) % n3
    p1 = poscar['base'][a1] + np.dot(s, poscar['cell'])

    # atom 2
    s = np.zeros(3)
    s[0] = l2 % n1
    s[1] = (l2 // n1) % n2
    s[2] = ((l2 // n1) // n2) % n3
    p2 = poscar['base'][a2] + np.dot(s, poscar['cell'])

    dmin = 1e9
    pp = [[0, 0], [-1, 0], [-1, -1], [1, 0], [1, 1], [-1, 1], [1, -1], [0, 1], [0, -1]]
    for p in pp:
        tr = p[0] * forces['periodicity'][0] + p[1] * forces['periodicity'][1]
        d = (p2 + tr) - p1
        if np.linalg.norm(d) < dmin:
            dmin = np.linalg.norm(d)
            dfinal = d

    return dfinal


def compute_distances(poscar,forces):

    nsc = forces['nsc']
    na = poscar['na']
    distances = np.zeros((na,na,nsc,3))
    for a1 in range(na):
        for a2 in range(na):
            for l in range(nsc):
                distances[a1, a2, l] = distance_between_two_atoms(poscar, forces, a1, a2, 0, l)

    return distances


def load_2nd_FC(base, poscar, displ, path='.'):

    # Load FCs

    [n1, n2, n3] = displ
    nsc = n1 * n2 * n3
    periodicity = np.zeros((3, 3))
    periodicity[0] = poscar['cell'][0] * n1
    periodicity[1] = poscar['cell'][1] * n2
    periodicity[2] = poscar['cell'][2] * n3

    na = poscar['na']
    f = open(path + '/FORCE_CONSTANTS', 'r')
    nas = int(f.readline())
    force_2nd = np.zeros((nas * 3, nas * 3))
    for i in range(nas):
        for j in range(nas):
            tmp = f.readline().split()
            n1 = int(tmp[0]) - 1
            n2 = int(tmp[1]) - 1
            for l in range(3):
                tmp = f.readline().split()
                for k in range(3):
                    force_2nd[3 * i + l, 3 * j + k] = float(tmp[k])
    f.close()

    force_2nd_n = np.zeros((nsc, nsc, na * 3, na * 3))
    for n1 in range(nas):
        for n2 in range(nas):
            (sa1, sb1, sc1, l1, a1) = get_indices(displ, n1)
            (sa2, sb2, sc2, l2, a2) = get_indices(displ, n2)
            force_2nd_n[l1, l2, a1 * 3:(a1 + 1) * 3, a2 * 3:(a2 + 1) * 3] = force_2nd[n1 * 3:(n1 + 1) * 3,
                                                                            n2 * 3:(n2 + 1) * 3] / np.sqrt(
                base['pt'][poscar['elements'][a1]]['mass'] * base['pt'][poscar['elements'][a2]]['mass'])

    forces = {'supercell': [n1, n2, n3], 'nsc': nsc, 'periodicity': periodicity,
              'forces': force_2nd_n * np.power(base['L'], 2) / base['J'] * base['Kg']}

    distances = compute_distances(poscar, forces)

    forces.update({'distances': distances})

    return forces













