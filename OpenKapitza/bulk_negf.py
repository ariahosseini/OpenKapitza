"""compute Green's function in bulk materials."""

import numpy as np
import numpy.linalg as LA


def reciprocal(vec):

    recip_vec = np.zeros([3, 3])
    vol = np.dot(vec[0], (np.cross(vec[1], vec[2])))

    recip_vec[0] = np.cross(vec[1], vec[2])
    recip_vec[1] = np.cross(vec[2], vec[0])
    recip_vec[2] = np.cross(vec[0], vec[1])

    recip_vec *= 2 * np.pi / vol

    return recip_vec


def sposcar(vasp_poscar='POSCAR', phonopy_sposcar='SPOSCAR'):

    # Read 'POSCAR' and 'SPOSCAR' files
    poscar_file = open(vasp_poscar, "r")
    sposcar_file = open(phonopy_sposcar, "r")
    _pos = poscar_file.readlines()
    _spos = sposcar_file.readlines()

    # Read scaling_factor
    pos_scaling_factor = float(_pos[1].strip().split()[0])
    spos_scaling_factor = float(_spos[1].strip().split()[0])

    # Define unit vectors in POSCAR
    _unit_vec = [_pos[_].strip().split() for _ in range(2,5)]
    unit_vec = pos_scaling_factor * np.array([[float(_) for _ in __] for __ in _unit_vec])
    recip_univec = reciprocal(unit_vec)  # Define unit vectors in reciprocal space

    # Define lattice vectors in SPOSCAR
    _lat_vec = [_spos[_].strip().split() for _ in range(2,5)]
    lat_vec = spos_scaling_factor * np.array([[float(_) for _ in __] for __ in _lat_vec])
    conversion_factor = LA.inv(LA.inv(lat_vec).T)

    # Atoms' types and number of each atom in the unit cell
    atoms_types = _spos[5].strip().split()
    num_each_elm_sc = [int(_) for _ in _spos[6].strip().split()]
    atoms_identity = list(zip(atoms_types, num_each_elm_sc))

    num_atoms_uc = sum(map(int, _pos[6].strip().split()))  # Number of atoms in the unitcell
    num_atoms_sup_cell = sum(map(int, _spos[6].strip().split()))  # Number of atoms in the supercell
    spos_direct = []

    for i in range(num_atoms_sup_cell):
        spos_direct.append([float(_) for _ in _spos[8+i].strip().split()])

    poscar_file.close()
    sposcar_file.close()

    def convert(pos):
        return np.dot(conversion_factor, pos)

    spos = np.array(list(map(convert, spos_direct)))

    return spos, unit_vec, recip_univec, lat_vec, num_atoms_uc, num_atoms_sup_cell, atoms_identity


def unitcell(spos, unit_vec):

    def _unitcell(pos):

        margin = 1e-7

        return 0 <= pos[0] < unit_len[0] - margin and \
               0 <= pos[1] < unit_len[1] - margin and \
               0 <= pos[2] < unit_len[2] - margin

    unit_len = np.sum(unit_vec, axis=0)
    atom_uc = list(map(_unitcell, spos))
    atom_uc_index = [idx+1 for idx, val in enumerate(atom_uc) if val]

    return atom_uc_index


def make_shift_list(rep):  # Needs update

    shift_max = [1, 1, 1]

    for i in range(3):
        if rep[i] > 3:
            shift_max[i] = int(rep[i] * 0.5)

    global xrng_u, yrng_u, zrng_u
    global lat_vec_shf, uni_vec_shf

    def make_rng(m):
        rng_l = [_ for _ in range(-m, 1)]
        rng_u = [_ for _ in range(-m, m+1)]
        return rng_l, rng_u

    xrng_l, xrng_u = make_rng(shift_max[0])
    yrng_l, yrng_u = make_rng(shift_max[1])
    zrng_l, zrng_u = make_rng(shift_max[2])

    lat_vec_shf = [[i, j, k] for i in xrng_l for j in yrng_l for k in zrng_l]
    uni_vec_shf = [[i, j, k] for i in xrng_u for j in yrng_u for k in zrng_u]

    pass


def bool_uc(pos, vec_len):

    margin = 1e-7

    return 0 <= pos[0] < vec_len[0]-margin and 0 <= pos[1] < vec_len[1]-margin and 0 <= pos[2] < vec_len[2]-margin


def shift_func(shift, vec):

    return shift[0] * vec[0] + shift[1] * vec[1] + shift[2] * vec[2]


def distance(spos, lat_vec, p, q, num_shift):

    dist = []

    for i in range(num_shift):

        shift = shift_func(lat_vec_shf[i], lat_vec)
        _dist = LA.norm(spos[p-1] - (spos[q-1] + shift))
        dist.append(_dist)

    return dist


def symmetry_func(lat_vec, unit_vec, spos, p, q, cutoff):

    symmetry = []
    unit_len = np.sum(unit_vec, axis=0)

    num_shift = len(lat_vec_shf)
    dist = distance(spos, lat_vec, p, q, num_shift)

    if min(dist) <= cutoff:
        index = [i for i, x in enumerate(
            dist) if abs(x - min(dist)) < 1e-5]
        num_shift = len(uni_vec_shf)
        for i in index:
            x_la = spos[q-1] + shift_func(lat_vec_shf[i], lat_vec)
            for j in range(num_shift):
                x_uni = x_la - shift_func(uni_vec_shf[j], lat_vec)

                if bool_uc(x_uni, unit_len):
                    symmetry.append(j)
                    break

    return symmetry


def generate_pairs(atom_uc, spos, lat_vec, unit_vec, num_atoms_sup_cell, cutoff):

    pairs = {}

    for p in atom_uc:
        pairs[p] = []
        for q in range(p, num_atoms_sup_cell + 1):
            sym = symmetry_func(lat_vec, unit_vec, spos, p, q, cutoff)
            pairs[p].append([q, sym])

    return pairs


def mass_uc(mass, atoms_identity):

    if len(mass) != len(atoms_identity):
        raise ValueError("make sure size of mass and atoms_identity are the same")

    identity = []
    for idx, elm in enumerate(atoms_identity):
        elm += (mass[idx],)
        identity.append(elm)

    return identity


def map_func(spos, unit_vec, atom_uc, num_atoms_sup_cell, rep):
    map_uc = []
    unit_len = np.sum(unit_vec, axis=0)
    for x in range(num_atoms_sup_cell):
        break_frag = False
        for i in range(rep[0]):
            for j in range(rep[1]):
                for k in range(rep[2]):
                    x_uni = spos[x] - shift_func([i, j, k], unit_vec)
                    if bool_uc(x_uni, unit_len):
                        break_frag = True
                        break  # break for loop k

                if break_frag:
                    break  # break for loop j

            if break_frag:
                break  # break for loop i

        for p in atom_uc:
            r = LA.norm(x_uni - spos[p-1])
            if r < 1e-3:
                map_uc.append(p)
                break

        if len(map_uc) != x + 1:
            print("mapping atom error.")
            print("Please check NAT, &cell, and &unit_cell fields in negf file.")
            exit(1)

    return map_uc


def generate_qmesh(num_kpoints, revec):

    def get_qpoint(qmesh):
        return qmesh[0] * revec[0] + qmesh[1] * revec[1] + qmesh[2] * revec[2]

    kpoints = []
    for i in range(len(num_kpoints)):
        dq = 1 / (num_kpoints[i] + 1)
        kpoints += [np.linspace(-0.5 + dq, 0.5, num_kpoints[i], endpoint=False),]

    k_grid = np.meshgrid(kpoints[0], kpoints[1])
    points = np.array([k_grid[1].flatten(), k_grid[0].flatten()]).T
    print(np.shape(points))
    qpoints = list(map(get_qpoint, kpoints))
    print(np.shape(qpoints))

    return qpoints



# def force_constant(forces="FORCE_CONSTANTS", rep=5):
#
#     forces_file = open(forces, "r")
#     forces = forces_file.readlines()
#     number_atoms = int(forces.pop(0))
#     del forces[::4]
#     number_atoms_unitcell = int(number_atoms / rep**3)
#     ifc = []
#     for elm in forces:
#         _ifc = elm.split()
#         ifc.append(np.asarray(_ifc, dtype=float))
#     ifc = np.array(ifc)
#
#     hsn_matrix = ifc[:3 * number_atoms, :]
#
#     for idx in np.arange(1, number_atoms):
#         _hessian = ifc[3 * idx * number_atoms:(idx + 1) * 3 * number_atoms, :]
#         hsn_matrix = np.hstack((hsn_matrix, _hessian))
#
#     return hsn_matrix, number_atoms, number_atoms_unitcell
#
#
# def phonopy_matrix_decomposition(hsn_matrix: np.ndarray, num_atoms_uc, rep: int = 5):
#
#     """
#     A method to decompose the Hessian matrix in to sub-matrices (onsite and hopping matrices)
#
#     Parameters
#     ----------
#     hsn_matrix: np.ndarray
#         Hessian matrix is the "read_hessian" method return
#     rep: list
#         This term shows how many times the unit cell is replicated in each lead
#     num_atoms_uc: int
#         Number of atoms per unit cell
#
#     Returns
#     ----------
#         H0, H1, H2, H3, H4, T1, T2, T3, T4 showing onsite and hopping matrices
#     """
#
#     H0 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2)]
#
#     H1 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-1+rep): 3*num_atoms_uc*(int(rep**3/2)+rep)]
#
#     H2 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-1-rep): 3*num_atoms_uc*(int(rep**3/2)-rep)]
#
#     H3 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-1+rep**2): 3*num_atoms_uc*(int(rep**3/2)+rep**2)]
#
#     H4 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-1-rep**2): 3*num_atoms_uc*(int(rep**3/2)-rep**2)]
#
#     T1 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)+rep): 3*num_atoms_uc*(int(rep**3/2)+1+rep)]
#
#     T2 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-rep): 3*num_atoms_uc*(int(rep**3/2)+1-rep)]
#
#     T3 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)+rep**2): 3*num_atoms_uc*(int(rep**3/2)+1+rep**2)]
#
#     T4 = hsn_matrix[3*num_atoms_uc*(int(rep**3/2)-1): 3*num_atoms_uc*int(rep**3/2),
#                     3*num_atoms_uc*(int(rep**3/2)-rep**2): 3*num_atoms_uc*(int(rep**3/2)+1-rep**2)]
#
#     Hsn = {'H0': H0, 'H1': H1, 'H2': H2, 'H3': H3, 'H4': H4, 'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4}
#
#     return Hsn
#
#
# def wave_vectors(wave_len: float, num_kpoints: int) -> dict:
#
#     """
#     A function to define wave vectors in the surface perpendicular to the transport direction
#
#     Parameters
#     ----------
#     wave_len: float
#         The periodicity length along the transverse direction
#     num_kpoints : int
#         Number of kpoints
#
#     Returns
#     ----------
#     dic_output : dict
#         First key includes the kpoints, and the second one includes the periodicity length
#     """
#
#     kpoints_i = np.linspace(-np.sqrt(2) * np.pi / wave_len, np.sqrt(2) * np.pi / wave_len,
#                             num_kpoints,
#                             endpoint=True)
#     kpoints_j = np.linspace(-np.sqrt(2) * np.pi / wave_len, np.sqrt(2) * np.pi / wave_len,
#                             num_kpoints,
#                             endpoint=True)
#
#     ki_grid, kj_grid = np.meshgrid(kpoints_i, kpoints_j)
#     kpoints = np.array([kj_grid.flatten(), ki_grid.flatten()])
#
#     dict_output = dict(kpoints=kpoints, wave_len=wave_len)
#
#     return dict_output
