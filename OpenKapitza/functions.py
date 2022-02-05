"""Provide the primary functions."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functools

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

    kpoints_y = np.linspace(-np.sqrt(2)*np.pi/periodicity_lenght, np.sqrt(2)*np.pi/periodicity_lenght,num_kpoints,
                            endpoint=True)
    kpoints_x = np.linspace(-np.sqrt(2)*np.pi/periodicity_lenght, np.sqrt(2)*np.pi/periodicity_lenght,num_kpoints,
                            endpoint=True)

    kx_grid, ky_grid = np.meshgrid(kpoints_x, kpoints_y)

    kpoints = np.array([kx_grid.flatten(), ky_grid.flatten()])

    return kpoints


def hessian_fourier_form(Hsn: dict, periodicity_lenght: float, wavevector: np.ndarray) -> np.ndarray:
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



    distance_vector = periodicity_lenght*np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    unit_planewave = np.exp(-1j*(np.matmul(distance_vector, wavevector)).T)
    print(Hsn.keys())
    print(np.shape(unit_planewave))
    Hsn_keys = np.arange(np.shape(wavevector)[1])
    Hsn_fourier = {}
    Hopping_fourier = {}

    for i in range(len(Hsn_keys)):

        Hsn_fourier[Hsn_keys[i]] = Hsn['H0']*unit_planewave[i, 0] + Hsn['H1']*unit_planewave[i, 1]
        + Hsn['H2']*unit_planewave[i, 2] + Hsn['H3']*unit_planewave[i, 3] + Hsn['H4']*unit_planewave[i, 4]

        Hopping_fourier[Hsn_keys[i]] = Hsn['T1']*unit_planewave[i, 1] + Hsn['T2']*unit_planewave[i, 2] + \
                                       Hsn['T3']*unit_planewave[i, 3] + Hsn['T4']*unit_planewave[i, 4]
    Hsn_matrix_fourier = {'Hsn_fourier': Hsn_fourier, 'Hopping_fourier': Hopping_fourier}

    return Hsn_matrix_fourier


def hessian_fourier_form_two(Hsn: dict, periodicity_lenght: float, wavevector: np.ndarray) -> np.ndarray:

    distance_vector = periodicity_lenght*np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])
    unit_planewave = np.exp(-1j*(np.matmul(distance_vector, wavevector)).T)
    Hsn_fourier = {}
    Hopping_fourier = {}

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

    return Hsn_matrix_fourier


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print('Done')
