"""Provide the primary functions."""
import numpy as np


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


def plot_2Darray(array_to_plot) -> None:
    """
    A function to plot two dimennsionnal arrays

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


def canvas(with_attribution=True):
    """
    Placeholder function to show example docstring (NumPy format).

    Replace this function and doc string for your own project.

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from.

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution.
    """

    quote = "The code is but a canvas to our imagination."
    if with_attribution:
        quote += "\n\t- Adapted from Henry David Thoreau"
    return quote


if __name__ == "__main__":
    # Do something if this file is invoked on its own
    print(canvas())
