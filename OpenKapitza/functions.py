"""Provide the primary functions."""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_2Darray(array_to_plot: np.ndarray, set_title: str, x_label: str = None, y_label: str = None) -> None:
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
    ax = sns.heatmap(array_to_plot, linewidth=0.1, cbar=False)
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
    plt.show()

def read_crystal(file_name: str, skiprows: int = 9) -> np.ndarray:
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

    crystal_points = np.loadtxt(file_name, delimiter=None, skiprows=skip_rows)

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
