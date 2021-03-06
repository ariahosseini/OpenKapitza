"""Provide the visualizing functions."""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("white", {"xtick.major.size": 2, "ytick.major.size": 2})
sns.set_context("paper", font_scale=2, rc={"lines.linewidth": 4})


def plot_2darray(array_to_plot: np.ndarray, pic_name: str, set_title: str,
                 x_label: str = None, y_label: str = None) -> None:
    """
    A function to plot two-dimensional heat map

    Parameters
    ----------
    array_to_plot : np.ndarray
        Numpy two-dimensional arrays we seek to plot
    pic_name: str
        Name of the figure
    set_title : str
        Title of the plot
    x_label : str
        Heatmap x label
    y_label : str
        Heatmap y label

    Returns
    ----------
    None
    """

    dirName = 'Figs'
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        pass

    plt.figure(figsize=(6.5, 6.5))
    ax = sns.heatmap(array_to_plot, cbar=True)
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
    plt.savefig(os.path.join(dirName, pic_name), dpi=100)
    plt.close()


def plot_1darray(array_to_plot: np.ndarray, linestyle: str, marker: str, pic_name: str, set_title: str,
                 x_label: str = None, y_label: str = None) -> None:
    """
    A function to plot two-dimensional numpy arrays

    Parameters
    ----------
    array_to_plot : np.ndarray
        Numpy two-dimensional arrays we seek to plot
    linestyle: str
        Line style
    marker: str
        Marker shape
    set_title : str
        Title of the plot
    pic_name: str
        Name of the figure
    x_label : str
        x label
    y_label : str
        y label

    Returns
    ----------
    None
    """

    dirName = 'Figs'
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        pass

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = plt.axes()
    ax.set_axis_on()
    ax.grid(True)

    ax.plot(array_to_plot[0], array_to_plot[1], linestyle=linestyle, marker=marker, color='maroon',
            markersize=6, linewidth=1.5,
            markerfacecolor='white',
            markeredgecolor='maroon',
            markeredgewidth=1)

    ax.set_xlabel(x_label, fontsize=24)
    ax.set_ylabel(y_label, fontsize=24, labelpad=15)
    ax.tick_params(axis="y", labelsize=24)
    ax.tick_params(axis="x", labelsize=24)
    ax.set_title(set_title)
    fig.tight_layout()
    plt.ticklabel_format(axis="both", style="sci")
    plt.savefig(os.path.join(dirName, pic_name), dpi=100)
    plt.close()
