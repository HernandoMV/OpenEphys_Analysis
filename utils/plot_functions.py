# Functions to support the analysis of ephys data
# Hernando Martinez Vergara
# December 2020

import numpy as np
from matplotlib.lines import Line2D


def plot_traces_on_probe(ax, intervalTimes, traces, idx, xcoords, ycoords,
                         normalized_slope,
                         x_scaler=1, y_scaler=1, points_to_plot=None,
                         timing_test=None, color='k', alpha=1, linewidth=1,
                         include_electrode_numbers=False, new_label='None',
                         grouping=None, spread=False):
    """
    TODO: this needs cleaning and add functionalities
    TODO: grouping

    # ax: axis on where to plot
    # intervalTimes: array of time points (M)
    # traces: signal, NxM array, where N is number of electrodes
    # idx: time indeces of the traces to plot
    # xcoords: x coordinates (N)
    # ycoords: y coordinates (N)
    # x_scaler: zoom of the traces
    # y_scaler: zoom of the traces
    # points_to_plot: list of points to plot in blue (each element of size N)
    # normalized_slope: uses the first point in points_to_plot (N)
    # timing_test: subselection to plot some features on only some traces (boolean array)
    # color: color of the trace
    # alpha: alpha of the trace
    # linewidth: width of the trace
    # include_electrode_numbers: boolean (overridden by grouping)
    # new_label: legend label
    # grouping: average of traces (typically tetrodes)
    """
    # spread the traces
    if spread:
        xcoords = array_spreader(xcoords)
        ycoords = array_spreader(ycoords)
    # plot a big circle in the back of each trace color-coded for the slope
    # Select only those that pass the test
    if timing_test is None:
        timing_test = np.ones(len(xcoords), dtype=bool)  # everything gets plotted

    if points_to_plot is not None:
        DerMinIdx = points_to_plot[0]
        x_s = [(xcoords[j] + intervalTimes[DerMinIdx[j] + idx[0][0]] * x_scaler) for j in range(len(DerMinIdx))]
        y_s = [(ycoords[j] + traces[j][DerMinIdx[j] + idx[0][0]] * y_scaler) for j in range(len(DerMinIdx))]

        ax.scatter(np.array(x_s)[timing_test],
                   np.array(y_s)[timing_test],
                   c=normalized_slope[timing_test],
                   cmap='viridis',
                   s=4000,
                   alpha=0.3)

    # loop through the traces
    for j in range(len(xcoords)):
        ax.plot(xcoords[j] + intervalTimes[idx] * x_scaler,
                ycoords[j] + traces[j][idx] * y_scaler,
                color=color,
                alpha=alpha,
                linewidth=linewidth)

        # plot the index of the electrodes, in colors depending on whether they pass the test above
        if include_electrode_numbers:
            if timing_test[j]:
                ax.text(xcoords[j] + intervalTimes[idx][0] * x_scaler,
                        ycoords[j] + traces[j][idx][0] * y_scaler,
                        j, color='green')
            else:
                ax.text(xcoords[j] + intervalTimes[idx][0] * x_scaler,
                        ycoords[j] + traces[j][idx][0] * y_scaler,
                        j, color='red')

        # plot points if asked to
        if points_to_plot is not None:
            for point in points_to_plot:
                ax.plot(xcoords[j] + intervalTimes[point[j] + idx[0][0]] * x_scaler,
                        ycoords[j] + traces[j][point[j] + idx[0][0]] * y_scaler,
                        marker='.',
                        markersize=10,
                        color='b')
    # set the custom legend
    new_handle = Line2D([0], [0], color=color, alpha=alpha, lw=linewidth)

    return ax, new_handle, new_label


def array_spreader(points):
    '''
    Takes a 1D array of points and spreads them evenly by category
    This is used for plotting purposes on electrode traces
    '''
    # categories
    un_points = np.unique(points)
    # number of categories
    n_cat = len(un_points)
    # new categories
    new_cat = np.linspace(min(un_points), max(un_points), n_cat)
    # new points
    spread_points = np.empty(len(points))
    spread_points[:] = np.NaN
    # loop through categories and substitute
    for un_idx, cat in enumerate(un_points):
        # which indeces are in that category
        el_idx = np.where(points == cat)[0]
        # populate with new points
        spread_points[el_idx] = new_cat[un_idx]
    return spread_points
