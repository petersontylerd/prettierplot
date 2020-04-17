import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import prettierplot.style as style
import prettierplot.util as util


def line(self, x, y, label=None, df=None, linecolor=style.style_grey, linestyle=None, bbox=(1.2, 0.9), x_units="f",
        x_ticks=None, y_units="f", y_ticks=None, marker_on=False, plot_buffer=False, axis_limits=False, ax=None):
    """
    Documentation:

        ---
        Description:
            Create single line plot.

        ---
        Parameters:
            x : list, array or string
                1-dimensional array of values to plot along x-axis
            y : list, array or string
                1-dimensional array of values to plot along y-axis
            label : str : default=None
                Legend label for line.
            df : Pandas DataFrame, default=None
                Pandas DataFrame containing data to plot. can be any size, as plotted columns will be chosen
                by columns names specified in x and y parameters.
            linecolor : str, default=style.style_grey
                Line color.
            linestyle : str, default=None
                Line style.
            bbox : tuple of floats, default=(1.2, 0.9)
                Coordinates for determining legend position.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 's' displays string. 'f' displays float.
                'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for
                additional decimal places.
            x_ticks : array, default=None
                Custom x-tick labels.
            y_units : str, default='f'
                Determines unit of measurement for y-axis tick labels. 's' displays string. 'f' displays float.
                'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for
                additional decimal places.
            y_ticks : array, default=None
                Custom y_tick labels.
            marker_on : bool, default=False
                Controls whether to show line with markers for each data element.
            plot_buffer : bool, default=False
                Controls whether dynamic plot buffer function is executed to ensure visual elements are
                not cut-off at the figure borders.
            axis_limits : bool, default=False
                Controls whether dynamic axis limit setting function is executed.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # if a Pandas DataFrame is passed to function, create x and y arrays using columns names passed into function
    if df is not None:
        if isinstance(df.index, pd.core.indexes.base.Index):
            x = df.index.values
        else:
            x = df[x].values

        y = df[y].values
    else:
        # convert input list to array
        x = np.array(x) if isinstance(x, list) else x
        y = np.array(y) if isinstance(y, list) else y

        # reshape arrays if necessar
        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y

    # add line to plot
    plt.plot(
        x,
        y * 100 if "p" in y_units else y,
        color=linecolor,
        linestyle=linestyle,
        linewidth=0.247 * self.chart_scale,
        label=label,
        marker="." if marker_on else None,
        markersize=17 if marker_on else None,
        markerfacecolor="w" if marker_on else None,
        markeredgewidth=2.2 if marker_on else None,
    )

    # add legend to figure
    if label is not None:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=bbox,
            ncol=1,
            frameon=True,
            fontsize=1.1 * self.chart_scale,
        )

    # optionally set axis lower / upper limits
    if axis_limits:
        x_min, x_max, y_min, y_max = util.util_set_axes(x=x, y=y)
        plt.axis([x_min, x_max, y_min, y_max])

    # optionally create smaller buffer around plot area to prevent cutting off elements
    if plot_buffer:
        util.util_plot_buffer(ax=ax, x=0.02, y=0.02)

    # optionally creates custom x-tick labels
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    # optionally creates custom y-tick labels
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=1.0 * self.chart_scale,
        color=style.style_grey,
    )
    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=1.0 * self.chart_scale,
        color=style.style_grey,
    )

    # axis tick label formatting
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)


def multi_line(self, x, y, label=None, df=None, linecolor=None, linestyle=None, bbox=(1.2, 0.9), x_units="f",
                x_ticks=None, y_units="f", y_ticks=None, marker_on=False, plot_buffer=False, axis_limits=False,
                color_map="viridis", ax=None):
    """
    Documentation:

        Description:
            Create single plot with multiple lines. Capable of adjusting which axis will have the same
            data for each line and which will have different data for each line.

        ---
        Parameters:
            x : array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            label : list of strings : default=None
                Custom legend label for each line.
            df : Pandas DataFrame, default=None
                Pandas DataFrame containing data to plot. Can be any size, as plotted columns will be chosen
                by columns names specified in x and y parameters.
            linecolor : str, default=None
                Line colors. If None, utilizes color_map
            linestyle : str, default=None
                Line style.
            bbox : tuple, default=(1.2, 0.9)
                Coordinates for determining legend position.
            x_units : str, default='d'
                Determines unit of measurement for x-axis tick labels. 's' displays string. 'f' displays float.
                'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for
                additional decimal places.
            x_ticks : array, default=None
                Custom x-tick labels.
            y_units : str, default='d'
                Determines unit of measurement for x-axis tick labels. 's' displays string. 'f' displays float.
                'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for
                additional decimal places.
            y_ticks : array, default=None
                Custom y-tick labels.
            marker_on : bool, default=False
                Controls whether to show line with markers for each data element.
             plot_buffer : bool, default=False
                Controls whether dynamic plot buffer function is executed to ensure visual elements are
                not cut-off at the figure borders.
            axis_limits : bool, default=False
                Controls whether dynamic axis limit setting function is executed.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # if a Pandas DataFrame is passed to function, create x and y arrays using columns names passed into function
    if df is not None:
        if isinstance(df.index, pd.core.indexes.base.Index):
            x = df.index.values
        else:
            x = df[x].values

        y = df[y].values
    else:
        # convert input list to array
        x = np.array(x) if isinstance(x, list) else x
        y = np.array(y) if isinstance(y, list) else y

        x = x.reshape(-1, 1) if len(x.shape) == 1 else x
        y = y.reshape(-1, 1) if len(y.shape) == 1 else y

    # generate color list
    color_list = style.color_gen(name=color_map, num=y.shape[1])

    # add multiple lines to plot
    for ix in np.arange(y.shape[1]):
        y_col = y[:, ix]
        plt.plot(
            x,
            y_col * 100 if "p" in y_units else y_col,
            color=linecolor if linecolor is not None else color_list[ix],
            linestyle=linestyle if linestyle is not None else style.style_line_style[0],
            linewidth=0.247 * self.chart_scale,
            label=label[ix] if label is not None else None,
            marker="." if marker_on else None,
            markersize=17 if marker_on else None,
            markerfacecolor="w" if marker_on else None,
            markeredgewidth=2.2 if marker_on else None,
        )

    # add legend to figure
    if label is not None:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=bbox,
            ncol=1,
            frameon=True,
            fontsize=1.1 * self.chart_scale,
        )

    # optionally set axis lower / upper limits
    if axis_limits:
        x_min, x_max, y_min, y_max = util.util_set_axes(x=x, y=y)
        plt.axis([x_min, x_max, y_min, y_max])

    # optionally create smaller buffer around plot area to prevent cutting off elements
    if plot_buffer:
        util.util_plot_buffer(ax=ax, x=0.02, y=0.02)

    # optionally creates custom x-tick labels
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    # optionally creates custom y-tick labels
    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=1.1 * self.chart_scale,
        color=style.style_grey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=1.1 * self.chart_scale,
        color=style.style_grey,
    )

    # axis tick label formatting
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)
