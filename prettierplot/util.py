import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib import cm


def util_plot_buffer(ax, x, y):
    """
    Documentation:

        ---
        Description:
            Creates narrow border around plot area. Prevents plot elements
            from being cut-off by plot edges.
    """
    # identify current x-axis and y-axis limits
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    # calculate margins by subtracing min from max and multiply by x/y
    x_margin = (x_lim[1] - x_lim[0]) * x
    y_margin = (y_lim[1] - y_lim[0]) * y

    # reset x-axis and y-axis limits
    ax.set_xlim(x_lim[0] - x_margin, x_lim[1] + x_margin)
    ax.set_ylim(y_lim[0] - y_margin, y_lim[1] + y_margin)


def util_label_formatter(ax, x_units=None, y_units=None, x_size=None, y_size=None, x_rotate=None,
                            y_rotate=None):
    """
    Documentation:

        ---
        Description:
            Formats tick labels as dollars, percentages, or decimals. Applies varying levels
            of precision to labels

        ---
        Parameters:
            ax : axes object, default=None
                Axis object for the visualization.
            x_units : str, default=None
                Determines unit of measurement for x-axis tick labels. None displays float.
                'p' displays percentages, '$' displays dollars.
            x_size : int or float, default=None
                x-axis label size.
            y_units : str, default=None
                Determines unit of measurement for y-axis tick labels. None displays float.
                'p' displays percentages, '$' displays dollars.
            y_size : int or float, default=None
                y-axis label size.
    """
    ## x-axis
    # format as dollars
    if x_units == "d":
        fmt = "${x:,.0f}"
    elif x_units == "dd":
        fmt = "${x:,.1f}"
    elif x_units == "ddd":
        fmt = "${x:,.2f}"
    elif x_units == "dddd":
        fmt = "${x:,.3f}"
    elif x_units == "ddddd":
        fmt = "${x:,.4f}"

    # format as percent
    elif x_units == "p":
        fmt = "{x:,.0f}%"
    elif x_units == "pp":
        fmt = "{x:,.1f}%"
    elif x_units == "ppp":
        fmt = "{x:,.2f}%"
    elif x_units == "pppp":
        fmt = "{x:,.3f}%"
    elif x_units == "ppppp":
        fmt = "{x:,.4f}%"

    # format as float
    elif x_units == "f":
        fmt = "{x:,.0f}"
    elif x_units == "ff":
        fmt = "{x:,.1f}"
    elif x_units == "fff":
        fmt = "{x:,.2f}"
    elif x_units == "ffff":
        fmt = "{x:,.3f}"
    elif x_units == "fffff":
        fmt = "{x:,.4f}"

    # apply tick label formatting to x-tick labels
    if x_units is not None and x_units != "s":
        tick = tkr.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)

    # apply x-tick rotation
    if x_rotate is not None:
        ax.tick_params(labelrotation=x_rotate, axis="x")

    # resize x-tick label
    if x_size is not None:
        for tk in ax.get_xticklabels():
            tk.set_fontsize(x_size)

    ## y_axis
    # format as dollars
    if y_units == "d":
        fmt = "${x:,.0f}"
    elif y_units == "dd":
        fmt = "${x:,.1f}"
    elif y_units == "ddd":
        fmt = "${x:,.2f}"
    elif y_units == "dddd":
        fmt = "${x:,.3f}"
    elif y_units == "ddddd":
        fmt = "${x:,.4f}"

    # format as percent
    elif y_units == "p":
        fmt = "{x:,.0f}%"
    elif y_units == "pp":
        fmt = "{x:,.1f}%"
    elif y_units == "ppp":
        fmt = "{x:,.2f}%"
    elif y_units == "pppp":
        fmt = "{x:,.3f}%"
    elif y_units == "ppppp":
        fmt = "{x:,.4f}%"

    # format as float
    elif y_units == "f":
        fmt = "{x:,.0f}"
    elif y_units == "ff":
        fmt = "{x:,.1f}"
    elif y_units == "fff":
        fmt = "{x:,.2f}"
    elif y_units == "ffff":
        fmt = "{x:,.3f}"
    elif y_units == "ffff":
        fmt = "{x:,.4f}"

    # apply tick label formatting to y-tick labels
    if y_units is not None and y_units != "s":
        tick = tkr.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)

    # apply y-tick rotation
    if y_rotate is not None:
        ax.tick_params(labelrotation=y_rotate, axis="y")

    # resize y-tick label
    if y_size is not None:
        for tk in ax.get_yticklabels():
            tk.set_fontsize(y_size)


def util_set_axes(x, y, x_thresh=0.75, y_thresh=0.75):
    """
    Documentation:

        ---
        Description:
            Dynamically set lower/upper limits of x and y axes.

        ---
        Parameters:
            x : list or array
                1-dimensional array of values
            y : list or array
                1-dimensional array of values
            x_thresh : float
                Controls x-axis adjustment amount
            y_thresh : float
                Controls y-axis adjustment amount

    """
    x_min = round(np.nanmin(x), 5)
    x_max = round(np.nanmax(x), 5)
    x_change = (x_max - x_min) / x_max
    x_min = 0 if 1.00 >= x_change >= x_thresh else np.round(x_min, 1)
    x_max = x_max + x_max * 0.01

    y_min = round(np.nanmin(y), 5)
    y_max = round(np.nanmax(y), 5)
    y_change = (y_max - y_min) / y_max
    y_min = 0 if 1.00 >= y_change >= y_thresh else np.round(y_min, 1)
    y_max = y_max + y_max * 0.01
    return x_min, x_max, y_min, y_max


def number_coerce(df, columns=None):
    """
    Documentation:

        ---
        Description:
            Convert categorical columns that include only numeric data to
            float or int data type.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing columns to convert
            columns : list of strings
                List of column names to convert.
        Returns:
            Pandas DataFrame with converted columns.
    """
    # if no subset of columns is provided, use all columns in df
    if columns is None:
        columns = df.columns

    # iterate through all columns
    for col in columns:

        # exclude columns that contain only nulls
        if not df[col].isnull().all():
            try:
                df[col] = df[col].apply(pd.to_numeric)
            except ValueError:
                pass
        else:
            pass
    return df
