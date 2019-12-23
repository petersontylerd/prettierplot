import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib import cm


def util_plot_buffer(ax, x, y):
    """
    documentation:
        description:
            creates narrow border around plot arrow.
            prevents plot icons from being clipped by plot edges.
    """
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    x_margin = (x_lim[1] - x_lim[0]) * x
    y_margin = (y_lim[1] - y_lim[0]) * y

    ax.set_xlim(x_lim[0] - x_margin, x_lim[1] + x_margin)
    ax.set_ylim(y_lim[0] - y_margin, y_lim[1] + y_margin)


def util_label_formatter(
    ax,
    x_units=None,
    y_units=None,
    x_size=None,
    y_size=None,
    x_rotate=None,
    y_rotate=None,
):
    """
    documentation:
        description:
            formats tick labels as dolloars, percentages, or decimals.
        parameters:
            ax : axes object, default=None
                axis on which to place visual..
            x_units : str, default=None
                determines units of x_axis tick labels. none displays float. 'p' displays percentages,
                '$' displays dollars.
            x_size : int or float, default=None
                x_axis label size.
            y_units : str, default=None
                determines units of y_axis tick labels. none displays float. 'p' displays percentages,
                '$' displays dollars.
            y_size : int or float, default=None
                y_axis label size.
    """
    ## x_axis
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

    if x_units is not None and x_units != "s":
        tick = tkr.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)

    if x_units is not None and x_rotate is not None:
        ax.tick_params(labelrotation=45, axis="x")

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

    if y_units is not None and y_units != "s":
        tick = tkr.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)

    if y_units is not None and y_rotate is not None:
        ax.tick_params(labelrotation=45, axis="y")

    if y_size is not None:
        for tk in ax.get_yticklabels():
            tk.set_fontsize(y_size)


def util_set_axes(x, y, x_thresh=0.75, y_thresh=0.75):
    """
    documentation:
        description:
            dynamically set lower/upper limits of x/y axes.
    """
    x_min = round(np.min(x), 5)
    x_max = round(np.max(x), 5)
    x_change = (x_max - x_min) / x_max
    x_min = 0 if 1.00 >= x_change >= x_thresh else np.round(x_min, 1)
    x_max = x_max + x_max * 0.1

    y_min = round(np.min(y), 5)
    y_max = round(np.max(y), 5)
    y_change = (y_max - y_min) / y_max
    y_min = 0 if 1.00 >= y_change >= y_thresh else np.round(y_min, 1)
    y_max = y_max + y_max * 0.1
    return x_min, x_max, y_min, y_max


def number_coerce(df, columns=None):
    """
    documentation:
        description:
            convert object columns that include number data to float or int
            data type.
        paramters:
            df : pandas DataFrame
                input dataset
            columns : list of strings
                list of column names to convert.
        returns:
            pandas DataFrame with converted columns.
    """
    # if no columns specified, set columns equal to all non object columns
    if columns is None:
        columns = df.columns

    for col in columns:
        # exclude columsn that contain only nulls
        if not df[col].isnull().all():
            try:
                df[col] = df[col].apply(pd.to_number)
            except ValueError:
                pass
        else:
            pass
    return df
