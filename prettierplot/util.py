import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr


def utilPlotBuffer(ax, x, y):
    """
    Documentation:
        Description:
            Creates narrow border around plot arrow.
            Prevents plot icons from being clipped by plot edges.
    """
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()

    xMargin = (xLim[1] - xLim[0]) * x
    yMargin = (yLim[1] - yLim[0]) * y

    ax.set_xlim(xLim[0] - xMargin, xLim[1] + xMargin)
    ax.set_ylim(yLim[0] - yMargin, yLim[1] + yMargin)


def utilLabelFormatter(ax, xUnits=None, yUnits=None, xSize=None, ySize=None, xRotate=None, yRotate=None):
    """
    Documentation:
        Description:
            Formats tick labels as dolloars, percentages, or decimals.
        Parameters:
            ax : Axes object, default = None
                Axis on which to place visual..
            xUnits : str, default = None
                Determines units of x-axis tick labels. None displays float. 'p' displays percentages,
                '$' displays dollars.
            xSize : int or float, default = None
                x-axis label size.
            yUnits : str, default = None
                Determines units of y-axis tick labels. None displays float. 'p' displays percentages,
                '$' displays dollars.
            ySize : int or float, default = None
                y-axis label size.
    """
    ## x-axis
    # format as dollars
    if xUnits == "d":
        fmt = "${x:,.0f}"
    elif xUnits == "dd":
        fmt = "${x:,.1f}"
    elif xUnits == "ddd":
        fmt = "${x:,.2f}"
    elif xUnits == "dddd":
        fmt = "${x:,.3f}"
    elif xUnits == "ddddd":
        fmt = "${x:,.4f}"

    # format as percent
    elif xUnits == "p":
        fmt = "{x:,.0f}%"
    elif xUnits == "pp":
        fmt = "{x:,.1f}%"
    elif xUnits == "ppp":
        fmt = "{x:,.2f}%"
    elif xUnits == "pppp":
        fmt = "{x:,.3f}%"
    elif xUnits == "ppppp":
        fmt = "{x:,.4f}%"

    # format as float
    elif xUnits == "f":
        fmt = "{x:,.0f}"
    elif xUnits == "ff":
        fmt = "{x:,.1f}"
    elif xUnits == "fff":
        fmt = "{x:,.2f}"
    elif xUnits == "ffff":
        fmt = "{x:,.3f}"
    elif xUnits == "fffff":
        fmt = "{x:,.4f}"

    if xUnits is not None and xUnits != "s":
        tick = tkr.StrMethodFormatter(fmt)
        ax.xaxis.set_major_formatter(tick)

    if xUnits is not None and xRotate is not None:
        ax.tick_params(labelrotation=45, axis="x")

    if xSize is not None:
        for tk in ax.get_xticklabels():
            tk.set_fontsize(xSize)

    ## y-axis
    # format as dollars
    if yUnits == "d":
        fmt = "${x:,.0f}"
    elif yUnits == "dd":
        fmt = "${x:,.1f}"
    elif yUnits == "ddd":
        fmt = "${x:,.2f}"
    elif yUnits == "dddd":
        fmt = "${x:,.3f}"
    elif yUnits == "ddddd":
        fmt = "${x:,.4f}"

    # format as percent
    elif yUnits == "p":
        fmt = "{x:,.0f}%"
    elif yUnits == "pp":
        fmt = "{x:,.1f}%"
    elif yUnits == "ppp":
        fmt = "{x:,.2f}%"
    elif yUnits == "pppp":
        fmt = "{x:,.3f}%"
    elif yUnits == "ppppp":
        fmt = "{x:,.4f}%"

    # format as float
    elif yUnits == "f":
        fmt = "{x:,.0f}"
    elif yUnits == "ff":
        fmt = "{x:,.1f}"
    elif yUnits == "fff":
        fmt = "{x:,.2f}"
    elif yUnits == "ffff":
        fmt = "{x:,.3f}"
    elif yUnits == "ffff":
        fmt = "{x:,.4f}"

    if yUnits is not None and yUnits != "s":
        tick = tkr.StrMethodFormatter(fmt)
        ax.yaxis.set_major_formatter(tick)

    if yUnits is not None and yRotate is not None:
        ax.tick_params(labelrotation=45, axis="y")

    if ySize is not None:
        for tk in ax.get_yticklabels():
            tk.set_fontsize(ySize)


def utilSetAxes(x, y, xThresh=0.75, yThresh=0.75):
    """
    Documentation:
        Description:
            Dynamically set lower/upper limits of x/y axes.
    """
    xMin = round(np.min(x), 5)
    xMax = round(np.max(x), 5)
    xChange = (xMax - xMin) / xMax
    xMin = 0 if 1.00 >= xChange >= xThresh else np.round(xMin, 1)
    xMax = xMax + xMax * 0.1

    yMin = round(np.min(y), 5)
    yMax = round(np.max(y), 5)
    yChange = (yMax - yMin) / yMax
    yMin = 0 if 1.00 >= yChange >= yThresh else np.round(yMin, 1)
    yMax = yMax + yMax * 0.1
    return xMin, xMax, yMin, yMax


def numericCoerce(df, cols=None):
    """
    Documentation:
        Description:
            Convert object columns that include numeric data to float or int
            data type.
        Paramters:
            df : Pandas DataFrame
                Input dataset
            cols : list of strings
                List of column names to convert.
        Returns:
            Pandas DataFrame with converted columns.
    """
    # if no columns specified, set cols equal to all non object columns
    if cols is None:
        cols = df.columns

    for col in cols:
        # exclude columsn that contain only nulls
        if not df[col].isnull().all():
            try:
                df[col] = df[col].apply(pd.to_numeric)
            except ValueError:
                pass
        else:
            pass
    return df