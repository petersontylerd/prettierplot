import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import prettierplot.style as style
import prettierplot.util as util


def prettyLine(self, x, y, label=None, df=None, linecolor=style.styleGrey, linestyle=None, bbox=(1.2, 0.9), xUnits="f", xTicks=None,
                yUnits="f", yTicks=None, markerOn=False, plotBuffer=False, axisLimits=False, ax=None,):
    """
    Documentation:
        Description:
            Create line plot. Capable of plotting multile lines on the same figure. Also capable of
            adjusting which axis will have the same data for each line and which will have different
            data for each line.
        Parameters:
            x : list, array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            y : list, array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            label : string : default = None
                Name to create legend entry.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size, as plotted columns will be chosen
                by columns names specified in x, y.
            linecolor : string, default = reference to list
                Determine color of line.
            linestyle : string, default = reference to list
                Determine style of line.
            bbox : tuple, default = (1.2, 0.9)
                Override bbox value for legend
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            xTicks : array, default = None
                Specify custom x-tick labels.
            yUnits : string, default = 'f'
                Determines units of y-axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            yTicks : array, default = None
                Specify custom y-tick labels.
            markerOn : boolean, default = False
                Determines whether to show line with markers at each element.
            plotBuffer : boolean, default = False
                Switch for determining whether dynamic plot buffer function is executed.
            axisLimits : boolean, default = False
                Switch for determining whether dynamic axis limit setting function is executed.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # if a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
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
        y * 100 if "p" in yUnits else y,
        color=linecolor,
        linestyle=linestyle,
        linewidth=0.247 * self.chartProp,
        label=label,
        marker="." if markerOn else None,
        markersize=17 if markerOn else None,
        markerfacecolor="w" if markerOn else None,
        markeredgewidth=2.2 if markerOn else None,
    )

    # add legend to figure
    if label is not None:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=bbox,
            ncol=1,
            frameon=True,
            fontsize=1.1 * self.chartProp,
        )

    # dynamically set axis lower / upper limits
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x=x, y=y)
        plt.axis([xMin, xMax, yMin, yMax])

    # create smaller buffer around plot area to prevent cutting off elements
    if plotBuffer:
        util.utilPlotBuffer(ax=ax, x=0.02, y=0.02)

    # tick label control
    if xTicks is not None:
        ax.set_xticks(xTicks)

    if yTicks is not None:
        ax.set_yticks(yTicks)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in yUnits else ax.get_yticklabels(),
        rotation=0,
        fontsize=1.0 * self.chartProp,
        color=style.styleGrey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in yUnits else ax.get_xticklabels(),
        rotation=0,
        fontsize=1.0 * self.chartProp,
        color=style.styleGrey,
    )

    # axis tick label formatting
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)


def prettyMultiLine(self, x, y, label=None, df=None, linecolor=None, linestyle=None, bbox=(1.2, 0.9), xUnits="f", xTicks=None,
                    yUnits="f", yTicks=None, markerOn=False, plotBuffer=False, axisLimits=False, colorMap="viridis", ax=None,):
    """
    Documentation:
        Description:
            Create line plot. Capable of plotting multile lines on the same figure. Also capable of
            adjusting which axis will have the same data for each line and which will have different
            data for each line.
        Parameters:
            x : array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            label : list of strings : default = None
                List of names of used to create legend entries for each line.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size, as plotted columns will be chosen
                by columns names specified in x, y.
            linecolor : string, default = reference to list
                Determine color of line.
            linestyle : string, default = reference to list
                Determine style of line.
            bbox : tuple, default = (1.2, 0.9)
                Override bbox value for legend
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            xTicks : array, default = None
                Specify custom x-tick labels.
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            yTicks : array, default = None
                Specify custom y-tick labels.
            markerOn : boolean, default = False
                Determines whether to show line with markers at each element.
            plotBuffer : boolean, default = False
                Switch for determining whether dynamic plot buffer function is executed.
            axisLimits : boolean, default = False
                Switch for determining whether dynamic axis limit setting function is executed.
            colorMap : string specifying built-in matplotlib colormap, default = "viridis"
                Colormap from which to draw plot colors.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # if a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
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
    colorList = style.colorGen(name=colorMap, num=y.shape[1])

    # add multiple lines
    for ix in np.arange(y.shape[1]):
        yCol = y[:, ix]
        plt.plot(
            x,
            yCol * 100 if "p" in yUnits else yCol,
            color=linecolor if linecolor is not None else colorList[ix],
            linestyle=linestyle if linestyle is not None else style.styleLineStyle[0],
            linewidth=0.247 * self.chartProp,
            label=label[ix] if label is not None else None,
            marker="." if markerOn else None,
            markersize=17 if markerOn else None,
            markerfacecolor="w" if markerOn else None,
            markeredgewidth=2.2 if markerOn else None,
        )

    # add legend to figure
    if label is not None:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=bbox,
            ncol=1,
            frameon=True,
            fontsize=1.1 * self.chartProp,
        )

    # dynamically set axis lower / upper limits
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x=x, y=y)
        plt.axis([xMin, xMax, yMin, yMax])

    # create smaller buffer around plot area to prevent cutting off elements
    if plotBuffer:
        util.utilPlotBuffer(ax=ax, x=0.02, y=0.02)

    # tick label control
    if xTicks is not None:
        ax.set_xticks(xTicks)

    if yTicks is not None:
        ax.set_yticks(yTicks)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in yUnits else ax.get_yticklabels(),
        rotation=0,
        fontsize=1.1 * self.chartProp,
        color=style.styleGrey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in yUnits else ax.get_xticklabels(),
        rotation=0,
        fontsize=1.1 * self.chartProp,
        color=style.styleGrey,
    )

    # axis tick label formatting
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)