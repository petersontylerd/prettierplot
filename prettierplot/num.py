import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.stats import linregress

import prettierplot.style as style
import prettierplot.util as util


def pretty2dScatter(self, x, y, df=None, xUnits="f", xTicks=None, yUnits="f", yTicks=None, plotBuffer=True,
                    size=5, axisLimits=True, color=style.styleGrey, facecolor="w", ax=None):
    """
    Documentation:
        Description:
            Create 2-dimensional scatter plot.
        Parameters:
            x : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size - plotted columns will be
                chosen by columns names specified in x, y.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            xTicks : array, default = None
                Specify custom x-tick labels.
            yUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            yTicks : array, default = None
                Specify custom y-tick labels.
            plotBuffer : boolean, default = True
                Switch for determining whether dynamic plot buffer function is executed.
            size : int or float, default = 5
                Determines scatter dot size.
            axisLimits : boolean, default = True
                Switch for determining whether dynamic axis limit setting function is executed.
            color : string (color code of some sort), default = style.styleGrey
                Determine color of scatter dots
            facecolor : string (color code of some sort), default = 'w'
                Determine face color of scatter dots.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # if a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
    if df is not None:
        x = df[x].values.reshape(-1, 1)
        y = df[y].values.reshape(-1, 1)
    # else reshape arrays.
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    # generate color

    # plot 2-d scatter.
    plt.scatter(
        x=x,
        y=y * 100 if "p" in yUnits else y,
        color=color,
        s=size * self.chartProp,
        alpha=0.7,
        facecolor=facecolor,
        linewidth=0.167 * self.chartProp,
    )

    # dynamically set axis lower / upper limits.
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x=x, y=y)
        plt.axis([xMin, xMax, yMin, yMax])

    # vreate smaller buffer around plot area to prevent cutting off elements.
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

    # use label formatter utility function to customize chart labels
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)


def pretty2dScatterHue(self, x, y, target, label, df=None, xUnits="f", xTicks=None, yUnits="f", yTicks=None, plotBuffer=True, size=10,
                        axisLimits=True, color=style.styleGrey, facecolor="w", bbox=(1.2, 0.9), colorMap="viridis", ax=None):
    """
    Documentation:
        Description:
            Create 2-dimensional scatter plot with a third dimension represented as a color hue in the
            scatter dots.
        Parameters:
            x : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            target : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            label : list
                List of labels describing hue.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size - plotted columns will be
                chosen by columns names specified in x, y.
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            xTicks : array, default = None
                Specify custom x-tick labels.
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            yTicks : array, default = None
                Specify custom y-tick labels.
            plotBuffer : boolean, default = True
                Switch for determining whether dynamic plot buffer function is executed.
            size : int or float, default = 10
                Determines scatter dot size.
            axisLimits : boolean, default = True
                Switch for determining whether dynamic axis limit setting function is executed.
            color : string (color code of some sort), default = style.styleGrey
                Determine color of scatter dots.
            facecolor : string (color code of some sort), default = 'w'
                Determine face color of scatter dots.
            bbox : tuple of floats, default = (1.2, 0.9)
                Coordinates for determining legend position.
            colorMap : string specifying built-in matplotlib colormap, default = "viridis"
                Colormap from which to draw plot colors.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # if a Pandas DataFrame is passed to function, create x, y and target arrays using columns names
    # passed into function. Also create X, which is a matrix containing the x, y and target columns.
    if df is not None:
        X = df[[x, y, target]].values
        x = df[x].values
        y = df[y].values
        target = df[target].values
    # concatenate the x, y and target arrays.
    else:
        X = np.c_[x, y, target]

    # unique target values.
    targetIds = np.unique(X[:, 2])

    # generate color list
    colorList = style.colorGen(name=colorMap, num=len(targetIds))

    # loop through sets of target values, labels and colors to create 2-d scatter with hue.
    for targetId, targetName, color in zip(
        targetIds, label, colorList
    ):
        plt.scatter(
            x=X[X[:, 2] == targetId][:, 0],
            y=X[X[:, 2] == targetId][:, 1],
            color=color,
            label=targetName,
            s=size * self.chartProp,
            alpha=0.7,
            facecolor="w",
            linewidth=0.234 * self.chartProp,
        )

    # add legend to figure.
    if label is not None:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=bbox,
            ncol=1,
            frameon=True,
            fontsize=1.1 * self.chartProp,
        )

    # dynamically set axis lower / upper limits.
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x=x, y=y)
        plt.axis([xMin, xMax, yMin, yMax])

    # create smaller buffer around plot area to prevent cutting off elements.
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

    # use label formatter utility function to customize chart labels
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)


def prettyDistPlot(self, x, color, xUnits="f", yUnits="f", fit=None, xRotate=None, ax=None):
    """
    Documentation:
        Description:
            Creates distribution plot for numeric variables, showing counts of a single
            variable. Also overlays a kernel density estimation curve.
        Parameters:
            x : array
                Data to be plotted.
            color : string (some sort of color code)
                Determines color of bars, KDE lines.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            yUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            fit : random variabe object, default = None
                Allows for the addition of another curve. Utilizing 'norm' overlays a normal distribution
                over the distribution bar chart. Useful for seeing how well, or not, the distribution tracks
                with a normal distrbution.
            xRotate : int, default = None
                Rotates x-axis tick mark labels X degrees.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # create distribution plot with an optional fit curve
    g = sns.distplot(a=x, kde=False, color=color, axlabel=False, fit=fit, ax=ax)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in yUnits else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chartProp,
        color=style.styleGrey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in yUnits else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chartProp,
        color=style.styleGrey,
    )

    # use label formatter utility function to customize chart labels
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits, xRotate=xRotate)


def prettyKdePlot(self, x, color, yUnits="f", xUnits="f", ax=None):
    """
    Documentation:
        Description:
            Create kernel density curve for a feature.
        Parameters:
            x : array
                Data to be plotted.
            color : string (some sort of color code)
                Determines color of KDE lines.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            yUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # create kernel density estimation line
    g = sns.kdeplot(data=x, shade=True, color=color, legend=None, ax=ax)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in yUnits else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chartProp,
        color=style.styleGrey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in yUnits else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chartProp,
        color=style.styleGrey,
    )
    # use label formatter utility function to customize chart labels
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)


def prettyRegPlot(self, x, y, data, dotColor=style.styleGrey, lineColor=style.styleBlue, x_jitter=None, xUnits="f", yUnits="f",
                    xRotate=None, ax=None):
    """
    Documentation:
        Description:
            Create scatter plot with regression line.
        Parameters:
            x : string
                Name of independent variable in dataframe. Represents a category
            y : string
                Name of numeric target variable.
            data : Pandas DataFrame
                Pandas DataFrame including both indepedent variable and target variable.
            dotColor : string
                Determines color of dots.
            lineColor : string
                Determines color of regression line.
            x_jitter : float, default = None
                Optional paramter for randomly displacing dots along the x-axis to enable easier visibility
                of dots.
            labelRotate : float or int, default = 45
                Degrees by which the xtick labels are rotated.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            yUnits : string, default = 'f'
                Determines units of y-axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            xRotate : int, default = None
                Rotates x-axis tick mark labels X degrees.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # create regression plot.
    g = sns.regplot(
        x=x,
        y=y,
        data=data,
        x_jitter=x_jitter,
        scatter_kws={"alpha": 0.3, "color": dotColor},
        line_kws={"color": lineColor},
        ax=ax,
    ).set(xlabel=None, ylabel=None)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in yUnits else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chartProp,
        color=style.styleGrey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in yUnits else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chartProp,
        color=style.styleGrey,
    )

    # use label formatter utility function to customize chart labels
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits, xRotate=xRotate)


def prettyPairPlotCustom(self, df, cols=None, color = style.styleBlue, gradientCol=None):
    """
    Documentation:
        Description:
            Create pair plot that produces a grid of scatter plots for all unique pairs of
            numeric features and a series of KDE or histogram plots along the diagonal.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data of interest.
            cols : list, default = None
                List of strings describing columns in Pandas DataFrame to be visualized.
            color : string, default = style.styleBlue
                Color to serve as high end of gradient when gradientCol is specified.
            gradientCol : string, default = None
                Introduce third dimension to scatter plots through a color hue that differentiates
                dots based on the target's value.
            diag_kind : string, default = 'auto.
                Type of plot created along diagonal.
    """
    # Custom plot formatting settings for this particular chart.
    with plt.rc_context(
        {
            "axes.titlesize": 3.5 * self.chartProp,
            "axes.labelsize": 0.9 * self.chartProp,  # Axis title font size
            "xtick.labelsize": 0.8 * self.chartProp,
            "xtick.major.size": 0.5 * self.chartProp,
            "xtick.major.width": 0.05 * self.chartProp,
            "xtick.color": style.styleGrey,
            "ytick.labelsize": 0.8 * self.chartProp,
            "ytick.major.size": 0.5 * self.chartProp,
            "ytick.major.width": 0.05 * self.chartProp,
            "ytick.color": style.styleGrey,
            "figure.facecolor": style.styleWhite,
            "axes.facecolor": style.styleWhite,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": style.styleGrey,
            "axes.grid": False,
        }
    ):

        # limit to columns of interest if provided
        if cols is not None:
            df = df[cols]

        df = util.numericCoerce(df, cols=cols)


        # create figure and axes
        fig, axes = plt.subplots(
            ncols=len(df.columns),
            nrows=len(df.columns),
            constrained_layout=True,
            figsize=(20, 14),
        )

        # unpack axes
        for (i, j), ax in np.ndenumerate(axes):
            # turn of axes on upper triangle
            # if i < j:
            #     plt.setp(ax.get_xticklabels(), visible=False)
            #     plt.setp(ax.get_yticklabels(), visible=False)
            # set diagonal plots as KDE plots
            if i == j:
                sns.kdeplot(df.iloc[:, i], ax=ax, legend=False, shade=True, color=color)
            # set lower triangle plots as scatter plots
            else:
                sns.scatterplot(
                    x=df.iloc[:, j],
                    y=df.iloc[:, i],
                    hue=gradientCol if gradientCol is None else df[gradientCol],
                    data=df,
                    palette=LinearSegmentedColormap.from_list(
                        name="", colors=["white", color]
                    ),
                    legend=False,
                    ax=ax,
                )
        plt.show()


def prettyPairPlot(self, df, cols=None, target=None, diag_kind="auto", legendLabels=None, bbox=None, colorMap="viridis"):
    """
    Documentation:
        Description:
            Create pair plot that produces a grid of scatter plots for all unique pairs of
            numeric features and a series of KDE or histogram plots along the diagonal.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data of interest.
            cols : list, default = None
                List of strings describing columns in Pandas DataFrame to be visualized.
            target : Pandas Series, default = None
                Introduce third dimension to scatter plots through a color hue that differentiates
                dots based on the target's value.
            diag_kind : string, default = 'auto.
                Type of plot created along diagonal.
            legendLabels : list, default = None
                List containing strings of custom labels to display in legend.
            bbox : tuple of floats, default = None
                Coordinates for determining legend position.
            colorMap : string specifying built-in matplotlib colormap, default = "viridis"
                Colormap from which to draw plot colors.
    """
    # Custom plot formatting settings for this particular chart.
    with plt.rc_context(
        {
            "axes.titlesize": 3.5 * self.chartProp,
            "axes.labelsize": 1.5 * self.chartProp,  # Axis title font size
            "xtick.labelsize": 1.2 * self.chartProp,
            "xtick.major.size": 0.5 * self.chartProp,
            "xtick.major.width": 0.05 * self.chartProp,
            "xtick.color": style.styleGrey,
            "ytick.labelsize": 1.2 * self.chartProp,
            "ytick.major.size": 0.5 * self.chartProp,
            "ytick.major.width": 0.05 * self.chartProp,
            "ytick.color": style.styleGrey,
            "figure.facecolor": style.styleWhite,
            "axes.facecolor": style.styleWhite,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.edgecolor": style.styleGrey,
            "axes.grid": False,
        }
    ):
        # remove object columns
        df = df.select_dtypes(exclude=[object])

        # limit to columns of interest if provided
        if cols is not None:
            df = df[cols]

        # merge df with target if target is provided
        if target is not None:
            df = df.merge(target, left_index=True, right_index=True)

        # create pair plot.
        g = sns.pairplot(
            data=df if target is None else df.dropna(),
            vars=df.columns
            if target is None
            else [x for x in df.columns if x is not target.name],
            hue=target if target is None else target.name,
            diag_kind=diag_kind,
            height=0.2 * self.chartProp,
            plot_kws={
                "s": 2.0 * self.chartProp,
                "edgecolor": None,
                "linewidth": 1,
                "alpha": 0.4,
                "marker": "o",
                "facecolor": style.styleGrey if target is None else None,
            },
            diag_kws={"facecolor": style.styleGrey if target is None else None},
            palette=None if target is None else sns.color_palette(style.colorGen(colorMap, num = len(np.unique(target)))),
        )

        # plot formatting
        for ax in g.axes.flat:
            # _ = ax.set_ylabel(ax.get_ylabel(), rotation=0)
            # _ = ax.set_xlabel(ax.get_xlabel(), rotation=0)
            _ = ax.set_ylabel(ax.get_ylabel(), rotation=40, ha="right")
            _ = ax.set_xlabel(ax.get_xlabel(), rotation=40, ha="right")
            _ = ax.xaxis.labelpad = 20
            _ = ax.yaxis.labelpad = 40
            _ = ax.xaxis.label.set_color(style.styleGrey)
            _ = ax.yaxis.label.set_color(style.styleGrey)
            # _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        # add custom legend describing hue labels
        if target is not None:
            g._legend.remove()

            ## create custom legend
            # create labels
            if legendLabels is None:
                legendLabels = np.unique(df[df[target.name].notnull()][target.name])
            else:
                legendLabels = np.array(legendLabels)

            # generate colors
            colorList = style.colorGen("viridis", num = len(legendLabels))

            labelColor = {}
            for ix, i in enumerate(legendLabels):
                labelColor[i] = colorList[ix]

            # create patches
            patches = [Patch(color=v, label=k) for k, v in labelColor.items()]

            # draw legend
            leg = plt.legend(
                handles=patches,
                fontsize=1.3 * self.chartProp,
                loc="upper right",
                markerscale=0.5 * self.chartProp,
                ncol=1,
                bbox_to_anchor=bbox,
            )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color="Grey")


def prettyHist(self, x, color, label, alpha=0.8):
    """
    Documentation:
        Description:
            Create histogram of numeric variable. Simple function capable of easy
            iteration through several groupings of a numeric variable that is
            separated out based on a categorical label. This results in several overlapping
            histograms and can reveal differences in distributions.
        Parameters:
            x : array
                1-dimensional array of values to be plotted on x-axis.
            color : string (some sort of color code)
                Determines color of histogram.
            label : string
                Category value label.
            alpha : float, default = 0.8
                Fades histogram bars to create transparent bars.
    """
    # Create histogram.
    plt.hist(x=x, color=color, label=label, alpha=alpha)