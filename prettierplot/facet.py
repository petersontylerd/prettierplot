import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import prettierplot.style as style
import prettierplot.util as util

import textwrap


def prettyFacetCat(
    self, df, feature, labelRotate=0, yUnits="f", xUnits="s", bbox=(1.2, 0.9), ax=None
):
    """
    Documentation:
        Description:
            Creates a count plot for a categorical variable and facets the variable by a 
            categorical label.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            feature : string
                String describing column name containing target values
            labelRotate : float or int, default = 0
                Degrees by which the xtick labels are rotated.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 's' displays string. 'f' displays float. 'p' displays 
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional 
                decimal places.
            yUnits : string, default = 's'
                Determines units of y-axis tick labels. 's' displays string. 'f' displays float. 'p' displays 
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional 
                decimal places.
            bbox : tuple of floats, default = (1.2, 0.9)
                Coordinates for determining legend position.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    ixs = np.arange(df.shape[0])
    bar_width = 0.35

    featureDict = {}
    for feature in df.columns[1:]:
        featureDict[feature] = df[feature].values.tolist()
    for featureIx, (k, v) in enumerate(featureDict.items()):
        plt.bar(
            ixs + (bar_width * featureIx),
            featureDict[k],
            bar_width,
            alpha=0.75,
            color=style.styleHexMid[featureIx],
            label=str(k),
        )

    # custom x-tick labels.
    plt.xticks(ixs[: df.shape[0]] + bar_width / 2,
                ['\n'.join(textwrap.wrap(str(i).replace('_'," "),12)) for i in df.iloc[:, 0].values])
    plt.xticks(rotation=labelRotate)

    # add legend to figure.
    plt.legend(
        loc="upper right",
        bbox_to_anchor=bbox,
        ncol=1,
        frameon=True,
        fontsize=1.1 * self.chartProp,
    )

    # use label formatter utility function to customize chart labels
    util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)

    # resize x-axis labels as needed.
    if len(featureDict[feature]) > 10 and len(featureDict[feature]) <= 20:
        ax.tick_params(axis="x", colors=style.styleGrey, labelsize=1.2 * self.chartProp)
    elif len(featureDict[feature]) > 20:
        ax.tick_params(axis="x", colors=style.styleGrey, labelsize=0.6 * self.chartProp)

    plt.show()


def prettyFacetTwoCatBar(
    self,
    df,
    x,
    y,
    split,
    xUnits=None,
    yUnits=None,
    bbox=None,
    legendLabels=None,
    filterNaN=True,
    ax=None,
):
    """
    Documentation:
        Description:
            Creates a series of bar plots that count a variable along the y-axis and separate the counts
            into bins based on by two categorical variables.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            x : string
                Categorical variable to be plotted along x-axis.
            y : string
                Variable to be counted along y-axis.
            split : string
                Categorical variable on which to differentiate the numCol variable.
            xUnits : string, default = None
                Determines units of x-axis tick labels. 's' displays string. 'f' displays float. 'p' displays 
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional 
                decimal places.
            yUnits : string, default = None
                Determines units of x-axis tick labels. 's' displays string. 'f' displays float. 'p' displays 
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional 
                decimal places.            
            bbox : tuple of floats, default = None
                Coordinates for determining legend position.
            legendLabels : list, default = None
                Custom legend labels.
            filterNan : boolean, default = True
                Remove record that have a null value in the column specified by the 'x' parameter.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    if filterNaN:
        df = df.dropna(subset=[x])

    g = sns.barplot(
        x=x,
        y=y,
        hue=split,
        data=df,
        palette=style.styleHexMid,
        order=df[x].sort_values().drop_duplicates().values.tolist(),
        hue_order=df[split].sort_values().drop_duplicates().values.tolist()
        if split is not None
        else None,
        ax=ax,
        ci=None,
    )

    # Format x and y-tick labels
    g.set_yticklabels(
        g.get_yticklabels(),
        rotation=0,
        fontsize=1.25 * self.chartProp,
        color=style.styleGrey,
    )
    g.set_xticklabels(
        g.get_xticklabels(),
        rotation=0,
        fontsize=1.25 * self.chartProp,
        color=style.styleGrey,
    )
    g.set_ylabel(
        g.get_ylabel(),
        rotation=90,
        fontsize=1.75 * self.chartProp,
        color=style.styleGrey,
    )
    g.set_xlabel(
        g.get_xlabel(),
        rotation=0,
        fontsize=1.75 * self.chartProp,
        color=style.styleGrey,
    )

    ## create custom legend
    # create labels
    if split is not None:
        if legendLabels is None:
            legendLabels = (
                df[df[split].notnull()][split]
                .sort_values()
                .drop_duplicates()
                .values.tolist()
            )
        else:
            legendLabels = np.array(legendLabels)

        labelColor = {}
        for ix, i in enumerate(legendLabels):
            labelColor[i] = style.styleHexMid[ix]

        # create patches
        patches = [Patch(color=v, label=k) for k, v in labelColor.items()]

        # draw legend
        leg = plt.legend(
            handles=patches,
            fontsize=1.25 * self.chartProp,
            loc="upper right",
            markerscale=0.5 * self.chartProp,
            ncol=1,
            bbox_to_anchor=bbox,
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color="Grey")

        # use label formatter utility function to customize chart labels
        util.utilLabelFormatter(ax=ax, xUnits=xUnits, yUnits=yUnits)

    plt.show()


def prettyFacetCatNumScatter(
    self,
    df,
    xNum,
    yNum,
    catRow=None,
    catCol=None,
    split=None,
    bbox=None,
    aspect=1,
    height=4,
    legendLabels=None,
):
    """
    Documentation:
        Description:
            Creates scatter plots of two continuous variables and allows for faceting by up to two
            categorical variables along the column and/or row axes of the figure.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            xNum : string
                Continuous variable to be plotted along x-axis.
            yNum : string
                Continuous variable to be plotted along y-axis.
            catRow : string
                Categorical variable faceted along the row axis.
            catCol : string 
                Categorical variable faceted along the column axis.
            split : string
                Categorical variable on which to differentiate the numCol variable.
            bbox : tuple of floats, default = None
                Coordinates for determining legend position.
            aspect : float, default = 1 
                Higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            height : float, default = 4
                Height in inches of each facet.
            legendLabels : list, default = None
                Custom legend labels.
    """
    g = sns.FacetGrid(
        df,
        col=catCol,
        row=catRow,
        hue=split,
        palette=style.styleHexMid,
        hue_order=df[split].sort_values().drop_duplicates().values.tolist()
        if split is not None
        else None,
        height=height,
        aspect=aspect,
        margin_titles=True,
    )
    g = g.map(
        plt.scatter,
        xNum,
        yNum
        #   ,**kws
    )

    for ax in g.axes.flat:
        _ = ax.set_ylabel(
            ax.get_ylabel(),
            rotation=90,
            fontsize=1.25 * self.chartProp,
            color=style.styleGrey,
        )
        _ = ax.set_xlabel(
            ax.get_xlabel(),
            rotation=0,
            fontsize=1.25 * self.chartProp,
            color=style.styleGrey,
        )
        _ = ax.set_title(
            ax.get_title(),
            rotation=0,
            fontsize=1.05 * self.chartProp,
            color=style.styleGrey,
        )

        if ax.texts:
            # This contains the right ylabel text
            txt = ax.texts[0]
            ax.text(
                txt.get_unitless_position()[0],
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize=1.25 * self.chartProp,
                color=style.styleGrey,
                rotation=-90,
            )
            # Remove the original text
            ax.texts[0].remove()

    ## create custom legend
    # create labels
    if split is not None:
        if legendLabels is None:
            legendLabels = (
                df[df[split].notnull()][split]
                .sort_values()
                .drop_duplicates()
                .values.tolist()
            )
        else:
            legendLabels = np.array(legendLabels)

        labelColor = {}
        for ix, i in enumerate(legendLabels):
            labelColor[i] = style.styleHexMid[ix]

        # create patches
        patches = [Patch(color=v, label=k) for k, v in labelColor.items()]

        # draw legend
        leg = plt.legend(
            handles=patches,
            fontsize=1.0 * self.chartProp,
            loc="upper right",
            markerscale=0.5 * self.chartProp,
            ncol=1,
            bbox_to_anchor=bbox,
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color="Grey")

    plt.show()


def prettyFacetCatNumHist(
    self,
    df,
    catRow,
    catCol,
    numCol,
    split,
    bbox=None,
    aspect=1,
    height=4,
    legendLabels=None,
):
    """
    Documentation:
        Description:
            Creates histograms of one continuous variable, and each can optionally be split by a categorical to 
            show two or more distributions. Allows for faceting by up to two categorical variables along the 
            column and/or row axes of the figure.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            catRow : string
                Categorical variable faceted along the row axis.
            catCol : string 
                Categorical variable faceted along the column axis.
            numCol : string
                Continuous variable to be plotted along x-axis.
            split : string
                Categorical variable on which to differentiate the numCol variable.
            bbox : tuple of floats, default = None
                Coordinates for determining legend position.
            aspect : float, default = 1 
                Higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            height : float, default = 4
                Height in inches of each facet.
            legendLabels : list, default = None
                Custom legend labels.
    """
    g = sns.FacetGrid(
        df,
        row=catRow,
        col=catCol,
        hue=split,
        hue_order=df[split].sort_values().drop_duplicates().values.tolist()
        if split is not None
        else None,
        palette=style.styleHexMid,
        despine=True,
        height=height,
        aspect=aspect,
        margin_titles=True,
    )
    g.map(
        plt.hist,
        numCol
        #  ,bins = np.arange(0, 20)
        ,
        alpha=0.5,
    )

    for ax in g.axes.flat:
        _ = ax.set_ylabel(
            ax.get_ylabel(),
            rotation=90,
            fontsize=1.25 * self.chartProp,
            color=style.styleGrey,
        )
        _ = ax.set_xlabel(
            ax.get_xlabel(),
            rotation=0,
            fontsize=1.25 * self.chartProp,
            color=style.styleGrey,
        )
        _ = ax.set_title(
            ax.get_title(),
            rotation=0,
            fontsize=1.05 * self.chartProp,
            color=style.styleGrey,
        )

        if ax.texts:
            # This contains the right ylabel text
            txt = ax.texts[0]
            ax.text(
                txt.get_unitless_position()[0],
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize=1.25 * self.chartProp,
                color=style.styleGrey,
                rotation=-90,
            )
            # Remove the original text
            ax.texts[0].remove()

    ## create custom legend
    # create labels
    if split is not None:
        if legendLabels is None:
            legendLabels = (
                df[df[split].notnull()][split]
                .sort_values()
                .drop_duplicates()
                .values.tolist()
            )
        else:
            legendLabels = np.array(legendLabels)

        labelColor = {}
        for ix, i in enumerate(legendLabels):
            labelColor[i] = style.styleHexMid[ix]

        # create patches
        patches = [Patch(color=v, label=k) for k, v in labelColor.items()]

        # draw legend
        leg = plt.legend(
            handles=patches,
            fontsize=1.0 * self.chartProp,
            loc="upper right",
            markerscale=0.5 * self.chartProp,
            ncol=1,
            bbox_to_anchor=bbox,
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color="Grey")

    plt.show()


def prettyFacetTwoCatPoint(
    self,
    df,
    x,
    y,
    split,
    catCol=None,
    catRow=None,
    bbox=None,
    aspect=1,
    height=4,
    legendLabels=None,
):
    """
    Documentation:
        Description:
            Creates point plots that 
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            x : string
                Categorical variable to be plotted along x-axis.
            y : string
                Variable to be counted along y-axis.
            split : string
                Categorical variable on which to differentiate the 'x' variable.
            catRow : string
                Categorical variable faceted along the row axis.
            catCol : string 
                Categorical variable faceted along the column axis.
            bbox : tuple of floats, default = None
                Coordinates for determining legend position.
            aspect : float, default = 1 
                Higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            height : float, default = 4
                Height in inches of each facet.
            legendLabels : list, default = None
                Custom legend labels.
    """
    g = sns.FacetGrid(
        df, row=catCol, col=catRow, aspect=aspect, height=height, margin_titles=True
    )
    g.map(
        sns.pointplot,
        x,
        y,
        split,
        order=df[x].sort_values().drop_duplicates().values.tolist(),
        hue_order=df[split].sort_values().drop_duplicates().values.tolist(),
        palette=style.styleHexMid,
        alpha=0.75,
        ci=None,
    )

    for ax in g.axes.flat:
        _ = ax.set_ylabel(
            ax.get_ylabel(),
            rotation=90,
            fontsize=0.95 * self.chartProp,
            color=style.styleGrey,
        )
        _ = ax.set_xlabel(
            ax.get_xlabel(),
            rotation=0,
            fontsize=0.95 * self.chartProp,
            color=style.styleGrey,
        )
        _ = ax.set_title(
            ax.get_title(),
            rotation=0,
            fontsize=1.05 * self.chartProp,
            color=style.styleGrey,
        )

        if ax.texts:
            # This contains the right ylabel text
            txt = ax.texts[0]

            ax.text(
                txt.get_unitless_position()[0],
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize=1.25 * self.chartProp,
                color=style.styleGrey,
                rotation=-90,
            )
            # Remove the original text
            ax.texts[0].remove()

    ## create custom legend
    # create labels
    if legendLabels is None:
        legendLabels = np.unique(df[df[split].notnull()][split])
    else:
        legendLabels = np.array(legendLabels)

    labelColor = {}
    for ix, i in enumerate(legendLabels):
        labelColor[i] = style.styleHexMid[ix]

    # create patches
    patches = [Patch(color=v, label=k) for k, v in labelColor.items()]

    # draw legend
    leg = plt.legend(
        handles=patches,
        fontsize=1.0 * self.chartProp,
        loc="upper right",
        markerscale=0.5 * self.chartProp,
        ncol=1,
        bbox_to_anchor=bbox,
    )

    # label font color
    for text in leg.get_texts():
        plt.setp(text, color="Grey")

    plt.show()
