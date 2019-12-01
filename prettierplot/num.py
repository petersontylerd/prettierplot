import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.stats import linregress

import prettierplot.style as style
import prettierplot.util as util


def pretty2d_scatter(
    self,
    x,
    y,
    df=None,
    x_units="f",
    x_ticks=None,
    y_units="f",
    y_ticks=None,
    plot_buffer=True,
    size=5,
    axis_limits=True,
    color=style.style_grey,
    facecolor="w",
    ax=None,
):
    """
    documentation:
        description:
            create 2_dimensional scatter plot.
        parameters:
            x : array or string
                either 1_dimensional array of values or a column name in a pandas DataFrame.
            y : array or string
                either 1_dimensional array of values or a column name in a pandas DataFrame.
            df : pandas DataFrame, default =None
                dataset containing data to be plotted. can be any size - plotted columns will be
                chosen by columns names specified in x, y.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            x_ticks : array, default =None
                specify custom x_tick labels.
            y_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_ticks : array, default =None
                specify custom y_tick labels.
            plot_buffer : boolean, default=True
                switch for determining whether dynamic plot buffer function is executed.
            size : int or float, default = 5
                determines scatter dot size.
            axis_limits : boolean, default=True
                switch for determining whether dynamic axis limit setting function is executed.
            color : string (color code of some sort), default = style.style_grey
                determine color of scatter dots
            facecolor : string (color code of some sort), default = 'w'
                determine face color of scatter dots.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # if a pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
    if df is not None:
        x = df[x].values.reshape(-1, 1)
        y = df[y].values.reshape(-1, 1)
    # else reshape arrays.
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    # generate color

    # plot 2_d scatter.
    plt.scatter(
        x=x,
        y=y * 100 if "p" in y_units else y,
        color=color,
        s=size * self.chart_prop,
        alpha=0.7,
        facecolor=facecolor,
        linewidth=0.167 * self.chart_prop,
    )

    # dynamically set axis lower / upper limits.
    if axis_limits:
        x_min, x_max, y_min, y_max = util.util_set_axes(x=x, y=y)
        plt.axis([x_min, x_max, y_min, y_max])

    # vreate smaller buffer around plot area to prevent cutting off elements.
    if plot_buffer:
        util.util_plot_buffer(ax=ax, x=0.02, y=0.02)

    # tick label control
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=1.0 * self.chart_prop,
        color=style.style_grey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=1.0 * self.chart_prop,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)


def pretty2d_scatter_hue(
    self,
    x,
    y,
    target,
    label,
    df=None,
    x_units="f",
    x_ticks=None,
    y_units="f",
    y_ticks=None,
    plot_buffer=True,
    size=10,
    axis_limits=True,
    color=style.style_grey,
    facecolor="w",
    bbox=(1.2, 0.9),
    color_map="viridis",
    ax=None,
):
    """
    documentation:
        description:
            create 2_dimensional scatter plot with a third dimension represented as a color hue in the
            scatter dots.
        parameters:
            x : array or string
                either 1_dimensional array of values or a column name in a pandas DataFrame.
            y : array or string
                either 1_dimensional array of values or a column name in a pandas DataFrame.
            target : array or string
                either 1_dimensional array of values or a column name in a pandas DataFrame.
            label : list
                list of labels describing hue.
            df : pandas DataFrame, default =None
                dataset containing data to be plotted. can be any size - plotted columns will be
                chosen by columns names specified in x, y.
            x_units : string, default = 'd'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            x_ticks : array, default =None
                specify custom x_tick labels.
            y_units : string, default = 'd'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_ticks : array, default =None
                specify custom y_tick labels.
            plot_buffer : boolean, default=True
                switch for determining whether dynamic plot buffer function is executed.
            size : int or float, default = 10
                determines scatter dot size.
            axis_limits : boolean, default=True
                switch for determining whether dynamic axis limit setting function is executed.
            color : string (color code of some sort), default = style.style_grey
                determine color of scatter dots.
            facecolor : string (color code of some sort), default = 'w'
                determine face color of scatter dots.
            bbox : tuple of floats, default = (1.2, 0.9)
                coordinates for determining legend position.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # if a pandas DataFrame is passed to function, create x, y and target arrays using columns names
    # passed into function. also create x, which is a matrix containing the x, y and target columns.
    if df is not None:
        x = df[[x, y, target]].values
        x = df[x].values
        y = df[y].values
        target = df[target].values
    # concatenate the x, y and target arrays.
    else:
        x = np.c_[x, y, target]

    # unique target values.
    target_ids = np.unique(x[:, 2])

    # generate color list
    color_list = style.color_gen(name=color_map, num=len(target_ids))

    # loop through sets of target values, labels and colors to create 2_d scatter with hue.
    for target_id, target_name, color in zip(target_ids, label, color_list):
        plt.scatter(
            x=x[x[:, 2] == target_id][:, 0],
            y=x[x[:, 2] == target_id][:, 1],
            color=color,
            label=target_name,
            s=size * self.chart_prop,
            alpha=0.7,
            facecolor="w",
            linewidth=0.234 * self.chart_prop,
        )

    # add legend to figure.
    if label is not None:
        plt.legend(
            loc="upper right",
            bbox_to_anchor=bbox,
            ncol=1,
            frameon=True,
            fontsize=1.1 * self.chart_prop,
        )

    # dynamically set axis lower / upper limits.
    if axis_limits:
        x_min, x_max, y_min, y_max = util.util_set_axes(x=x, y=y)
        plt.axis([x_min, x_max, y_min, y_max])

    # create smaller buffer around plot area to prevent cutting off elements.
    if plot_buffer:
        util.util_plot_buffer(ax=ax, x=0.02, y=0.02)

    # tick label control
    if x_ticks is not None:
        ax.set_xticks(x_ticks)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=1.0 * self.chart_prop,
        color=style.style_grey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=1.0 * self.chart_prop,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)


def pretty_dist_plot(
    self, x, color, x_units="f", y_units="f", fit=None, x_rotate=None, ax=None
):
    """
    documentation:
        description:
            creates distribution plot for numeric variables, showing counts of a single
            variable. also overlays a kernel density estimation curve.
        parameters:
            x : array
                data to be plotted.
            color : string (some sort of color code)
                determines color of bars, kde lines.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            fit : random variabe object, default =None
                allows for the addition of another curve. utilizing 'norm' overlays a normal distribution
                over the distribution bar chart. useful for seeing how well, or not, the distribution tracks
                with a normal distrbution.
            x_rotate : int, default =None
                rotates x_axis tick mark labels x degrees.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # create distribution plot with an optional fit curve
    g = sns.distplot(a=x, kde=False, color=color, axlabel=False, fit=fit, ax=ax)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(
        ax=ax, x_units=x_units, y_units=y_units, x_rotate=x_rotate
    )


def pretty_kde_plot(self, x, color, y_units="f", x_units="f", ax=None):
    """
    documentation:
        description:
            create kernel density curve for a feature.
        parameters:
            x : array
                data to be plotted.
            color : string (some sort of color code)
                determines color of kde lines.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # create kernel density estimation line
    g = sns.kdeplot(data=x, shade=True, color=color, legend=None, ax=ax)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )
    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)


def pretty_reg_plot(
    self,
    x,
    y,
    data,
    dot_color=style.style_grey,
    line_color=style.style_blue,
    x_jitter=None,
    x_units="f",
    y_units="f",
    x_rotate=None,
    ax=None,
):
    """
    documentation:
        description:
            create scatter plot with regression line.
        parameters:
            x : string
                name of independent variable in dataframe. represents a category
            y : string
                name of numeric target variable.
            data : pandas DataFrame
                pandas DataFrame including both indepedent variable and target variable.
            dot_color : string
                determines color of dots.
            line_color : string
                determines color of regression line.
            x_jitter : float, default =None
                optional paramter for randomly displacing dots along the x_axis to enable easier visibility
                of dots.
            label_rotate : float or int, default = 45
                degrees by which the xtick labels are rotated.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_units : string, default = 'f'
                determines units of y_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            x_rotate : int, default =None
                rotates x_axis tick mark labels x degrees.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # create regression plot.
    g = sns.regplot(
        x=x,
        y=y,
        data=data,
        x_jitter=x_jitter,
        scatter_kws={"alpha": 0.3, "color": dot_color},
        line_kws={"color": line_color},
        ax=ax,
    ).set(xlabel=None, ylabel=None)

    # format x and y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in y_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(
        ax=ax, x_units=x_units, y_units=y_units, x_rotate=x_rotate
    )


def pretty_pair_plot_custom(
    self, df, cols=None, color=style.style_blue, gradient_col=None
):
    """
    documentation:
        description:
            create pair plot that produces a grid of scatter plots for all unique pairs of
            numeric features and a series of kde or histogram plots along the diagonal.
        parameters:
            df : pandas DataFrame
                pandas DataFrame containing data of interest.
            cols : list, default =None
                list of strings describing columns in pandas DataFrame to be visualized.
            color : string, default = style.style_blue
                color to serve as high end of gradient when gradient_col is specified.
            gradient_col : string, default =None
                introduce third dimension to scatter plots through a color hue that differentiates
                dots based on the target's value.
            diag_kind : string, default = 'auto.
                type of plot created along diagonal.
    """
    # custom plot formatting settings for this particular chart.
    with plt.rc_context(
        {
            "axes.titlesize": 3.5 * self.chart_prop,
            "axes.labelsize": 0.9 * self.chart_prop,  # axis title font size
            "xtick.labelsize": 0.8 * self.chart_prop,
            "xtick.major.size": 0.5 * self.chart_prop,
            "xtick.major.width": 0.05 * self.chart_prop,
            "xtick.color": style.style_grey,
            "ytick.labelsize": 0.8 * self.chart_prop,
            "ytick.major.size": 0.5 * self.chart_prop,
            "ytick.major.width": 0.05 * self.chart_prop,
            "ytick.color": style.style_grey,
            "figure.facecolor": style.style_white,
            "axes.facecolor": style.style_white,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": style.style_grey,
            "axes.grid": False,
        }
    ):

        # limit to columns of interest if provided
        if cols is not None:
            df = df[cols]

        df = util.numeric_coerce(df, cols=cols)

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
            # set diagonal plots as kde plots
            if i == j:
                sns.kdeplot(df.iloc[:, i], ax=ax, legend=False, shade=True, color=color)
            # set lower triangle plots as scatter plots
            else:
                sns.scatterplot(
                    x=df.iloc[:, j],
                    y=df.iloc[:, i],
                    hue=gradient_col if gradient_col is None else df[gradient_col],
                    data=df,
                    palette=LinearSegmentedColormap.from_list(
                        name="", colors=["white", color]
                    ),
                    legend=False,
                    ax=ax,
                )
        plt.show()


def pretty_pair_plot(
    self,
    df,
    cols=None,
    target=None,
    diag_kind="auto",
    legend_labels=None,
    bbox=None,
    color_map="viridis",
):
    """
    documentation:
        description:
            create pair plot that produces a grid of scatter plots for all unique pairs of
            numeric features and a series of kde or histogram plots along the diagonal.
        parameters:
            df : pandas DataFrame
                pandas DataFrame containing data of interest.
            cols : list, default =None
                list of strings describing columns in pandas DataFrame to be visualized.
            target : pandas series, default =None
                introduce third dimension to scatter plots through a color hue that differentiates
                dots based on the target's value.
            diag_kind : string, default = 'auto.
                type of plot created along diagonal.
            legend_labels : list, default =None
                list containing strings of custom labels to display in legend.
            bbox : tuple of floats, default =None
                coordinates for determining legend position.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
    """
    # custom plot formatting settings for this particular chart.
    with plt.rc_context(
        {
            "axes.titlesize": 3.5 * self.chart_prop,
            "axes.labelsize": 1.5 * self.chart_prop,  # axis title font size
            "xtick.labelsize": 1.2 * self.chart_prop,
            "xtick.major.size": 0.5 * self.chart_prop,
            "xtick.major.width": 0.05 * self.chart_prop,
            "xtick.color": style.style_grey,
            "ytick.labelsize": 1.2 * self.chart_prop,
            "ytick.major.size": 0.5 * self.chart_prop,
            "ytick.major.width": 0.05 * self.chart_prop,
            "ytick.color": style.style_grey,
            "figure.facecolor": style.style_white,
            "axes.facecolor": style.style_white,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.edgecolor": style.style_grey,
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
            height=0.2 * self.chart_prop,
            plot_kws={
                "s": 2.0 * self.chart_prop,
                "edgecolor": None,
                "linewidth": 1,
                "alpha": 0.4,
                "marker": "o",
                "facecolor": style.style_grey if target is None else None,
            },
            diag_kws={"facecolor": style.style_grey if target is None else None},
            palette=None
            if target is None
            else sns.color_palette(
                style.color_gen(color_map, num=len(np.unique(target)))
            ),
        )

        # plot formatting
        for ax in g.axes.flat:
            # _ = ax.set_ylabel(ax.get_ylabel(), rotation=0)
            # _ = ax.set_xlabel(ax.get_xlabel(), rotation=0)
            _ = ax.set_ylabel(ax.get_ylabel(), rotation=40, ha="right")
            _ = ax.set_xlabel(ax.get_xlabel(), rotation=40, ha="right")
            _ = ax.xaxis.labelpad = 20
            _ = ax.yaxis.labelpad = 40
            _ = ax.xaxis.label.set_color(style.style_grey)
            _ = ax.yaxis.label.set_color(style.style_grey)
            # _ = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

        plt.subplots_adjust(hspace=0.0, wspace=0.0)

        # add custom legend describing hue labels
        if target is not None:
            g._legend.remove()

            ## create custom legend
            # create labels
            if legend_labels is None:
                legend_labels = np.unique(df[df[target.name].notnull()][target.name])
            else:
                legend_labels = np.array(legend_labels)

            # generate colors
            color_list = style.color_gen("viridis", num=len(legend_labels))

            label_color = {}
            for ix, i in enumerate(legend_labels):
                label_color[i] = color_list[ix]

            # create patches
            patches = [Patch(color=v, label=k) for k, v in label_color.items()]

            # draw legend
            leg = plt.legend(
                handles=patches,
                fontsize=1.3 * self.chart_prop,
                loc="upper right",
                markerscale=0.5 * self.chart_prop,
                ncol=1,
                bbox_to_anchor=bbox,
            )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color="grey")


def pretty_hist(self, x, color, label, alpha=0.8):
    """
    documentation:
        description:
            create histogram of numeric variable. simple function capable of easy
            iteration through several groupings of a numeric variable that is
            separated out based on a categorical label. this results in several overlapping
            histograms and can reveal differences in distributions.
        parameters:
            x : array
                1_dimensional array of values to be plotted on x_axis.
            color : string (some sort of color code)
                determines color of histogram.
            label : string
                category value label.
            alpha : float, default = 0.8
                fades histogram bars to create transparent bars.
    """
    # create histogram.
    plt.hist(x=x, color=color, label=label, alpha=alpha)
