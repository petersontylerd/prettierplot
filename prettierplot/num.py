import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from scipy.stats import linregress

import prettierplot.style as style
import prettierplot.util as util

import textwrap


def scatter_2d(self, x, y, df=None, x_units="f", x_ticks=None, y_units="f", y_ticks=None, plot_buffer=True,
                        size=5, axis_limits=True, color=style.style_grey, facecolor="w", alpha=0.8,
                        x_rotate=None, ax=None):
    """
    Documentation:

        ---
        Description:
            Create 2-dimensional scatter plot.

        ---
        Parameters:
            x : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            df : Pandas DataFrame, default=None
                Pandas DataFrame containing data to plot. Can be any size - plotted columns will be
                chosen by columns names specified in x and y parameters.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            x_ticks : array, default=None
                Custom x-tick labels.
            y_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_ticks : array, default=None
                Custom y-tick labels.
            plot_buffer : bool, default=True
                Controls whether dynamic plot buffer function is executed.
            size : int or float, default=5
                Size of scattered dots.
            axis_limits : bool, default=True
                Controls whether dynamic axis limit setting function is executed.
            color : str (color code of some sort), default=style.style_grey
                Color of scattered dots
            facecolor : str (color code of some sort), default='w'
                Face color of scattered dots.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            x_rotate : int, default=None
                Rotates x-axis tick mark labels x degrees.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # if a Pandas DataFrame is passed to function, create x and y arrays using columns names passed into function
    if df is not None:
        x = df[x].values.reshape(-1, 1)
        y = df[y].values.reshape(-1, 1)
    # else reshape arrays
    else:
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

    # plot 2-dimensional scatter
    plt.scatter(
        x=x,
        y=y * 100 if "p" in y_units else y,
        color=color,
        s=size * self.chart_scale,
        alpha=alpha,
        facecolor=facecolor,
        linewidth=0.167 * self.chart_scale,
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

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units, x_rotate=x_rotate)

def scatter_2d_hue(self, x, y, target, label, df=None, x_units="f", x_ticks=None, y_units="f", y_ticks=None,
                        plot_buffer=True, size=10, axis_limits=True, color=style.style_grey, facecolor="w",
                        bbox=(1.2, 0.9), color_map="viridis", alpha=0.8, x_rotate=None, ax=None):
    """
    Documentation:

        ---
        Description:
            Create 2-dimensional scatter plot with a third dimension represented as a color hue in the
            scatter dots.

        ---
        Parameters:
            x : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            target : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            label : list
                Labels corresponding to color hue.
            df : Pandas DataFrame, default=None
                Pandas DataFrame containing data to plot. Can be any size - plotted columns will be
                chosen by columns names specified in x and y parameters.
            x_units : str, default='d'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            x_ticks : array, default=None
                Custom x-tick labels.
            y_units : str, default='d'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_ticks : array, default=None
                Custom y-tick labels.
            plot_buffer : bool, default=True
                Controls whether dynamic plot buffer function is executed.
            size : int or float, default=10
                Size of scattered dots.
            axis_limits : bool, default=True
                Controls whether dynamic axis limit setting function is executed.
            color : str (color code of some sort), default=style.style_grey
                Color of scattered dots
            facecolor : str (color code of some sort), default='w'
                Face color of scattered dots
            bbox : tuple of floats, default=(1.2, 0.9)
                Coordinates for determining legend position.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            x_rotate : int, default=None
                Rotates x-axis tick mark labels x degrees.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # if a Pandas DataFrame is passed to function, create x and y and target arrays using columns names
    # passed into function. Also concatenates columns into single object
    if df is not None:
        x = df[[x, y, target]].values
        x = df[x].values
        y = df[y].values
        target = df[target].values
    # concatenate the x, y and target arrays
    else:
        x = np.c_[x, y, target]

    # unique target values
    target_ids = np.unique(x[:, 2])

    # generate color list
    color_list = style.color_gen(name=color_map, num=len(target_ids))

    # loop through sets of target values, labels and colors to create 2_d scatter with hue
    for target_id, target_name, color in zip(target_ids, label, color_list):
        plt.scatter(
            x=x[x[:, 2] == target_id][:, 0],
            y=x[x[:, 2] == target_id][:, 1],
            color=color,
            label=target_name,
            s=size * self.chart_scale,
            alpha=alpha,
            facecolor="w",
            linewidth=0.234 * self.chart_scale,
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

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units, x_rotate=x_rotate)

def dist_plot(self, x, color, x_units="f", y_units="f", fit=None, kde=False, x_rotate=None, alpha=0.8,
                    bbox=(1.2, 0.9), legend_labels=None, color_map="viridis", ax=None):
    """
    Documentation:

        ---
        Description:
            Creates distribution plot for numeric variable. Optionally overlays a kernel density
            estimation curve.

        ---
        Parameters:
            x : array
                Data for plotting.
            color : str (some sort of color code)
                Color of bars and KDE lines.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            fit : random variabe object, default=None
                Allows for the addition of another curve. utilizing 'norm' overlays a normal distribution
                over the distribution bar chart. Useful for seeing how well, or not, the distribution tracks
                with a specified distrbution.
            kde : boolean, default=False
                Controls whether kernel density is plotted over distribution.
            x_rotate : int, default=None
                Rotates x_axis tick mark labels x degrees.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            bbox : tuple of floats, default=(1.2, 0.9)
                Coordinates for determining legend position.
            legend_labels : list, default=None
                Custom legend labels.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # create distribution plot with an optional fit curve
    g = sns.distplot(
        a=x,
        kde=kde,
        color=color,
        axlabel=False,
        fit=fit,
        kde_kws={"lw": 0.2 * self.chart_scale},
        hist_kws={"alpha": alpha},
        ax=ax,
    )

    # tick label font size
    ax.tick_params(axis="both", colors=style.style_grey, labelsize=1.2 * self.chart_scale)

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

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(
        ax=ax, x_units=x_units, y_units=y_units, x_rotate=x_rotate
    )

    ## create custom legend
    if legend_labels is None:
        legend_labels = legend_labels
    else:
        legend_labels = np.array(legend_labels)

        # generate colors
        color_list = style.color_gen(color_map, num=len(legend_labels))

        label_color = {}
        for ix, i in enumerate(legend_labels):
            label_color[i] = color_list[ix]

        # create legend Patches
        patches = [Patch(color=v, label=k, alpha=alpha) for k, v in label_color.items()]

        # draw legend
        leg = plt.legend(
            handles=patches,
            fontsize=1.0 * self.chart_scale,
            loc="upper right",
            markerscale=0.5 * self.chart_scale,
            ncol=1,
            bbox_to_anchor=bbox,
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color="grey")

def kde_plot(self, x, color, x_units="f", y_units="f", shade=False, line_width=0.25, bw=1.0, ax=None):
    """
    Documentation:

        ---
        Description:
            Create kernel density curve for a feature.

        ---
        Parameters:
            x : array
                Data for plotting.
            color : str (some sort of color code)
                Color of KDE lines.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            shade : boolean, default=True
                Controls whether area under KDE curve is shaded.
            line_width : float or int, default= 0.25
                Controlsthickness of kde lines
            bw : float, default=1.0
                Scaling factor for the KDE curve. Smaller values create more detailed curves
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # create kernel density estimation line
    g = sns.kdeplot(
        data=x,
        shade=shade,
        color=color,
        legend=None,
        linewidth=self.chart_scale * line_width,
        ax=ax
    )

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
    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)

def reg_plot(self, x, y, data, dot_color=style.style_grey, dot_size=2.0, line_color=style.style_blue, line_width = 0.3,
            x_jitter=None, x_units="f", y_units="f", x_rotate=None, alpha=0.3, ax=None):
    """
    Documentation:

        ---
        Description:
            create scatter plot with regression line.

        ---
        Parameters:
            x : str
                Name of independent variable in dataframe.
            y : str
                Name of numeric target variable.
            data : Pandas DataFrame
                Pandas DataFrame including both x and y columns.
            dot_color : str
                Color of scattered dots.
            dot_size : float or int
                Size of scattered dots.
            line_color : str
                Regression line color.
            line_width : float or int
                Regression line width.
            x_jitter : float, default=None
                optional paramter for randomly displacing dots along the x_axis to enable easier
                visibility of individual dots.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_units : str, default='f'
                Determines unit of measurement for y-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            x_rotate : int, default=None
                Rotates x_axis tick mark labels x degrees.
            alpha : float, default=0.3
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # create regression plot
    g = sns.regplot(
        x=x,
        y=y,
        data=data,
        x_jitter=x_jitter,
        scatter_kws={
            "alpha": alpha,
            "color": dot_color,
            "s": dot_size * self.chart_scale,
            },
        line_kws={
            "color": line_color,
            "linewidth": self.chart_scale * line_width,
            },
        ax=ax,
    ).set(xlabel=None, ylabel=None)

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

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(
        ax=ax, x_units=x_units, y_units=y_units, x_rotate=x_rotate
    )

def pair_plot_custom(self, df, columns=None, color=style.style_blue, gradient_col=None):
    """
    Documentation:

        ---
        Description:
            Create pair plot that produces a grid of scatter plots for all unique pairs of
            numeric features and a series of KDE plots along the diagonal.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data for plotting.
            columns : list, default=None
                List of strings describing columns in Pandas DataFrame to be visualized. If None,
                all columns are visualized.
            color : str, default=style.style_blue
                Color applied to KDE along diagonal. Also used as the  high end of gradient if
                a gradient_col is specified.
            gradient_col : str, default=None
                Introduce third dimension to scatter plots through a color hue that differentiates
                dots based on the category.
    """
    # custom plot formatting settings for this particular chart.
    with plt.rc_context(
        {
            "axes.titlesize": 3.5 * self.chart_scale,
            "axes.labelsize": 0.9 * self.chart_scale,  # axis title font size
            "xtick.labelsize": 0.8 * self.chart_scale,
            "xtick.major.size": 0.5 * self.chart_scale,
            "xtick.major.width": 0.05 * self.chart_scale,
            "xtick.color": style.style_grey,
            "ytick.labelsize": 0.8 * self.chart_scale,
            "ytick.major.size": 0.5 * self.chart_scale,
            "ytick.major.width": 0.05 * self.chart_scale,
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

        # optionally limit to a subset of columns
        if columns is not None:
            df = df[columns]

        # ensure values are numeric to ensure that scattering works
        df = util.number_coerce(df, columns=columns)

        # create figure and axes
        fig, axes = plt.subplots(
            ncols=len(df.columns),
            nrows=len(df.columns),
            constrained_layout=True,
            figsize=(1.2 * self.chart_scale, 0.9 * self.chart_scale),
        )

        # unpack axes
        for (i, j), ax in np.ndenumerate(axes):
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

def pair_plot(self, df, columns=None, target=None, diag_kind="auto", legend_labels=None, drop_na=True,
                    bbox=(2.0, 1.0), alpha=0.7, color_map="viridis"):
    """
    Documentation:

        ---
        Description:
            Create pair plot that produces a grid of scatter plots for all unique pairs of
            number features and a series of KDE or histogram plots along the diagonal.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data of interest.
            columns : list, default=None
                List of strings describing columns in Pandas DataFrame to be visualized. If None,
                all columns are visualized.
            target : Pandas Series, default=None
                Introduce third dimension to scatter plots through a color hue that differentiates
                dots based on the category value.
            diag_kind : str, default='auto.
                Type of plot created along diagonal.
            drop_na : boolean, default=True
                Controls whether rows with null values are dropped.
            legend_labels : list, default=None
                List containing strings of custom labels to display in legend.
            bbox : tuple of floats, default=None
                Coordinates for determining legend position.
            alpha : float, default=0.7
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
    """
    # custom plot formatting settings for this particular chart.
    with plt.rc_context(
        {
            "axes.titlesize": 3.5 * self.chart_scale,
            "axes.labelsize": 1.5 * self.chart_scale,  # axis title font size
            "xtick.labelsize": 1.2 * self.chart_scale,
            "xtick.major.size": 0.5 * self.chart_scale,
            "xtick.major.width": 0.05 * self.chart_scale,
            "xtick.color": style.style_grey,
            "ytick.labelsize": 1.2 * self.chart_scale,
            "ytick.major.size": 0.5 * self.chart_scale,
            "ytick.major.width": 0.05 * self.chart_scale,
            "ytick.color": style.style_grey,
            "figure.facecolor": style.style_white,
            "axes.facecolor": style.style_white,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "axes.edgecolor": style.style_grey,
            "axes.grid": False,
        }
    ):
        # optionally drop rows with nulls
        if drop_na:
            df = df.dropna()

        # optionally limit to a subset of columns
        if columns is not None:
            df = df[columns]

        # merge df with target if target is provided
        if target is not None:
            df = df.merge(target, left_index=True, right_index=True)

        # create pair plot
        g = sns.pairplot(
            data=df if target is None else df.dropna(),
            vars=df.columns
            if target is None
            else [x for x in df.columns if x is not target.name],
            hue=target if target is None else target.name,
            diag_kind=diag_kind,
            height=0.2 * self.chart_scale,
            plot_kws={
                "s": 2.0 * self.chart_scale,
                "edgecolor": None,
                "linewidth": 1,
                "alpha": alpha,
                "marker": "o",
                "facecolor": style.style_grey if target is None else None,
            },
            diag_kws={
                "facecolor": style.style_grey if target is None else style.style_white,
                "linewidth": 2,
                },
            # diag_kws={"facecolor": style.style_grey if target is None else None},
            palette=None
            if target is None
            else sns.color_palette(
                style.color_gen(color_map, num=len(np.unique(target)))
            ),
        )

        # plot formatting
        for ax in g.axes.flat:

            _ = ax.set_xlabel(
                    "\n".join(textwrap.wrap(str(ax.get_xlabel()).replace("_", " "), 12))
                , rotation=40, ha="right")
            _ = ax.set_ylabel(
                    "\n".join(textwrap.wrap(str(ax.get_ylabel()).replace("_", " "), 12))
                , rotation=40, ha="right")
            _ = ax.xaxis.labelpad = 20
            _ = ax.yaxis.labelpad = 40
            _ = ax.xaxis.label.set_color(style.style_grey)
            _ = ax.yaxis.label.set_color(style.style_grey)

            # wrap long x-tick labels
            plt.xlabel(
                # 0,
                [
                    "\n".join(textwrap.wrap(str(i).replace("_", " "), 12))
                    for i in ax.get_xlabel()
                ],
                # ha="center",
            )

            # wrap long y-tick labels
            plt.ylabel(
                # 0,
                [
                    "\n".join(textwrap.wrap(str(i).replace("_", " "), 12))
                    for i in ax.get_xlabel()
                ],
                # va="center_baseline",
            )

        # adjust subplot relative positioning
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

            # create legend Patches
            patches = [Patch(color=v, label=k, alpha=alpha) for k, v in label_color.items()]

            # draw legend
            leg = plt.legend(
                handles=patches,
                fontsize=0.6 * self.chart_scale * np.log1p(len(g.axes.flat)),
                loc="upper right",
                markerscale=0.15 * self.chart_scale * np.log1p(len(g.axes.flat)),
                ncol=1,
                bbox_to_anchor=bbox,
            )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color="grey")

def hist(self, x, color, label, alpha=0.8):
    """
    Documentation:

        ---
        Description:
            Create histogram of numeric variable.

        ---
        Parameters:
            x : array
                1-dimensional array of values to plot on x_axis.
            color : str (some sort of color code)
                Histogram color.
            label : str
                Legend label.
            alpha : float, default=0.8
                Controls transparency of bars. Accepts value between 0.0 and 1.0.
    """
    # create histogram
    plt.hist(x=x, color=color, label=label, alpha=alpha)

