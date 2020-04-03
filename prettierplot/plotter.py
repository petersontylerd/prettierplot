import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import prettierplot.style as style
import prettierplot.util as util


class PrettierPlot:
    """
    Documentation:

        ---
        Description:
            PrettierPlot creates high_-quality data visualizations quickly and easily.
            Initialization of this class creates a plotting object of a chosen size and
            shape. Once the figure is initialized, the method make_canvas should be called
            to add an axis to the plotting object. Multiple axes can be plotted on a single
            figure, or the position variable can be utilized to create a subplot arrangement.
    """

    from .cat import (
        bar_v,
        bar_h,
        box_plot_v,
        box_plot_h,
        stacked_bar_h,
        tree_map,
    )
    from .data import titanic, attrition, housing
    from .eval import (
        prob_plot,
        corr_heatmap,
        corr_heatmap_target,
        roc_curve_plot,
        decision_region,
    )
    from .facet import (
        facet_cat,
        facet_two_cat_bar,
        facet_cat_num_hist,
        facet_two_cat_point,
        facet_cat_num_scatter,
    )
    from .line import line, multi_line
    from .num import (
        scatter_2d,
        scatter_2d_hue,
        dist_plot,
        kde_plot,
        reg_plot,
        pair_plot,
        pair_plot_custom,
        hist,
    )

    def __init__(self, chart_scale=15, plot_orientation=None):
        """
        Documentation:
            ---
            Description:
                Initialize PrettierPlot and dynamically set chart size and shap.

            ---
            Parameters:
                chart_scale : float or int, default=15
                    Chart proportionality control. Determines relative size of figure size, axis labels,
                    chart title, tick labels, tick marks, among other things.
                plot_orientation : str, default=None
                    Default value produces a plot that is twice as wide as it is tall. Additional plot shapes
                    can be specified by passing certain strings. Options include:
                    - 'tall' - plot that is taller than it is wide
                    - 'square' - plot that is as tall as it is wide
                    - 'wide_narrow' - plot that is much wider than it is tall
                    - 'wide_standard' - plot that is wider than it is tall
        """
        self.chart_scale = chart_scale
        self.plot_orientation = plot_orientation

        # force white plot facecolor
        self.fig = plt.figure(facecolor="white")

        # set graphic style
        # plt.rcParams['figure.facecolor'] = 'white'
        sns.set(rc=style.rc_grey)

        # dynamically set chart width and height parameters
        if plot_orientation == "tall":
            chart_width = self.chart_scale * 0.7
            chart_height = self.chart_scale * 1.2
        elif plot_orientation == "square":
            chart_width = self.chart_scale
            chart_height = self.chart_scale * 0.8
        elif plot_orientation == "wide_narrow":
            chart_width = self.chart_scale * 2.0
            chart_height = self.chart_scale * 0.42
        elif plot_orientation == "wide_standard":
            chart_width = self.chart_scale * 1.6
            chart_height = self.chart_scale * 0.75
        else:
            chart_width = self.chart_scale
            chart_height = self.chart_scale * 0.5
        self.fig.set_figheight(chart_height)
        self.fig.set_figwidth(chart_width)

    def make_canvas(self, title="", x_label="", x_shift=0.0, y_label="", y_shift=0.8, position=111, nrows=None,
                    ncols=None, index=None, sharex=None, sharey=None, title_scale=1.0):
        """
        Documentation:
            ---
            Description:
                Create axes object. Adds descriptive attributes such as titles and axis labels,
                sets font size and font color. Removes grid. Removes top and right spine.

            ---
            Parameters:
                title : str, default="" (blank)
                    The title for the chart.
                x_label : str, default="" (blank)
                    x-axis label.
                x_shift : float, default=0.8
                    Controls position of x-axis label. Higher values move label right along axis.
                    Intent is to align with left side of x-axis.
                y_label : str, default="" (blank)
                    y-axis label.
                y_shift : float, default=0.8
                    Controls position of y-axis label. Higher values move label higher along axis.
                    Intent is to align with top of y-axis.
                position : int (nrows, ncols, index), default=111
                    Determine subplot position of plot.
                nrows : int, default=None
                    Number of rows in subplot grid.
                ncols : int, default=None
                    Number of columns in subplot grid.
                index : int, default=None
                    Axis position on subplot grid.
                sharex : bool or none, default=None
                    Conditional controlling whether to share x-axis across all subplots in a column.
                sharey : bool or none, default=None
                    Conditional controlling whether to share y-axis across all subplots in a row.
                title_scale : float, default=1.0
                    Controls the scaling up (higher value) and scaling down (lower value) of the size of
                    the main chart title, the x-axis title and the y-axis title.
            returns
                ax : axes object
                    Axes object containing visual elements
        """
        # add subplot
        if index is not None:
            ax = self.fig.add_subplot(nrows, ncols, index, sharex=sharex, sharey=sharey)
        else:
            ax = self.fig.add_subplot(position)

        ## add title
        # dynamically determine font size based on string length and scale by title_scale
        if len(title) >= 45:
            font_adjust = 1.0 * title_scale
        elif len(title) >= 30 and len(title) < 45:
            font_adjust = 1.25 * title_scale
        elif len(title) >= 20 and len(title) < 30:
            font_adjust = 1.5 * title_scale
        else:
            font_adjust = 1.75 * title_scale

        # set title
        ax.set_title(
            title,
            fontsize=(2.0 * self.chart_scale) * title_scale
            if position == 111
            else font_adjust * self.chart_scale,
            color=style.style_grey,
            loc="left",
            pad=0.4 * self.chart_scale,
        )

        # remove grid line and right/top spines
        ax.grid(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # add axis labels
        plt.xlabel(
            x_label,
            fontsize=1.667 * self.chart_scale * title_scale,
            labelpad=1.667 * self.chart_scale,
            position=(x_shift, 0.5),
            horizontalalignment="left",
        )
        plt.ylabel(
            y_label,
            fontsize=1.667 * self.chart_scale * title_scale,
            labelpad=1.667 * self.chart_scale,
            position=(1.0, y_shift),
            horizontalalignment="left",
        )

        return ax
