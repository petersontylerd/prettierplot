import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import prettierplot.style as style
import prettierplot.util as util

import textwrap


def facet_cat(self, df, feature, label_rotate=0, x_units="s", y_units="f", bbox=(1.2, 0.9), alpha=0.8,
                legend_labels=None, color_map="viridis", ax=None):
    """
    Documentation:

        ---
        Description:
            Creates a count plot for a categorical variable and facet the variable by another
            categorical variable.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data for plotting.
            feature : str
                Name of column that contains the category values to be used for faceting/
            label_rotate : float or int, default=0
                Number of degrees to rotate the x-tick labels.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 's' displays string. 'f' displays
                float. 'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd')
                for additional decimal places.
            y_units : str, default='s'
                Determines unit of measurement for y-axis tick labels. 's' displays string. 'f' displays
                float. 'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd')
                for additional decimal places.
            bbox : tuple of floats, default=(1.2, 0.9)
                Coordinates for determining legend position.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            legend_labels : list, default=None
                Custom legend labels.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    ixs = np.arange(df.shape[0])
    bar_width = 0.35

    feature_dict = {}
    for feature in df.columns[1:]:
        feature_dict[feature] = df[feature].values.tolist()

    # generate color list
    if isinstance(color_map, str):
        color_list = style.color_gen(name=color_map, num=len(feature_dict.keys()))
    elif isinstance(color_map, list):
        color_list = color_map

    for feature_ix, (k, v) in enumerate(feature_dict.items()):
        plt.bar(
            ixs + (bar_width * feature_ix),
            feature_dict[k],
            bar_width,
            alpha=alpha,
            color=color_list[feature_ix],
            label=str(k),
        )

    # wrap long x-tick labels
    plt.xticks(
        ixs[: df.shape[0]] + bar_width / 2,
        [
            "\n".join(textwrap.wrap(str(i).replace("_", " "), 12))
            for i in df.iloc[:, 0].values
        ],
    )
    plt.xticks(rotation=label_rotate)

    ## create custom legend
    # create labels
    if legend_labels is None:
        legend_labels = np.arange(len(color_list))
    else:
        legend_labels = np.array(legend_labels)

    # define colors
    label_color = {}
    for ix, i in enumerate(legend_labels):
        label_color[i] = color_list[ix]

    # create legend Patches
    patches = [Patch(color=v, label=k, alpha=alpha) for k, v in label_color.items()]

    # draw legend
    leg = plt.legend(
        handles=patches,
        fontsize=0.95 * self.chart_scale,
        loc="upper right",
        markerscale=0.3 * self.chart_scale,
        ncol=1,
        bbox_to_anchor=bbox,
    )

    # label font color
    for text in leg.get_texts():
        plt.setp(text, color="grey")

    ### general formatting
    # if data is float dtype, then format as a number
    if df.iloc[:, 0].values.dtype == np.float:
        x_units = "f"
    # otherwise represent data as a string
    else:
        x_units = "s"

    # use label formatter utility function to customize chart labels
    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)

    # tick label font size
    ax.tick_params(axis="both", colors=style.style_grey, labelsize=1.2 * self.chart_scale)

    # dynamically set x-axis label size
    if 7 < len(feature_dict[feature]) <= 10:
        ax.tick_params(axis="x", colors=style.style_grey, labelsize=0.9 * self.chart_scale)
    elif 10 < len(feature_dict[feature]) <= 20:
        ax.tick_params(axis="x", colors=style.style_grey, labelsize=0.75 * self.chart_scale)
    elif len(feature_dict[feature]) > 20:
        ax.tick_params(axis="x", colors=style.style_grey, labelsize=0.6 * self.chart_scale)

def facet_two_cat_bar(self, df, x, y, split, x_units=None, y_units=None, bbox=None, alpha=0.8,
                        legend_labels=None, filter_nan=True, color_map="viridis", ax=None):
    """
    Documentation:

        Description:
            Creates a series of bar plots that count a variable along the y_axis and separate the counts
            into bins based on two category variables.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data for plotting.
            x : str
                Categorical variable to plot along x-axis.
            y : str
                Pandas DataFrame containing data for plotting.
                ariable to be counted along y-axis.
            split : str
                Categorical variable for faceting the num_col variable.
            x_units : str, default=None
                Determines unit of measurement for x-axis tick labels. 's' displays string. 'f' displays
                float. 'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd')
                for additional decimal places.
            y_units : str, default=None
                Determines unit of measurement for x-axis tick labels. 's' displays string. 'f' displays
                float. 'p' displays percentages, 'd' displays dollars. Repeat character (e.g 'ff' or 'ddd')
                for additional decimal places.
            bbox : tuple of floats, default=None
                Coordinates for determining legend position.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            legend_labels : list, default=None
                Custom legend labels.
            filter_nan : bool, default=True
                Remove records that have a null value in the column specified by the 'x' parameter.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    if ax is None:
        ax = self.ax

    # remove nans from x columns
    if filter_nan:
        df = df.dropna(subset=[x])

    # create bar plot
    g = sns.barplot(
        x=x,
        y=y,
        hue=split,
        data=df,
        palette=sns.color_palette(
            style.color_gen("viridis", num=len(np.unique(df[split].values)))
        ),
        order=df[x].sort_values().drop_duplicates().values.tolist(),
        hue_order=df[split].sort_values().drop_duplicates().values.tolist()
        if split is not None
        else None,
        ax=ax,
        ci=None,
    )

    # format x-tick labels
    g.set_xticklabels(
        g.get_xticklabels(),
        rotation=0,
        fontsize=1.05 * self.chart_scale,
        color=style.style_grey,
    )
    # format y-tick labels
    g.set_yticklabels(
        g.get_yticklabels() * 100 if "p" in y_units else g.get_yticklabels(),
        rotation=0,
        fontsize=1.05 * self.chart_scale,
        color=style.style_grey,
    )
    # format x-axis label
    g.set_xlabel(
        g.get_xlabel(),
        rotation=0,
        fontsize=1.35 * self.chart_scale,
        color=style.style_grey,
    )
    # format y-axis label
    g.set_ylabel(
        g.get_ylabel(),
        rotation=90,
        fontsize=1.35 * self.chart_scale,
        color=style.style_grey,
    )
    # format title
    g.set_title(
        g.get_title(),
        rotation=0,
        fontsize=1.5 * self.chart_scale,
        color=style.style_grey,
    )

    ## create custom legend
    # create labels
    if split is not None:
        if legend_labels is None:
            legend_labels = (
                df[df[split].notnull()][split]
                .sort_values()
                .drop_duplicates()
                .values.tolist()
            )
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
            fontsize=1.25 * self.chart_scale,
            loc="upper right",
            markerscale=0.5 * self.chart_scale,
            ncol=1,
            bbox_to_anchor=bbox,
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color="grey")

        # use label formatter utility function to customize chart labels
        util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)

def facet_cat_num_scatter(self, df, x, y, cat_row=None, cat_col=None, split=None, bbox=None, aspect=1, alpha=0.8,
                                height=4, legend_labels=None, x_units="f", y_units="f", color_map="viridis"):
    """
    Documentation:

        ---
        Description:
            Creates scatter plots of two numeric variables and allows for faceting by up to two
            categorical variables along the column and/or row axes of the figure.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data for plotting.
            x : str
                Numeric variable to plot along x-axis.
            y : str
                Numeric variable to plot along y-axis.
            cat_row : str
                Categorical variable faceted along the row axis.
            cat_col : str
                Categorical variable faceted along the column axis.
            split : str
                Categorical variable for faceting the num_col variable.
            bbox : tuple of floats, default=None
                Coordinates for determining legend position.
            aspect : float, default=1
                Higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            height : float, default=4
                Height in inches of each facet.
            legend_labels : list, default=None
                Custom legend labels.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p'
                displays percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd')
                for additional decimal places.
            y_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p'
                displays percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd')
                for additional decimal places.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
    """
    # create FacetGrid object
    g = sns.FacetGrid(
        df,
        col=cat_col,
        row=cat_row,
        hue=split,
        palette=sns.color_palette(
            style.color_gen(color_map, num=len(np.unique(df[split].values)))
        ),
        hue_order=df[split].sort_values().drop_duplicates().values.tolist()
        if split is not None
        else None,
        height=height,
        aspect=aspect,
        margin_titles=True,
    )

    # map scatter plot to FacetGrid object
    g = g.map(plt.scatter, x, y, s=1.2 * self.chart_scale)

    # format x any y ticklabels, x and y labels, and main title
    for ax in g.axes.flat:
        _ = ax.set_yticklabels(
            ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
            rotation=0,
            fontsize=0.8 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0,
            fontsize=0.8 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_ylabel(
            ax.get_ylabel(),
            rotation=90,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_xlabel(
            ax.get_xlabel(),
            rotation=0,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_title(
            ax.get_title(),
            rotation=0,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )

        # custom tick label formatting
        util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)

        if ax.texts:
            # this contains the right ylabel text
            txt = ax.texts[0]
            ax.text(
                txt.get_unitless_position()[0],
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize=1.05 * self.chart_scale,
                color=style.style_grey,
                rotation=-90,
            )
            # remove the original text
            ax.texts[0].remove()

    ## create custom legend
    # create labels
    if split is not None:
        if legend_labels is None:
            legend_labels = (
                df[df[split].notnull()][split]
                .sort_values()
                .drop_duplicates()
                .values.tolist()
            )
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

def facet_cat_num_hist(self, df, cat_row, cat_col, num_col, split, bbox=None, aspect=1, height=4, alpha=0.8,
                        legend_labels=None, x_units="f", y_units="f", color_map="viridis"):
    """
    Documentation:
        
        ---
        Description:
            Creates histograms of one numeric variable, and each can optionally be split by a category to
            show two or more distributions. Allows for faceting by up to two category variables along the
            column and/or row axes of the figure.
        
        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data for plotting.
            cat_row : str
                Categorical variable faceted along the row axis.
            cat_col : str
                Categorical variable faceted along the column axis.
            num_col : str
                number variable to plot along x_axis.
            split : str
                Categorical variable on which to differentiate the num_col variable.
            bbox : tuple of floats, default=None
                Coordinates for determining legend position.
            aspect : float, default=1
                higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            height : float, default=4
                height in inches of each facet.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            legend_labels : list, default=None
                Custom legend labels.
            x_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_units : str, default='f'
                Determines unit of measurement for x-axis tick labels. 'f' displays float. 'p' displays
                percentages, d' displays dollars. Repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.

    """
    # create FacetGrid object
    g = sns.FacetGrid(
        df,
        row=cat_row,
        col=cat_col,
        hue=split,
        hue_order=df[split].sort_values().drop_duplicates().values.tolist()
        if split is not None
        else None,
        palette=sns.color_palette(
            style.color_gen(color_map, num=len(np.unique(df[split].values)))
        ),
        despine=True,
        height=height,
        aspect=aspect,
        margin_titles=True,
    )

    # map histogram to FacetGrid object
    g.map(
        plt.hist,
        num_col,
        alpha=alpha,
    )

    # format x any y ticklabels, x and y labels, and main title
    for i, ax in enumerate(g.axes.flat):
        _ = ax.set_ylabel(
            ax.get_ylabel(),
            rotation=90,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_xlabel(
            ax.get_xlabel(),
            rotation=0,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_title(
            ax.get_title(),
            rotation=0,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )

        # resize y tick labels
        labels = ax.get_yticklabels()
        if len(labels) > 0:
            _ = ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=0.8 * self.chart_scale,
                color=style.style_grey,
            )
        # resize x tick labels
        labels = ax.get_xticklabels()
        if len(labels) > 0:
            _ = ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=0,
                fontsize=0.8 * self.chart_scale,
                color=style.style_grey,
            )

        if ax.texts:
            # this contains the right ylabel text
            txt = ax.texts[0]
            ax.text(
                txt.get_unitless_position()[0],
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize=1.05 * self.chart_scale,
                color=style.style_grey,
                rotation=-90,
            )
            # remove the original text
            ax.texts[0].remove()

    ## create custom legend
    # create labels
    if split is not None:
        if legend_labels is None:
            legend_labels = (
                df[df[split].notnull()][split]
                .sort_values()
                .drop_duplicates()
                .values.tolist()
            )
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

def facet_two_cat_point(self, df, x, y, split, cat_col=None, cat_row=None, bbox=None, aspect=1,
                                alpha=0.8, height=4, legend_labels=None, color_map="viridis"):
    """
    Documentation:
        
        ---
        Description:
            Creates pointplots of one categorical variable, and each can optionally be split by
            two additional categories along the column and/or row axes of the figure.
        
        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data for plotting.
            x : str
                Categorical variable to plot along x_axis.
            y : str
                Variable to be counted along y_axis.
            split : str
                Categorical variable for faceting the 'x' variable.
            cat_col : str
                Categorical variable faceted along the column axis.
            cat_row : str
                Categorical variable faceted along the row axis.
            bbox : tuple of floats, default=None
                Coordinates for determining legend position.
            aspect : float, default=1
                higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            alpha : float, default=0.8
                Controls transparency of objects. Accepts value between 0.0 and 1.0.
            height : float, default=4
                height in inches of each facet.
            legend_labels : list, default=None
                Custom legend labels.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
    """
    # create FacetGrid object
    g = sns.FacetGrid(
        df, row=cat_row, col=cat_col, aspect=aspect, height=height, margin_titles=True
    )

    # map pointplot to FacetGrid object
    g.map(
        sns.pointplot,
        x,
        y,
        split,
        order=df[x].sort_values().drop_duplicates().values.tolist(),
        hue_order=df[split].sort_values().drop_duplicates().values.tolist(),
        palette=sns.color_palette(
            style.color_gen(color_map, num=len(np.unique(df[split].values)))
        ),
        alpha=alpha,
        ci=None,
    )

    # format x any y ticklabels, x and y labels, and main title
    for ax in g.axes.flat:
        _ = ax.set_ylabel(
            ax.get_ylabel(),
            rotation=90,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_xlabel(
            ax.get_xlabel(),
            rotation=0,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )
        _ = ax.set_title(
            ax.get_title(),
            rotation=0,
            fontsize=1.05 * self.chart_scale,
            color=style.style_grey,
        )

        # resize y tick labels
        labels = ax.get_yticklabels()
        if len(labels) > 0:
            _ = ax.set_yticklabels(
                ax.get_yticklabels(),
                rotation=0,
                fontsize=0.8 * self.chart_scale,
                color=style.style_grey,
            )
        # resize x tick labels
        labels = ax.get_xticklabels()
        if len(labels) > 0:
            _ = ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=0,
                fontsize=0.8 * self.chart_scale,
                color=style.style_grey,
            )

        if ax.texts:
            # this contains the right ylabel text
            txt = ax.texts[0]

            ax.text(
                txt.get_unitless_position()[0],
                txt.get_unitless_position()[1],
                txt.get_text(),
                transform=ax.transAxes,
                va="center",
                fontsize=1.05 * self.chart_scale,
                color=style.style_grey,
                rotation=-90,
            )
            # remove the original text
            ax.texts[0].remove()

    ## create custom legend
    # create labels
    if legend_labels is None:
        legend_labels = np.unique(df[df[split].notnull()][split])
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
