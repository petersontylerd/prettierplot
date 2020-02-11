import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import prettierplot.style as style
import prettierplot.util as util

import textwrap


def facet_cat(self, df, feature, label_rotate=0, y_units="f", x_units="s", bbox=(1.2, 0.9),
                        alpha=0.8, legend_labels=None, color_map="viridis", ax=None):
    """
    Documentation:
        Description:
            creates a count plot for a object variable and facets the variable by a
            object label.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            feature : string
                string describing column name containing target values
            label_rotate : float or int, default=0
                degrees by which the xtick labels are rotated.
            x_units : string, default='f'
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_units : string, default='s'
                determines units of y_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            bbox : tuple of floats, default=(1.2, 0.9)
                coordinates for determining legend position.
            alpha : float, default=0.8
                controls transparency of objects. accepts value between 0.0 and 1.0.
            legend_labels : list, default=None
                custom legend labels.
            color_map : string specifying built_in matplotlib colormap, default="viridis"
                colormap from which to draw plot colors.
            ax : axes object, default=None
                axis on which to place visual.
    """
    ixs = np.arange(df.shape[0])
    bar_width = 0.35

    feature_dict = {}
    for feature in df.columns[1:]:
        feature_dict[feature] = df[feature].values.tolist()

    # generate color list
    color_list = style.color_gen(name=color_map, num=len(feature_dict.keys()))

    for feature_ix, (k, v) in enumerate(feature_dict.items()):
        plt.bar(
            ixs + (bar_width * feature_ix),
            feature_dict[k],
            bar_width,
            alpha=alpha,
            color=color_list[feature_ix],
            label=str(k),
        )

    # custom x_tick labels.
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

    # create patches
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
    # use label formatter utility function to customize chart labels

    if df.iloc[:, 0].values.dtype == np.float:
        x_units = "f"
    else:
        x_units = "s"

    util.util_label_formatter(ax=ax, x_units=x_units, y_units=y_units)

    # tick label font size
    ax.tick_params(axis="both", colors=style.style_grey, labelsize=1.2 * self.chart_scale)

    # resize x_axis labels as needed.
    if 7 < len(feature_dict[feature]) <= 10:
        ax.tick_params(axis="x", colors=style.style_grey, labelsize=0.9 * self.chart_scale)
    elif 10 < len(feature_dict[feature]) <= 20:
        ax.tick_params(axis="x", colors=style.style_grey, labelsize=0.75 * self.chart_scale)
    elif len(feature_dict[feature]) > 20:
        ax.tick_params(axis="x", colors=style.style_grey, labelsize=0.6 * self.chart_scale)


def facet_two_cat_bar(self, df, x, y, split, x_units=None, y_units=None, bbox=None, alpha=0.8,
                        legend_labels=None, filter_na_n=True, color_map="viridis", ax=None):
    """
    Documentation:
        Description:
            creates a series of bar plots that count a variable along the y_axis and separate the counts
            into bins based on by two object variables.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            x : string
                object variable to be plotted along x_axis.
            y : string
                variable to be counted along y_axis.
            split : string
                object variable on which to differentiate the num_col variable.
            x_units : string, default=None
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            y_units : string, default=None
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            alpha : float, default=0.8
                controls transparency of objects. accepts value between 0.0 and 1.0.
            bbox : tuple of floats, default=None
                coordinates for determining legend position.
            legend_labels : list, default=None
                custom legend labels.
            filter_nan : bool, default=True
                remove record that have a null value in the column specified by the 'x' parameter.
            color_map : string specifying built_in matplotlib colormap, default="viridis"
                colormap from which to draw plot colors.
            ax : axes object, default=None
                axis on which to place visual.
    """
    if filter_na_n:
        df = df.dropna(subset=[x])

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

    # format x and y_tick labels
    g.set_yticklabels(
        g.get_yticklabels() * 100 if "p" in y_units else g.get_yticklabels(),
        rotation=0,
        fontsize=1.05 * self.chart_scale,
        color=style.style_grey,
    )
    g.set_xticklabels(
        g.get_xticklabels(),
        rotation=0,
        fontsize=1.05 * self.chart_scale,
        color=style.style_grey,
    )
    g.set_ylabel(
        g.get_ylabel(),
        rotation=90,
        fontsize=1.35 * self.chart_scale,
        color=style.style_grey,
    )
    g.set_xlabel(
        g.get_xlabel(),
        rotation=0,
        fontsize=1.35 * self.chart_scale,
        color=style.style_grey,
    )
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

        # create patches
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

    #plt.show()


def facet_cat_num_scatter(self, df, x, y, cat_row=None, cat_col=None, split=None, bbox=None, aspect=1, alpha=0.8,
                                height=4, legend_labels=None, x_units="f", y_units="f", color_map="viridis"):
    """
    Documentation:
        Description:
            creates scatter plots of two number variables and allows for faceting by up to two
            object variables along the column and/or row axes of the figure.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            x : string
                number variable to be plotted along x_axis.
            y : string
                number variable to be plotted along y_axis.
            cat_row : string
                object variable faceted along the row axis.
            cat_col : string
                object variable faceted along the column axis.
            split : string
                object variable on which to differentiate the num_col variable.
            bbox : tuple of floats, default=None
                coordinates for determining legend position.
            aspect : float, default=1
                higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            alpha : float, default=0.8
                controls transparency of objects. accepts value between 0.0 and 1.0.
            height : float, default=4
                height in inches of each facet.
            legend_labels : list, default=None
                custom legend labels.
            x_units : string, default='f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_units : string, default='f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            color_map : string specifying built_in matplotlib colormap, default="viridis"
                colormap from which to draw plot colors.
    """
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

        # create patches
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

    #plt.show()


def facet_cat_num_hist(self, df, cat_row, cat_col, num_col, split, bbox=None, aspect=1, height=4, alpha=0.8,
                                legend_labels=None, x_units="f", y_units="f", color_map="viridis"):
    """
    Documentation:
        Description:
            creates histograms of one number variable, and each can optionally be split by a object to
            show two or more distributions. allows for faceting by up to two object variables along the
            column and/or row axes of the figure.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            cat_row : string
                object variable faceted along the row axis.
            cat_col : string
                object variable faceted along the column axis.
            num_col : string
                number variable to be plotted along x_axis.
            split : string
                object variable on which to differentiate the num_col variable.
            bbox : tuple of floats, default=None
                coordinates for determining legend position.
            aspect : float, default=1
                higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            height : float, default=4
                height in inches of each facet.
            alpha : float, default=0.8
                controls transparency of objects. accepts value between 0.0 and 1.0.
            legend_labels : list, default=None
                custom legend labels.
            x_units : string, default='f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            y_units : string, default='f'
                determines units of x_axis tick labels. 'f' displays float. 'p' displays percentages,
                'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional decimal places.
            color_map : string specifying built_in matplotlib colormap, default="viridis"
                colormap from which to draw plot colors.
    """
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
    g.map(
        plt.hist,
        num_col,
        alpha=alpha,
    )

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

        # create patches
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

    #plt.show()


def facet_two_cat_point(self, df, x, y, split, cat_col=None, cat_row=None, bbox=None, aspect=1,
                                alpha=0.8, height=4, legend_labels=None, color_map="viridis"):
    """
    Documentation:
        Description:
            creates point plots that
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            x : string
                object variable to be plotted along x_axis.
            y : string
                variable to be counted along y_axis.
            split : string
                object variable on which to differentiate the 'x' variable.
            cat_row : string
                object variable faceted along the row axis.
            cat_col : string
                object variable faceted along the column axis.
            bbox : tuple of floats, default=None
                coordinates for determining legend position.
            aspect : float, default=1
                higher values create wider plot, lower values create narrow plot, while
                keeping height constant.
            alpha : float, default=0.8
                controls transparency of objects. accepts value between 0.0 and 1.0.
            height : float, default=4
                height in inches of each facet.
            legend_labels : list, default=None
                custom legend labels.
            color_map : string specifying built_in matplotlib colormap, default="viridis"
                colormap from which to draw plot colors.
    """
    g = sns.FacetGrid(
        df, row=cat_row, col=cat_col, aspect=aspect, height=height, margin_titles=True
    )
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

    # create patches
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

    #plt.show()
