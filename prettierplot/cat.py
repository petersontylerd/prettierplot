import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm

import prettierplot.style as style
import prettierplot.util as util


import textwrap


def pretty_bar_v(
    self,
    x,
    counts,
    color=style.style_grey,
    x_labels=None,
    x_tick_wrap=True,
    label_rotate=0,
    y_units="f",
    ax=None,
):
    """
    documentation:
        description:
            create vertical bar plot.
        parameters:
            x : array
                1_dimensional array of values to be plotted on x_axis representing distinct categories.
            counts : array or string
                1_dimensional array of value counts for categories.
            color : string (some sort of color code), default = style.style_hex_mid[0]
                bar color.
            x_labels : list, default =None
                custom x_axis text labels.
            x_tick_wrap : boolean, default=True
                wrap x_axis tick labels.
            label_rotate : float or int, default = 0
                degrees by which the xtick labels are rotated.
            y_units : string, default = 'f'
                determines units of y_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # custom labela
    labels = x_labels if x_labels is not None else x

    # create vertical bar plot.
    plt.bar(
        x=x, height=counts, color=color, tick_label=labels, alpha=0.8,
    )

    # rotate x_tick labels.
    plt.xticks(rotation=label_rotate)

    # resize x_axis labels as needed.
    if len(x) > 10 and len(x) <= 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.0 * self.chart_prop
        )
    elif len(x) > 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=0.9 * self.chart_prop
        )
    else:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.2 * self.chart_prop
        )

    if x_tick_wrap and type(labels):
        try:
            x = ["\n".join(textwrap.wrap(i.replace("_", " "), 12)) for i in labels]
            ax.set_xticklabels(x)
        except AttributeError:
            pass

    # format y ticklabels
    ax.set_yticklabels(
        ax.get_yticklabels() * 100 if "p" in y_units else ax.get_yticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, y_units=y_units)


def pretty_bar_h(
    self, y, counts, color=style.style_grey, label_rotate=45, x_units="f", ax=None
):
    """
    documentation:
        description:
            create vertical bar plot.
        parameters:
            y : array
                1_dimensional array of values to be plotted on x_axis representing distinct categories.
            counts : array or string
                1_dimensional array of value counts for categories.
            color : string (some sort of color code), default = style.style_hex_mid[0]
                bar color.
            label_rotate : float or int, default = 45
                degrees by which the xtick labels are rotated.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # plot horizontal bar plot.
    plt.barh(y=y, width=counts, color=color, tick_label=y, alpha=0.8)

    # rotate x_tick labels.
    plt.xticks(rotation=label_rotate)

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in x_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_prop,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, x_units=x_units)


def pretty_box_plot_v(
    self, x, y, data, color, label_rotate=0, y_units="f", color_map="viridis", ax=None
):
    """
    documentation:
        description:
            create vertical box plots. useful for evaluated a numeric target on the y_axis
            vs. several different category segments on the x_axis
        parameters:
            x : string
                name of independent variable in dataframe. represents a category.
            y : string
                name of numeric target variable.
            data : pandas DataFrame
                pandas DataFrame including both indpedent variable and target variable.
            color : string
                determines color of box plot figures. ideally this object is a color palette,
                which can be a default seaborn palette, a custom seaborn palette, or a custom
                matplotlib cmap.
            label_rotate : float or int, default = 45
                degrees by which the xtick labels are rotated.
            y_units : string, default = 'f'
                determines units of y_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # create vertical box plot.
    g = sns.boxplot(
        x=x,
        y=y,
        data=data,
        orient="v",
        palette=sns.color_palette(
            style.color_gen(color_map, num=len(np.unique(data[x].values)))
        ),
        ax=ax,
    ).set(xlabel=None, ylabel=None)

    # # resize x_axis labels as needed.
    # unique = np.unique(data[x])
    # if len(unique) > 10 and len(unique) <= 20:
    #     ax.tick_params(axis="x", labelsize=0.9 * self.chart_prop)
    # elif len(unique) > 20:
    #     ax.tick_params(axis="x", labelsize=0.9 * self.chart_prop)

    # resize x_axis labels as needed.
    unique = np.unique(data[x])
    if len(unique) > 10 and len(unique) <= 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.0 * self.chart_prop
        )
    elif len(unique) > 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=0.9 * self.chart_prop
        )
    else:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.2 * self.chart_prop
        )

    # resize y_axis
    ax.tick_params(axis="y", labelsize=0.9 * self.chart_prop)

    # fade box plot figures by reducing alpha.
    plt.setp(ax.artists, alpha=0.8)

    # rotate x_tick labels.
    plt.xticks(rotation=label_rotate)
    ax.yaxis.set_visible(True)

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, y_units=y_units)


def pretty_box_plot_h(
    self,
    x,
    y,
    data,
    color=style.style_grey,
    x_units="f",
    bbox=(1.05, 1),
    color_map="viridis",
    ax=None,
):
    """
    documentation:
        description:
            create horizontal box plots. useful for evaluating a categorical target on the y_axis
            vs. a numeric independent variable on the x_axis.
        parameters:
            x : string
                name of independent variable in dataframe. represents a category.
            y : string
                name of numeric target variable.
            data : pandas DataFrame
                pandas DataFrame including both indpedent variable and target variable.
            color : string (some sort of color code), default = style.style_hex_mid
                determines color of box plot figures. ideally this object is a color palette,
                which can be a default seaborn palette, a custom seaborn palette, or a custom
                matplotlib cmap.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            bbox : tuple of floats, default = (1.05, 1.0)
                coordinates for determining legend position.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # create horizontal box plot.
    g = sns.boxplot(
        x=x,
        y=y,
        hue=y,
        data=data,
        orient="h",
        palette=sns.color_palette(
            style.color_gen(color_map, num=len(np.unique(data[y].values)))
        ),
        # palette=sns.color_palette(style.color_gen(color_map, num=len(np.unique(y)))),
        ax=ax,
    ).set(xlabel=None, ylabel=None)

    # fade box plot figures by reducing alpha.
    plt.setp(ax.artists, alpha=0.8)
    ax.yaxis.set_visible(False)

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, x_units=x_units)

    # legend placement.
    plt.legend(bbox_to_anchor=bbox, loc=2, borderaxespad=0.0)
