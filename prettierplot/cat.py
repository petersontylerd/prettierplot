import numpy as np
import seaborn as sns
import squarify
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.patches import Patch

import prettierplot.style as style
import prettierplot.util as util


import textwrap


def bar_v(self, x, counts, color=style.style_grey, x_labels=None, x_tick_wrap=True, label_rotate=0,
                    y_units="f", alpha=0.8, ax=None):
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
            x_labels : list, default=None
                custom x_axis text labels.
            x_tick_wrap : bool, default=True
                wrap x_axis tick labels.
            label_rotate : float or int, default = 0
                degrees by which the xtick labels are rotated.
            y_units : string, default = 'f'
                determines units of y_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            alpha : float, default = 0.8
                controls transparency of bars. accepts value between 0.0 and 1.0.
            ax : axes object, default=None
                axis on which to place visual.
    """
    # custom labela
    labels = x_labels if x_labels is not None else x

    # create vertical bar plot.
    plt.bar(
        x=x,
        height=counts,
        color=color,
        tick_label=labels,
        alpha=alpha,
    )

    # rotate x_tick labels.
    plt.xticks(rotation=label_rotate)

    # resize x_axis labels as needed.
    if len(x) > 10 and len(x) <= 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.0 * self.chart_scale
        )
    elif len(x) > 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=0.9 * self.chart_scale
        )
    else:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.2 * self.chart_scale
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
        fontsize=1.2 * self.chart_scale,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, y_units=y_units)


def bar_h(self, y, counts, color=style.style_grey, label_rotate=45, x_units="f", alpha=0.8, ax=None):
    """
    documentation:
        description:
            create vertical bar plot.
        parameters:
            y : array
                1_dimensional array of values to be plotted on y-axis representing distinct categories.
            counts : array or string
                1_dimensional array of value counts for categories.
            color : string (some sort of color code), default=style.style_grey
                bar color.
            label_rotate : float or int, default = 45
                degrees by which the xtick labels are rotated.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            alpha : float, default = 0.8
                control transparency of bars. accepts value between 0.0 and 1.0.
            ax : axes object, default=None
                axis on which to place visual.
    """
    # plot horizontal bar plot.
    plt.barh(y=y, width=counts, color=color, tick_label=y, alpha=alpha)

    # rotate x_tick labels.
    plt.xticks(rotation=label_rotate)

    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in x_units else ax.get_xticklabels(),
        rotation=0,
        fontsize=0.9 * self.chart_scale,
        color=style.style_grey,
    )

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, x_units=x_units)

def stacked_bar_h(self, df, label_rotate=0, x_units="p", alpha=0.8, color_map="viridis", bbox=(1.2,0.9),
                        legend_labels=None, ax=None):
    """
    documentation:
        description:
            create vertical bar plot.
        parameters:
            df : Pandas DataFrame
                1_dimensional array of values to be plotted on y-axis representing distinct categories.
            label_rotate : float or int, default = 45
                degrees by which the xtick labels are rotated.
            x_units : string, default = 'f'
                determines units of x_axis tick labels. 's' displays string. 'f' displays float. 'p' displays
                percentages, 'd' displays dollars. repeat character (e.g 'ff' or 'ddd') for additional
                decimal places.
            alpha : float, default = 0.8
                control transparency of bars. accepts value between 0.0 and 1.0.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
            bbox : tuple of floats, default = (1.2, 0.9)
                coordinates for determining legend position.
            legend_labels : list, default=None
                custom legend labels.
            ax : axes object, default=None
                axis on which to place visual.
    """

    # define class label count and bar color list
    y = np.arange(len(df.index))
    color_list = style.color_gen(color_map, num=len(y))

    # define category labels
    category_levels = np.arange(len(df.columns))

    # plot stacked bars
    for class_label, color in zip(np.arange(len(y)), color_list):
        if class_label == 0:
            plt.barh(
                y=category_levels,
                width=df.loc[class_label],
                color=color,
                alpha=alpha,
            )
        # elif class_label == 1:
        else:
            plt.barh(
                y=category_levels,
                width=df.loc[class_label],
                left=df.drop([x for x in df.index if x >= class_label]).sum(axis=0),
                color=color,
                alpha=alpha,
            )

    # convert x-axis tick labels to percentages
    ax.set_xticklabels(
        ax.get_xticklabels() * 100 if "p" in x_units else ax.get_xticklabels(),
        rotation=0,
        color=style.style_grey,
    )

    ## create custom legend
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

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, x_units=x_units)

    # overwrite y-axis labels with category labels
    try:
        columns = df.columns.map(np.int)
    except ValueError:
        columns = df.columns

    #  dynamically size y-labels
    if 7 < len(category_levels) <= 10:
        ax.tick_params(axis="y", colors=style.style_grey, labelsize=0.9 * self.chart_scale)
    elif 10 < len(category_levels) <= 20:
        ax.tick_params(axis="y", colors=style.style_grey, labelsize=0.75 * self.chart_scale)
    elif len(category_levels) > 20:
        ax.tick_params(axis="y", colors=style.style_grey, labelsize=0.6 * self.chart_scale)

    ax.tick_params(axis="x", colors=style.style_grey, labelsize=1.2 * self.chart_scale)

    # custom x_tick labels.
    plt.yticks(
        category_levels,
        [
            "\n".join(textwrap.wrap(str(i).replace("_", " "), 12))
            for i in columns
        ],
    )

def box_plot_v(self, x, y, data, color, label_rotate=0, y_units="f", color_map="viridis", alpha=0.8,
                        suppress_outliers=False, ax=None):
    """
    documentation:
        description:
            create vertical box plots. useful for evaluated a number target on the y_axis
            vs. several different category segments on the x_axis
        parameters:
            x : string
                name of independent variable in dataframe. represents a category.
            y : string
                name of number target variable.
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
            alpha : float, default = 0.8
                controls transparency of objects. accepts value between 0.0 and 1.0.
            suppress_outliers : boolean, default=False
                controls removal of outliers from box/whisker plots
            ax : axes object, default=None
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
        showfliers=suppress_outliers,
        ax=ax,
    ).set(xlabel=None, ylabel=None)

    # # resize x_axis labels as needed.
    # unique = np.unique(data[x])
    # if len(unique) > 10 and len(unique) <= 20:
    #     ax.tick_params(axis="x", labelsize=0.9 * self.chart_scale)
    # elif len(unique) > 20:
    #     ax.tick_params(axis="x", labelsize=0.9 * self.chart_scale)

    # tick label font size
    ax.tick_params(axis="both", colors=style.style_grey, labelsize=1.2 * self.chart_scale)


    # resize x_axis labels as needed.
    unique = np.unique(data[x])
    if len(unique) > 10 and len(unique) <= 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.0 * self.chart_scale
        )
    elif len(unique) > 20:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=0.9 * self.chart_scale
        )
    else:
        ax.tick_params(
            axis="x", colors=style.style_grey, labelsize=1.2 * self.chart_scale
        )

    # resize y_axis
    ax.tick_params(axis="y", labelsize=1.2 * self.chart_scale)

    # fade box plot figures by reducing alpha.
    plt.setp(ax.artists, alpha=alpha)

    # rotate x_tick labels.
    plt.xticks(rotation=label_rotate)
    ax.yaxis.set_visible(True)

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, y_units=y_units)

def box_plot_h(self, x, y, data, color=style.style_grey, x_units="f", bbox=(1.05, 1), color_map="viridis",
                        suppress_outliers=False, alpha=0.8, legend_labels=None, ax=None):
    """
    documentation:
        description:
            create horizontal box plots. useful for evaluating a object target on the y_axis
            vs. a number independent variable on the x_axis.
        parameters:
            x : string
                name of independent variable in dataframe. represents a category.
            y : string
                name of number target variable.
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
            suppress_outliers : boolean, default=False
                controls removal of outliers from box/whisker plots
            alpha : float, default = 0.8
                controls transparency of bars. accepts value between 0.0 and 1.0.
            ax : axes object, default=None
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
        showfliers=suppress_outliers,
        ax=ax,
    ).set(xlabel=None, ylabel=None)

    # fade box plot figures by reducing alpha.
    plt.setp(ax.artists, alpha=alpha)
    ax.yaxis.set_visible(False)

    # tick label font size
    ax.tick_params(axis="both", colors=style.style_grey, labelsize=1.2 * self.chart_scale)

    # use label formatter utility function to customize chart labels.
    util.util_label_formatter(ax=ax, x_units=x_units)

    # legend placement
    if legend_labels is None:
        legend_labels = np.unique(data[y].values)
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

def tree_map(self, counts, labels, colors, alpha=0.8, ax=None):
    """

    """

    squarify.plot(
        sizes=counts,
        label=labels,
        color=colors,
        alpha=alpha,
        text_kwargs={
            "fontsize" : 1.2 * self.chart_scale,
            "color" : "black",
            # 'weight' : 'bold',
            },
        ax=ax,
    )
    plt.axis('off')