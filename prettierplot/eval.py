import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sklearn.metrics as metrics

from scipy import stats

import prettierplot.style as style
import prettierplot.util as util


def pretty_prob_plot(self, x, plot):
    """
        documentation:
            description:
                create plot that visualizes how well a numeric feature's distribution
                conforms to a normal distribution
            parameters:
                x : array
                    1_dimensional array containing data of a numeric feature.
                plot : plot object
                    plotting object for applying additional formatting.
        """
    stats.probplot(x, plot=plot)

    # override title and axis labels.
    plot.set_title("")
    plt.xlabel("")
    plt.ylabel("")

    # format scattered dots.
    plot.get_lines()[0].set_markerfacecolor(style.style_white)
    plot.get_lines()[0].set_color(style.style_grey)
    plot.get_lines()[0].set_markersize(5.0)

    # format line representing normality.
    plot.get_lines()[1].set_linewidth(3.0)
    plot.get_lines()[1].set_color(style.style_grey)


def pretty_corr_heatmap(
    self, df, annot=False, cols=None, mask=False, ax=None, vmin=-1.0, vmax=1.0
):
    """
    documentation:
        description:
            using numeric features, create correlation heatmap. produces correlation
            with all numerical features, and can be limited to certain features using 'cols'.
        parameters:
            df : pandas DataFrame
                pandas DataFrame containing all features of interest. will be transformed into
                a correlation matrix.
            annot : boolean, default=False
                determines whether or not correlation table is annotated with correlation
                value or not.
            cols : list, default =None
                list of strings describing dataframe columns. limits dataframe to select columns.
            mask : boolean, default=False
                determines whether or not correlation table is masked such that only the lower
                triangle appears.
            ax : axes object, default =None
                axis on which to place visual.
            vmin : float, default = _1.0
                minimum anchor value for color map.
            vmax : float, default = 1.0
                maximum anchor value for color map.
    """
    # create correlation matrix
    corr_matrix = df[cols].corr() if cols is not None else df.corr()
    cols = corr_matrix.columns

    # generate a mask for the upper triangle
    mask_grid = np.zeros_like(corr_matrix, dtype=np.bool)
    mask_grid[np.triu_indices_from(mask_grid)] = True

    # adjust font size as needed
    if len(cols) <= 5:
        font_adjust = 1.25
    elif len(cols) > 5 and len(cols) <= 10:
        font_adjust = 0.95
    elif len(cols) > 10 and len(cols) <= 20:
        font_adjust = 0.85
    elif len(cols) > 20 and len(cols) <= 30:
        font_adjust = 0.75
    elif len(cols) > 30 and len(cols) <= 40:
        font_adjust = 0.65
    else:
        font_adjust = 0.45

    # create heatmap using correlation matrix.
    g = sns.heatmap(
        corr_matrix,
        mask=mask_grid if mask else None,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        annot_kws={"size": font_adjust * self.chart_prop},
        square=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cmap="viridis",
    )

    # format x_tick and y_tick labels
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=0.8 * self.chart_prop)
    g.set_xticklabels(g.get_xticklabels(), rotation=-90, fontsize=0.8 * self.chart_prop)

    # workaround for matplotlib 3.1.1 bug
    if matplotlib.__version__ == "3.1.1":
        g.set_ylim(corr_matrix.shape[1] + 0.1, -0.1)

    # customize color bar formatting and labeling.
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=font_adjust * self.chart_prop, colors=style.style_grey, length=0
    )
    cbar.set_ticks([vmax, 0.0, vmin])


def pretty_corr_heatmap_target(
    self, df, target=None, annot=False, thresh=0.2, ax=None, vmin=-1.0, vmax=1.0
):
    """
    documentation:
        description:
            using numeric features, create correlation heatmap. capable of dropping
            zeros in select features, where zeros potentially indicate a complete absence
            of the feature.
        parameters:
            df : pandas DataFrame
                pandas DataFrame containing all features of interest. will be transformed into
                a correlation matrix.
            annot : boolean, default=False
                determines whether or not correlation table is annotated with correlation
                value or not.
            cols : list, default =None
                list of strings describing dataframe columns. limits dataframe to select columns.
            thresh : float, default = 0.2
                minimum correlation coefficient value needed.
            corr_focus : string, default = self.target[0]
                the feature of focus in the supplemental correlation visualization. used
                to determine the feature for which the nlargest correlation coefficients
                are returned.
            ax : axes object, default =None
                axis on which to place visual.
            vmin : float, default = _1.0
                minimum anchor value for color map.
            vmax : float, default = 1.0
                maximum anchor value for color map.
    """
    df = df.merge(target, left_index=True, right_index=True)

    # limit to top correlated features relative to specified target.
    corr_matrix = df.corr()
    corr_top = corr_matrix[target.name]  # [:_1]
    corr_top = corr_top[abs(corr_top) > thresh].sort_values(ascending=False)

    if len(corr_top) <= 5:
        font_adjust = 1.25
    elif len(corr_top) > 5 and len(corr_top) <= 10:
        font_adjust = 1.15
    elif len(corr_top) > 10 and len(corr_top) <= 20:
        font_adjust = 1.05
    elif len(corr_top) > 20 and len(corr_top) <= 30:
        font_adjust = 0.95
    elif len(corr_top) > 30 and len(corr_top) <= 40:
        font_adjust = 0.85
    else:
        font_adjust = 0.65

    # create heatmap using correlation matrix.
    g = sns.heatmap(
        df[corr_top.index].corr().iloc[:, :1],
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        annot_kws={"size": font_adjust * self.chart_prop},
        square=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cmap="viridis",
    )

    # format y_tick labels and turn off xticks.
    g.set_yticklabels(
        g.get_yticklabels(), rotation=0, fontsize=font_adjust * self.chart_prop
    )
    plt.xticks([])

    # # workaround for matplotlib 3.1.1 bug
    # if matplotlib.__version__ == "3.1.1":
    #     g.set_ylim(corr_matrix.shape[1] + 0.1, _0.1)

    # customize color bar formatting and labeling.
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=font_adjust * self.chart_prop, colors=style.style_grey, length=0
    )
    cbar.set_ticks([vmax, 0.0, vmin])

    plt.show()


def pretty_confusion_matrix(
    self,
    y_pred,
    y_true,
    labels,
    cmap="viridis",
    ax=None,
    textcolors=["black", "white"],
    threshold=None,
    reverse_labels=False,
    valfmt="{x:.0f}",
):
    """
    documentation:
        description:

        parameters:
            reverse_labels : boolean, default=False
                reverse the direction of the labels. puts the True positives in the upper left hand corner in
                binary classification problems.

    """
    if ax is None:
        ax = plt.gca()

    # create confusion matrix using predictions and True labels
    cm = pd.DataFrame(metrics.confusion_matrix(y_true=y_true, y_pred=y_pred))

    # sort rows and columns in descending order
    # cm.sort_index(axis = 1, ascending=False, inplace=True)
    # cm.sort_index(axis = 0, ascending=False, inplace=True)

    # plot heatmap and color bar
    im = ax.imshow(cm, cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)

    # set ticks and custom labels
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(labels, fontsize=15)
    ax.set_yticklabels(labels, fontsize=15)

    # customize tick and label positions
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

    # turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)

    if not isinstance(cm, (list, np.ndarray)):
        cm = im.get_array()

    # normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(cm.max()) / 2.0

    # set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")

    # get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = tkr.StrMethodFormatter(valfmt)

    # loop over the cm and create a `text` for each "pixel".
    # change the text's color depending on the cm.
    texts = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            kw.update(color=textcolors[int(im.norm(cm[i, j]) < threshold)])
            # kw.update(color=textcolors[int(im.norm(cm[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(cm[i, j], None), **kw)
            texts.append(text)

    #
    ax.set_ylabel(
        ax.get_ylabel(), rotation=-90, fontsize=18, color=style.style_grey,
    )
    ax.set_xlabel(
        ax.get_xlabel(), rotation=0, fontsize=18, color=style.style_grey,
    )

    plt.show()


def pretty_roc_curve(
    self,
    model,
    x_train,
    y_train,
    x_valid=None,
    y_valid=None,
    linecolor=style.style_grey,
    bbox=(1.0, 0.4),
    ax=None,
):
    """
    documentation:
        description:
            plot roc curve and report auc in
        parameters:
            model : sklearn model or pipeline
                model to fit and generate prediction probabilities.
            x_train : array
                training data for model fitting. also used to return predict_probas
                when x_valid is None.
            y_train : array
                training labels for model fitting. also used to create roc curve when
                x_valid is None.
            x_valid : array, default =None
                test data for returning predict_probas.
            y_valid : array, default =None
                test data for creating roc curve
            linecolor : str, default = style.style_hex_mid[0]
                curve line color
            bbox : tuple of floats, default = (1.2, 0.8)
                coordinates for determining legend position
            ax : axes object, default =None
                axis on which to place visual.
    """
    ## return prediction probabilities.
    # if x_test is None then fit the model using training data and return roc curve for training data.
    if x_valid is None:
        probas = model.fit(x_train, y_train).predict_proba(x_train)
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=y_train, y_score=probas[:, 1], pos_label=1
        )
    # otherwise fit the model using training data and return roc curve for test data.
    else:
        probas = model.fit(x_train, y_train).predict_proba(x_valid)
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=y_valid, y_score=probas[:, 1], pos_label=1
        )

    # calculate area under the curve using fpr and tpr.
    roc_auc = metrics.auc(fpr, tpr)

    # plot roc curve.
    self.pretty_line(
        x=fpr,
        y=tpr,
        label="auc: {:.4f}".format(roc_auc),
        linecolor=linecolor,
        x_units="fff",
        y_units="fff",
        bbox=bbox,
        ax=ax,
    )

    # plot 'random guess' line for reference.
    self.pretty_line(
        x=np.array([0, 1]),
        y=np.array([0, 1]),
        linecolor=style.style_grey,
        linestyle="__",
        x_units="fff",
        y_units="fff",
        ax=ax,
    )

    # plot 'perfection' line for reference.
    self.pretty_line(
        x=np.array([0, 0, 1]),
        y=np.array([0, 1, 1]),
        linecolor=style.style_grey,
        linestyle=":",
        x_units="fff",
        y_units="fff",
        ax=ax,
    )


def pretty_decision_region(
    self,
    x,
    y,
    classifier,
    test_idx=None,
    resolution=0.1,
    bbox=(1.2, 0.9),
    color_map="viridis",
    ax=None,
):
    """
    documentation:
        description:
            create 2_dimensional chart with shading used to highlight decision regions.
        parameters:
            x : array
                m x 2 array containing 2 features.
            y : array
                m x 1 array containing labels for observations.
            classifier : sklearn model or pipeline
                classifier used to create decision regions.
            test_idx :  tuple, default =None
                optional parameter for specifying observations to be highlighted as test examples.
            resolution : float, default = 0.1
                controls clarity of the graph by setting interval of the arrays passed into np.meshgrid.
            bbox : tuple of floats, default = (1.2, 0.9)
                coordinates for determining legend position.
            color_map : string specifying built_in matplotlib colormap, default = "viridis"
                colormap from which to draw plot colors.
            ax : axes object, default =None
                axis on which to place visual.
    """
    # generate color list
    color_list = style.color_gen(name=color_map, num=len(np.unique(y)))

    # objects for marker generator and color map
    cmap = ListedColormap(color_list)
    # cmap = ListedColormap(style.style_hex_light[: len(np.unique(y))])

    # plot decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    # generate predictions using classifier for all points on grid
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # reshape the predictions and apply coloration
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=x[y == cl, 0],
            y=x[y == cl, 1],
            alpha=1.0,
            c=color_list[idx],
            marker=style.style_markers[1],
            label=cl,
            s=12.5 * self.chart_prop,
            # edgecolor=style.style_hex_mid_dark[idx],
        )

    # highlight test samples
    if test_idx:
        x_test = x[test_idx, :]
        plt.scatter(
            x_test[:, 0],
            x_test[:, 1],
            facecolor="none",
            edgecolor="white",
            alpha=1.0,
            linewidth=1.4,
            marker="o",
            s=12.75 * self.chart_prop,
            label="test set",
        )
    # add legend to figure
    plt.legend(
        loc="upper right",
        bbox_to_anchor=bbox,
        ncol=1,
        frameon=True,
        fontsize=1.1 * self.chart_prop,
    )

    plt.tight_layout()
