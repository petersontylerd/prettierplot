import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import textwrap

from sklearn.metrics import (
    auc,
    precision_score,
    recall_score,
    f1_score,
    explained_variance_score,
    mean_squared_log_error,
    mean_absolute_error,
    median_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    roc_curve,
    accuracy_score,
    roc_auc_score,
    homogeneity_score,
    completeness_score,
    classification_report,
    silhouette_samples,
)

from scipy import stats

import prettierplot.style as style
import prettierplot.util as util


def prob_plot(self, x, plot):
    """
        Documentation:

            ---
            Description:
                Create QQ plot that visualizes how well a numeric feature's distribution
                conforms to a normal distribution

            --
            Parameters:
                x : array
                    1-dimensional array containing data of a numeric feature.
                plot : plot object
                    Plotting object for applying additional formatting.
        """
    stats.probplot(x, plot=plot)

    # override title and axis labels.
    plot.set_title("")
    plt.xlabel("")
    plt.ylabel("")

    # format scattered dots.
    plot.get_lines()[0].set_markerfacecolor(style.style_white)
    plot.get_lines()[0].set_color(style.style_grey)
    plot.get_lines()[0].set_markersize(0.4 * self.chart_scale)

    # format line representing normality.
    plot.get_lines()[1].set_linewidth(0.15 * self.chart_scale)
    plot.get_lines()[1].set_color(style.style_grey)

    # tick label font size
    plot.tick_params(axis="both", colors=style.style_grey, labelsize=1.1 * self.chart_scale)

def corr_heatmap(self, df, annot=False, columns=None, mask=False, color_map="viridis", vmin=-1.0, vmax=1.0,
                        ax=None):
    """
    Documentation:

        ---
        Description:
            using number features, create correlation heatmap. produces correlation
            with all numberal features, and can be limited to certain features using 'columns'.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing all features of interest.
            annot : bool, default=False
                Determines whether or not correlation table is annotated with correlation
                coefficients.
            columns : list, default=None
                List of strings describing DataFrame columns. Limits DataFrame to select columns.
            mask : bool, default=False
                Determines whether or not correlation table is masked such that only the lower
                triangle appears.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            vmin : float, default=_1.0
                Minimum anchor value for color map.
            vmax : float, default=1.0
                Maximum anchor value for color map.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    # create correlation matrix
    corr_matrix = df[columns].corr() if columns is not None else df.corr()
    columns = corr_matrix.columns

    # generate a mask for the upper triangle
    mask_grid = np.zeros_like(corr_matrix, dtype=np.bool)
    mask_grid[np.triu_indices_from(mask_grid)] = True

    # dynamically adjust font size based on number of columns in dataset
    if len(columns) <= 5:
        font_adjust = 1.25
    elif len(columns) > 5 and len(columns) <= 10:
        font_adjust = 0.95
    elif len(columns) > 10 and len(columns) <= 20:
        font_adjust = 0.85
    elif len(columns) > 20 and len(columns) <= 30:
        font_adjust = 0.75
    elif len(columns) > 30 and len(columns) <= 40:
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
        annot_kws={"size": font_adjust * self.chart_scale},
        square=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cmap=color_map,
    )

    # format x_tick and y_tick labels
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=font_adjust * self.chart_scale)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=font_adjust * self.chart_scale)

    # wrap lables if necessary
    x_labels =[item.get_text() for item in ax.get_xticklabels()]
    y_labels =[item.get_text() for item in ax.get_yticklabels()]

    # wrap long x-tick labels
    plt.xticks(
        np.arange(len(x_labels)) + 0.5,
        [
            "\n".join(textwrap.wrap(str(i).replace("_", " "), 12))
            for i in x_labels
        ],
        ha="center",
    )

    # wrap long y-tick labels
    plt.yticks(
        np.arange(len(y_labels)) + 0.5,
        [
            "\n".join(textwrap.wrap(str(i).replace("_", " "), 12))
            for i in y_labels
        ],
        va="center_baseline",
    )

    # customize color bar formatting and labeling.
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=font_adjust * self.chart_scale, colors=style.style_grey, length=0
    )
    cbar.set_ticks([vmax, 0.0, vmin])

def corr_heatmap_target(self, df, target, annot=False, thresh=0.2, color_map="viridis", vmin=-1.0, vmax=1.0,
                                ax=None):
    """
    Documentation:

        ---
        Description:
            Create correlation heatmap that visualizes correlation coefficients relative to one
            target feature.

        ---
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing all features of interest.
            target : str
                The focus feature in the visualization. Output is limited to correlation
                coefficients relative to this feature.
            annot : bool, default=False
                Determines whether or not correlation table is annotated with correlation
                coefficient.
            thresh : float, default=0.2
                Minimum correlation coefficient value required to be in the visualization.
            columns : list, default=None
                List of strings describing DataFrame columns. Limits DataFrame to select
                columns.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            vmin : float, default=_1.0
                Minimum anchor value for color map.
            vmax : float, default=1.0
                Maximum anchor value for color map.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    # combine dataset of independent variables with target variable
    df = df.merge(target, left_index=True, right_index=True)

    # create correlation coefficient matrix, limit to target feature, and
    # filter by threshold values.
    corr_matrix = df.corr()
    corr_top = corr_matrix[target.name]  # [:_1]
    corr_top = corr_top[abs(corr_top) > thresh].sort_values(ascending=False)

    # dynamically adjust font size based on number of columns in dataset
    if len(corr_top) <= 5:
        font_adjust = 1.90
    elif len(corr_top) > 5 and len(corr_top) <= 10:
        font_adjust = 1.80
    elif len(corr_top) > 10 and len(corr_top) <= 20:
        font_adjust = 1.70
    elif len(corr_top) > 20 and len(corr_top) <= 30:
        font_adjust = 1.60
    elif len(corr_top) > 30 and len(corr_top) <= 40:
        font_adjust = 1.50
    else:
        font_adjust = 1.40

    # create heatmap using correlation matrix
    g = sns.heatmap(
        df[corr_top.index].corr().iloc[:, :1],
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        annot_kws={"size": font_adjust * self.chart_scale},
        square=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cmap=color_map,
    )

    # format y-tick labels and turn off xticks
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=font_adjust * self.chart_scale)
    plt.xticks([])

    # customize color bar formatting and labeling
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=font_adjust * self.chart_scale, colors=style.style_grey, length=0
    )
    cbar.set_ticks([vmax, 0.0, vmin])

    plt.show()

def roc_curve_plot(self, model, X_train, y_train, X_valid=None, y_valid=None, linecolor=style.style_grey,
                        bbox=(1.0, 0.4), ax=None):
    """
    Documentation:

        ---
        Description:
            Plot ROC curve and display AUC in legend.

        ---
        Parameters:
            model : sklearn model or pipeline
                Model to fit and generate prediction probabilities.
            X_train : array
                Training data for model fitting. Also used to return predict_probas
                when X_valid is None.
            y_train : array
                Training labels for model fitting. also used to create ROC curve when
                X_valid is None.
            X_valid : array, default=None
                Test data for returning predict_probas.
            y_valid : array, default=None
                Test data for creating ROC curve
            linecolor : str, default=style.style_grey
                Curve line color
            bbox : tuple of floats, default=(1.0, 0.4)
                Coordinates for determining legend position
            ax : axes object, default=None
                Axis object for the visualization.
    """
    ## return prediction probabilities
    # if X_valid is None then fit the model using training data and return ROC curve for training data
    if X_valid is None:
        probas = model.fit(X_train, y_train).predict_proba(X_train)
        fpr, tpr, thresholds = roc_curve(
            y_true=y_train, y_score=probas[:, 1], pos_label=1
        )
    # otherwise fit the model using training data and return ROC curve for validation data
    else:
        probas = model.fit(X_train, y_train).predict_proba(X_valid)
        fpr, tpr, thresholds = roc_curve(
            y_true=y_valid, y_score=probas[:, 1], pos_label=1
        )

    # calculate area under the curve using fpr and tpr
    roc_auc = auc(fpr, tpr)

    # plot ROC curve
    self.line(
        x=fpr,
        y=tpr,
        label="AUC: {:.4f}".format(roc_auc),
        linecolor=linecolor,
        x_units="fff",
        y_units="fff",
        bbox=bbox,
        ax=ax,
    )

    # plot 'random guess' line for reference
    self.line(
        x=np.array([0, 1]),
        y=np.array([0, 1]),
        linecolor=style.style_grey,
        linestyle="--",
        x_units="fff",
        y_units="fff",
        ax=ax,
    )

    # plot 'perfection' line for reference
    self.line(
        x=np.array([0, 0, 1]),
        y=np.array([0, 1, 1]),
        linecolor=style.style_grey,
        linestyle=":",
        x_units="fff",
        y_units="fff",
        ax=ax,
    )

def decision_region(self, x, y, estimator, test_idx=None, resolution=0.1, bbox=(1.2, 0.9),
                            color_map="viridis", ax=None):
    """
    Documentation:
        Description:
            Create 2-dimensional chart with shading used to highlight decision regions.
        Parameters:
            x : array
                m x 2 array containing 2 features.
            y : array
                m x 1 array containing labels for observations.
            estimator : sklearn model
                Estimator used to create decision regions.
            test_idx :  tuple, default=None
                Optional parameter for specifying observations to be highlighted as test examples.
            resolution : float, default=0.1
                Controls clarity of the graph by setting interval of the arrays passed into np.meshgrid.
                Higher resolution will take longer to generate because predictions have to be generated
                for each point on the grid.
            bbox : tuple of floats, default=(1.2, 0.9)
                Coordinates for determining legend position.
            color_map : str specifying built-in matplotlib colormap, default="viridis"
                Color map applied to plots.
            ax : axes object, default=None
                Axis object for the visualization.
    """
    # generate color list
    color_list = style.color_gen(name=color_map, num=len(np.unique(y)))

    # objects for marker generator and color map
    cmap = ListedColormap(color_list)

    # plot decision surface
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1

    # generate meshgrid indices
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    # generate predictions using estimator for all points on grid
    z = estimator.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

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
            s=12.5 * self.chart_scale,
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
            s=12.75 * self.chart_scale,
            label="test set",
        )

    # add legend to figure
    plt.legend(
        loc="upper right",
        bbox_to_anchor=bbox,
        ncol=1,
        frameon=True,
        fontsize=1.1 * self.chart_scale,
    )

    plt.tight_layout()
