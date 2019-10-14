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


def prettyProbPlot(self, x, plot):
    """
        Documentation:
            Description:
                Create plot that visualizes how well a numeric feature's distribution
                conforms to a normal distribution
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
    plot.get_lines()[0].set_markerfacecolor(style.styleWhite)
    plot.get_lines()[0].set_color(style.styleGrey)
    plot.get_lines()[0].set_markersize(5.0)

    # format line representing normality.
    plot.get_lines()[1].set_linewidth(3.0)
    plot.get_lines()[1].set_color(style.styleGrey)


def prettyCorrHeatmap(self, df, annot=False, cols=None, mask = False, ax=None, vmin=-1.0, vmax=1.0):
    """
    Documentation:
        Description:
            Using numeric features, create correlation heatmap. Produces correlation
            with all numerical features, and can be limited to certain features using 'cols'.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing all features of interest. Will be transformed into
                a correlation matrix.
            annot : boolean, default = False
                Determines whether or not correlation table is annotated with correlation
                value or not.
            cols : list, default = None
                List of strings describing dataframe columns. Limits dataframe to select columns.
            mask : boolean, default = False
                Determines whether or not correlation table is masked such that only the lower
                triangle appears.
            ax : Axes object, default = None
                Axis on which to place visual.
            vmin : Float, default = -1.0
                Minimum anchor value for color map.
            vmax : Float, default = 1.0
                Maximum anchor value for color map.
    """
    # create correlation matrix
    corrMatrix = df[cols].corr() if cols is not None else df.corr()
    cols = corrMatrix.columns

    # generate a mask for the upper triangle
    maskGrid = np.zeros_like(corrMatrix, dtype=np.bool)
    maskGrid[np.triu_indices_from(maskGrid)] = True

    # adjust font size as needed
    if len(cols) <= 5:
        fontAdjust = 1.25
    elif len(cols) > 5 and len(cols) <= 10:
        fontAdjust = 0.95
    elif len(cols) > 10 and len(cols) <= 20:
        fontAdjust = 0.85
    elif len(cols) > 20 and len(cols) <= 30:
        fontAdjust = 0.75
    elif len(cols) > 30 and len(cols) <= 40:
        fontAdjust = 0.65
    else:
        fontAdjust = 0.45

    # create heatmap using correlation matrix.
    g = sns.heatmap(
        corrMatrix,
        mask=maskGrid if mask else None,
        vmin=vmin,
        vmax=vmax,
        annot=annot,
        annot_kws={"size": fontAdjust * self.chartProp},
        square=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cmap="viridis",
    )

    # format x-tick and y-tick labels
    g.set_yticklabels(g.get_yticklabels(), rotation=0, fontsize=0.8 * self.chartProp)
    g.set_xticklabels(g.get_xticklabels(), rotation=90, fontsize=0.8 * self.chartProp)

    # workaround for matplotlib 3.1.1 bug
    if matplotlib.__version__ == "3.1.1":
        g.set_ylim(corrMatrix.shape[1] + 0.1, -0.1)

    # customize color bar formatting and labeling.
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=fontAdjust * self.chartProp, colors=style.styleGrey, length=0
    )
    cbar.set_ticks([vmax, 0.0, vmin])


def prettyCorrHeatmapTarget(self, df, target=None, annot=False, thresh=0.2, ax=None, vmin=-1.0, vmax=1.0):
    """
    Documentation:
        Description:
            Using numeric features, create correlation heatmap. Capable of dropping
            zeros in select features, where zeros potentially indicate a complete absence
            of the feature.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing all features of interest. Will be transformed into
                a correlation matrix.
            annot : boolean, default = False
                Determines whether or not correlation table is annotated with correlation
                value or not.
            cols : list, default = None
                List of strings describing dataframe columns. Limits dataframe to select columns.
            thresh : float, default = 0.2
                Minimum correlation coefficient value needed.
            corrFocus : string, default = self.target[0]
                The feature of focus in the supplemental correlation visualization. Used
                to determine the feature for which the nlargest correlation coefficients
                are returned.
            ax : Axes object, default = None
                Axis on which to place visual.
            vmin : Float, default = -1.0
                Minimum anchor value for color map.
            vmax : Float, default = 1.0
                Maximum anchor value for color map.
    """
    df = df.merge(target, left_index=True, right_index=True)

    # limit to top correlated features relative to specified target.
    corrMatrix = df.corr()
    corrTop = corrMatrix[target.name]  # [:-1]
    corrTop = corrTop[abs(corrTop) > thresh].sort_values(ascending=False)

    if len(corrTop) <= 5:
        fontAdjust = 1.25
    elif len(corrTop) > 5 and len(corrTop) <= 10:
        fontAdjust = 1.15
    elif len(corrTop) > 10 and len(corrTop) <= 20:
        fontAdjust = 1.05
    elif len(corrTop) > 20 and len(corrTop) <= 30:
        fontAdjust = 0.95
    elif len(corrTop) > 30 and len(corrTop) <= 40:
        fontAdjust = 0.85
    else:
        fontAdjust = 0.65

    # create heatmap using correlation matrix.
    g = sns.heatmap(
        df[corrTop.index].corr().iloc[:,:1],
        vmin=-1.0,
        vmax=1.0,
        annot=annot,
        annot_kws={"size": fontAdjust * self.chartProp},
        square=False,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
        cmap="viridis",
    )

    # format y-tick labels and turn off xticks.
    g.set_yticklabels(
        g.get_yticklabels(), rotation=0, fontsize=fontAdjust * self.chartProp
    )
    plt.xticks([])

    # # workaround for matplotlib 3.1.1 bug
    # if matplotlib.__version__ == "3.1.1":
    #     g.set_ylim(corrMatrix.shape[1] + 0.1, -0.1)

    # customize color bar formatting and labeling.
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(
        labelsize=fontAdjust * self.chartProp, colors=style.styleGrey, length=0
    )
    cbar.set_ticks([vmax, 0.0, vmin])

    plt.show()


def prettyConfusionMatrix(self, yPred, yTrue, labels, cmap="viridis", ax=None, textcolors=["black", "white"],
                            threshold=None, reverseLabels=False, valfmt="{x:.0f}"):
    """
    Documentation:
        Description:

        Parameters:
            reverseLabels : boolean, default = False
                Reverse the direction of the labels. Puts the True Positives in the upper left hand corner in
                binary classification problems.

    """
    if ax is None:
        ax = plt.gca()

    # create confusion matrix using predictions and true labels
    cm = pd.DataFrame(metrics.confusion_matrix(y_true=yTrue, y_pred=yPred))

    # Sort rows and columns in descending order
    # cm.sort_index(axis = 1, ascending = False, inplace = True)
    # cm.sort_index(axis = 0, ascending = False, inplace = True)

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

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(cm.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(cm.shape[0] + 1) - 0.5, minor=True)
    ax.grid(False)
    ax.tick_params(which="minor", bottom=False, left=False)

    if not isinstance(cm, (list, np.ndarray)):
        cm = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(cm.max()) / 2.0

    # Set default alignment to center, but allow it to be overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = tkr.StrMethodFormatter(valfmt)

    # Loop over the cm and create a `Text` for each "pixel".
    # Change the text's color depending on the cm.
    texts = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            kw.update(color=textcolors[int(im.norm(cm[i, j]) < threshold)])
            # kw.update(color=textcolors[int(im.norm(cm[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(cm[i, j], None), **kw)
            texts.append(text)

    #
    ax.set_ylabel(
        ax.get_ylabel(),
        rotation=90,
        fontsize=18,
        color=style.styleGrey,
    )
    ax.set_xlabel(
        ax.get_xlabel(),
        rotation=0,
        fontsize=18,
        color=style.styleGrey,
    )

    plt.show()


def prettyRocCurve(self, model, XTrain, yTrain, XValid=None, yValid=None, linecolor=style.styleGrey,
                    bbox=(1.0, 0.4), ax=None):
    """
    Documentation:
        Description:
            Plot ROC curve and report AUC in
        Parameters:
            model : sklearn model or pipeline
                Model to fit and generate prediction probabilities.
            XTrain : array
                Training data for model fitting. Also used to return predict_probas
                when XValid is None.
            yTrain : array
                Training labels for model fitting. Also used to create ROC curve when
                XValid is None.
            XValid : array, default = None
                Test data for returning predict_probas.
            yValid : array, default = None
                Test data for creating ROC curve
            linecolor : str, default = style.styleHexMid[0]
                Curve line color
            bbox : tuple of floats, default = (1.2, 0.8)
                Coordinates for determining legend position
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    ## return prediction probabilities.
    # if XTest is None then fit the model using training data and return ROC curve for training data.
    if XValid is None:
        probas = model.fit(XTrain, yTrain).predict_proba(XTrain)
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=yTrain, y_score=probas[:, 1], pos_label=1
        )
    # otherwise fit the model using training data and return ROC curve for test data.
    else:
        probas = model.fit(XTrain, yTrain).predict_proba(XValid)
        fpr, tpr, thresholds = metrics.roc_curve(
            y_true=yValid, y_score=probas[:, 1], pos_label=1
        )

    # calculate area under the curve using FPR and TPR.
    roc_auc = metrics.auc(fpr, tpr)

    # plot ROC curve.
    self.prettyLine(
        x=fpr,
        y=tpr,
        label="AUC: {:.4f}".format(roc_auc),
        linecolor=linecolor,
        xUnits="fff",
        yUnits="fff",
        bbox=bbox,
        ax=ax,
    )

    # plot 'random guess' line for reference.
    self.prettyLine(
        x=np.array([0, 1]),
        y=np.array([0, 1]),
        linecolor=style.styleGrey,
        linestyle="--",
        xUnits="fff",
        yUnits="fff",
        ax=ax,
    )

    # plot 'perfection' line for reference.
    self.prettyLine(
        x=np.array([0, 0, 1]),
        y=np.array([0, 1, 1]),
        linecolor=style.styleGrey,
        linestyle=":",
        xUnits="fff",
        yUnits="fff",
        ax=ax,
    )

def prettyDecisionRegion(self, x, y, classifier, testIdx=None, resolution=0.1, bbox=(1.2, 0.9), colorMap="viridis", ax=None):
    """
    Documentation:
        Description:
            Create 2-dimensional chart with shading used to highlight decision regions.
        Parameters:
            X : array
                m x 2 array containing 2 features.
            y : array
                m x 1 array containing labels for observations.
            classifier : sklearn model or pipeline
                Classifier used to create decision regions.
            testIdx :  tuple, default = None
                Optional parameter for specifying observations to be highlighted as test examples.
            resolution : float, default = 0.1
                Controls clarity of the graph by setting interval of the arrays passed into np.meshgrid.
            bbox : tuple of floats, default = (1.2, 0.9)
                Coordinates for determining legend position.
            colorMap : string specifying built-in matplotlib colormap, default = "viridis"
                Colormap from which to draw plot colors.
            ax : Axes object, default = None
                Axis on which to place visual.
    """
    # generate color list
    colorList = style.colorGen(name=colorMap, num=len(np.unique(y)))

    # objects for marker generator and color map
    cmap = ListedColormap(colorList)
    # cmap = ListedColormap(style.styleHexLight[: len(np.unique(y))])

    # plot decision surface
    x1Min, x1Max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2Min, x2Max = x[:, 1].min() - 1, x[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1Min, x1Max, resolution), np.arange(x2Min, x2Max, resolution)
    )

    # generate predictions using classifier for all points on grid
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)

    # reshape the predictions and apply coloration
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


    # plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=x[y == cl, 0],
            y=x[y == cl, 1],
            alpha=1.0,
            c=colorList[idx],
            marker=style.styleMarkers[1],
            label=cl,
            s=12.5 * self.chartProp,
            # edgecolor=style.styleHexMidDark[idx],
        )

    # highlight test samples
    if testIdx:
        XTest = x[testIdx, :]
        plt.scatter(
            XTest[:, 0],
            XTest[:, 1],
            facecolor="none",
            edgecolor="white",
            alpha=1.0,
            linewidth=1.4,
            marker="o",
            s=12.75 * self.chartProp,
            label="test set",
        )
    # add legend to figure
    plt.legend(
        loc="upper right",
        bbox_to_anchor=bbox,
        ncol=1,
        frameon=True,
        fontsize=1.1 * self.chartProp,
    )

    plt.tight_layout()