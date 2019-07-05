
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import sklearn.metrics as metrics
import sklearn.preprocessing as prepocessing

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

import prettierplot.style as style
import prettierplot.util as util


def prettyProbPlot(self, x, plot):
        """
        Documentation:
            Description:
                Create plot that visualizes how well a continuous feature's distribution
                conforms to a normal distribution
            Parameters:
                x : array
                    1-dimensional array containing data of a continuous feature
                plot : plot object
                    Plotting object for applying additional formatting.
        """
        stats.probplot(x, plot = plot)
        
        # Override title and axis labels.
        plot.set_title('')
        plt.xlabel('')
        plt.ylabel('')
        
        # Format scatter dots.
        plot.get_lines()[0].set_markerfacecolor(style.styleWhite)
        plot.get_lines()[0].set_color(style.styleHexMid[2])
        plot.get_lines()[0].set_markersize(5.0)

        # Format line representing normality.
        plot.get_lines()[1].set_linewidth(3.0)
        plot.get_lines()[1].set_color(style.styleGrey)


def prettyCorrHeatmap(self, df, target = None, targetName = None, annot = True, cols = None, ax = None, vmin = -1.0, vmax = 1.0):
    """
    Documentation:
        Description:
            Using continuous features, create correlation heatmap. Produces correlation
            with all numerical features, and can be limited to certain features using 'cols'.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            annot : boolean, default = True
                Determines whether or not correlation table is annoted wil correlation
                value or not.
            cols : list, default = None
                List of strings describing dataframe column. Limits dataframe to select
                columns.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
        Returns:
            corrSummDf : Pandas DataFrame
                Pandas DataFrame summarizing highest correlation coefficients between 
                features and target.
    """
    if target is not None:
        df = pd.merge(df, pd.DataFrame(target, columns = [targetName]), left_index = True, right_index = True)
    
    if cols is None:
        cols = df.columns
    
    # Create correlation matrix
    corrMatrix = df[cols].corr() if cols is not None else df.corr() 
    
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

    # Create heatmap using correlation matrix
    g = sns.heatmap(corrMatrix
                ,vmin = vmin
                ,vmax = vmax
                ,annot = annot
                ,annot_kws = {'size' : fontAdjust * self.chartProp}
                ,square = False
                ,ax = ax
                ,xticklabels = True
                ,yticklabels = True
                ,cmap = LinearSegmentedColormap.from_list(name = ''
                                                        ,colors = [style.styleHexMid[2], 'white', style.styleHexMid[0]])
                )

    # Format x and y-tick labels
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 1.0 * self.chartProp)
    g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = 1.0 * self.chartProp)

    # Customize color bar formatting
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize = fontAdjust * self.chartProp, colors = style.styleGrey, length = 0)
    cbar.set_ticks([1.0, 0.0, -1.0])


def prettyCorrHeatmapRefine(self, df, target = None, targetName = None, annot = True, cols = None, thresh = 0.5, ax = None):
    """
    Documentation:
        Description:
            Using continuous features, create correlation heatmap. Capable of dropping 
            zeros in select features, where zeros potentially indicate a complete absence
            of the feature.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            annot : boolean, default = True
                Determines whether or not correlation table is annoted wil correlation
                value or not.
            cols : list, default = None
                List of strings describing dataframe column. Limits dataframe to select
                columns.
            corrFocus : string, default = self.target[0]
                The feature of focus in the supplemental correlation visualization. Used
                to determine the feature for which the nlargest correlation coefficients
                are returned.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
        Returns:
            corrSummDf : Pandas DataFrame
                Pandas DataFrame summarizing highest correlation coefficients between 
                features and target.
    """
    df = pd.merge(df, pd.DataFrame(target, columns = [targetName]), left_index = True, right_index = True)
            

    # Limit to top correlated features relative to targetName
    corrMatrix = df[cols].corr() if cols is not None else df.corr() 
    corrTop = corrMatrix[targetName]#[:-1]
    corrTop = corrTop[abs(corrTop) > thresh].sort_values(ascending = False)
    display(pd.DataFrame(corrTop))        
    
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


    # Create heatmap using correlation matrix
    g = sns.heatmap(df[corrTop.index].corr()
                ,vmin = -1.0
                ,vmax = 1.0
                ,annot = annot
                ,annot_kws = {'size' : fontAdjust * self.chartProp}
                ,square = False
                ,ax = ax
                ,xticklabels = True
                ,yticklabels = True
                ,cmap = LinearSegmentedColormap.from_list(name = ''
                                                        ,colors = [style.styleHexMid[2], 'white', style.styleHexMid[0]])
                )

    # Format x and y-tick labels
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = fontAdjust * self.chartProp)
    g.set_xticklabels(g.get_xticklabels(), rotation = 90, fontsize = fontAdjust * self.chartProp)

    # Customize color bar formatting
    cbar = g.collections[0].colorbar
    cbar.ax.tick_params(labelsize = fontAdjust * self.chartProp, colors = style.styleGrey, length = 0)
    cbar.set_ticks([1.0, 0.0, -1.0])

    plt.show()          


def prettyConfusionMatrix(self, yTest, yPred, ax = None):
    """
    Documentation:
        Description:
            yTest : array
                1-dimensional array containing true values
            yPred : array
                1-dimensional array containing predictions
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within `function.
    """
    # Create confution matrix.
    cm = pd.DataFrame(metrics.confusion_matrix(y_true = yTest
                                                , y_pred = yPred)
                                                )
    
    # Sort rows and columns in descending order
    cm.sort_index(axis = 1, ascending = False, inplace = True)
    cm.sort_index(axis = 0, ascending = False, inplace = True)
    
    # Apply heatmap styling to confusion matrix.
    sns.heatmap(data = cm
                ,annot = True
                ,square = True
                ,cbar = False
                ,cmap = 'Blues'
                ,annot_kws = {'size' : 2.5 * self.chartProp})
    
    # Remove top tick marks and add labels
    ax.xaxis.tick_top()
    plt.xlabel('predicted', size = 40)
    plt.ylabel('actual', size = 40)


def prettyRocCurve(self, model, xTrain, yTrain, xTest, yTest, linecolor, ax = None):
    """
    Documentation:
        Description:
            Plot ROC curve and report AUC
        Parameters:
            model : sklearn model or pipeline
                model to fit
            xTrain : array
                Training data to fit model
            yTrain : array
                Training data to fit model
            xTest : array
                Test data to return predict_probas
            yTest : array
                Test data for creating roc_curve
            linecolor : str
                line color
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within `function.
    """
    # Return prediction probabilities.
    probas = model.fit(xTrain, yTrain).predict_proba(xTest)
    
    # Return false positive rate, true positive rate and thresholds.
    fpr, tpr, thresholds = metrics.roc_curve(y_true = yTest, y_score = probas[:, 1], pos_label = 1)
    
    # Calculate area under the curve using FPR and TPR.
    roc_auc = metrics.auc(fpr, tpr)
    
    # Plot ROC curve.
    self.prettyLine(x = fpr
                    ,y = tpr
                    ,label = ['ROC AUC = {:.3f}'.format(roc_auc)]
                    ,linecolor = linecolor
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,bbox = (1.0, 0.8)
                    ,ax = ax
                )
    
    # Plot 'random guess' line.
    self.prettyLine(x = np.array([0, 1])
                    ,y = np.array([0, 1])
                    ,linecolor = style.styleGrey
                    ,linestyle = '--'
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,ax = ax
                )
    
    # Plot 'perfection' line.
    self.prettyLine(x = np.array([0, 0, 1])
                    ,y = np.array([0, 1, 1])
                    ,linecolor = style.styleGrey
                    ,linestyle = ':'
                    ,xUnits = 'fff'
                    ,yUnits = 'fff'
                    ,ax = ax
                )


def prettyDecisionRegion(self, x, y, classifier, testIdx = None, resolution = 0.1, bbox = (1.2, 0.9), ax = None):
    """
    Documentation:
        Description:
            Create 2-dimensional chart with shading used to highlight decision regions.
        Parameters:
            X : array
                m x 2 array containing 2 features.
            y : array
                m x 1 array containing labels for observations.
            classifier : sklearn model
                Classifier used to create decision regions.
            testIdx :  tuple, default = None
                Option parameter for specifying observations to be highlighted
                as test examples.
            resolution : float, default = 0.001
                Controls clarity of the graph by setting interval of the arrays 
                passed into np.meshgrid.
            bbox : tuple of floats, default = (1.2, 0.9)
                Coordinates for determining legend position
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # objects for marker generator and color map
    cmap = ListedColormap(style.styleHexLight[:len(np.unique(y))])
    
    # plot decision surface
    x1Min, x1Max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2Min, x2Max = x[:, 1].min() - 1, x[:, 1].max() + 1
    
    xx1, xx2 = np.meshgrid(np.arange(x1Min, x1Max, resolution)
                        ,np.arange(x2Min, x2Max, resolution))
    
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap = cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Plot samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x = x[y == cl, 0]
                ,y = x[y == cl, 1]
                ,alpha = 1.0
                ,c = style.styleHexMid[idx]
                ,marker = style.styleMarkers[1]
                ,label = cl
                ,s = 12.5 * self.chartProp
                ,edgecolor = style.styleHexMidDark[idx]
                )
    
    # Highlight test samples
    if testIdx:
        xTest = x[testIdx, :]
        plt.scatter(xTest[:,0]
                    ,xTest[:,1]
                    ,facecolor = 'none'
                    ,edgecolor = 'white'
                    ,alpha = 1.0
                    ,linewidth = 1.4
                    ,marker = 'o'
                    ,s = 12.75 * self.chartProp
                    ,label = 'test set'                   
                )
    # Add legend to figure
    plt.legend(loc = 'upper right'
                ,bbox_to_anchor = bbox
                ,ncol = 1
                ,frameon = True
                ,fontsize = 1.1 * self.chartProp
                )
    plt.tight_layout()


def prettyResidualPlot(self):
    pass