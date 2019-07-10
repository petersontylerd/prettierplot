
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

            
def prettyFacetCat(self, df, feature, labelRotate = 0, yUnits = 'f', xUnits = 's', bbox = (1.2, 0.9), ax = None):       
    """
    Documentation:
        Description:
            Creates a count plot of a categorical feature and facets that feature by a categorical label.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame
            feature : string
                String describing column name containing target values
            labelRotate : float or int, default = 45
                Degrees by which the xtick labels are rotated.
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            bbox : tuple of floats, default = (1.2, 0.9)
                Coordinates for determining legend position
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    ixs = np.arange(df.shape[0])
    bar_width = 0.35
    
    featureDict = {}
    for feature in df.columns[1:]:
        featureDict[feature] = df[feature].values.tolist()
    for featureIx, (k, v) in enumerate(featureDict.items()):
        plt.bar(ixs + (bar_width * featureIx)
                ,featureDict[k]
                ,bar_width
                ,alpha = 0.75
                ,color = style.styleHexMid[featureIx]
                ,label = str(k)
                )
    
    # Custom x-tick labels.
    plt.xticks(ixs[:df.shape[0]] + bar_width / 2, df.iloc[:,0].values)
    
    # Rotate x-tick labels.
    plt.xticks(rotation = labelRotate)
            
    # Add legend to figure.
    plt.legend(loc = 'upper right'
                ,bbox_to_anchor = bbox
                ,ncol = 1
                ,frameon = True
                ,fontsize = 1.1 * self.chartProp
                )
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)
    
    # Resize x-axis labels as needed.
    if len(featureDict[feature]) > 10 and len(featureDict[feature]) <= 20:
        ax.tick_params(axis = 'x', colors = style.styleGrey, labelsize = 1.2 * self.chartProp)
        
    elif len(featureDict[feature]) > 20:
        ax.tick_params(axis = 'x', colors = style.styleGrey, labelsize = 0.6 * self.chartProp)


def prettyFacetTwoCatBar(self, df, x, y, split, xUnits = None, yUnits = None, bbox_to_anchor = None, legendLabels = None, filterNaN = True, ax = None):
    """
    Documentation:
        Description:
            desc
        Parameters:
    """
    if filterNaN:
        df = df.dropna(subset = [x])
    
    g = sns.barplot(x = x
                    ,y = y
                    ,hue = split
                    ,data = df
                    ,palette = style.styleHexMid
                    ,order = df[x].sort_values().drop_duplicates().values.tolist()
                    ,hue_order = df[split].sort_values().drop_duplicates().values.tolist() if split is not None else None
                    ,ax = ax
                    ,ci = None)
    
    # Format x and y-tick labels
    g.set_yticklabels(g.get_yticklabels(), rotation = 0, fontsize = 1.25 * self.chartProp, color = style.styleGrey)
    g.set_xticklabels(g.get_xticklabels(), rotation = 0, fontsize = 1.25 * self.chartProp, color = style.styleGrey)
    g.set_ylabel(g.get_ylabel(), rotation = 90, fontsize = 1.75 * self.chartProp, color = style.styleGrey)
    g.set_xlabel(g.get_xlabel(), rotation = 0, fontsize = 1.75 * self.chartProp, color = style.styleGrey)
    

    ## create custom legend
    # create labels
    if split is not None:
        if legendLabels is None:
            legendLabels = df[df[split].notnull()][split].sort_values().drop_duplicates().values.tolist()
        else:
            legendLabels = np.array(legendLabels)

        labelColor = {}
        for ix, i in enumerate(legendLabels):
            labelColor[i] = style.styleHexMid[ix]

        # create patches
        patches = [matplotlib.patches.Patch(color = v, label = k) for k, v in labelColor.items()]
        
        # draw legend
        leg = plt.legend(handles = patches
                    ,fontsize = 1.25 * self.chartProp
                    ,loc = 'upper right'
                    ,markerscale = 0.5 * self.chartProp
                    ,ncol = 1
                    ,bbox_to_anchor = bbox_to_anchor
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color = 'Grey')
        
        # Axis tick label formatting.
        util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)            


def prettyFacetCatNumScatter(self, df, xNum, yNum, catRow = None, catCol = None, split = None, bbox_to_anchor = None
                            ,aspect = 1, height = 4, legendLabels = None):
    """
    Documentation:
        Description:
            desc
        Parameters:
            df : Pandas DataFrame
                desc
            catRow : 
                desc
            catCol : 
                desc
            split : 
                desc
            numCol : 
                desc
            bbox_to_anchor : 
                desc
            aspect : float, default = 1 
                desc
            height : float, default = 4
                desc
            legendLabels : list, default = None
                desc
    """
    g = sns.FacetGrid(df
                    ,col = catCol
                    ,row = catRow
                    ,hue = split
                    ,palette = style.styleHexMid
                    ,hue_order = df[split].sort_values().drop_duplicates().values.tolist() if split is not None else None
                    )
    g = (g.map(plt.scatter
              ,xNum
              ,yNum
            #   ,**kws
            )
        )
    
    for ax in g.axes.flat:
        _ = ax.set_ylabel(ax.get_ylabel(), rotation = 90, fontsize = 1.25 * self.chartProp, color = style.styleGrey)
        _ = ax.set_xlabel(ax.get_xlabel(), rotation = 0, fontsize = 1.25 * self.chartProp, color = style.styleGrey)
        # _ = ax.xaxis.labelpad = 5
        # _ = ax.yaxis.labelpad = 5
        # _ = ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 1.05 * self.chartProp, color = style.styleGrey)
        # _ = ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 1.05 * self.chartProp, color = style.styleGrey)
        _ = ax.set_title(ax.get_title(), rotation = 0, fontsize = 0.65 * self.chartProp, color = style.styleGrey)
    
    ## create custom legend
    # create labels
    if split is not None:
        if legendLabels is None:
            legendLabels = df[df[split].notnull()][split].sort_values().drop_duplicates().values.tolist()
        else:
            legendLabels = np.array(legendLabels)

        labelColor = {}
        for ix, i in enumerate(legendLabels):
            labelColor[i] = style.styleHexMid[ix]

        # create patches
        patches = [matplotlib.patches.Patch(color = v, label = k) for k, v in labelColor.items()]
        
        # draw legend
        leg = plt.legend(handles = patches
                    ,fontsize = 1.0 * self.chartProp
                    ,loc = 'upper right'
                    ,markerscale = 0.5 * self.chartProp
                    ,ncol = 1
                    ,bbox_to_anchor = bbox_to_anchor
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color = 'Grey')


def prettyFacetCatNumHist(self, df, catRow, catCol, numCol, split, bbox_to_anchor = None, aspect = 1, height = 4, legendLabels = None):
    """
    Documentation:
        Description:
            desc
        Parameters:
            df : Pandas DataFrame
                desc
            catRow : 
                desc
            catCol : 
                desc
            split : 
                desc
            numCol : 
                desc
            bbox_to_anchor : 
                desc
            aspect : float, default = 1 
                desc
            height : float, default = 4
                desc
            legendLabels : list, default = None
                desc
    """
    
    g = sns.FacetGrid(df
                     ,row = catRow
                     ,col = catCol
                     ,hue = split
                     ,hue_order = df[split].sort_values().drop_duplicates().values.tolist() if split is not None else None
                     ,palette = style.styleHexMid
                     ,despine = True
                     ,height = height
                     ,aspect = aspect
        )
    g.map(plt.hist
         ,numCol
        #  ,bins = np.arange(0, 20)
         ,alpha = .5
        )
    
    for ax in g.axes.flat:
        # _ = ax.set_ylabel(ax.get_ylabel(), rotation = 90, fontsize = 1.25 * self.chartProp, color = style.styleGrey)
        _ = ax.set_xlabel(ax.get_xlabel(), rotation = 0, fontsize = 1.25 * self.chartProp, color = style.styleGrey)
        # _ = ax.xaxis.labelpad = 25
        # _ = ax.yaxis.labelpad = 25
        # _ = ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 1.05 * self.chartProp, color = style.styleGrey)
        # _ = ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 1.05 * self.chartProp, color = style.styleGrey)
        _ = ax.set_title(ax.get_title(), rotation = 0, fontsize = 1.05 * self.chartProp, color = style.styleGrey)
    
    ## create custom legend
    # create labels
    if split is not None:
        if legendLabels is None:
            legendLabels = df[df[split].notnull()][split].sort_values().drop_duplicates().values.tolist()
        else:
            legendLabels = np.array(legendLabels)

        labelColor = {}
        for ix, i in enumerate(legendLabels):
            labelColor[i] = style.styleHexMid[ix]

        # create patches
        patches = [matplotlib.patches.Patch(color = v, label = k) for k, v in labelColor.items()]
        
        # draw legend
        leg = plt.legend(handles = patches
                    ,fontsize = 1.0 * self.chartProp
                    ,loc = 'upper right'
                    ,markerscale = 0.5 * self.chartProp
                    ,ncol = 1
                    ,bbox_to_anchor = bbox_to_anchor
        )

        # label font color
        for text in leg.get_texts():
            plt.setp(text, color = 'Grey')

    
def prettyFacetTwoCatPoint(self, df, target, targetName, catLine, catPoint, catGrid, bbox_to_anchor = None, aspect = 1, height = 4, legendLabels = None):
    """
    Documentation:
        Description:
            desc
        Parameters:
    """
    df = pd.merge(df[[catLine, catPoint, catGrid]]
                    ,pd.DataFrame(target
                                 ,columns = [targetName])
                ,left_index = True
                ,right_index = True
        )
    g = sns.FacetGrid(df
                     ,catGrid
                     ,aspect = aspect
                     ,height = height
        )
    g.map(sns.pointplot
         ,catPoint
         ,targetName
         ,catLine
         ,order = df[catPoint].sort_values().drop_duplicates().values.tolist()
         ,hue_order = df[catLine].sort_values().drop_duplicates().values.tolist()
         ,palette = style.styleHexMid
         ,alpha = .75
        )
    
    for ax in g.axes.flat:
        _ = ax.set_ylabel(ax.get_ylabel(), rotation = 90, fontsize = 0.95 * self.chartProp, color = style.styleGrey)
        _ = ax.set_xlabel(ax.get_xlabel(), rotation = 0, fontsize = 0.95 * self.chartProp, color = style.styleGrey)
        _ = ax.xaxis.labelpad = 5
        _ = ax.yaxis.labelpad = 5
        # _ = ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 0.85 * self.chartProp, color = style.styleGrey)
        # _ = ax.set_xticklabels(ax.get_xticklabels(), rotation = 0, fontsize = 0.85 * self.chartProp, color = style.styleGrey)
        _ = ax.set_title(ax.get_title(), rotation = 0, fontsize = 1.05 * self.chartProp, color = style.styleGrey)
    
    ## create custom legend
    # create labels
    if legendLabels is None:
        legendLabels = np.unique(df[df[catLine].notnull()][catLine])
    else:
        legendLabels = np.array(legendLabels)

    labelColor = {}
    for ix, i in enumerate(legendLabels):
        labelColor[i] = style.styleHexMid[ix]

    # create patches
    patches = [matplotlib.patches.Patch(color = v, label = k) for k, v in labelColor.items()]
    
    # draw legend
    leg = plt.legend(handles = patches
                ,fontsize = 1.0 * self.chartProp
                ,loc = 'upper right'
                ,markerscale = 0.5 * self.chartProp
                ,ncol = 1
                ,bbox_to_anchor = bbox_to_anchor
    )

    # label font color
    for text in leg.get_texts():
        plt.setp(text, color = 'Grey')
