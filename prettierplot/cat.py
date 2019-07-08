
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


def prettyBarV(self, x, counts, color = style.styleHexMid[0], xLabels = None, labelRotate = 0, yUnits = 'f', ax = None):
    """
    Documentation:
        Description: 
            Create vertical bar plot.
        Parameters:
            x : array
                1-dimensional array of values to be plotted on x-axis representing distinct categories.
            counts : array or string
                1-dimensional array of value counts for categories.
            color : string (some sort of color code), default = style.styleHexMid[0]
                Bar color.
            xLabels : list, default = None
                Custom x-axis test labels
            labelRotate : float or int, default = 0
                Degrees by which the xtick labels are rotated.
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # Create vertical bar plot.
    plt.bar(x = x
            ,height = counts
            ,color = color
            ,tick_label = xLabels if xLabels is not None else x
            ,alpha = 0.8
        )

    # Rotate x-tick labels.
    plt.xticks(rotation = labelRotate)
    
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, yUnits = yUnits)

    # Resize x-axis labels as needed
    if len(x) > 10 and len(x) <= 20:
        ax.tick_params(axis = 'x', colors = style.styleGrey, labelsize = 1.2 * self.chartProp)
    elif len(x) > 20:
        ax.tick_params(axis = 'x', colors = style.styleGrey, labelsize = 0.6 * self.chartProp)
    

def prettyBarH(self, y, counts, color = style.styleHexMid[0], labelRotate = 45, xUnits = 'f', ax = None):
    """
    Documentation:
        Description: 
            Create vertical bar plot.
        Parameters:
            y : array
                1-dimensional array of values to be plotted on x-axis representing distinct categories.
            counts : array or string
                1-dimensional array of value counts for categories.
            color : string (some sort of color code), default = style.styleHexMid[0]
                Bar color.
            labelRotate : float or int, default = 45
                Degrees by which the xtick labels are rotated.
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within `function.
    """
    # Plot horizontal bar plot.
    plt.barh(y = y
            ,width = counts
            ,color = color
            ,tick_label = y
            ,alpha = 0.8
        )
    
    # Rotate x-tick labels.
    plt.xticks(rotation = labelRotate)
    
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits)


def prettyBoxPlotV(self, x, y, data, color, labelRotate = 0, yUnits = 'f', ax = None):
    """
    Documentation:
        Description:
            Create vertical box plots. Useful for evaluated a continuous target on the y-axis
            vs. several different category segments on the x-axis
        Parameters:
            x : string
                Name of independent variable in dataframe. Represents a category
            y : string
                Name of continuous target variable. 
            data : Pandas DataFrame
                Pandas DataFrame including both indpedent variable and target variable.
            color : string
                Determines color of box plot figures. Ideally this object is a color palette,
                which can be a default seaborn palette, a custom seaborn palette, or a custom
                matplotlib cmap.
            labelRotate : float or int, default = 45
                Degrees by which the xtick labels are rotated.
            yUnits : string, default = 'f'
                Determines units of y-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # Create vertical box plot.
    g = sns.boxplot(x = x
                    ,y = y
                    ,data = data
                    ,orient = 'v'
                    ,palette = color
                    ,ax = ax).set(
                                xlabel = None
                                ,ylabel = None
                            )
    
    # Resize x-axis labels as needed.
    unique = np.unique(data[x])
    if len(unique) > 10 and len(unique) <= 20:
        ax.tick_params(axis = 'x', labelsize = 1.2 * self.chartProp)
    elif len(unique) > 20:
        ax.tick_params(axis = 'x', labelsize = 0.6 * self.chartProp)
    
    # Fade box plot figures by reducing alpha.
    plt.setp(ax.artists, alpha = 0.8)
    
    # Rotate x-tick labels.
    plt.xticks(rotation = labelRotate)
    ax.yaxis.set_visible(True)

    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, yUnits = yUnits)

        
def prettyBoxPlotH(self, x, y, data, color = style.styleHexMid, xUnits = 'f', ax = None):
    """
    Documentation:
        Description:
            Create horizontal box plots. Useful for evaluating a categorical target on the y-axis
            vs. a continuous independent variable on the x-axis
        Parameters:
            x : string
                Name of independent variable in dataframe. Represents a category
            y : string
                Name of continuous target variable. 
            data : Pandas DataFrame
                Pandas DataFrame including both indpedent variable and target variable.
            color : string (some sort of color code), default = style.styleHexMid
                Determines color of box plot figures. Ideally this object is a color palette,
                which can be a default seaborn palette, a custom seaborn palette, or a custom
                matplotlib cmap.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # Create horizontal box plot.
    g = sns.boxplot(x = x
                    ,y = y
                    ,hue = y
                    ,data = data
                    ,orient = 'h'
                    ,palette = color
                    ,ax = ax).set(
                                xlabel = None
                                ,ylabel = None
                            )
    
    # Fade box plot figures by reducing alpha.
    plt.setp(ax.artists, alpha = 0.8)
    ax.yaxis.set_visible(False)
    
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits)
    plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad = 0.)