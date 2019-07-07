
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


def prettyLine(self, x, y, label = None, df = None, linecolor = None, linestyle = None
                ,bbox = (1.2, 0.9), xUnits = 'f', xTicks = None, yUnits = 'f', yTicks = None
                ,markerOn = False, plotBuffer = False, axisLimits = False, ax = None):
    """
    Documentation:
        Description: 
            Create line plot. Capable of plotting multile lines on the same figure. Also capable of
            adjusting which axis will have the same data for each line and which will have different
            data for each line.
        Parameters:
            x : list, array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns 
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            y : list, array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns 
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            label : string : default = None
                Name to create legend entry.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size, as plotted columns will be chosen 
                by columns names specified in x, y. 
            linecolor : string, default = reference to list
                Determine color of line.
            linestyle : string, default = reference to list
                Determine style of line.
            bbox : tuple, default = (1.2, 0.9)
                Override bbox value for legend
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            xTicks : array, default = None
                Specify custom x-tick labels. 
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yTicks : array, default = None
                Specify custom y-tick labels. 
            markerOn : boolean, default = False
                Determines whether to show line with markers at each element.
            plotBuffer : boolean, default = False
                Switch for determining whether dynamic plot buffer function is executed.
            axisLimits : boolean, default = False
                Switch for determining whether dynamic axis limit setting function is executed.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within `function.
    """
    # If a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
    if df is not None:
        x = df[x].values
        y = df[y].values
    else:
        # convert input list to array
        x = np.array(x) if type(x) == list else x
        y = np.array(y) if type(y) == list else y

        # reshape arrays if necessar
        x = x.reshape(-1,1) if len(x.shape) == 1 else x
        y = y.reshape(-1,1) if len(y.shape) == 1 else y
    
    # Add line 
    plt.plot(x
            ,y
            ,color = linecolor if linecolor is not None else style.styleHexMid[0]
            ,linestyle = linestyle if linestyle is not None else style.styleLineStyle[0]
            ,linewidth = 0.247 * self.chartProp
            ,label = label
            # ,label = label[0] if label is not None else None
            ,marker = '.' if markerOn else None
            ,markersize = 25 if markerOn else None
            ,markerfacecolor = 'w' if markerOn else None
            ,markeredgewidth = 2.5 if markerOn else None
        )                
    
    # Add legend to figure
    if label is not None:
        plt.legend(loc = 'upper right'
                    ,bbox_to_anchor = bbox
                    ,ncol = 1
                    ,frameon = True
                    ,fontsize = 1.1 * self.chartProp
                    )
        
    # Axis tick label formatting
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

    # Dynamically set axis lower / upper limits
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x = x, y = y)
        plt.axis([xMin, xMax, yMin, yMax])   
    
    # Create smaller buffer around plot area to prevent cutting off elements
    if plotBuffer:
        util.utilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

    # x-axis tick label control
    if xTicks is not None:
        ax.set_xticks(xTicks)
    
    # y-axis tick label control
    if yTicks is not None:
        ax.set_yticks(yTicks)
    
    # Show figure with tight layout
    plt.tight_layout()

def prettyMultiLine(self, x, y, label = None, df = None, linecolor = None, linestyle = None, bbox = (1.2, 0.9)
                ,xUnits = 'f', xTicks = None, yUnits = 'f', yTicks = None, markerOn = False, plotBuffer = False
                ,axisLimits = False, ax = None):
    """
    Documentation:
        Description: 
            Create line plot. Capable of plotting multile lines on the same figure. Also capable of
            adjusting which axis will have the same data for each line and which will have different
            data for each line.
        Parameters:
            x : array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns 
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values, a multidimensional array of values, a list of columns 
                in a Pandas DataFrame, or a column name in a Pandas DataFrame.
            label : list of strings : default = None
                List of names of used to create legend entries for each line.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size, as plotted columns will be chosen 
                by columns names specified in x, y. 
            linecolor : string, default = reference to list
                Determine color of line.
            linestyle : string, default = reference to list
                Determine style of line.
            bbox : tuple, default = (1.2, 0.9)
                Override bbox value for legend
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            xTicks : array, default = None
                Specify custom x-tick labels. 
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yTicks : array, default = None
                Specify custom y-tick labels. 
            markerOn : boolean, default = False
                Determines whether to show line with markers at each element.
            plotBuffer : boolean, default = False
                Switch for determining whether dynamic plot buffer function is executed.
            axisLimits : boolean, default = False
                Switch for determining whether dynamic axis limit setting function is executed.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within `function.
    """
    # If a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
    if df is not None:
        x = df[x].values
        y = df[y].values
    else:
        # convert input list to array
        x = np.array(x) if type(x) == list else x
        y = np.array(y) if type(y) == list else y
        
        x = x.reshape(-1,1) if len(x.shape) == 1 else x
        y = y.reshape(-1,1) if len(y.shape) == 1 else y
    
    # add multiple lines
    for ix in np.arange(y.shape[1]):
        yCol = y[:, ix]
        plt.plot(x
                ,yCol
                ,color = linecolor if linecolor is not None else style.styleHexMid[ix]
                ,linestyle = linestyle if linestyle is not None else style.styleLineStyle[0]
                ,linewidth = 0.247 * self.chartProp
                ,label = label[ix] if label is not None else None
                ,marker = '.' if markerOn else None
                ,markersize = 25 if markerOn else None
                ,markerfacecolor = 'w' if markerOn else None
                ,markeredgewidth = 2.5 if markerOn else None
            )

    # Add legend to figure
    if label is not None:
        plt.legend(loc = 'upper right'
                  ,bbox_to_anchor = bbox
                  ,ncol = 1
                  ,frameon = True
                  ,fontsize = 1.1 * self.chartProp
            )
        
    # Axis tick label formatting
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

    # Dynamically set axis lower / upper limits
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x = x, y = y)
        plt.axis([xMin, xMax, yMin, yMax])   
    
    # Create smaller buffer around plot area to prevent cutting off elements
    if plotBuffer:
        util.utilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

    # x-axis tick label control
    if xTicks is not None:
        ax.set_xticks(xTicks)
    
    # y-axis tick label control
    if yTicks is not None:
        ax.set_yticks(yTicks)

    # Show figure with tight layout
    plt.tight_layout()