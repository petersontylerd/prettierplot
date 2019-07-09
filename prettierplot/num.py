
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
# import prettierplot.style as style
# import prettierplot.util as util


def pretty2dScatter(self, x, y, df = None, xUnits = 'f', xTicks = None, yUnits = 'f', yTicks = None
                   ,plotBuffer = True, size = 10, axisLimits = True, color = style.styleGrey
                   ,facecolor = 'w', ax = None):
    """
    Documentation:
        Description: 
            Create 2-dimensional scatter plot.
        Parameters:
            x : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size - plotted columns will be 
                chosen by columns names specified in x, y. 
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            plotBuffer : boolean, default = True
                Switch for determining whether dynamic plot buffer function is executed.
            size : int or float, default = 10
                Determines scatter dot size
            axisLimits : boolean, default = True
                Switch for determining whether dynamic axis limit setting function is executed.
            color : string (color code of some sort), default = style.styleGrey
                Determine color of scatter dots
            facecolor : string (color code of some sort), default = 'w'
                Determine face color of scatter dots
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # If a Pandas DataFrame is passed to function, create x, y arrays using columns names passed into function.
    if df is not None:
        x = df[x].values.reshape(-1,1)
        y = df[y].values.reshape(-1,1)
    # Else reshape arrays.
    else:
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
    
    # Plot 2-d scatter.
    plt.scatter(x = x
                ,y = y
                ,color = color
                ,s = size * self.chartProp
                ,alpha = 0.7
                ,facecolor = facecolor
                ,linewidth = 0.167 * self.chartProp
                )
    
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)        

    # Dynamically set axis lower / upper limits.
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x = x, y = y)        
        plt.axis([xMin, xMax, yMin, yMax])   

    # Create smaller buffer around plot area to prevent cutting off elements.
    if plotBuffer:
        util.utilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

    # x-axis tick label control
    if xTicks is not None:
        ax.set_xticks(xTicks)
    
    # y-axis tick label control
    if yTicks is not None:
        ax.set_yticks(yTicks)

    # Show figure with tight layout.
    plt.tight_layout()


def pretty2dScatterHue(self, x, y, target, label, df = None, xUnits = 'f', xTicks = None, yUnits = 'f', yTicks = None
                      ,plotBuffer = True, size = 10, axisLimits = True, color = style.styleGrey, facecolor = 'w'
                      ,bbox = (1.2, 0.9), ax = None):
    """
    Documentation:
        Description: 
            Create 2-dimensional scatter plot with a third dimension represented as a color hue in the
            scatter dots.
        Parameters:
            x : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            y : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            target : array or string
                Either 1-dimensional array of values or a column name in a Pandas DataFrame.
            label : list
                List of labels describing hue.
            df : Pandas DataFrame, default = None
                Dataset containing data to be plotted. Can be any size - plotted columns will be 
                chosen by columns names specified in x, y. 
            xUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yUnits : string, default = 'd'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            plotBuffer : boolean, default = True
                Switch for determining whether dynamic plot buffer function is executed.
            size : int or float, default = 10
                Determines scatter dot size
            axisLimits : boolean, default = True
                Switch for determining whether dynamic axis limit setting function is executed.
            color : string (color code of some sort), default = style.styleGrey
                Determine color of scatter dots
            facecolor : string (color code of some sort), default = 'w'
                Determine face color of scatter dots
            bbox : tuple of floats, default = (1.2, 0.9)
                Coordinates for determining legend position
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # If a Pandas DataFrame is passed to function, create x, y and target arrays using columns names 
    # passed into function. Also create X, which is a matrix containing the x, y and target columns.
    if df is not None:
        X = df[[x, y, target]].values
        x = df[x].values
        y = df[y].values
        target = df[target].values
    # Concatenate the x, y and target arrays.
    else:
        X = np.c_[x, y, target]

    # Unique target values.
    targetIds =  np.unique(X[:, 2])
        
    # Loop through sets of target values, labels and colors to create 2-d scatter with hue.
    for targetId, targetName, color in zip(targetIds, label, style.styleHexMid[:len(targetIds)]):
        plt.scatter(x = X[X[:,2] == targetId][:,0]
                    ,y = X[X[:,2] == targetId][:,1]
                    ,color = color
                    ,label = targetName
                    ,s = size * self.chartProp
                    ,alpha = 0.7
                    ,facecolor = 'w'
                    ,linewidth = 0.234 * self.chartProp
                )
    
    # Add legend to figure.
    if label is not None:
        plt.legend(loc = 'upper right'
                    ,bbox_to_anchor = bbox
                    ,ncol = 1
                    ,frameon = True
                    ,fontsize = 1.1 * self.chartProp
                    )
        
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)

    # Dynamically set axis lower / upper limits.
    if axisLimits:
        xMin, xMax, yMin, yMax = util.utilSetAxes(x = x, y = y)
        plt.axis([xMin, xMax, yMin, yMax])   
    
    # Create smaller buffer around plot area to prevent cutting off elements.
    if plotBuffer:
        util.utilPlotBuffer(ax = ax, x = 0.02, y = 0.02)

    # x-axis tick label control
    if xTicks is not None:
        ax.set_xticks(xTicks)
    
    # y-axis tick label control
    if yTicks is not None:
        ax.set_yticks(yTicks)

    # Show figure with tight layout.
    plt.tight_layout()


def prettyDistPlot(self, x, color, yUnits = 'f', xUnits = 'f', fit = None, xRotate = None, ax = None):
    """
    Documentation:
        Description:
            Creates distribution plot for continuous variables, showing counts of a single
            variable. Also overlays a kernel density estimation curve.
        Parameters:
            x : array
                Data to be plotted.
            color : string (some sort of color code)
                Determines color of bars, KDE lines.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            xRotate : int, default = None
                Rotates x-axis tick mark labels X degrees
            fit : random variabe object, default = None
                Allows for the addition of another curve. Utilizing 'norm' overlays a normal distribution
                over the distribution bar chart. Useful for seeing how well, or not, distribution tracks
                with a normal distrbution.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    g = sns.distplot(a = x
                    ,kde = True
                    ,color = color
                    ,axlabel = False
                    ,fit = fit
                    ,ax = ax)

    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits, xRotate = xRotate)


def prettyKdePlot(self, x, color, yUnits = 'f', xUnits = 'f', ax = None):
    """
    Documentation:
        Description:
            Create kernel density curve for a feature.
        Parameters:
            x : array
                Data to be plotted.
            color : string (some sort of color code)
                Determines color of KDE lines.
            xUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            yUnits : string, default = 'f'
                Determines units of x-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # Create kernel density estimation line
    g = sns.kdeplot(data = x
                    ,shade = True
                    ,color = color
                    ,legend = None
                    ,ax = ax)
    
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits)


def prettyRegPlot(self, x, y, data, color = style.styleHexMid[0], x_jitter = None, xUnits = 'f', xRotate = None
                 ,yUnits = 'f', ax = None):
    """
    Documentation:
        Description:

        Parameters:
            x : string
                Name of independent variable in dataframe. Represents a category
            y : string
                Name of continuous target variable. 
            data : Pandas DataFrame
                Pandas DataFrame including both indepedent variable and target variable.
            color : string
                Determines color of dots and regression line.
            labelRotate : float or int, default = 45
                Degrees by which the xtick labels are rotated.
            yUnits : string, default = 'f'
                Determines units of y-axis tick labels. 'f' displays float. '%' displays percentages, 
                '$' displays dollars. 'd' displays real numbers.
            xRotate : int, default = None
                Rotates x-axis tick mark labels X degrees
            ax : Axes object, default = None
                Axes object containing figure elements to be adjusted within function.
    """
    # Create vertical box plot.
    g = sns.regplot(x = x
                    ,y = y
                    ,data = data
                    ,x_jitter = x_jitter
                    # ,color = color
                    ,scatter_kws = {'alpha' : 0.3
                                    ,'color' : style.styleHexMid[0]}
                    ,line_kws = {'color' : style.styleHexMid[1]}
                    ,ax = ax).set(
                                xlabel = None
                                ,ylabel = None
                            )        
    
    # Axis tick label formatting.
    util.utilLabelFormatter(ax = ax, xUnits = xUnits, yUnits = yUnits, xRotate = xRotate)


def prettyPairPlot(self, df, cols = None, target = None, targetName = None, diag_kind = 'auto', legendLabels = None, bbox_to_anchor = None):
    """
    Documentation:
        Description: 
            Create pair plot that produces a grid of scatter plots for all unique pairs of
            continuous features.
        Parameters:
            df : Pandas DataFrame
                Pandas DataFrame containing data of interest.
            vars : list, default = None
                List of strings describing columns in Pandas DataFrame to be visualized.
            hue : string, default = None
                Variable name to be used to introduce third dimension to scatter plots through
                a color hue.
            diag_kind : string, default = 'auto.
                Type of plot created along diagonal.
    """
    # Custom plot formatting settings for this particular chart.
    with plt.rc_context({'axes.titlesize' : 3.5 * self.chartProp
                        ,'axes.labelsize' : 1.5 * self.chartProp  # Axis title font size
                        ,'xtick.labelsize' : 1.2 * self.chartProp
                        ,'xtick.major.size' : 0.5 * self.chartProp
                        ,'xtick.major.width' : 0.05 * self.chartProp
                        ,'xtick.color' : style.styleGrey
                        ,'ytick.labelsize' : 1.2 * self.chartProp
                        ,'ytick.major.size' : 0.5 * self.chartProp
                        ,'ytick.major.width' : 0.05 * self.chartProp
                        ,'ytick.color' : style.styleGrey
                        ,'figure.facecolor' : style.styleWhite
                        ,'axes.facecolor': style.styleWhite
                        ,'axes.spines.left': False
                        ,'axes.spines.bottom': False
                        ,'axes.edgecolor': style.styleGrey
                        ,'axes.grid': False
        }):
        
        # capture all predictor columns
        if cols is None:
            cols = df.select_dtypes(exclude = [object]).columns
            
        if target is not None:
            df = pd.merge(df[cols]
                        ,pd.DataFrame(target
                                        ,columns = [targetName])
                    ,left_index = True
                    ,right_index = True
                )

        # Create pair plot.
        g = sns.pairplot(data = df if target is None else df.dropna()
                        ,vars = cols
                        ,hue = targetName
                        ,diag_kind = diag_kind
                        ,height = 0.2 * self.chartProp
                        ,plot_kws = {'s' : 2.0 * self.chartProp
                                    ,'edgecolor' : None
                                    ,'linewidth' : 1
                                    ,'alpha' : 0.4
                                    ,'marker' : 'o'
                                    ,'facecolor' : style.styleHexMid[0] if target is None else None
                            }
                        ,diag_kws = {'facecolor' : style.styleHexMid[1] if target is None else None
                            }
                        ,palette = style.styleHexMid
            )
        

        for ax in g.axes.flat:
            _ = ax.set_ylabel(ax.get_ylabel(), rotation = 0)
            _ = ax.set_xlabel(ax.get_xlabel(), rotation = 0)
            _ = ax.xaxis.labelpad = 20
            _ = ax.yaxis.labelpad = 75
            _ = ax.xaxis.label.set_color(style.styleGrey)
            _ = ax.yaxis.label.set_color(style.styleGrey)            
        
        plt.subplots_adjust(hspace = 0.0, wspace = 0.0)
        
        # Add custom legend describing hue labels
        if target is not None:
            g._legend.remove()
            
            ## create custom legend
            # create labels
            if legendLabels is None:
                legendLabels = np.unique(df[df[targetName].notnull()][targetName])
            else:
                legendLabels = np.array(legendLabels)

            labelColor = {}
            for ix, i in enumerate(legendLabels):
                labelColor[i] = style.styleHexMid[ix]

            # create patches
            patches = [matplotlib.patches.Patch(color = v, label = k) for k, v in labelColor.items()]
            
            # draw legend
            leg = plt.legend(handles = patches
                        ,fontsize = 1.3 * self.chartProp
                        ,loc = 'upper right'
                        ,markerscale = 0.5 * self.chartProp
                        ,ncol = 1
                        ,bbox_to_anchor = bbox_to_anchor
                )

            # label font color
            for text in leg.get_texts():
                plt.setp(text, color = 'Grey')


def prettyHist(self, x, color, label, alpha = 0.8):
    """
    Documentation:
        Description:
            Create histogram of continuous variable. Simple function capable of easy
            iteration through several groupings of a continuous variable that is 
            separated out based on a categorical label. This results in several overlapping
            histograms and can reveal differences in distributions.
        Parameters:
            x : array
                1-dimensional array of values to be plotted on x-axis.
            color : string (some sort of color code)
                Determines color of histogram
            label : string
                Category value label
            alpha : float, default = 0.8
                Fades histogram bars to create transparent bars            
    """
    # Create histogram.
    plt.hist(x = x
            ,color = color
            ,label = label
            ,alpha = alpha
            )