
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


class PrettierPlot():
    """
    Documentation:
        Description:
            PrettierPlot is a class that contain methods for creating data visualization.
            Initialization of this class creates a plotting object of a chosen size and
            orientation. Once the figure is initialized, the method makeCanvas is called
            to create the figure axis or chosen number of axes. If multiple axes are 
            specified, then multiple axes can be plotted on a single figure, or the 
            position variable can be utilized to create a subplot arrangement.        
    """        
    
    from .cat import prettyBarV, prettyBarH, prettyBoxPlotV, prettyBoxPlotH
    from .eval import prettyProbPlot, prettyCorrHeatmap, prettyCorrHeatmapRefine, prettyConfusionMatrix, \
                    prettyRocCurve, prettyDecisionRegion, prettyResidualPlot
    from .facet import prettyFacetCat, prettyFacetTwoCatBar, prettyFacetCatNumHist, prettyFacetTwoCatPoint
    from .line import prettyLine
    from .num import pretty2dScatter, pretty2dScatterHue, prettyDistPlot, prettyKdePlot, prettyRegPlot, \
                    prettyPairPlot, prettyHist
    

    # Foundation
    def __init__(self, chartProp = 15):
        """
        Documentation:
            Description: 
                Initialize PrettierPlot, create figure and determine chart proportions, orientation.        
            Parameters:
                chartProp : float or int, default = 15
                    Chart proportionality control. Determines relative size of figure size, axis labels, 
                    chart title, tick labels, tick marks.
        """
        self.chartProp = chartProp
    
        
    def makeCanvas(self, title = '', xLabel = '', yLabel = '', yShift = 0.8, position = 111, plotOrientation = None):
        """
        Documentation:
            Description: 
                Create Axes object. Add descriptive attributes such as titles and axis labels,
                set font size and font color. Remove grid. Remove top and right spine. 
            Parameters:
                fig : figure object, default = plt.figure()
                    matplotlib.pyplot figure object is a top level container for all plot elements.
                title : string, default = '' (blank)
                    The title for the chart.
                xLabel : string, default = '' (blank)
                    x-axis label.
                yLabel : string, default = '' (blank)
                    y-axis label.
                yShift : float, default = 0.8
                    Controls position of y-axis label. Higher values move label higher along axis. 
                    Intent is to align with top of axis.
                position int (nrows, ncols, index) : default = 111
                    Determine subplot position of plot.
                orientation : string, default = None
                    Default value produces a plot that is wider than it is tall. Specifying 'tall' will 
                    produce a taller, less wide plot. 'square' produces a square plot. 'wide' produces a
                    plot that is much wide than it is tall.

            Returns 
                ax : Axes object
                    Contain figure elements
        """        
        plt.rcParams['figure.facecolor'] = 'white'
        fig = plt.figure()
        sns.set(rc = style.rcGrey)

        # Dynamically set chart width and height parameters.
        if plotOrientation == 'tall':
            chartWidth = self.chartProp * .7
            chartHeight = self.chartProp * .8
        elif plotOrientation == 'square':
            chartWidth = self.chartProp
            chartHeight = self.chartProp * .8
        elif plotOrientation == 'wide':
            chartWidth = self.chartProp * 1.7
            chartHeight = self.chartProp * .32
        else:            
            chartWidth = self.chartProp
            chartHeight = self.chartProp * .5
        fig.set_figheight(chartHeight)
        fig.set_figwidth(chartWidth)
        
        ax = fig.add_subplot(position)
        
        # Add title.
        if len(title) >= 45:
            fontAdjust = 1.0
        elif len(title) >= 30 and len(title) < 45:
            fontAdjust = 1.25
        elif len(title) >= 20 and len(title) < 30:
            fontAdjust = 1.5
        else:
            fontAdjust = 1.75

        ax.set_title(title
                    ,fontsize = 2.0 * self.chartProp if position == 111 else fontAdjust * self.chartProp
                    ,color = style.styleGrey
                    ,loc = 'left'
                    ,pad = 0.4 * self.chartProp)
        
        # Remove grid line and right/top spines.
        ax.grid(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # ax.set_facecolor('white')

        # Add axis labels.
        plt.xlabel(xLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (0.5, 0.5))
        plt.ylabel(yLabel, fontsize = 1.667 * self.chartProp, labelpad = 1.667 * self.chartProp, position = (1.0, yShift))
        return ax