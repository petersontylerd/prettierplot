
import seaborn as sns
import matplotlib.cm
import matplotlib.colors

styleHexLight = ['#8BA4D9', '#D8F894', '#FFC799', '#E78BC1', '#BC86D9', '#7DD1D1']
styleHexMidLight = ['#466DC1', '#BCF347', '#FF9C4B', '#D84099', '#9340C1', '#35B4B4']
styleHexMid = ['#1449BB', '#02e218', '#FF7401', '#D5017D', '#7C0BBB', '#01ADAD']
styleHexMidDark = ['#0A3389', '#82C000', '#CD5D00', '#A3005F', '#5A0589', '#007B7B']
styleHexDark = ['#04215D', '#588200', '#8B3F00', '#6E0041', '#3D025D', '#005353']

styleRgbLight = [(139, 164, 217), (216, 248, 148), (255, 199, 153), (231, 139, 193), (188, 134, 217), (125, 209, 209)]
styleRgbMidLight = [( 70, 109, 193), (188, 243, 71), (255, 156, 75), (216, 64, 153), (147, 64, 193), ( 53, 180, 180)]
styleRgbMid = [( 20, 73, 187), (2, 226, 24), (255, 116,  1), (213, 1, 125), (124, 11, 187), (  1, 173, 173)]
styleRgbMidDark = [( 10, 51, 137), (130, 192,  0), (205, 93,  0), (163, 0, 95), ( 90, 5, 137), (  0, 123, 123)]
styleRgbDark = [( 4, 33, 93), ( 88, 130, 0), (139, 63, 0), (110, 0, 65), ( 61, 2, 93), ( 0, 83, 83)]

styleRgb0Light = [(0.545, 0.643, 0.851), (0.847, 0.973, 0.58), (1, 0.78, 0.6), (0.906, 0.545, 0.757), (0.737, 0.525, 0.851), (0.49, 0.82, 0.82)]
styleRgb0MidLight = [(0.275, 0.427, 0.757), (0.737, 0.953, 0.278), (1, 0.612, 0.294), (0.847, 0.251, 0.6), (0.576, 0.251, 0.757), (0.208, 0.706, 0.706)]
styleRgb0Mid = [(0.078, 0.286, 0.733), (0.643, 0.949, 0.004), (1, 0.455, 0.004), (0.835, 0.004, 0.49), (0.486, 0.043, 0.733), (0.004, 0.678, 0.678)]
styleRgb0MidDark = [(0.039, 0.2, 0.537), (0.51, 0.753, 0), (0.804, 0.365, 0), (0.639, 0, 0.373), (0.353, 0.02, 0.537), (0, 0.482, 0.482)]
styleRgb0Dark = [(0.016, 0.129, 0.365), (0.345, 0.51, 0), (0.545, 0.247, 0), (0.431, 0, 0.255), (0.239, 0.008, 0.365), (0, 0.325, 0.325)]

styleLineStyle = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'] 

styleMarkers = ('s', 'o', 'v', 'x', '^')
        
styleWhite = (255 / 255, 255 / 255, 255 / 255)
styleGrey = (105 / 255, 105 / 255, 105 / 255)

def genCmap(nColors, colorList):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('', colorList)
    matplotlib.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors = nColors, desat = 1.0)
    return cpal


# rc parameters
rcGrey = {'axes.titlesize' : 50.0
            ,'axes.labelsize' : 40.0   # Axis title font size
            ,'axes.facecolor': styleWhite
            ,'axes.edgecolor': styleWhite
            ,'axes.grid': False
            ,'axes.axisbelow': True
            ,'axes.labelcolor': styleGrey
            ,'axes.spines.left': True
            ,'axes.spines.bottom': True
            ,'axes.spines.right': False
            ,'axes.spines.top': False
            ,'xtick.labelsize' : 25.0
            ,'xtick.color': styleGrey
            ,'xtick.direction': 'out'
            ,'xtick.bottom': True
            ,'xtick.top': False
            ,'xtick.major.size' : 10.0
            ,'xtick.major.width' : 3.0
            ,'ytick.labelsize' : 25.0
            ,'ytick.color': styleGrey
            ,'ytick.direction': 'out'
            ,'ytick.left': True
            ,'ytick.right': False
            ,'ytick.major.size' : 10.0
            ,'ytick.major.width' : 3.0
            ,'figure.facecolor' : styleWhite
            ,'font.family': ['Arial']
            }