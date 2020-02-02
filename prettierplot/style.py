import seaborn as sns
import matplotlib.cm
import matplotlib.colors


def color_gen(name="viridis", num=2):
    """
    documentation:
        description:
            generates a list of hex color codes of a specified length from a specified
             color map.
        parameters
            name: string, default="viridis"
                name of built_in colormap
            num : int, default=2
                an integer specifying the number of entries desired in the lookup table.
        returns:
            color_list : list
                list containing specified number of hex codes.
    """
    cmap = matplotlib.cm.get_cmap(name=name, lut=num)

    color_list = []
    for i in range(cmap.N):
        rgb = cmap(i)[:3]
        color_list.append(matplotlib.colors.rgb2hex(rgb))
    return color_list


style_line_style = ["-", "--", "-.", ":","-", "--", "-.", ":","-", "--", "-.", ":"]

style_markers = ("s", "o", "v", "x", "^")

style_white = (255 / 255, 255 / 255, 255 / 255)
style_grey = (105 / 255, 105 / 255, 105 / 255)
style_blue = (20 / 255, 73 / 255, 187 / 255)
style_green = (2 / 255, 226 / 255, 24 / 255)
style_orange = (255 / 255, 115 / 255, 1 / 255)


def gen_cmap(n_colors, color_list):
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", color_list)
    matplotlib.cm.register_cmap("mycolormap", cmap)
    cpal = sns.color_palette("mycolormap", n_colors=n_colors, desat=1.0)
    return cpal


# rc parameters
rc_grey = {
    "axes.titlesize": 50.0,
    "axes.labelsize": 40.0,  # axis title font size
    "axes.facecolor": "white",
    "axes.edgecolor": style_white,
    "axes.grid": False,
    "axes.axisbelow": True,
    "axes.labelcolor": style_grey,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "grid.color": "white",
    "xtick.labelsize": 15.0,
    "xtick.color": style_grey,
    "xtick.direction": "out",
    "xtick.bottom": True,
    "xtick.top": False,
    "xtick.major.size": 6.0,
    "xtick.major.width": 1.25,
    "ytick.labelsize": 15.0,
    "ytick.color": style_grey,
    "ytick.direction": "out",
    "ytick.left": True,
    "ytick.right": False,
    "ytick.major.size": 6.0,
    "ytick.major.width": 1.25,
    "figure.facecolor": "white",
    "font.family": ["DejaVu Sans"]
}