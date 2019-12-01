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
            name: string, default = "viridis"
                name of built_in colormap
            num : int, default = 2
                an integer specifying the number of entries desired in the lookup table.
        returns:
            color_list : list
                list containing specified number of hex codes.
    """
    cmap = matplotlib.cm.get_cmap(name=name, lut=num)

    color_list = []
    for i in range(cmap.n):
        rgb = cmap(i)[:3]
        color_list.append(matplotlib.colors.rgb2hex(rgb))
    return color_list

style_line_style = ["_", "__", "_.", ":", "_", "__", "_.", ":", "_", "__", "_.", ":"]

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
    "font.family": ["deja_vu sans"]
    # ,'font.family': ['arial']
}

# style_hex_light = ["#8_ba4_d9", "#d8_f894", "#ffc799", "#e78_bc1", "#bc86_d9", "#7_dd1_d1"]
# style_hex_mid_light = ["#466_dc1", "#bcf347", "#ff9_c4_b", "#d84099", "#9340_c1", "#35_b4_b4"]
# style_hex_mid = ["#1449_bb", "#02e218", "#ff7401", "#d5017_d", "#7_c0_bbb", "#01_adad"]
# style_hex_mid_dark = ["#0_a3389", "#82_c000", "#cd5_d00", "#a3005_f", "#5_a0589", "#007_b7_b"]
# style_hex_dark = ["#04215_d", "#588200", "#8_b3_f00", "#6_e0041", "#3_d025_d", "#005353"]

# style_rgb_light = [
#     (139, 164, 217),
#     (216, 248, 148),
#     (255, 199, 153),
#     (231, 139, 193),
#     (188, 134, 217),
#     (125, 209, 209),
# ]
# style_rgb_mid_light = [
#     (70, 109, 193),
#     (188, 243, 71),
#     (255, 156, 75),
#     (216, 64, 153),
#     (147, 64, 193),
#     (53, 180, 180),
# ]
# style_rgb_mid = [
#     (20, 73, 187),
#     (2, 226, 24),
#     (255, 116, 1),
#     (213, 1, 125),
#     (124, 11, 187),
#     (1, 173, 173),
# ]
# style_rgb_mid_dark = [
#     (10, 51, 137),
#     (130, 192, 0),
#     (205, 93, 0),
#     (163, 0, 95),
#     (90, 5, 137),
#     (0, 123, 123),
# ]
# style_rgb_dark = [
#     (4, 33, 93),
#     (88, 130, 0),
#     (139, 63, 0),
#     (110, 0, 65),
#     (61, 2, 93),
#     (0, 83, 83),
# ]

# style_rgb0_light = [
#     (0.545, 0.643, 0.851),
#     (0.847, 0.973, 0.58),
#     (1, 0.78, 0.6),
#     (0.906, 0.545, 0.757),
#     (0.737, 0.525, 0.851),
#     (0.49, 0.82, 0.82),
# ]
# style_rgb0_mid_light = [
#     (0.275, 0.427, 0.757),
#     (0.737, 0.953, 0.278),
#     (1, 0.612, 0.294),
#     (0.847, 0.251, 0.6),
#     (0.576, 0.251, 0.757),
#     (0.208, 0.706, 0.706),
# ]
# style_rgb0_mid = [
#     (0.078, 0.286, 0.733),
#     (0.643, 0.949, 0.004),
#     (1, 0.455, 0.004),
#     (0.835, 0.004, 0.49),
#     (0.486, 0.043, 0.733),
#     (0.004, 0.678, 0.678),
# ]
# style_rgb0_mid_dark = [
#     (0.039, 0.2, 0.537),
#     (0.51, 0.753, 0),
#     (0.804, 0.365, 0),
#     (0.639, 0, 0.373),
#     (0.353, 0.02, 0.537),
#     (0, 0.482, 0.482),
# ]
# style_rgb0_dark = [
#     (0.016, 0.129, 0.365),
#     (0.345, 0.51, 0),
#     (0.545, 0.247, 0),
#     (0.431, 0, 0.255),
#     (0.239, 0.008, 0.365),
#     (0, 0.325, 0.325),
# ]