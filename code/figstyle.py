"""One look for every figure: serif type, Okabe-Ito colours, each model keeps its colour and marker."""
import matplotlib as mpl

MODELS = ["lr", "ridge", "lgbm", "tabpfn", "tabicl"]
LABEL = {"lr": "Logistic regression", "ridge": "Ridge regression", "lgbm": "LightGBM",
         "tabpfn": "TabPFN", "tabicl": "TabICL"}
SHORT = {"lr": "Logistic", "ridge": "Ridge", "lgbm": "LightGBM", "tabpfn": "TabPFN", "tabicl": "TabICL"}
# classical models in cool colours, the AI foundation models in warm ones
COLOUR = {"lr": "#0072B2", "ridge": "#56B4E9", "lgbm": "#009E73", "tabpfn": "#E69F00", "tabicl": "#D55E00"}
MARKER = {"lr": "o", "ridge": "s", "lgbm": "D", "tabpfn": "^", "tabicl": "v"}
FOUNDATION = {"tabpfn", "tabicl"}
GREY = "#6E6E6E"
LIGHT = "#D9D9D9"

TEXT_WIDTH_IN = 6.15     # the manuscript's text block is 446.7 pt = 6.18 in

COHORT = {"AAA_2013J_2014J": "AAA Oct", "BBB_2013B_2014B": "BBB Feb", "BBB_2013J_2014J": "BBB Oct",
          "DDD_2013B_2014B": "DDD Feb", "DDD_2013J_2014J": "DDD Oct", "EEE_2013J_2014J": "EEE Oct",
          "FFF_2013B_2014B": "FFF Feb", "FFF_2013J_2014J": "FFF Oct", "GGG_2013J_2014J": "GGG Oct"}


def use():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.edgecolor": "#333333", "xtick.color": "#333333", "ytick.color": "#333333",
        "axes.labelcolor": "#222222", "text.color": "#222222",
        "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    })
