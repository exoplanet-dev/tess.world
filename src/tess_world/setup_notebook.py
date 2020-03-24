__all__ = ["setup_notebook"]

import logging
import multiprocessing as mp
import warnings

import matplotlib.pyplot as plt


def setup_notebook():
    # Deal with warnings from Theano
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    logging.getLogger("astropy").setLevel(logging.ERROR)
    logging.getLogger("theano.gof.compilelock").setLevel(logging.ERROR)
    logging.getLogger("exoplanet").setLevel(logging.DEBUG)

    # Temporary workaround for multiprocessing on macOS
    mp.set_start_method("fork")

    # Set up plotting the way we want it
    plt.style.use("default")
    plt.rcParams["savefig.dpi"] = 100
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["font.size"] = 16
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
    plt.rcParams["font.cursive"] = ["Liberation Sans"]
    plt.rcParams["mathtext.fontset"] = "custom"
