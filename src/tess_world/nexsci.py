__all__ = ["get_toi_list"]

import pathlib

import pandas as pd
import pkg_resources

PATH = pathlib.Path(pkg_resources.resource_filename(__name__, "nexsci"))
PATH.mkdir(parents=True, exist_ok=True)
PATH = PATH / "toi.csv"


def get_toi_list():
    return pd.read_csv(PATH)
