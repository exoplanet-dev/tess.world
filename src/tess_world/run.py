__all__ = ["run_one", "run_multiple", "run_all"]

import multiprocessing as mp
import os
import pathlib
import re
import time
from functools import partial

import jupytext
import nbformat
import pkg_resources
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

from .nexsci import get_toi_list
from .tess_world_version import __version__

PATH = (
    pathlib.Path(pkg_resources.resource_filename(__name__, "templates"))
    / "template.py"
)


def run_one(toi_num, output_directory="./results"):
    working_path = pathlib.Path(output_directory).resolve() / __version__
    output_path = working_path / f"{toi_num}"
    output_path.mkdir(parents=True, exist_ok=True)
    filename = output_path / f"toi{toi_num}.ipynb"

    # Load the template and insert the TOI number where it is required
    with open(PATH, "r") as f:
        txt = f.read()

    txt = txt.replace("{{{TOINUMBER}}}", "{0}".format(toi_num))
    txt = txt.replace("{{{VERSIONNUMBER}}}", "{0}".format(__version__))
    txt = re.sub(r"toi_num = [0-9]+", "toi_num = {0}".format(toi_num), txt)

    # Set the required environment variables
    compiledir = working_path / f"cache/{os.getpid()}"
    os.environ["THEANO_FLAGS"] = f"compiledir={compiledir}"
    os.environ["LIGHTKURVE_DIRECTORY"] = str(working_path / "cache/lightkurve")

    # Load the notebook as jupytext format
    notebook = jupytext.reads(txt, fmt="py:percent")

    # Execute the notebook
    ep = ExecutePreprocessor(timeout=-1)

    print("running: {0}".format(filename))
    try:
        strt = time.time()
        ep.preprocess(notebook, {"metadata": {"path": str(output_path)}})
    except CellExecutionError as e:
        total_time = time.time() - strt
        msg = "error while running: {0}\n\n".format(filename)
        msg += e.traceback
        print(msg)
        with open(output_path / "error.log", "w") as f:
            f.write(msg)
    else:
        total_time = time.time() - strt
        with open(filename, mode="wt") as f:
            nbformat.write(notebook, f)

    with open(output_path / "time.txt", "w") as f:
        f.write(f"{total_time} seconds\n")


def run_multiple(toi_nums, output_directory="./results"):
    num_toi = len(toi_nums)
    num_cpu = mp.cpu_count()
    num_jobs = num_cpu // 2

    print(f"Running {num_toi} fits in {num_jobs} jobs on {num_cpu} CPUs...")

    func = partial(run_one, output_directory=output_directory)
    with mp.Pool(num_jobs) as pool:
        for result in pool.imap_unordered(func, toi_nums):
            pass


def run_all(output_directory="./results"):
    tois = get_toi_list()
    run_multiple(tois.toipfx.unique(), output_directory=output_directory)
