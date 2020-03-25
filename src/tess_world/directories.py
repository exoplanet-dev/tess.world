__all__ = ["get_lightkurve_directory"]

import os


def get_lightkurve_directory():
    return os.environ.get("LIGHTKURVE_DIRECTORY", "./cache/lightkurve")
