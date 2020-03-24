__all__ = ["NumpyEncoder"]

import json


class NumpyEncoder(json.JSONEncoder):
    """Encode numpy arrays to JSON - use at your own risk!"""

    def default(self, o):
        try:
            return o.tolist()
        except AttributeError:
            return json.JSONEncoder.default(self, o)
