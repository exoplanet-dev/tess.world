__all__ = ["LightCurveModels"]

import theano.tensor as tt


class LightCurveModels:
    def __init__(self, mean, star, orbit, ror):
        self.mean = mean
        self.star = star
        self.orbit = orbit
        self.ror = ror

    def light_curves(self, t):
        return 1e3 * self.star.get_light_curve(
            orbit=self.orbit, r=self.ror, t=t
        )

    def __call__(self, t):
        return self.mean + tt.sum(self.light_curves(t), axis=-1)
