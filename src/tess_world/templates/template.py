# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TESS Atlas fit for TOI {{{TOINUMBER}}}
#
# **Version: {{{VERSIONNUMBER}}}**
#
# **Note: This notebook was automatically generated as part of the tess.world project. More information can be found at:** [tess.world](https://tess.world)
#
# In this notebook, we do a full probabilistic characterization of the TESS Objects of Interest (TOIs) in the system number {{{TOINUMBER}}}.
# To do this fit, we use the [exoplanet](https://docs.exoplanet.codes) library and you can find more information about that project at [docs.exoplanet.codes](https://docs.exoplanet.codes).
#
# ## Caveats
#
# There are many caveats associated with an automated bulk analysis that should be kept in mind.
# Here are some things that come to mind:
#
# 1. Transit timing variations, correlated noise, and (probably) your favorite systematics are ignored. Sorry!
#
# 2. This notebook was generated automatically without human intervention. Use at your own risk!
#
# ## Table of Contents
#
# 1. [Getting started](#Getting-started)
# 2. [Data access and pre-processing](#Data-access-and-pre-processing)
# 3. [The probabilistic model and initialization](#The-probabilistic-model-and-initialization)
# 5. [Inference](#Inference)
# 6. [Results](#Results)
# 7. [Attribution and environment](#Attribution-and-environment)
#
# ## Getting started
#
# To get going, we'll need to make out plots show up inline and import all the required packages.
# Note that this notebook is mostly self contained, but for technical reasons, there are a few helper functions implemented in the [tess_world package](https://github.com/exoplanet-dev/tess.world) so, if you're running this notebook locally, you'll need to install that and its dependencies.

# %%
# %matplotlib inline

# %%
# %config InlineBackend.figure_format = "retina"

import json
import os
import pathlib
import pickle

import corner
import exoplanet as xo
import h5py
import lightkurve as lk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
from astroquery.mast import Observations

import tess_world

tess_world.setup_notebook()

# %%
tois = tess_world.get_toi_list()
len(tois.toipfx.unique())

# %% [markdown]
# ## Data access and pre-processing
#
# Now we will get the information about this TOI from the [Exoplanet Archive TOI table](https://exoplanetarchive.ipac.caltech.edu/docs/API_toi_columns.html).
# These parameters will be used as an initial guess for the fit.

# %%
toi_num = 514

tois = tess_world.get_toi_list()

# Select all of the rows in the TOI table that are associated with this target
toi = tois[tois.toi == toi_num + 0.01]
if not len(toi):
    raise RuntimeError(f"no TOI entry for {toi_num}")
toi = toi.iloc[0]
tic = toi.tid
tois = tois[tois.tid == tic].sort_values("toi")
num_toi = len(tois)

# Extract the planet periods
period_guess = np.array(tois.pl_orbper, dtype=float)

# Convert the phase to TBJD from BJD
t0_guess = np.array(tois.pl_tranmid, dtype=float) - 2457000

# Convert the depth to parts per thousand from parts per million
depth_guess = 1e-3 * np.array(tois.pl_trandep, dtype=float)

# Convert the duration to days from hours
duration_guess = np.array(tois.pl_trandurh, dtype=float) / 24.0

tois[["tid", "toi", "pl_orbper", "pl_trandep", "pl_trandurh"]]

# %% [markdown]
# Then we can search for and download the light curves.
# Note that this will fail if there is no 2-minute cadence light curve for this target.
# Typically this would be executed using [lightkurve](https://docs.lightkurve.org) directly, but we'll use the MAST API directly because the lightkurve light curve search is a little slow currently.

# %%
# Temporary workaround for slow MAST queries with lightkurve
observations = Observations.query_criteria(
    target_name=f"{tic}",
    radius=0.0001,
    project=["TESS"],
    obs_collection=["TESS"],
    provenance_name="SPOC",
    dataproduct_type="timeseries",
)
if not len(observations):
    raise RuntimeError("no 2-minute cadence data")
products = Observations.get_product_list(observations)
products = products[products["productSubGroupDescription"] == "LC"]
files = Observations.download_products(
    products, download_dir=tess_world.get_lightkurve_directory()
)
lcfs = lk.LightCurveCollection(
    [lk.open(file).PDCSAP_FLUX for file in files["Local Path"]]
)
lc = lcfs.stitch().remove_nans()

# Extract the data in the correct format
x = np.ascontiguousarray(lc.time, dtype=np.float64)
y = np.ascontiguousarray(1e3 * (lc.flux - 1), dtype=np.float64)
yerr = np.ascontiguousarray(1e3 * lc.flux_err, dtype=np.float64)

# Plot the light curve
plt.plot(x, y, "k", linewidth=0.5)
plt.xlabel("time [days]")
plt.ylabel("relative flux [ppt]")
plt.title(f"TOI {toi_num}; TIC {tic}", fontsize=14)

# Label the transits on the plot
for n in range(num_toi):
    t = float(t0_guess[n])
    label = f"TOI {toi_num}.{n + 1:02d}"
    while t < x.max():
        plt.axvline(t, color=f"C{n}", alpha=0.3, lw=3, label=label)
        t += period_guess[n]
        label = None

plt.xlim(x.min(), x.max())
_ = plt.legend(fontsize=10)

# %% [markdown]
# We need to make one last adjustment to our initial parameters if any of the TOIs are labeled as a single transit because we'll handle those differently.
# In particular, for single transits, we'll make the assumption that the period must be *at least* long enough that a second transit could not have occurred in the observational window.
# This is a strong assumption (because the second transit could have been in a data gap), but it'll do for this analysis.

# %%
# Deal with single transits
single_transit = period_guess <= 0.0
period_guess[single_transit] = x.max() - x.min()
period_min = np.maximum(np.abs(t0_guess - x.max()), np.abs(x.min() - t0_guess))

# %% [markdown]
# Finally, we extract just the data near the transits.
# This helps speed up the analysis and will only limit our precision for stars with extremely coherent variability, and then probably only marginally.

# %%
transit_mask = np.zeros_like(x, dtype=bool)
for n in range(num_toi):
    delta = max(1.5 * duration_guess[n], 0.1)
    if single_transit[n]:
        delta = 1.0
    x_fold = (x - t0_guess[n] + 0.5 * period_guess[n]) % period_guess[
        n
    ] - 0.5 * period_guess[n]
    m = np.abs(x_fold) < delta
    transit_mask |= m

    plt.figure(figsize=(8, 4))
    plt.scatter(x_fold[m], y[m], c=x[m], s=3)
    plt.xlabel("time since transit [days]")
    plt.ylabel("relative flux [ppt]")
    plt.colorbar(label="time [days]")
    plt.title(
        f"TOI {toi_num}.{n + 1:02d}, PDC flux, period = {period_guess[n]:.3f} d",
        fontsize=14,
    )
    plt.xlim(-delta, delta)

x = np.ascontiguousarray(x[transit_mask])
y = np.ascontiguousarray(y[transit_mask])
yerr = np.ascontiguousarray(yerr[transit_mask])


# %% [markdown]
# ## The probabilistic model and initialization
#
# Here's how we set up the transit model using [exoplanet](https://docs.exoplanet.codes) and [PyMC3](https://docs.pymc.io).
# For more information about how to use these libraries take a look at the docs that are linked above.
# In this model, the parameters that we're fitting are:
#
# * `mean`: the mean (out-of-transit) flux of the star,
# * `u`: the quadratic limb darkening parameters, parameterized following [Kipping (2013)](https://arxiv.org/abs/1308.0009)
# * `sigma`: a jitter parameter that captures excess white noise or underestimated error bars,
# * `S_tot`: the total power in a [celerite](https://celerite.readthedocs.io) Gaussian process model (a [SHOTerm](https://docs.exoplanet.codes/en/stable/user/api/#exoplanet.gp.terms.SHOTerm) to be precise) for low-frequency variability,
# * `ell`: the characteristic time scale of the Gaussian Process model,
# * `period`: the orbital period with either a log-normal or power-law prior (the latter if the TOI is a single transit),
# * `t0`: the mid-transit time of a reference transit for each planet,
# * `transit_depth`: the transit depth in parts per thousand, assuming the small-planet approximation,
# * `transit_duration`: the transit duration in days, and
# * `b`: the impact parameter of the orbit (note that this is constrained to the range $0 < b < 1$ so this won't deal gracefully with grazing transits).
#
# A few key assumptions include:
#
# * The orbits are assumed to be circular, but we fit for both period and duration so this model is flexible enough to fit eccentric orbits to first order (see, for example, [Dawson & Johnson 2012](https://arxiv.org/abs/1203.5537)).
# * We are neglecting transit times (the ephemeris is assumed to be linear) which should be sufficient for most cases with the short TESS baseline, but transit timing variations could be important for some targets.
#
# Finally, note that the model is implemented inside of a model "factory" so that we can iteratively clip outliers.

# %%
def build_model(mask=None):
    if mask is None:
        mask = np.ones_like(x, dtype=bool)

    with pm.Model() as model:
        # Stellar parameters
        mean = pm.Normal("mean", mu=0.0, sigma=10.0)
        u = xo.distributions.QuadLimbDark("u")

        # Gaussian process noise model
        sigma = pm.InverseGamma("sigma", alpha=3.0, beta=2 * np.median(yerr))
        S_tot = pm.Lognormal(
            "S_tot",
            mu=np.log(np.median((y[mask] - np.median(y[mask])) ** 2)),
            sigma=5.0,
        )
        ell = pm.Lognormal("ell", mu=np.log(1.0), sigma=5.0)
        Q = 1.0 / 3.0
        w0 = 2 * np.pi / ell
        S0 = S_tot / (w0 * Q)
        kernel = xo.gp.terms.SHOTerm(S0=S0, w0=w0, Q=Q)

        # Dealing with period, treating single transits properly
        period_params = []
        for n in range(num_toi):
            if single_transit[n]:
                period = pm.Pareto(
                    f"period_{n}",
                    m=period_min[n],
                    alpha=2.0 / 3,
                    testval=period_guess[n],
                )
            else:
                period = pm.Lognormal(
                    f"period_{n}", mu=np.log(period_guess[n]), sigma=1.0
                )
            period_params.append(period)
        period = pm.Deterministic("period", tt.stack(period_params))

        # Transit parameters
        t0 = pm.Normal("t0", mu=t0_guess, sigma=1.0, shape=num_toi)
        depth = pm.Lognormal(
            "transit_depth", mu=np.log(depth_guess), sigma=5.0, shape=num_toi
        )
        duration = pm.Lognormal(
            "transit_duration",
            mu=np.log(duration_guess),
            sigma=5.0,
            shape=num_toi,
        )
        b = xo.distributions.UnitUniform("b", shape=num_toi)

        # Compute the radius ratio from the transit depth, impact parameter, and
        # limb darkening parameters making the small-planet assumption
        u1 = u[0]
        u2 = u[1]
        mu = tt.sqrt(1 - b ** 2)
        ror = pm.Deterministic(
            "ror",
            tt.sqrt(
                1e-3
                * depth
                * (1 - u1 / 3 - u2 / 6)
                / (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)
            ),
        )

        # Set up the orbit
        orbit = xo.orbits.KeplerianOrbit(
            period=period, duration=duration, t0=t0, b=b
        )

        # We're going to track the implied density for reasons that will become clear later
        pm.Deterministic("rho_circ", orbit.rho_star)

        # Set up the mean transit model
        star = xo.LimbDarkLightCurve(u)

        lc_model = tess_world.LightCurveModels(mean, star, orbit, ror)

        # Finally the GP observation model
        gp = xo.gp.GP(
            kernel, x[mask], yerr[mask] ** 2 + sigma ** 2, mean=lc_model
        )
        gp.marginal("obs", observed=y[mask])

        # Double check that everything looks good - we shouldn't see any NaNs!
        print(model.check_test_point())

        # Optimize the model
        map_soln = model.test_point
        map_soln = xo.optimize(map_soln, [sigma])
        map_soln = xo.optimize(map_soln, [mean, depth, b, duration])
        map_soln = xo.optimize(map_soln, [sigma, S_tot, ell])
        map_soln = xo.optimize(map_soln, [mean, u])
        map_soln = xo.optimize(map_soln, period_params)
        map_soln = xo.optimize(map_soln)

        # Save some of the key parameters
        model.map_soln = map_soln
        model.lc_model = lc_model
        model.gp = gp
        model.mask = mask
        model.x = x[mask]
        model.y = y[mask]
        model.yerr = yerr[mask]

    return model


def build_model_with_sigma_clipping(sigma=5.0, maxiter=10):
    ntot = len(x)
    mask = np.ones_like(x, dtype=bool)
    pred = np.zeros_like(y)
    for i in range(maxiter):
        print(f"Sigma clipping round {i + 1}")

        with build_model(mask) as model:
            pred[mask] = xo.eval_in_model(
                model.gp.predict() + model.lc_model(x[mask]), model.map_soln
            )
            if np.any(~mask):
                pred[~mask] = xo.eval_in_model(
                    model.gp.predict(x[~mask]) + model.lc_model(x[~mask]),
                    model.map_soln,
                )

        resid = y - pred
        rms = np.sqrt(np.median(resid ** 2))
        mask = np.abs(resid) < sigma * rms

        print(
            f"... clipping {(~mask).sum()} of {len(x)} ({100 * (~mask).sum() / len(x):.1f}%)"
        )

        if ntot == mask.sum():
            break
        ntot = mask.sum()

    return model


model = build_model_with_sigma_clipping()

# %% [markdown]
# Now, after building the model, clipping outliers, and optimizing to estimate the maximum a posteriori (MAP) parameters, we can visualize our initial fit.

# %%
with model:
    gp_pred, lc_pred = xo.eval_in_model(
        [model.gp.predict(), model.lc_model.light_curves(model.x)],
        model.map_soln,
    )

for n in range(num_toi):
    t0 = model.map_soln["t0"][n]
    period = model.map_soln["period"][n]
    x_fold = (model.x - t0 + 0.5 * period) % period - 0.5 * period

    plt.figure(figsize=(8, 4))
    plt.scatter(
        x_fold, model.y - gp_pred - model.map_soln["mean"], c=model.x, s=3
    )

    inds = np.argsort(x_fold)
    plt.plot(x_fold[inds], lc_pred[inds, n], "k")

    plt.xlabel("time since transit [days]")
    plt.ylabel("de-trended flux [ppt]")
    plt.colorbar(label="time [days]")
    plt.title(
        f"TOI {toi_num}.{n + 1:02d}, map model, period = {period:.3f} d",
        fontsize=14,
    )
    delta = max(1.5 * duration_guess[n], 0.1)
    if single_transit[n]:
        delta = 1.0
    plt.xlim(-delta, delta)

# %% [markdown]
# In this figure, the colors represent the time at which the datapoint was collected.
# This can show us if a subset of the transits are systematically off.
#
# ## Inference
#
# Now we can get to the good stuff and fit our transit light curve using this probabilistic model and PyMC3's support for Markov chain Monte Carlo (MCMC).
# The settings here have been chosen to be reasonably sensible defaults in most cases, but you might be able to get better perfomance for your favorite system by tweaking them.

# %%
np.random.seed(toi_num)
with model:
    trace = pm.sample(
        tune=2000,
        draws=2000,
        start=model.map_soln,
        chains=2,
        cores=2,
        step=xo.get_dense_nuts_step(target_accept=0.9),
    )

# %% [markdown]
# After running an MCMC, it's good to look at some summary statistics to see how things went.
# If any of the `r_hat` values are greater than `1` or if any of the entries in the `ess_mean` column are very small, that suggests that something went wrong while sampling (maybe it's a grazing transit?).
# Another thing to look for is the number of "divergences" listed above.
# In most cases, there won't be any divergences, but when the model assumptions are not satisfied, this number can indicate that something went wrong.
# If there were more than about 10 divergences per chain, you should be cautious about the results.
# Again, this often suggests that the transit shape is consistent with a grazing transit.

# %%
pm.summary(trace)

# %% [markdown]
# Then, before we go any further, let's save the results of the MCMC to disk so that we don't lose them.

# %%
output_directory = pathlib.Path(os.environ.get("OUTPUT_DIRECTORY", "."))

# Save the model
with open(output_directory / "model.pkl", "wb") as f:
    pickle.dump(model, f, -1)

# Save the MAP solution
with open(output_directory / "map_soln.json", "w") as f:
    json.dump(model.map_soln, f, indent=2, cls=tess_world.NumpyEncoder)

# Save the summary statistics
summary = pm.summary(trace, round_to="none")
summary.to_csv(output_directory / "summary.csv")

# Save the trace
df = pm.trace_to_dataframe(trace, include_transformed=True)
stats = pd.DataFrame(
    dict((name, trace.get_sampler_stats(name)) for name in trace.stat_names)
)
with h5py.File(output_directory / "trace.h5", "w") as f:
    f.create_dataset("trace", data=df.to_records(index=False))
    f.create_dataset("stats", data=stats.to_records(index=False))

# %% [markdown]
# ## Results
#
# Finally, we can look at some of the results of our inference.
# One of the key figures is the posterior distribution of transit models overplotted on the light curve.
# Here we're removing the MAP prediction from the Gaussian Process and then plotting 100 random transit models from the chain to get a sense for the uncertainty.

# %%
fig, axes = plt.subplots(
    nrows=num_toi, figsize=(8, 3 * num_toi), squeeze=False
)
axes = axes[:, 0]

for n in range(num_toi):
    t0 = np.median(trace["t0"][:, n])
    period = np.median(trace["period"][:, n])
    x_fold = (model.x - t0 + 0.5 * period) % period - 0.5 * period

    delta = max(1.5 * duration_guess[n], 0.1)
    if single_transit[n]:
        delta = 1.0

    m = np.abs(x_fold) < delta
    x0 = 24 * x_fold[m]
    y0 = (model.y - gp_pred - np.median(trace["mean"]))[m]
    axes[n].plot(x0, y0, ".k", label="data", alpha=0.3, mec="none")

    bins = np.linspace(-24 * delta, 24 * delta, 36)
    num, _ = np.histogram(x0, bins, weights=y0)
    denom, _ = np.histogram(x0, bins)
    axes[n].plot(
        0.5 * (bins[1:] + bins[:-1]), num / denom, "ok", label="binned data"
    )

    axes[n].set_ylabel("de-trended flux [ppt]")
    axes[n].set_xlim(-24 * delta, 24 * delta)

ylim = [ax.get_ylim() for ax in axes]

with model:
    period = np.median(trace["period"], axis=0)
    func = xo.utils.get_theano_function_for_var(
        model.lc_model.light_curves(model.x)
    )
    labels = [
        f"TOI {toi_num}.{n + 1:02d}, P = {period[n]:.3f} d"
        for n in range(num_toi)
    ]
    for point in xo.utils.get_samples_from_trace(trace, size=100):
        lcs = func(*xo.utils.get_args_for_theano_function(point))
        for n in range(num_toi):
            t0 = point["t0"][n]
            period = point["period"][n]
            x_fold = (model.x - t0 + 0.5 * period) % period - 0.5 * period
            inds = np.argsort(x_fold)
            axes[n].plot(
                24 * x_fold[inds],
                lcs[inds, n],
                f"C{n}",
                alpha=0.1,
                lw=0.75,
                label=labels[n],
            )
            labels[n] = None

[ax.legend(fontsize=10, loc=3) for ax in axes]
axes[-1].set_xlabel("time since transit [hours]")
axes[0].set_title(f"TOI {toi_num}, posterior inference", fontsize=14)
_ = [axes[n].set_ylim(ylim[n]) for n in range(num_toi)]

fig.savefig(output_directory / "figure.png", dpi=300, bbox_inches="tight")

# %% [markdown]
# Then we have the posterior constraint on the period (or periods, if there are multiple TOIs).
# For display purposes, it is useful to plot the distribution of the difference (measured in minutes) between the sampled period and a fiducial period (the posterior median in this case).
# If any of the TOIs just have a single transit, we plot the logarithm of the period in days instead since it won't be well constrained.

# %%
median_period = np.median(trace["period"], axis=0)
samples = np.array(trace["period"])
samples[:, ~single_transit] = (
    24
    * 60
    * (samples[:, ~single_transit] - median_period[None, ~single_transit])
)
samples[:, single_transit] = np.log10(samples[:, single_transit])
labels = [
    f"$\log_{{10}} P_{n + 1} / \mathrm{{day}}$"
    if single_transit[n]
    else f"$\Delta P_{n + 1}$ [min]"
    for n in range(num_toi)
]

fig = corner.corner(samples, labels=labels)
for n, ax in enumerate(np.diag(np.array(fig.axes).reshape(num_toi, num_toi))):
    if single_transit[n]:
        continue
    ax.set_title(f"$P_\mathrm{{ref}} = {median_period[n]:.6f}$", fontsize=14)

# %% [markdown]
# The other physical parameters that have relevant covariances are the radius ratio, the impact parameter, and the transit duration.
# (Remember that we're fitting in transit depth, not radius ratio, so the ratio is a derived quantity.)

# %%
samples = np.concatenate(
    (trace["ror"], trace["b"], trace["transit_duration"] * 24), axis=1
)
labels = [f"$R_{n + 1} / R_S$" for n in range(num_toi)]
labels += [f"$b_{n + 1}$" for n in range(num_toi)]
labels += [f"$\\tau_{n + 1}$ [hr]" for n in range(num_toi)]

_ = corner.corner(samples, labels=labels)

# %% [markdown]
# Those are all the plots that we'll make for now, but remember that you can use the samples that we save above to make more figures after the fact if you find something that you want to see.
#
# ## Attribution and environment
#
# If you use these results or this code, please consider citing the relevant sources.
# First, you can [cite the lightkurve package](https://docs.lightkurve.org/about/citing.html):
#
# ```bibtex
# @misc{lightkurve,
#    author = {{Lightkurve Collaboration} and {Cardoso}, J.~V.~d.~M. and
#              {Hedges}, C. and {Gully-Santiago}, M. and {Saunders}, N. and
#              {Cody}, A.~M. and {Barclay}, T. and {Hall}, O. and
#              {Sagear}, S. and {Turtelboom}, E. and {Zhang}, J. and
#              {Tzanidakis}, A. and {Mighell}, K. and {Coughlin}, J. and
#              {Bell}, K. and {Berta-Thompson}, Z. and {Williams}, P. and
#              {Dotson}, J. and {Barentsen}, G.},
#     title = "{Lightkurve: Kepler and TESS time series analysis in Python}",
#  keywords = {Software, NASA},
# howpublished = {Astrophysics Source Code Library},
#      year = 2018,
#     month = dec,
# archivePrefix = "ascl",
#    eprint = {1812.013},
#    adsurl = {http://adsabs.harvard.edu/abs/2018ascl.soft12013L},
# }
# ```
#
# You can also [cite the exoplanet project and its dependencies](https://docs.exoplanet.codes/en/stable/tutorials/citation/) using the following acknowledgement:

# %%
with model:
    txt, bib = xo.citations.get_citations_for_model()
print(txt)

# %% [markdown]
# Where the citations reference the BibTeX entries in the `bib` variable.
# See the [tess.world citation page](https://tess.world/citation) for more information.
#
# Finally, this notebook was executed in a [conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with the following environment:

# %%
# !conda env export | grep -v "name:" | grep -v "prefix:"

# %%
