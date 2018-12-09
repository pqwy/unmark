# Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
# See LICENSE.md

"""Estimators divine the "true" per-run value of counters from the data.

Estimators are `(mx, confidence) -> Estimate' functions.

`mx' is array-like, with `mx [:, 0]' independent and `mx [:, 1]' dependent,
while `confidence' is the interval probability."""

from .modstub import requires
import numpy as np
from numpy import *
from collections import namedtuple

__all__ = ['ols', 'tse', 'tse_bootstrap']

try:
    from scipy import stats

    def ppf (confidence):
        return stats.norm.ppf ((1 + confidence) / 2)

except ModuleNotFoundError:

    def ppf (confidence):
        if confidence != 0.99:
            raise Exception ("Could not load SciPy. Confidence must be 0.99.")
        return 2.575829303

Estimate = namedtuple ('Estimate', ['a', 'b', 'b_min', 'b_max', 'r2'])
Estimate.__repr__ = lambda self: \
    "#(a = %.10g, b = %.10g < %.10g > %.10g, R² = %.10g)" % (
        self.a, self.b_min, self.b, self.b_max, self.r2)

def name (estimator): return estimator.__dict__.get ('name', None)

# Use a gimped non-interpolating quantile to match Oh-Caml.
# Gives us divergence from SciPy, especially visible in the intercept.
#
def quantile (*a, **ka): return np.quantile (*a, **ka, interpolation = 'lower')
def median (xs, **ka): return quantile (xs, [0.5], **ka) [0]

def r2 (mx, a, b):
    yd = mx [:, 1] - mean (mx [:, 1])
    yy = yd @ yd
    if yy == 0: return 0
    ye = mx [:, 1] - a - b * mx [:, 0]
    return 1 - (ye @ ye) / yy

def ols (mx, confidence = 0.99):
    """Ordinary least-squares."""

    (n, _)       = shape (mx)
    (xs, ys)     = mx [:, 0], mx [:, 1]
    (xerr, yerr) = xs - mean (xs), ys - mean (ys)
    xx, yy, xy   = xerr @ xerr, yerr @ yerr, xerr @ yerr

    b            = xy / xx
    a            = mean (ys - xs * b)

    (err, sig)   = sqrt ((yy - b * xy) / xx / (n - 2)), ppf (confidence)
    (b0, b1)     = b - sig * err, b + sig * err

    return Estimate (a, b, b0, b1, r2 (mx, a, b))

ols.name = 'OLS'

# Check the corresponding OCaml source for justification and references.
#
def tse (mx, confidence = 0.99):
    """Theil-Sen estimator."""

    (n, _) = shape (mx)
    if n < 7: raise ValueError ('Number of samples must be > 6')

    (xs, ys)    = mx [:, 0], mx [:, 1]
    (xd, yd)    = xs - xs [:, newaxis], ys - ys [:, newaxis]
    bs          = yd [xd > 0] / xd [xd > 0]

    w           = ppf (confidence) * sqrt (n * (n - 1) * (2 * n + 5) / 18)
    N           = len (bs)    
    q0, q1      = (N - w) / 2, (N + w) / 2 + 1
    (b, b0, b1) = quantile (bs, [0.5, q0 / N, q1 / N], overwrite_input = True)
    a           = median (ys - b * xs)
    # a           = median (ys) - b * median (xs)

    return Estimate (a, b, b0, b1, r2 (mx, a, b))

tse.name = 'TSE'

# Wilcox, method 2.
#
def tse_bootstrap (mx, confidence = 0.99, resamples = 100):
    """Theil-Sen estimator, using bootstrap for confidence intervals."""

    (n, _)   = shape (mx)
    (xs, ys) = mx [:, 0], mx [:, 1]

    resample = random.randint (n, size = (resamples, n))
    resample = concatenate ([arange (n) [newaxis], resample], axis = 0)
    xss, yss = xs[resample], ys[resample]
    xds, yds = ( xss[:, :, newaxis] - xss[:, newaxis],
                 yss[:, :, newaxis] - yss[:, newaxis] )
    bs = []
    for s in range (resamples + 1):
        xd, yd = xds[s], yds[s]
        bs.append (median (yd [xd > 0] / xd [xd > 0], overwrite_input = True))

    b      = bs [0]
    b0, b1 = quantile (bs [1:], [(1 - confidence) / 2, (1 + confidence) / 2])
    a      = median (ys - b * xs)
    # a      = median (ys) - b * median (xs)

    return Estimate (a, b, b0, b1, r2 (mx, a, b))

tse_bootstrap.name = 'TSE (bootstrap)'

def ols_ransac (mx):
    xe, ye = mx [:, 0] - mean (mx [:, 0]), mx [:, 1] - mean (mx [:, 1])
    b = (xe @ ye) / (xe @ xe)
    a = mean (mx [:, 1] - b * mx [:, 0])
    return (a, b)

# RANSAC over OLS, using R^2 as the loss function.
#
# The two acceptance criteria are bad: a model is considered when more than 50%
# of the samples have the absolute error (wrt model) less than 30%. Too loose
# for tight fits, far too restrictive for noisy scenarios.
#
def ransac (mx, confidence = None, ratio = 0.5, resamples = 100):
    (n, _) = shape (mx)
    model, r, i = (np.inf, np.inf), 0, 0
    for i in range (resamples):
        sample = [0, 0]
        while sample[0] == sample[1]:
            sample = random.randint (n, size = 2)
        (a, b) = ols_ransac (mx [sample])
        err = mx [:, 1] - a - b * mx [:, 0]
        accept = mx [abs (err / mx [:, 1]) < 0.3]
        if shape (accept) [0] / n > ratio:
            (a, b) = ols_ransac (accept)
            r1     = r2 (accept, a, b)
            if r1 > r: model, r, i = (a, b), r1, i + 1
    (a, b) = model
    return Estimate (a, b, b, b, r2 (mx, a, b))

ransac.name = 'RANSAC'

@requires ('scipy')
def ols_scipy (mx, confidence = 0.99):
    """Ordinary least squares, SciPy version."""

    sigma             = ppf (confidence)
    (b, a, _, _, err) = stats.linregress (mx[:, 0], y = mx[:, 1])
    (b0, b1)          = b - sigma * err, b + sigma * err
    return Estimate (a, b, b0, b1, r2 (mx, a, b))

@requires ('scipy')
def tse_scipy (mx, confidence = 0.99):
    """Theil-Sen estimator, SciPy version."""

    (b, a, b0, b1) = stats.theilslopes (mx[:, 1], x = mx[:, 0], alpha = confidence)
    return Estimate (a, b, b0, b1, r2 (mx, a, b))

default = tse
