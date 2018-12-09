# Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
# See LICENSE.md

"""Visualisation.

- `reglines1' plots sample points and regression lines, for a single benchmark.
- `reglines' does the same, for a sequence of benchmarks.
- `barcompare' compares the expected values of a sequence of benchmarks.
- `trendlines' shows how the expected values behave as a function of a given
  variable.
"""



__all__ = ['reglines1', 'reglines', 'barcompare', 'trendlines']

from .modstub import requires
from . import est

from numpy import array, ceil, sqrt, arange
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import ticker
except ModuleNotFoundError: pass


annot_kw = {'verticalalignment': 'top',
            'bbox': {'facecolor': 'white', 'alpha': 0.2, 'pad': 5}}

@requires ('matplotlib')
def plot_reglines (mx, e, ax = None, intervals = True, annot = True, unit = None, log = False, **prop):
    """plot_reglines (mx, e, **keywords)

    Plots data points and regression lines.

    Arguments:
    - mx : Data matrix: mx [:, 0] is independent, mx [:, 1] is dependent.
    - e  : est.Estimate.

    Keyword arguments:
    - ax   : Matplotlib Axes to draw onto. pyplot.gca() when None.
    - unit : Unit to display.
    - log  :
        Use log scale:
        + "x"  : for x axis
        + "y"  : for y axis
        + True : for both
        + otherwise no.
    - intervals :
        Display confidence interval. One of:
        + True   : Always plot interval.
        + False  : Never plot interval.
        + "auto" : Plot interval if e.r2 < 0.99.
    - annot : 
        Display a summary. One of:
        + True   : Plot larger summary.
        + False  : Never plot the summary.
        + "auto" : Show e.r2 < 0.99.

    Other keyword arguments are passed to plotting functions."""

    ax     = ax or plt.gca ()
    color  = prop.pop ('color', None)
    xs, ys = mx [:, 0], mx [:, 1]    

    _ax_set_log_scale (ax, log)
    ax.plot (xs, ys, '.', color = 'black', **prop)

    (a, b, b0, b1, r2) = e
    [line] = ax.plot (xs, a + b * xs, color = color, **prop)

    bad_r2 = r2 < 0.99
    if (intervals == 'auto' and bad_r2) or intervals is True:
        (ys0, ys1) = a + b0 * xs, a + b1 * xs
        ax.fill_between (xs, ys0, ys1, alpha = 0.3, color = line.get_color (), **prop)

    fmt = ticker.EngFormatter (unit = unit or "")
    ax.yaxis.set_major_formatter (fmt)
    ax.xaxis.set_major_formatter (ticker.EngFormatter ())

    if annot:
        try:
            label = u"""$b = %s$""" % (fmt (b))
            if (annot == 'auto' and bad_r2) or annot is True:
                label = label + u"""\n$R^2 = %.02f$""" % r2
        except OverflowError: label = "No estimate."
        ax.text (0.05, 0.93, label, transform = ax.transAxes, **annot_kw)

@requires ('matplotlib')
def reglines1 (bench, *args, **kw):
    """reglines1 (bench, counters?, **keywords)

    Shows regression lines for counters of a single benchmark.

    Arguments:
    - bench     : Bench.
    - counters  : Counters to display. Optional, default `bench.dependent'.

    Keyword arguments:
    - est       : Regression estimator.
    - intervals : Like 'plot_reglines'. Default True.
    - annot     : Like 'plot_reglines'. Default True.
    - props     : cycler.Cycler or other iterable yielding prop dicts for each counter.
    """

    counters = _as_seq (_pos (args, 0, bench.dependent))
    estf     = kw.get ('est', est.default)
    props    = _counter_props (counters, kw.get ('props'))
    w        = int (ceil (sqrt (len (counters))))
    h        = int (ceil (len (counters) / w))
    fs       = plt.rcParams ['figure.figsize']

    fig, axes = plt.subplots (
        nrows = h, ncols = w, squeeze = False, sharex = 'all',
        figsize = (w * fs[0], h * fs[1]) )

    axes[-1][0].set_xlabel ('iterations')

    axes = [ax for row in axes for ax in row]
    for ax in axes [len(counters):]: ax.remove ()

    for ctr, ax in zip (counters, axes):
        mx = bench ['iterations', ctr]
        plot_reglines (
            mx, estf (mx), ax = ax, **props [ctr],
            unit      = bench.unit (ctr),
            annot     = kw.get ('annot', True),
            intervals = kw.get ('intervals', True),
            log       = kw.get ('log', False)
        )
        ax.set_ylabel (ctr)

    _fig_set_title (fig, bench.name, estf)

@requires ('matplotlib')
def reglines (benches, *args, **kw):
    """reglines (benchmarks, counters?, **keywords)

    Shows regression lines counters of a sequence of benchmarks.

    Arguments:
    - benches  : Sequence of Benches.
    - counters : Counters to display. Optional, default "time".

    Keyword arguments:
    - title     : Figure title.
    - est       : Regression estimator.
    - intervals : Like `plot_reglines'. Default "auto".
    - annot     : Like `plot_reglines'. Default "auto".
    - props     : cycler.Cycler or other iterable yielding prop dicts for each counter.
    """

    benches  = list (benches)
    counters = _as_seq (_pos (args, 0, 'time'))
    estf     = kw.get ('est', est.default)
    props    = _counter_props (counters, kw.get ('props'))
    (h, w)   = len (counters), len (benches)
    fs       = plt.rcParams ['figure.figsize']

    fig, axes = plt.subplots (
        ncols = w, nrows = h, squeeze = False, sharex = 'col',
        figsize = (w * fs[0], h * fs[1]) )

    for ctr, axrow in zip (counters, axes):
        for i in range (len (benches)):
            bench, ax = benches [i], axrow [i]
            if i == 0: ax.set_ylabel (ctr)
            mx = bench ['iterations', ctr]
            plot_reglines (
                mx, estf (mx), ax = ax, **props[ctr],
                unit      = bench.unit (ctr),
                annot     = kw.get ('annot', 'auto'),
                intervals = kw.get ('intervals', 'auto'),
                log       = kw.get ('log', False)
            )

    for bench, ax in zip (benches, axes [0]):
        ax.set_title (bench, pad = 4 * plt.rcParams ['axes.titlepad'])
    for ax in axes [-1]: ax.set_xlabel ('iterations')

    _fig_set_title (fig, kw.get ('title'), estf)

@requires ('matplotlib')
def barcompare (benches, *args, **kw):
    """barcompare (benchmarks, counters?, **keywords)

    Compares benchmark results as a bar-chart.

    Arguments:
    - benches  : Sequence of Benches.
    - counters : Counters to display.

    Keyword arguments:
    - title : Figure title.
    - est   : Regression estimator.
    - props : cycler.Cycler or other iterable yielding prop dicts for each counter.
    """

    benches  = list (benches)
    counters = _as_seq (_pos (args, 0, 'time'))
    estf     = kw.get ('est', est.default)
    props    = _counter_props (counters, kw.get ('props'))

    w  = int (ceil (sqrt (len (counters))))
    h  = int (ceil (len (counters) / w))
    fs = plt.rcParams ['figure.figsize']


    fig, axes = plt.subplots (
        nrows = h, ncols = w, squeeze = False, sharey = 'row',
        figsize = (w * fs[0], max (1, len (benches) * 0.1) * h * fs[1]) )

    axes = [ax for row in axes for ax in row]
    for ax in axes [len(counters):]: ax.remove ()

    names = [b.name for b in benches]
    ypos  = arange (len (benches) - 1, -1, -1)

    for ctr, ax in zip (counters, axes):

        es   = [bench.estimate (ctr, estf) for bench in benches]
        xerr = ([e.b - e.b_min for e in es], [e.b_max - e.b for e in es])

        ax.barh (ypos, [e.b for e in es], xerr = xerr, alpha = 0.3,
                    edgecolor = props[ctr]['color'], **props[ctr])

        ax.set_yticks (ypos)
        ax.set_yticklabels (names)

        unit = benches[0].unit (ctr) or ""
        ax.xaxis.set_major_formatter (ticker.EngFormatter (unit = unit))
        ax.set_xlabel (ctr)

    _fig_set_title (fig, kw.get ('title'), estf)

@requires ('matplotlib')
def plot_trendlines (mx, ax = None, unit = None, **prop):

    ax    = ax or plt.gca ()
    color = prop.pop ('color', None)

    xs, ys = mx [:, 0], mx [:, 1]
    yerr   = (ys - mx [:, 2], mx [:, 3] - ys)

    ax.errorbar (xs, ys, yerr = yerr, fmt = '-o', color = color)

    fmt = ticker.EngFormatter (unit = unit or "")
    ax.yaxis.set_major_formatter (fmt)
    ax.xaxis.set_major_formatter (ticker.EngFormatter ())

@requires ('matplotlib')
def trendlines (benches, predictor, counters, **kw):
    """trendlines (benchmarks, predictor, counters, **keywords)

    Shows dependence of benchmark results on `predictor'.
    
    Arguments:
    - benches   : Sequence of Benches.
    - predictor : Sequence of values that Benches depend on.
    - counters  : Counters to display.

    Keyword arguments:
    - attr  : Predictor name.
    - title : Figure title.
    - est   : Regression estimator.
    - props : cycler.Cycler or other iterable yielding prop dicts for each counter.
    """
    
    benches  = list (benches)
    counters = _as_seq (counters)
    estf     = kw.get ('est', est.default)
    props    = _counter_props (counters, kw.get ('props'))

    w  = int (ceil (sqrt (len (counters))))
    h  = int (ceil (len (counters) / w))
    fs = plt.rcParams ['figure.figsize']

    fig, axes = plt.subplots (
        nrows = h, ncols = w, squeeze = False, sharex = 'all',
        figsize = (w * fs[0], h * fs[1]) )

    if 'attr' in kw:
        axes[-1][0].set_xlabel (kw ['attr'])

    axes = [ax for row in axes for ax in row]
    for ax in axes [len(counters):]: ax.remove ()

    for ax in axes:
        _ax_set_log_scale (ax, kw.get ('log', False))

    for ctr, ax in zip (counters, axes):
        mx = []
        for (x, bench) in zip (predictor, benches):
            (_, b, b0, b1, _) = bench.estimate (ctr, est = estf)
            mx.append ([x, b, b0, b1])
        if len (benches) > 0: unit = benches[-1].unit (ctr)
        else: unit = None

        plot_trendlines (array (mx), ax = ax, **props [ctr], unit = unit)
        ax.set_ylabel (ctr)

    _fig_set_title (fig, kw.get ('title'), estf)

def _pos (args, pos, default):
    return (len (args) > pos and args [pos]) or default

def _as_seq (x):
    if hasattr (x, '__iter__') and not type (x) == str: return x
    return (x, )

def _fig_set_title (fig, title, estf):
    if title:
        if estf and est.name (estf):
            title = '%s  [%s]' % (title, est.name (estf))
        fig.suptitle (title, fontweight = 'bold')

def _ax_set_log_scale (ax, log):
    if log and not log == 'y':
        ax.set_xscale ('log')
    if log and not log == 'x':
        ax.set_yscale ('log')

def _counter_props (counters, props):
    if not props: return _def_props (counters)
    return dict (p for p in zip (counters, props))

def _def_props (counters):
    for ctr in counters:
        if not ctr in _cprop: _cprop [ctr] = next (_def_prop)
    return _cprop

_cprop = {}
_def_prop = iter (plt.rcParams ['axes.prop_cycle'])
# _def_props ( ['time', 'cycles', 'min', 'prom', 'maj'] )
