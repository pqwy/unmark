# Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
# See LICENSE.md

"""Unmark playground.

Module `est' contains estimators.
Module `viz' contains visualizations (requires Matplotlib).

Use `run', `of_json' and `of_json_file' to load benchmark results:

    >>> [run] = of_json (path)
    >>> run
    #(Run: ...)

`Run' objects are dictionaries. Use their methods to select the benchmarks of
interest:

    >>> run.filter ('g')
    #(Run: ... (only the `g' group) ...)
    >>> run ['g/a']
    #(Bench: g/a, ...)
    >>> list (run)
    [ ... benchmark names ... ]
    >>> for bench in run.filter ('g').values ():
    >>>   print (bench)
    #(Bench: g/a, ...)
    ...

`benchmarks' combines benchmarks from several `Run's:

    >>> bms = benchmarks ([run] + [run])
    >>> bms ['g/a']
    [#(Bench: g/a, ...), #(Bench: g/a, ...)]

Use `Bench.estimate' method to obtain an estimate for the "true" value of
counters:

    >>> bench.estimate ('time')
    #(a = ..., b = ..., ...)

... or extract the data using indexing:

    >>> bench ['iterations', 'time']
    array ([[1, T1], [2, T2], ...])
    >>> est.ols (bench ['iterations', 'time'])
    #(a = ..., b = ..., ...)

For a quick comparison of results, use `barcompare':

    >>> run.barcompare ('time')
    >>> viz.barcompare (run.values ())

For more insight into the regression behavior, use `reglines':

    >>> bench.reglines ()
    >>> run.reglines (['time', 'min'])
    >>> viz.reglines (run.values ())
"""

version = '%%VERSION_NUM%%'

from .data import *
from . import est, viz

__all__ = ['est', 'viz'] + data.__all__
