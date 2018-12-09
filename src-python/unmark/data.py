# Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
# See LICENSE.md

"""Data representation."""

from . import est, viz

from numpy import shape, array
from collections import Mapping, OrderedDict
import datetime, json
import subprocess, tempfile

__all__ = ['benchmarks', 'run', 'of_json', 'of_json_file']

class Bench (Mapping):
    """Outcome of running a single benchmark.

    Behaves as a mapping from counter names to samples. Indexing with a single
    counter returns the array of samples for that counter. Indexing with
    multiple counters returns a matrix where samples are rows.

        >>> list (bench)
        ['iterations', 'cycles', ...]
        >>> bench['iterations']
        array([1, 2, ...])
        >>> bench['iterations','cycles']
        array([[1, C1], [2, C2], ...])
    """

    def __init__ (self, name, samples, attr = {}, ctrs = None, units = None):
        self.name, self._mx, self.attr, self._ctrs, self._units = \
            name, samples, attr, ctrs, units
        self.dependent = _dependent_of_ctrs (ctrs)
        for k, v in self.attr.items ():
            try: self.attr [k] = float (v)
            except ValueError: pass

    def __repr__ (self):
        return "#(Bench: %s, samples: %d, counters: %s%s)" % (
                self.name, self.samples (), ','.join (self.dependent),
                self._repr_attr (fmt = ", %s"))

    def __str__ (self):
        return self.name + self._repr_attr (fmt = " (%s)")

    def _repr_attr (self, fmt = "%s"):
        if self.attr == {}: return ""
        return fmt % ', '.join (["%s: %s" % kv for kv in self.attr.items ()])

    def __getitem__ (self, ctr):
        if type (ctr) != tuple:
            return self._mx [:, self._ctrs [ctr]]
        return self._mx [:, [self._ctrs [i] for i in ctr]]

    def __iter__ (self): return iter (self._ctrs)

    def __len__ (self): return len (self._ctrs)

    def unit (self, counter): return self._units.get (counter)

    def matches (self, q):
        """Test if this benchmark matches the query."""

        (qs, ns) = q.split ('/'), self.name.split ('/')
        p        = all ([a == '' or a == b for a, b in zip (qs, ns)])
        return (len (q) <= len (self.name) and p)

    def samples (self):
        """Number of samples"""
        return shape (self._mx) [0]

    def estimate (self, counter, est = est.default, **kw):
        """Estimate the value of a single counter."""
        return est (self ['iterations', counter], **kw)

    def reglines (self, *args, **kw):
        """Bench.reglines (counters?, **keywords)

        Show regression. See `unmark.viz.reglines1'."""

        return viz.reglines1 (self, *args, **kw)

    def __group (self):
        if '/' in self.name: return self.name.split ('/', maxsplit = 1) [0]

class Run (Mapping):
    """Results of a single run of benchmark suite.

    Behaves as a mapping from benchmark names to Bench."""

    def __init__ (self, suite, note, time, dependent, benches):
        self.suite, self.note, self.time, self.dependent = suite, note, time, dependent
        self._benches = OrderedDict ([(b.name, b) for b in benches])

    def __repr__ (self):
        return "#(Run: %s, %d benches)" % (self, len (self))

    def __str__ (self):
        note = (self.note and " - %s" % self.note) or ""
        return "%s - %s%s" % (self.suite, self.time.isoformat (' '), note)

    def __getitem__ (self, bench): return self._benches [bench]

    def __iter__ (self): return iter (self._benches)

    def __len__ (self): return len (self._benches)

    def __copy (self, benches):
        return Run (suite = self.suite, note = self.note, time = self.time,
                    dependent = self.dependent, benches = benches)

    def filter (self, *filters):
        """Retain benchmarks matching `filters'.

        Each filter is either a predicate on `Bench', or a query for
        `Bench.matches'."""

        benches = self._benches.values ()
        for f in filters:
            if hasattr (f, '__call__'):
                benches = [b for b in benches if f (b)]
            else: benches = [b for b in benches if b.matches (f)]
        return self.__copy (benches = benches)

    def without (self, *filters):
        """Negates `Run.filter'."""
        def p (f): return lambda bench: not bench.matches (f)
        return self.filter (*[p (f) for f in filters])

    def groups (self):

        gs = _dict (((b.__group (), b) for b in self.values ()))
        return [self.__copy (benches) for benches in gs.values ()]

    def __def_ctrs (self, *args):
        return (len (args) > 0 and args [0]) or self.dependent

    def reglines (self, *args, **kw):
        """Run.reglines (counters?, **keywords)

        Show regression. See `unmark.viz.reglines'."""

        return viz.reglines (self.values (), self.__def_ctrs (*args), title = self, **kw)

    def barcompare (self, *args, **kw):
        """Run.barcompare (counters?, **keywords)

        Show comparison of results. See `unmark.viz.barcompare'."""

        return viz.barcompare (self.values (), self.__def_ctrs (*args), title = self, **kw)

    def trendlines (self, attr, *args, **kw):
        """Run.trendlines ("""

        (xs, benches) = unzip (
            [(bench.attr [attr], bench)
                for bench in self.values () if attr in bench.attr])
        if not benches:
            raise Exception ("No benchmarks contain the attribute `%s'." % attr)

        return viz.trendlines (benches, xs, self.__def_ctrs (*args), **kw,
                                attr = attr, title = self)

def benchmarks (runs, *filters):
    """Takes a sequence of `Run's and groups their benchmarks together.

    Returns a dict mapping benchmark names to sequences of benchmarks."""

    return _dict ([b for r in runs for b in r.filter (*filters).items ()])

def _dict (xs):
    d = OrderedDict ()
    for k, v in xs: d.setdefault (k, []).append (v)
    return d

def _dependent_of_ctrs (ctrs):
    c = list (ctrs)
    if 'iterations' in c: c.remove ('iterations')
    return c

def unzip (xys):
    xys = list (xys)
    return ([x for (x, _) in xys], [y for (_, y) in xys])

def _unpack_ctr (name):
    s = name.split (':')
    if len (s) > 1: return (s[0], s[1])
    return (s[0], None)

def _json_stream (s):
    d = json.JSONDecoder ()
    s = s.lstrip ()
    while s:
        (res, i) = d.raw_decode (s)
        s = s[i:].lstrip ()
        yield res

def _run_of_json (json):

    time = datetime.datetime.fromtimestamp (int (json ['time']))

    cs       = [(i, _unpack_ctr (n)) for i, n in enumerate (json ['counters'])]
    ctr_pos  = dict ([(n, i) for (i, (n, _)) in cs])
    ctr_unit = dict ([(n, u) for (_, (n, u)) in cs if u])

    bms = [ Bench ( j ['name'], array (j ['samples']),
                    j.get ('attr') or {}, ctr_pos, ctr_unit )
              for j in json ['benchmarks'] ]

    return Run (suite = json ['suite'], note = json.get ('note'), time = time,
                dependent = _dependent_of_ctrs (ctr_pos), benches = bms)

def of_json (string_or_file, sort = True):
    """Reads a sequence of measurement results from a stream of JSON documents.

    Argument is either an immediate string, or a file-like object."""

    s = string_or_file
    if type (s) != str: s = s.read ()

    runs = [_run_of_json (run) for run in _json_stream (s)]

    if sort: runs.sort (key = lambda r: r.time)
    return runs

def of_json_file (*paths, sort = True):
    """Reads sequences of measurement results from one or multiple files."""

    runs = []
    for path in paths:
        with open (path) as f:
            runs.extend (of_json (f, sort = sort))

    if sort: runs.sort (key = lambda r: r.time)
    return runs

def param (name, v):
    if v != None: return [name, v]
    return []

def run (executable, note = None, min_t = None, min_s = None, warmup = True, filter = ""):
    """Runs a benchmark suite built with `unmark.cli'. Returns `Run'."""

    f = tempfile.NamedTemporaryFile (mode = 'r')

    args = [executable, '--out', f.name, '--filter', filter]
    args += param ('--note', note)
    args += param ('--min-time', min_t) + param ('--min-samples', min_s)
    if not warmup: args += ['--no-warmup']

    subprocess.run ([str (x) for x in args], stderr = subprocess.PIPE, check = True)
    with f as f: return of_json (f)[0]
