# unmark — Painless micro-benchmarks

%%VERSION%%

**WORK-IN-PROGRESS. API COULD CHANGE.**

unmark is a benchmarking library with a focus on the small end of the scale.

Its essential purpose is to verify assumptions about the influence of certain
programming patterns on program efficiency, answering questions like "Is
avoiding this allocation worthwhile?", "Is this indirect call really bad?", or
"Is the loop thrashing the cache?"

It can also be used to establish the performance baselines, and to track the
evolution of performance in a given code base.

Special attention is devoted to evolving the benchmark code. It is easy to save
baselines, add ad-hoc benchmarks, or reuse poorly organized benchmarking code.
It is possible to compare results from a single, or across multiple runs.

unmark is less suitable for benchmarking entire systems, and particularly
unsuitable for benchmarking concurrency.

unmark is a product of stealing great ideas from [Criterion][criterion] and
[Core\_bench][core-bench] — in particular, benchmark-by-regression. It shares
many similarities with them as a consequence.

unmark is distributed under the ISC license.

Homepage: https://github.com/pqwy/unmark

<img src="https://github.com/pqwy/unmark/blob/promo/reglines.png" width="800"/>

[jupyter]: https://jupyter.org
[criterion]: http://www.serpentine.com/criterion
[core-bench]: https://github.com/janestreet/core_bench

## Library structure

The library contains five parts:

- `unmark` (`src/`) is needed to define, run, and analyse benchmarks.
- `unmark.cli` (`src-cli/`) wraps benchmarks into standalone programs.
- `unmark.papi` (`src-papi/`) provides access to hardware performance counters.
- `unmark` executable (`src-bin/`) prints reports on the command-line.
- `src-python/` provides access to the benchmark results from Python. It is
  intended to be used from [Jupyter][jupyter].

`unmark` depends only on `unix` and `logs`. Other parts are less conservative
with their dependencies.

## Documentation

Interface files or [online][doc].

Python files contain doc-strings, accessible from `ipython` with `??unmark`.

[doc]: https://pqwy.github.io/unmark/doc

## Examples

### Hello world

```OCaml
(* shebang.ml *)
open Unmark

let suite = [
  bench "eff" f;
  group "gee" [ bench "eye" i; bench "ohh" o ]
]

let () = Unmark_cli.main "The Whole Shebang" suite
```

```sh
$ ./shebang
$ ./shebang --help
```

### Hello world, now with a mess

Show only `time`:
```sh
$ ./shebang -- --counters time
```

Do a baseline run, saving the results:
```sh
$ nice -n -10 ./shebang --out bench.json --note 'when it worked'
```

Change the implementations. Decide it's too much work to get both versions into
the same executable. Instead run the benchmarks again, appending to the results:
```sh
$ ./shebang --out bench.json --note 'did stuff'
```

Show everything so far:
```sh
$ unmark < bench.json
```

Intensely work on the functions `i` and `o`. Nothing else changed, so run just
the group containing them, appending to the results:
```sh
$ ./shebang --filter gee --out bench.json --note turbo
```

Show `gee` across all three runs:
```sh
$ unmark --filter gee < bench.json
```

Change again. Compare the last run with a throwaway run of `gee`:
```sh
$ (tail -n1 bench.json && ./shebang --filter gee --out) | unmark
```

Check the measurement stability of `time` and `L1_TSC`:
```sh
for x in {1..5}; do
  ./shebang --note "run $x" --out
done | unmark --counters time,L1_TSC
```

### Python

The environment needs to point to the python code:

```sh
PYTHONPATH="$(opam var unmark:share)/python:${PYTHONPATH}" jupyter-notebook
```

Then start with

```python
%pylab inline
from unmark import *

runs = of_json_file ('bench.json')
```

Inspect the fit:

```python
r0 = runs[0]
r0.filter ('gee').reglines ()

eff = r0['eff']
eff.reglines ()
```

Do a clever analysis:

```python
mx = eff['iterations', 'time', 'L1_TSC']
secret_sauce (mx)
```

... and open an issue to share it. ;)
