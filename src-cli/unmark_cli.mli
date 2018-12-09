(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

(** Unmark CLI runner.

    Functions in this module consume {{!Unmark.bench}benchmarks} and turn them
    into ready-to-run executables.

    {e %%VERSION%% — {{:%%PKG_HOMEPAGE%% }homepage}} *)

open Unmark

(** {1 Interface} *)

val main : ?probe:Measurement.Probe.probe -> ?min_t:float -> ?min_s:int ->
            ?def_filter:Benchmarks.query -> string -> bench list ->
            unit
(** Benchmark entry point.

    [main suite benchmarks] implements the benchmark suite named [suite], which
    consists of benchmarks [benchmarks].

    [probe] is the measurement {{!Unmark.Measurement.Probe.probe}probe}.
    Default {!probe}.

    [min_t], [min_s] default per-benchmark minimum time and samples, which can
    be overridden from the command line. Default [1.] and [10].

    [def_filter] is a {{!Unmark.Benchmarks.filtering}query} applied to
    [benchmarks], which can be overridden from the command line. Default [[]].

    This invocation will:

    {ol
    {- parse the command-line arguments;}
    {- perform memasurements using [probe] on (potentially a subset of)
    [benchmarks];}
    {- output the data; and}
    {- terminate the process.}}


    For more information about the behavior of [main], compile it into an
    executable and invoke the executable with [--help].
    *)

val main_ext : ?probe:Measurement.Probe.probe -> ?min_t:float -> ?min_s:int ->
                  ?def_filter:Benchmarks.query ->
                  arg:'a Cmdliner.Term.t -> string -> ('a -> bench list) ->
                  unit
(** Entry point with a hook for command-line arguments.

    [main_p ~arg suite f] is [main suite (f p)], where [p] is obtained by
    evaluating the [Cmdliner] term [arg]. *)

val probe : Measurement.Probe.probe
(** Default measurement probe. *)

(** {1 Example usage}

    [shebang.ml]:
{[
let f () = ...
let o () = ...
let () = Unmark_cli.main "shebang" Unmark.[bench "eff" f; bench "ohh" o]
]}

{[
$ ocamlfind ocamlopt -linkpkg -package unmark.cli shebang.ml -o shebang
$ ./shebang --help
]}
*)
