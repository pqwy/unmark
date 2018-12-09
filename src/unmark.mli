(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

(** Painless micro-benchmarks.

    Theory of operation is TODO.

    Examples {{!examples}at the end}.

    {e %%VERSION%% — {{:%%PKG_HOMEPAGE%% }homepage}} *)

type 'a fmt = Format.formatter -> 'a -> unit

(** {1 Creating benchmarks}

    Benchmark construction is sufficient for most uses of the library.
    For a simple way to run the benchmarks, see {!Unmark_cli}.
    *)

type bench
(** A labelled tree of code.

    {{!bench}Leaves} contain annotated functions.
    {{!group}Inner nodes} are purely organisational. *)

(** Benchmark metadata. *)
module Attr: sig

  type t
  (** Attributes are dictionaries that can be attached to {{!bench}benchmarks}. *)

  val empty : t
  (** Empty set of attributes. *)

  val is_empty : t -> bool
  (** [is_empty t] is [true] iff [t] is [empty]. *)

  val key : 'a fmt -> name:string -> 'a -> t
  (** [key pp ~name v] is a set of attributes that contains only the mapping
      [name -> v]. *)

  val (++) : t -> t -> t
  (** [t1 ++ t2] is the right-biased union of [t1] and [t2]. *)

  val pp : t fmt
  (** [pp ppf t] pretty-prints [t] on [ppf]. *)
end

val bench : ?attr:Attr.t -> string -> (unit -> 'a) -> bench [@@inline]
(** [bench ?attr name f] is a named benchmark measuring [f].

    [attr] are optional {{!Attr.t}attributes}. Attributes serve to attach
    user-defined metadata to the benchmark.

    {b Note.} Occurrences of ['/'] in [name] are removed in an unspecified way.
    *)

val group : string -> bench list -> bench
(** [group name benchmarks] is a named group of benchmarks.

    Root nodes of [benchmarks] are renamed to avoid name clashes.

    [name] behaves like above.
    *)

val group_f : init:(unit -> 'a) -> fini:('a -> unit) -> string -> ('a -> bench list) -> bench
(** [group_f ~init ~fini name f] is a group of benchmarks that depend on
    a temporarily acquired resource.

    This is [group name (f (init ()))], except [init] and [f] are only invoked
    if the group is {e visited} (e.g. to run the benchmarks), and [fini] is
    called on the result of [init] after the visit.

    Apart from acquiring and releasing external resources, [init]/[fini] -- or
    simply constructing stuff in the body of [f] -- can be used to avoid
    expensive setup computations if the group is going to be skipped.
    *)

(** {1 Detailed interface}

    The rest of the API is of interest for changing what is being measured,
    creating alternative benchmark runners, directly inspecting benchmarks, or
    analysing the results. *)

val log: Logs.src
(** Log source for this module. *)

(** Low-level measurement machinery. *)
module Measurement : sig
  (** A {{!Probe.probe}[probe]} provides a set of measurements, called
      {{!Probe.counter}counters}. These are assumed to be global, monotonically
      increasing, and (ideally) affected by running code.

      A single measurement is {{!sample}performed} by invoking a probe to get
      the values of counters, running the target piece of code a certain number
      of times, getting the counters again, and reporting the difference.

      The {{!measure}measurement process} collects a series of measurements
      while varying the number of times the target code is ran.
      *)

  (** Probe construction and manipulation. *)
  module Probe : sig

    (** {1 Counters} *)

    type counter
    (** Individual measured quantity. *)

    val ctr: ?unit:string -> ?desc:string -> string -> counter
    (** [ctr ?unit ?desc name] is a description of counter named [name].

        [unit] is the counter's measurement unit. When absent, a simple count is
        assumed.

        [desc] is a human-readable description. *)

    val name : counter -> string

    val desc : counter -> string

    val unit : counter -> string option

    val pp_ctr : counter fmt
    (** [pp_ctr ppf c] pretty-prints [c] on [ppf]. *)

    (** {1 Probes} *)

    type probe
    (** Probes perform the actual measurements, producing a set of
        {{!counter}counters}. *)

    val probe : counters:counter list -> (float array -> int -> unit) -> probe
    (** [probe ~counters f] is a probe [p] that collects counters [counters]
        using the function [f].

        [f] is invoked as [f arr i] and must write the values of individual
        counters into [arr.(i) .. arr.(i + length counters - 1)].

        [counters] declares both the number and the order of counters that [p]
        measures.
        *)

    val nothing : probe
    (** [nothing] is the probe that measures nothing. *)

    val (++) : probe -> probe -> probe
    (** [p1 ++ p2] the probe that measures both the counters in [p1] and
        [p2]. *)

    val counters : probe -> counter list
    (** [counters p] are the counters that [p] measures. *)

    val pp : probe fmt
    (** [pp ppf p] pretty-prints [p] on [ppf]. *)

    (** {1 Predefined probes} *)

    val gc_q_stat : probe
    (** A probe for the GC subsystem, using {!Gc.quick_stat}.

        Counters:
        {ul
        {- [min] - words allocated on the minor heap ([Gc.minor_words]);}
        {- [prom] - words promoted to the major heap ([Gc.promoted_words]);}
        {- [maj] - words allocated directly on the major heap
        ([Gc.major_words - Gc.promoted_words]);}
        {- [gc_min] - number of minor collections ([Gc.minor_collections]); and}
        {- [gc_maj] - number of major collections ([Gc.major_collections]).}}

        *)

    val gc_counters : probe
    (** A probe for the GC subsystem, using {!Gc.counters}.

        Subset of {{!gc_q_stat}[gc_q_stat]}. *)

    val rdtsc : probe
    (** Reads the timestamp counter, using x86 RDTSC instruction.

        Produces the counter [tsc]. *)
  end

  open Probe

  type runnable
  (** [runnable] is the object of measurement. *)

  val runnable : (unit -> 'a) -> runnable

  type sample = float array
  (** An array of {{!Probe.counter}counter} values. *)

  val sample : probe:probe -> iters:int -> runnable -> sample
  (** [sample ~probe ~iters r] performs a single measurement of [r] and returns
      a sample [s].

      It runs [r] in a tight loop [iters] times, returning the [probe]'s
      counters' difference before and after the run.

      [s] has {{!core_counters}core counters} in the initial positions, followed
      by the [probe]'s counters as given by [Probe.counters probe]. In
      particular, [s.(0) = iters].
      *)

  val measure : ?probe:Probe.probe -> ?min_t:float -> ?min_s:int -> runnable -> sample list
  (** [measure ~probe ~min_t ~min_s r] performs a series of measurements of [r].

      Individual measurements are obtained by invoking {{!sample}[sample]} with
      varying [iters], until both the minimal number of samples are collected,
      and the minimal measurement time elapses.

      [probe] is the {{!Probe.probe}[probe]} used to produce measurements.
      Default {{!Probe.nothing}[nothing]}.

      [min_t] is the minimal measurement time in seconds. Default [1].

      [min_s] is the minimal number of samples to collect. Default [10].

      The exact sampling strategy is not part of the API and could change.
      Currently, it's an exponential function with slow start, approximating
      [iters -> iters^1.05].
      *)

  val warmup : ?seconds:float -> unit -> unit
  (** [warmup ~seconds] busy-loops the CPU to force it into full-power mode.

      [seconds] is the warmup period. Default [1]. *)

  val core_counters : Probe.counter list
  (** Wired-in counters that are always collected. *)
end

(** Running and elimination of whole benchmarks suites. *)
module Benchmarks : sig

  type name = string list * string
  (** Fully qualified names. *)

  (** {1:filtering Filtering} *)

  type query = string list

  (** Operations over benchmark suites support filtering, to restrict them to a
      subset of benchmarks.

      Queries contain query components. Components are strings with syntax
      [[NAME1[/NAME2...]]].

      Each benchmark is located in a tree of groups, and can be assigned a
      {{!name}qualified name} like [group1/group2/bench]. A query component is
      the full path to a benchmark or a group. As a special case, an empty
      [NAME] between slashes selects all groups at that level.

      For example:

      {ul
      {- ["a"] selects the top-level benchmark named [a], or all the benchmarks
      in the top-evel group [a].}
      {- ["a/b/x"] selects the benchmark (or group) named [x] in group [b] in
      group [a].}
      {- ["a//x"] selects the benchmark (or group) named [x] in any second-level
      subgroup of [a].}
      {- [""] selects everything.}}

      Query selects all benchmarks selected by any of its components. Empty
      lists selects everything.
      *)

  val matches : ?group:bool -> q:query -> name -> bool
  (** [matches ?group ~g name] checks if [name] matches the query [q] in the
      sense above.

      When [group] is [true], the match is strictly more permissive: [name] is
      also selected by query components more specific than it, making target's
      parent groups selectable. Default [false].
      *)

  (** {1 Running} *)

  type run = {
    suite      : string  (** Name of the benchmark suite. *)
  ; note       : string  (** A note for this run. *)
  ; time       : float   (** Run timestamp, UNIX time. *)
  ; counters   : (string * string option) list
  (** Counters and their units. *)
  ; benchmarks : (name * Attr.t * Measurement.sample list) list
  (** Benchmark names, attributes, and their sample series. *)
  }
  (** Raw results of running a benchmark suite. *)

  val run : ?probe:Measurement.Probe.probe ->
            ?min_t:float -> ?min_s:int ->
            ?q:query -> ?note:string -> suite:string -> bench list -> run
  (** [run ?probe ?min_t ?min_s ?q ?note ~suite benchmarks] runs the benchmark
      suite [benchmarks] named [suite]. Individual benchmarks are passed to
      {{!Measurement.measure}[measure]}.

      [q] - {{!filtering}Filtering} query.

      [note] - Text snippet attached to this particular run.

      [probe], [min_t], [min_s] - Passed to {!Measurement.measure}.
      *)

  (** Operations over {{!run}runs}. *)
  module Run : sig

    val filter : ?q:query -> run -> run
    (** [filter ?q run] is [run] with only the benchmarks that
        {{!filtering}match} [q]. *)

    (** {1 JSON}

        {{!run}Runs} can be converted to and from JSON text. This makes it easy
        to separate benchmark running and analysis. *)

    val add_json : Buffer.t -> run -> unit
    (** [add_json buf run] writes a JSON text representing [run] to [buf]. *)

    val of_json : string -> (run * string, [`Msg of string]) result
    (** [of_json string] is
        {ul
        {- [Ok (run, rest)] when [run] is encoded as JSON text in some prefix of
        [string], where [rest] is the remainder of [string]; or}
        {- [Error e] otherwise.}} *)
  end

  (** {1 Inspection} *)

  type 'a thunk
  (** Bracketed thunks.

      An ['a thunk] is [unit -> 'a] which uses additional external resources.
      Thunks are internalization of {!group_f}. *)

  val fold : bench:(name -> Attr.t -> Measurement.runnable -> 'r) ->
             group:(name -> 'r list thunk -> 'r) ->
             suite:('r list -> 'p) ->
             bench list -> 'p
  (** Benchmark eliminator. *)

  val eval : 'a thunk -> 'a
  (** [eval t] runs [t]. This can cause acquisition and release of associated
      resources. *)

  val (%%) : ('a -> 'b) -> 'a thunk -> 'b thunk
  (** [f %% t] extends [t] by applying [f] within the resource bracket. *)
end

(** Deepest numerology. *)
module Estimate : sig

  type estimator = float array -> float array -> e
  (** [estimator xs ys] is an estimate of the {e true} value of dependent [ys],
      given independent [xs]. *)

  and e = { a: float; b: float; r2: float; bounds: (float * float) Lazy.t }
  (** A linear estimate, modeling the dependent variable as [Y = a + bX].

      [r2] is
      {{: https://en.wikipedia.org/wiki/Coefficient_of_determination}[R²]} of the
      fit.

      [bounds] are the 99%
      {{: https://en.wikipedia.org/wiki/Confidence_interval}confidence interval}
      for [b].
      *)

  val pp_e : e fmt
  (** Pretty-prints {{!e}[e]}. *)

  val validity : e -> [`Undef | `Bad | `Meh | `Good]
  (** Gives an interpretation of [e]'s validity. One of:

      {ul
      {- [`Good] - [e] describes the data extremely well.}
      {- [`Meh] - [e] should be further investigated. It does not fully describe
      the data.}
      {- [`Bad] - [e] is likely meaningless. It poorly describes the data.}
      {- [`Undef] - [e] is not defined over the data.}}

      [e] can fail to describe the data for reasons ranging from successfully
      filtering out the noise, to data being non-linear and [e] being completely
      meaningless. Rules of thumb:

      {ul
      {- More samples should improve validity.}
      {- If [b] stays stable over repreated runs, the measurement is noisy, but
      the estimate is good.}}

      [e] is most commonly undefined when the data is all-0, in which case [0]
      is still a good summary of it.

      [validity] is currently implemented by checking [R²].
      [`Undef] is [R² = 0],
      [`Bad] is [0 < R² < 0.9],
      and [`Meh] is [0.9 <= R² < 0.98].
   *)

  (** {1:estimators Estimators} *)

  val ols : estimator
  (** {{: http://mathworld.wolfram.com/LeastSquaresFitting.html}
      Ordinary Least Squares} is the usual linear regression that minimizes
      average squared residuals (and maximizes [R²]).

      OLS assumptions are somewhat violated by the benchmarking procedure: the
      errors have a heavy positive tail (instead of being normally distributed),
      and their variance grows with the number of iterations (instead of being
      independent of [X]). As a result, the estimates are overly affected by
      noise and tend to have less stability across runs, while the confidence
      intervals tend to be too narrow to predict this instability.

      On the upside, OLS is quick to compute, well known, and readily
      interpretable. *)

  val tse : estimator
  (** {{: https://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator}
      Theil–Sen Estimator} is an order-based linear regression.

      It provides a robust estimation in the sense that
      {ul
      {- the result is unaffected by even a relatively large proportion of
      outliers (about 30%); and}
      {- there is no assumption on the error distribution,}}
      making TSE resilient against the skewed benchmark noise, and giving it
      good stability across runs.

      Note that by design, TSE does not maximize [R²]. For this reason, [R²] is
      not an entirely accurate way of assessing goodness-of-fit for this method,
      and perfectly good estimates are sometimes flagged as having poor
      {{!validity}[validity]}. A particular failure case happens with results
      resembling a step-function, where TSE sometimes (correctly) assigns a
      vanishingly small [b], and a biased [a] that offsets the line. Such
      predictions have an error larger than the sample mean, resulting in a
      negative [R²].

      Persistently low [R²] should be investigated by plotting the results.

      TSE is the default estimator in Unmark.
      *)
end

(** {1:examples Examples}

  A single benchmark measuring [f: unit -> t] for some [t]:

{[
let bm = bench "eff" f
]}

  A group of benchmarks:

{[
let bm = group "things" [
  bench "this" f;
  bench "that" g;
]
]}

  Group nesting:

{[
let bm = group "stuff" [
  group "more" [ bench "x" x ];
  group "less" [ bench "y" y ];
]
]}

  A group acquiring a resource:

{[
let bm path = group_f "files" (fun fd ->
  [ bench "f1" (fun () -> f1 fd);
    bench "f2" (fun () -> f2 fd);
  ])
  ~init:Unix.(fun () -> openfile path [O_RDONLY] 0)
  ~fini:Unix.close
]}

  Using a group to delay construction of data:

{[
let bm = group_f "big data" (fun () ->
  let really_big_value = ... in
  [ bench "f" (fun () -> f really_big_value) ]
  ) ~init:ignore ~fini:ignore
]}

  Independent variable.

  Creates a group, containing 3 subgroups. Each subgroup instantiates [f1] and
  [f2] with a different argument. Leaf-level benchmarks are annotated with the
  argument, keyed by [en].

{[
let pp_int ppf = Format.fprintf ppf "%d"
let en = Attr.key pp_int ~name:"en"

let bm =
  let g n = group (Format.sprintf "size %d" n) [
    bench ~attr:(en n) "f1" (fun () -> f1 n);
    bench ~attr:(en n) "f2" (fun () -> f2 n);
  ] in
  group "effs" @@ List.map g [1; 10; 100]
]}

*)
