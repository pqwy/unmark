(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

(** Unmark {{: https://github.com/pqwy/ocaml-papi}PAPI} integration.

    {e %%VERSION%% — {{:%%PKG_HOMEPAGE%% }homepage}} *)

open Unmark

val of_events: Papi.event list -> Measurement.Probe.probe
(** [of_events evts] is a probe that uses PAPI to count events [evts]. *)


(** {1 Usage}

  Measure L1, L2 and L3 cache miss behaviour:

{[let probe = of_events Papi.[L1_TCM; L2_TCM; L3_TCM]
let _ = Unmark_cli.main "stuff" ~probe [...]
]} *)
