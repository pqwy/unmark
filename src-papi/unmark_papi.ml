(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark.Measurement
open Papi

module Log = (val Logs.src_log Unmark.log)

let _ = init ()

let on_err ~trap f =
  try f () with Papi.Error err as exn -> trap err; raise exn

let add set e = on_err (fun () -> add set e)
  ~trap:(fun _ -> Log.err (fun k -> k "Adding event %s:" (name e)))

let log_err f = on_err f
  ~trap:(fun err -> Log.err (fun k -> k "@[PAPI setup:@ %a.@]" pp_exn_error err))

let supported =
  let log e = Log.warn (fun k ->
    k "PAPI event %s not available." (name e)); false in
  List.filter (fun e -> query e || log e)

let of_events = function
  [] -> Probe.nothing
| es -> log_err @@ fun () ->
    let es = supported es
    and set = create () in
    List.iter (add set) es;
    start set;
    Probe.probe (fun arr off -> read set ~off arr)
      ~counters:(es |> List.map @@ fun e ->
        Probe.ctr ~desc:(description e) (name e))
