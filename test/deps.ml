(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark

let rec fib = function 0 -> 0 | 1 -> 1 | n -> fib (n - 2) + fib (n - 1)

let rec (--) a b = if a <= b then a :: succ a -- b else []
let n = Attr.key ~name:"n" Fmt.int

let arg = Cmdliner.Arg.(
  value @@ opt int 5 @@ info ["r"; "bench-range"] ~docs:"BENCHMARK"
)

let () =
  Unmark_cli.main_ext "Fibs" ~arg @@ fun range ->
    0 -- range |> List.map @@ fun i ->
      bench "fib" (fun () -> fib i) ~attr:(n i)
