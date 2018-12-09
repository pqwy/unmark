(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark

let rec fib = function 0 -> 0 | 1 -> 1 | n -> fib (n - 2) + fib (n - 1)

let int_key = Attr.key Fmt.int
let bench ?attr fmt = Format.kasprintf (fun name f -> bench ?attr name f) fmt

let suite = [

  group "noop" [ bench "a" ignore; bench "b" (fun () -> ()) ];

  group "fib" (
    let n = int_key ~name:"n" in
    let bench i = bench "%d" i (fun () -> fib i) ~attr:(n i) in
    List.map bench [1; 2; 3; 4; 5; 7; 9; 12; 15; 20]
  );

  group "self" Measurement.(
    (* let probe = Unmark_papi.of_events Papi.[L1_ICM; L1_DCM; L1_TCM] *)
    let probe = Probe.rdtsc
    (* let probe = Probe.nothing *)
    and r     = runnable ignore
    and iters = int_key ~name:"iters" in
    let bench i =
      bench "%d" i ~attr:(iters i)
      (fun () -> sample ~probe ~iters:i r) in
    List.map bench [0; 1; 5; 10; 20]
  );

  group "array" [
    bench "create 50"  (fun () -> Array.make 50 17);
    bench "init 50"    (fun () -> Array.init 50 (fun _ -> 17));
    bench "create 300" (fun () -> Array.make 300 17);
    bench "init 300"   (fun () -> Array.init 300 (fun _ -> 17));
  ];

  group "random" [
    bench "int"   (fun () -> Random.int 1_000_000);
    bench "float" (fun () -> Random.float 1_000_000.);
  ];
]

let () =
  let papi  = Unmark_papi.of_events Papi.[TOT_CYC; REF_CYC; L1_TCM; L2_TCM; L3_TCM] in
  let probe = Measurement.Probe.(rdtsc ++ papi) in
  Unmark_cli.main "The basics" suite ~probe
