(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark
open Attr

let rec (--) a b = if a <= b then a :: succ a -- b else []
let this = key ~name:"this" Fmt.int
and that = key ~name:"that" Fmt.string

let suite = [
  bench "a" ignore
; group "g1" [
    bench "x" ignore
  ; group "g2" [
      bench "x" ignore
    ; group "g3" [
        bench "x" ignore
      ; bench "y" ignore
      ]
    ]
  ]
; bench "a" ignore
; group "g2" [
    bench "x" ignore
  ; group "x" [
      group "y" [ bench "a" ignore ]
    ; group "y" [ bench "b" ignore ]
    ]
  ; bench "x" ignore
  ; bench "y" ignore
  ; bench "y" ignore
  ]
; group "meta" [
    bench "a" ignore ~attr:(this 1 ++ that "thing")
  ; bench "b" ignore ~attr:(this 13)
  ; group "g" @@
    let bm i = bench "wamm" ignore ~attr:(this i ++ that "thing") in
    List.map bm [1; 10; 42; 117]
  ]
]

let () = Unmark_cli.main ~min_t:0. ~min_s:3 "Naming" suite
