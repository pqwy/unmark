(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark

let b = (bench [@inlined]) "noop" ignore
let () = Unmark_cli.main "Sanity" [b]
