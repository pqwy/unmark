(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark

let rec (--) a b = if a <= b then a :: succ a -- b else []
let size = Attr.key ~name:"size" Fmt.int
let group_f = group_f ~init:ignore ~fini:ignore

let suite (sizes, alloc) = [
  group_f "bytes" (fun () ->
    let create = Bytes.create in
    let refs = 0 -- alloc |> List.(map (fun _ -> map create sizes)) in
    ignore refs;
    sizes |> List.map @@ fun s ->
      bench "bytes" ~attr:(size s) (fun () -> create s)
  );
  group_f "bigarray" (fun () ->
    let create = Bigarray.(Array1.create char c_layout) in
    let refs = 0 -- alloc |> List.(map (fun _ -> map create sizes)) in
    ignore refs;
    sizes |> List.map @@ fun s ->
      bench "ba" ~attr:(size s) (fun () -> create s)
  );
]

let defsz = [ 1; 5; 10; 50; 100; 500; 1_000; 5_000; 10_000; 50_000;
              100_000; 500_000; 1_000_000 ]
let arg =
  let open Cmdliner in
  let sizes = Arg.(value @@ opt (list int) defsz @@ info ["sizes"])
  and alloc = Arg.(value @@ opt int ~vopt:1000 0 @@ info ["alloc"]) in
  Term.(const (fun a b -> a, b) $ sizes $ alloc)

let () = Unmark_cli.main_ext "Allocations" ~arg suite
