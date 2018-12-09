module S = Set.Make ( struct type t = int let compare (a: int) b = compare a b end)
let (--) a b =
  let rec go acc a b = if a > b then acc else go (a::acc) (succ a) b in
  go [] a b |> List.rev

open Unmark
let () = Unmark_cli.main "Test" [
  bench "ignore" ignore;
  bench "jitter" (fun () ->
    if Random.float 1. > 0.995 then Unix.sleepf 0.01
  );
  group_f "list" (fun () ->
    let xs = List.init 100_000 (fun _ -> Random.float 1.)
    and cmp a b = compare (a: float) b in
    [ bench "sort" (fun () -> List.sort cmp xs) ]
  ) ~init:ignore ~fini:ignore
]
