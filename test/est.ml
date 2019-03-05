open Unmark

let _ = Random.self_init ()

let create n = Array.init n (fun _ -> Random.float 1.)
let n = Cmdliner.Arg.(value @@ opt int 200 @@ info ["r"; "runs"])
let _ = Unmark_cli.main_ext "est" ~arg:n @@ fun n ->
  let xs, ys = create n, create n in
  [ bench "ols" (fun _ -> Estimate.ols xs ys);
    bench "tse" (fun _ -> Estimate.tse xs ys); ]
