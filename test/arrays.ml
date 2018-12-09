(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark

let () = Unmark_cli.main "Arrays" [

  group "array" [
    bench "create 50"  (fun () -> Array.make 50 17);
    bench "create 300" (fun () -> Array.make 300 17);
    bench "init 50"    (fun () -> Array.init 50 (fun _ -> 17));
    bench "init 300"   (fun () -> Array.init 300 (fun _ -> 17));
  ];

  group "int bigarray" (
    let open Bigarray in
    let create = Array1.create int c_layout in
    let a50 = create 50 and a300 = create 300 in
    [ bench "create 50"  (fun () -> create 50);
      bench "create 300" (fun () -> create 300);
      bench "fill 50"    (fun () -> Array1.fill a50 17);
      bench "fill 300"   (fun () -> Array1.fill a300 17);
    ]);

  group "char bigarray" (
    let open Bigarray in
    let create = Array1.create char c_layout in
    let a50 = create 50 and a300 = create 300 in
    [ bench "create 50"  (fun () -> create 50);
      bench "create 300" (fun () -> create 300);
      bench "fill 50"    (fun () -> Array1.fill a50 'x');
      bench "fill 300"   (fun () -> Array1.fill a300 'x');
    ]);

  group "bytes" [
    bench "create 50"  (fun () -> Bytes.create 50);
    bench "create 300" (fun () -> Bytes.create 300);
    bench "init 50"    (fun () -> Bytes.init 50 (fun _ -> 'x'));
    bench "init 300"   (fun () -> Bytes.init 300 (fun _ -> 'x'));
  ]

]
