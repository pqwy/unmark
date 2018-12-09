(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

type 'a fmt = Format.formatter -> 'a -> unit

let (%) f g x = f (g x)

let (pf, strf) = Format.(fprintf, asprintf)
let pp_seq ?(sep = fun ppf () -> pf ppf "@ ") pp ppf s =
  match s () with
    Seq.Nil          -> ()
  | Seq.Cons (x, xs) ->
      pf ppf "@[%a%a@]" pp x
        (fun ppf -> Seq.iter (pf ppf "%a%a" sep () pp)) xs

let log = Logs.Src.create "unmark" ~doc:"Logs benchmarking progress"
module Log = (val Logs.src_log log)

module SMap = Map.Make (String)

module Option = struct
  let fold ~none f = function Some x -> f x | _ -> none
  let iter f = fold ~none:() f
end

module Measurement = struct

  external get_mtime  : unit -> int = "caml_unmark_mtime_ns" [@@noalloc]
  external get_cycles : unit -> int = "caml_unmark_rdtsc" [@@noalloc]

  type counter = string * string * string option

  type probe = {
    size : int
  ; fs   : (int * (float array -> int -> unit)) list
  ; ctrs : counter list }

  let nothing = { size = 0; fs = []; ctrs = [] }

  let void = Array.create_float 0

  let trigger { size; fs; _} =
    let rec go res i =
      function [] -> res | (n, f)::fs -> f res i; go res (i + n) fs in
    match fs with [] -> void | _ -> go (Array.create_float size) 0 fs

  type runnable = int -> unit
  type sample   = float array

  (* This fundamentally exists since the loop needs a separate function to
     escape register spills/reloads of locals around it ¯\_(ツ)_/¯ .
     Might as well try to inline it upwards. *)
  let runnable f i =
    for _ = 1 to i do f () |> Sys.opaque_identity |> ignore done [@@inline]

  let sample ~probe ~iters run =
    let open Array in
    Gc.minor ();
    let r0    = trigger probe
    and time0 = get_mtime () in
    run iters;
    let time1 = get_mtime ()
    and r1    = trigger probe
    and res   = create_float (2 + probe.size) in
    unsafe_set res 0 @@ float iters;
    unsafe_set res 1 @@ float (time1 - time0) *. 1e-9;
    for i = 0 to probe.size - 1 do unsafe_set res (i + 2) @@ r1.(i) -. r0.(i) done;
    res

  let quot = 20

  let measure ?(probe = nothing) ?(min_t = 1.) ?(min_s = 10) run =
    let rec go ss n time iters =
      if n < min_s || time < min_t then
        let s = sample ~probe ~iters run in
        go (s::ss) (succ n) (time +. s.(1)) (iters + iters / quot + 1)
      else List.rev ss in
    ( Gc.major (); run 1; go [] 0 0. 1 )

  let duration f =
    let now () = float (get_mtime ()) *. 1e-9 in
    let t = now () and x = f () in (x, now () -. t)

  let warmup ?(seconds = 1.) () =
    Log.info (fun k -> k "Warming up for %.01fs." seconds);
    let rec chase n = function
      1 -> n
    | x -> chase (n + 1) (if x mod 2 = 0 then x / 2 else x * 3 + 1) in
    let rec go n x budget =
      if budget < 0. then n else
        let (n, t) = duration (fun () -> chase n x) in
        go n (x + 1) (budget -. t) in
    go 0 1 seconds |> Sys.opaque_identity |> ignore

  module Probe = struct

    type nonrec counter = counter
    type nonrec probe = probe

    let ctr ?unit ?(desc = "") name = (name, desc, unit)
    let name (n, _, _) = n
    and desc (_, d, _) = d
    and unit (_, _, u) = u

    let counters p = p.ctrs

    let probe ~counters f =
      let size = List.length counters in
      { size; fs = [size, f]; ctrs = counters }

    let nothing = nothing

    let (++) p1 p2 =
      { size = p1.size + p2.size; fs = p1.fs @ p2.fs; ctrs = p1.ctrs @ p2.ctrs }

    let pp_ctr ppf c =
      pf ppf "%s%a" (name c) (fun _ -> Option.iter (pf ppf " (%s)")) (unit c)

    let pp ppf p =
      let sep ppf () = pf ppf ",@ " in
      pf ppf "#(probe: %a)" (pp_seq ~sep pp_ctr) (List.to_seq p.ctrs)

    let rdtsc =
      let f arr off = Array.unsafe_set arr off (get_cycles () |> float) in
      probe f ~counters:[ctr "tsc" ~desc:"CPU timestamp counter (TSC)"]

    let gc1 res i =
      let open Array in
      let (minor, promoted, major) = Gc.counters () in
      unsafe_set res (i + 0) @@ minor;
      unsafe_set res (i + 1) @@ promoted;
      unsafe_set res (i + 2) @@ major -. promoted

    let gc2 res i =
      let open Array in
      let s = Gc.quick_stat () in
      unsafe_set res (i + 0) @@ s.Gc.minor_words;
      unsafe_set res (i + 1) @@ s.Gc.promoted_words;
      unsafe_set res (i + 2) @@ s.Gc.major_words -. s.Gc.promoted_words;
      unsafe_set res (i + 3) @@ float s.Gc.minor_collections;
      unsafe_set res (i + 4) @@ float s.Gc.major_collections

    let gc_counters =
      probe gc1 ~counters:[
        ctr "min"  ~unit:"w" ~desc:"Words allocated on the minor heap"
      ; ctr "prom" ~unit:"w" ~desc:"Words promoted to the major heap"
      ; ctr "maj"  ~unit:"w" ~desc:"Words allocated (directly) on the major heap" ]

    let gc_q_stat =
      probe gc2 ~counters:(gc_counters.ctrs @ [
        ctr "gc_min" ~desc:"Minor collections"
      ; ctr "gc_maj" ~desc:"Major collectoins" ])
  end

  let core_counters = Probe.[
    ctr "iterations" ~desc:"Number of iterations (pseudo-counter)"
  ; ctr "time"       ~desc:"Wall-clock time" ~unit:"s"
  ]

end

module Attr = struct
  type t = (Format.formatter -> unit -> unit) SMap.t
  let inj pp x ppf () = pp ppf x
  and prj v =
    let open Format in
    flush_str_formatter () |> ignore;
    pp_set_margin str_formatter 1_000_000; v str_formatter ();
    flush_str_formatter ()
  let key pp ~name x = SMap.singleton name (inj pp x)
  let (++) = SMap.union (fun _ _ v -> Some v)
  let empty = SMap.empty
  let of_list =
    let f = inj Format.pp_print_string in
    List.fold_left (fun m (k, x) -> SMap.add k (f x) m) empty
  let to_list m = SMap.bindings m |> List.map (fun (k, v) -> (k, prj v))
  let is_empty m = SMap.cardinal m = 0
  let pp ppf m =
    let item ppf (k, v) = pf ppf "@[%s:@ @[%a@]@]" k v () in
    match SMap.bindings m with
      []      -> ()
    | kv::kvs ->
        let pp_kvs ppf = List.iter (pf ppf ",@ %a" item) in
        pf ppf "@[%a%a@]" item kv pp_kvs kvs
end

let map_acc_l f s xs =
  let rec go acc s = function
    []    -> (List.rev acc, s)
  | x::xs -> let (x, s) = f x s in go (x::acc) s xs in
  go [] s xs

let rec last = function
  [] -> invalid_arg "last" | [x] -> x | _::xs -> last xs

let rec init = function
  [] -> invalid_arg "init" | [_] -> [] | x::xs -> x :: init xs

module Benchmarks = struct

  type name = string list * string
  type query = string list

  let string_of_name (ns, n) = String.concat "/" (ns @ [n])
  let pp_name ppf (ns, n) = List.iter (pf ppf "%s/") ns; pf ppf "%s" n
  let q_of_s = List.map (String.split_on_char '/')
  let name_of_string s =
    let xs = String.split_on_char '/' s in
    (init xs, last xs)
  let name = String.map (function '/' -> '!' | c -> c)

  let matchesv ?(group = false) ~q (p, n) =
    let rec go qs ns = match qs, ns with
      q::qs, n::ns -> (q = "" || q = n) && go qs ns
    | []   , _     -> true
    | _    , []    -> group
    and ns = p @ [n] in
    q = [] || List.exists (fun q -> go q ns) q

  let matches ?group ~q n = matchesv ?group ~q:(q_of_s q) n

  type run = {
    suite      : string
  ; note       : string
  ; time       : float
  ; counters   : (string * string option) list
  ; benchmarks : (name * Attr.t * Measurement.sample list) list
  }

  module Run = struct

    let filter ?(q = []) r = match q with
      [] -> r
    | _  -> let q = q_of_s q in
            { r with benchmarks =
                List.filter (fun (n, _, _) -> matchesv ~q n) r.benchmarks }

    let (>>=) a fb = match a with Ok x -> fb x | Error e -> Error e

    open Unmark_json

    let arr xs = G.(arr (List.fold_right (fun x xs -> el x ++ xs) xs empty))
    let arr_f f xs = List.map f xs |> arr
    let obj_ xs =
      G.(obj (List.fold_right (fun (n, x) xs -> mem n x ++ xs) xs empty))

    let pack_ctr = function (n, None) -> n | (n, Some u) -> n ^ ":" ^ u
    let unpack_ctr n = match String.rindex n ':' with
      exception Not_found -> (n, None)
    |  i -> String.(sub n 0 i, Some (sub n (i + 1) (length n - i - 1)))

    let attr_to m = match Attr.to_list m with
      [] -> G.null
    | xs -> List.map (fun (k, v) -> k, G.string v) xs |> obj_

    let attr_of = Q.(
      mems string |> nullable |>
      map @@ Option.fold ~none:Attr.empty Attr.of_list )

    let to_json res = G.(obj_ [
      "suite"      , string res.suite
    ; "note"       , string res.note
    ; "time"       , float res.time
    ; "counters"   , res.counters |> arr_f (string % pack_ctr)
    ; "benchmarks" , res.benchmarks |>
       arr_f (fun (name, attr, samples) ->
         obj_ [ "name"   , string (string_of_name name)
              ; "attr"   , attr_to attr
              ; "samples", samples |> arr_f (arr_f float % Array.to_list) ]
       )])

    let q =
      Q.(obj (fun suite note time counters benchmarks ->
        { suite; note; time; counters; benchmarks })
      |> mem "suite"     string
      |> mem "note"      string
      |> mem "time"      float
      |> mem "counters"  (array (map unpack_ctr string))
      |> mem "benchmarks"
         (array (obj (fun a b c -> (a, b, c))
         |> mem "name"    (map name_of_string string)
         |> mem "attr"    attr_of
         |> mem "samples" (array (array float |> map Array.of_list)))))

    let add_json buf x = G.buffer_add buf (to_json x)
    let of_json s =
      of_string_prefix s >>= fun (j, s) -> Q.query q j >>= fun x -> Ok (x, s)
  end

  type _ thunk = T : (unit -> 'b) * ('b -> unit) * ('b -> 'a) -> 'a thunk

  let (%%) f (T (init, fini, g)) = T (init, fini, fun x -> f (g x))

  let eval (T (init, fini, f)) =
    let x = init () in
    match f x with r -> fini x; r | exception exn -> fini x; raise exn

  type t = Bench of string * Attr.t * Measurement.runnable
         | Group of string * t list thunk

  let bench ?(attr = Attr.empty) n f =
    Bench (name n, attr, fun i -> (Measurement.runnable [@inlined]) f i)
    [@@inline]
  let group_f ~init ~fini n f =
    Group (name n, T (init, fini, f))
  let group n bs = group_f n (fun () -> bs) ~init:ignore ~fini:ignore

  let rename =
    let nmap f = function
        Bench (n, a, r) -> let (n, x) = f n in (Bench (n, a, r), x)
      | Group (n, t)    -> let (n, x) = f n in (Group (n, t), x) in
    let f x m = x |> nmap @@ fun n -> match SMap.find_opt n m with
        None   -> (n, SMap.add n 1 m)
      | Some i -> (strf "%s_%d" n i, SMap.add n (succ i) m) in
    fst % map_acc_l f SMap.empty

  let fold ~bench ~group ~suite =
    let rec list p = List.map (go p) % rename
    and go p = function
      Bench (n, a, r) -> bench (p, n) a r
    | Group (n, t)    -> group (p, n) (list (p @ [n]) %% t) in
    suite % list []

  let filter ?(q = []) xs = match q with
    [] -> xs
  | q  ->
      let q = q_of_s q in
      let bench p a r = if matchesv ~q p then [Bench (snd p, a, r)] else []
      and group p t = if matchesv ~q p ~group:true
                      then [Group (snd p, List.concat %% t)] else [] in
      fold xs ~suite:List.concat ~bench ~group

  let pp_counts ppf res =
    let f (n, i) s = (n + 1, i + truncate s.(0)) in
    let (n, i) = List.fold_left f (0, 0) res in
    pf ppf "samples: %d  runs: %d" n i

  let log_exn n f = try f () with exn ->
    let open Printexc in
    let bt = get_raw_backtrace () in
    Logs.err (fun k -> k "%a: exception: %s" pp_name n (to_string exn));
    raise_with_backtrace exn bt

  let concat, cons = List.fold_right (@@), fun x xs -> x::xs

  let ctr_desc c = Measurement.Probe.(name c, unit c)

  let run ?(probe = Measurement.nothing) ?min_t ?min_s ?q ?(note = "") ~suite xs =
    Log.info (fun k -> k "%a" Measurement.Probe.pp probe);
    let time = Unix.gettimeofday () (* XXX *)
    and benchmarks =
      let bench n a r =
        Log.info (fun k -> k "Benchmarking %a." pp_name n);
        let (res, t) = Measurement.(duration @@ fun () ->
          log_exn n @@ fun () -> measure ?min_t ?min_s ~probe r) in
        Log.debug (fun k -> k "time: %.02f  %a" t pp_counts res);
        cons (n, a, res)
      and group _ = concat % eval in
      fold ~bench ~group ~suite:concat (filter ?q xs) [] in
    { suite; note; time; benchmarks;
      counters = Measurement.(core_counters @ probe.ctrs) |> List.map ctr_desc }
end

module Estimate = struct

  (* Normal distribution's CDF[.] = 0.995.
   *
   * TODO: The actual inverse CDF.
   * https://stackedboxes.org/2017/05/01/acklams-normal-quantile-function *)
  let e_sigma = 2.575_829_303

  type estimator = float array -> float array -> e
  and e = { a: float; b: float; r2: float; bounds: (float * float) Lazy.t }

  let pp_e ppf { a; b; r2; bounds = lazy (b0, b1) } =
    pf ppf "#(@[a = %.10g,@ b = %.10g < %.10g > %.10g,@ R² = %.10g@])" a b0 b b1 r2

  let size = float % Array.length
  let sum  = Array.fold_left (+.) 0.
  let mean xs = sum xs /. size xs
  let amap2 f xs ys = Array.(init (length xs) @@ fun i -> f xs.(i) ys.(i))

  let (@) xs ys =
    let a = ref 0. in
    for i = 0 to Array.length xs - 1 do a := !a +. xs.(i) *. ys.(i) done;
    !a
  let (++.) c = Array.map (fun x -> x +. c)

  let r2 xs ys (a, b) =
    let yd = (-. mean ys) ++. ys in
    match yd @ yd with
      0.  -> 0.
    | ssy -> let ye = amap2 (fun x y -> y -. a -. b *. x) xs ys in
             1. -. (ye @ ye) /. ssy

  let validity e = match e.r2 with
    0.              -> `Undef
  | x when x < 0.9  -> `Bad
  | x when x < 0.98 -> `Meh
  | _               -> `Good

  let median3 a b c =
    let a = min a b and b = max a b in if a < c then min b c else a [@@inline]

  let kth k xs =
    let part xs i j =
      let rec scanu xs p i = if xs.(i) < p then scanu xs p (i + 1) else i in
      let rec scand xs p j = if xs.(j) > p then scand xs p (j - 1) else j in
      let rec meet xs p i j =
        let i = scanu xs p i and j = scand xs p j in
        if i < j then
          let x = xs.(i) in
          xs.(i) <- xs.(j); xs.(j) <- x; meet xs p (i + 1) (j - 1)
        else j in
      let p = median3 xs.(i) xs.(j) xs.((j - i) / 2 + i) in
      meet xs p i j in
    let rec go xs k i j =
      if i < j then
        let x = part xs i j in
        if k <= x then go xs k i x else go xs k (x + 1) j
      else xs.(i) in
    go xs k 0 (Array.length xs - 1)

  (* TODO: Interpolation. *)
  let quantile q xs =
    kth (float (Array.length xs - 1) *. q |> truncate) xs
  let median ?(preserve=false) xs =
    quantile 0.5 (if preserve then Array.copy xs else xs)

  (* "Ordinary least squares."
   *
   * http://mathworld.wolfram.com/LeastSquaresFitting.html
   *)
  let ols xs ys =
    let n = size xs in
    let (xd, yd)     = (-. mean xs ++. xs, -. mean ys ++. ys) in
    let (xx, yy, xy) = (xd @ xd, yd @ yd, xd @ yd) in
    let b   = xy /. xx in
    let a   = amap2 (fun x y -> y -. x *. b) xs ys |> mean in
    let err = sqrt ((yy -. b *. xy) /. xx /. (n -. 2.)) in
    { a; b; r2 = r2 xs ys (a, b);
      bounds = lazy (b -. e_sigma *. err, b +. e_sigma *. err) }

  (* Theil-Sen estimator.
   *
   * Slope is the usual median-pairwise-slope.
   *
   * Intercept is median-of-intercepts as opposed to intercept-of-medians as it
   * generally looks better. It's inconsequential otherwise.
   *
   * Confidence interval is like [3] (ch 5.5, eqn 9-11), and [2] (method 1).
   * Kendall T distribution is obtained by approximating Kendall τ with a
   * gaussian, variance [2 (2n + 5) / 9n (n - 1)], and scaling that up by
   * [n (n - 1) / 2], giving the magic for [w], below. This is same as [1]
   * (eqn 2.6), but without the "standard correction for ties observations."
   *
   * References:
   * 1 - P. K. Sen, "Estimates of the Regression Coefficient Based on Kendall's
   *     Tau", Journal of the American Statistical Association, Vol. 63,
   *     No. 324. (1968), pp. 1379-1389.
   * 2 - R. R. Wilcox, "A Note on the Theil-Sen Regression Estimator When the
   *     Regressor Is Random and the Error Term Is Heteroscedastic",
   *     Biometrical Journal 40 (1998) 3, pp. 261-268.
   * 3 - W. J. Conover, "Practical Nonparametric Statistics", 3rd ed., John
   *     Wiley and Sons, 1999, pp. 336.
   *)
  let tse xs ys =
    let n  = Array.length xs in
    let bs = Array.create_float (n * (n - 1) / 2) in
    let t  = ref 0 in
    for i = 0 to n - 1 do
      let xi = xs.(i) in
      for j = 0 to n - 1 do
        let dx = xs.(j) -. xi in
        if dx > 0. then (bs.(!t) <- (ys.(j) -. ys.(i)) /. dx; incr t)
      done
    done;
    let nb, bs = float !t, Array.sub bs 0 !t in
    let b = median bs in
    let a = median (amap2 (fun x y -> y -. b *. x) xs ys)
    and bounds =
      let n  = float n in
      let w  = e_sigma *. sqrt (n *. (n -. 1.) *. (2. *. n +. 5.) /. 18.) in
      let q0 = (1. -. w /. nb) /. 2.
      and q1 = (1. +. w /. nb) /. 2. +. 1. /. nb in
      lazy (quantile q0 bs, quantile q1 bs) in
    { a; b; bounds; r2 = r2 xs ys (a, b) }

end

type bench  = Benchmarks.t
let bench   = Benchmarks.bench
and group   = Benchmarks.group
and group_f = Benchmarks.group_f
