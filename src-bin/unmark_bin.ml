(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark
open Estimate

let (%) f g x = f (g x)

module Option = struct
  let (%%) f = function Some x -> Some (f x) | _ -> None
  let get ~none = function Some x -> x | _ -> none
  let cat ~none xs =
    if List.exists (function None -> false | _ -> true) xs then
      Some (List.map (get ~none) xs) else None
end

let qual_name (ns, n) = String.concat "/" (ns @ [n])
let pp_qual_name ppf (ns, n) = List.iter (Fmt.pf ppf "%s/") ns; Fmt.string ppf n
let pp_run_bench ppf (runid, bench) =
  Fmt.(pf ppf "run %d, %a" runid (styled `Bold pp_qual_name) bench)

let semi_g f = function
  []    -> invalid_arg "semi_g: empty list"
| x::xs -> List.fold_left f x xs
let maximum  = semi_g max
let max_by f = maximum  % List.map f

let rec map_f f = function
  []    -> []
| x::xs -> match f x with Some y -> y :: map_f f xs | _ -> map_f f xs

let rec map2_t f xs ys =
  match xs, ys with x::xs, y::ys -> f x y :: map2_t f xs ys | _ -> []

let combine_t xs ys = map2_t (fun x y -> (x, y)) xs ys

let rec enumerate ?(from = 0) = function
  []    -> []
| x::xs -> (from, x) :: enumerate ~from:(succ from) xs

let rec replicate n x = if n > 0 then x :: replicate (n - 1) x else []

let rec transpose = function
  []  -> []
| xss ->
    let hts =
      List.fold_right (fun xs hts -> match xs, hts with
        [], _ | _, None -> None
      | x::xs, Some (hs, ts) -> Some (x::hs, xs::ts)
      ) xss (Some ([], [])) in
    match hts with Some (hs, ts) -> hs :: transpose ts | _ -> []

(* let rec transpose = function *)
(*   [] | []::_ -> [] *)
(* | lls        -> List.(map hd) lls :: transpose List.(map tl lls) *)

let rec intersperse ~slip = function
  []    -> []
| [x]   -> [x]
| x::xs -> x :: slip :: intersperse ~slip xs

let of_list_f f xs =
  let arr = Array.create_float (List.length xs) in
  let _ = List.fold_left (fun i x -> arr.(i) <- f x; i + 1) 0 xs in
  arr

let partition (type a) ?(cmp=compare) xs =
  let module M = Map.Make (struct type t = a let compare = cmp end) in
  let aux (m, i) (k, v) =
    match M.find_opt k m with
      None          -> (M.add k (i, [v]) m, succ i)
    | Some (i', vs) -> (M.add k (i', v::vs) m, i) in
  let m = List.fold_left aux (M.empty, 0) xs |> fst in
  M.bindings m |> List.sort (fun (_, (i1, _)) (_, (i2, _)) -> compare i1 i2)
               |> List.map (fun (k, (_, xs)) -> (k, List.rev xs))

module IMap = struct
  include Map.Make (struct
    type t = int let compare (a: int) b = compare a b
  end)
  let of_list xs = List.fold_left (fun m (k, v) -> add k v m) empty xs
end

let estimates ~estimator ~counters id selector = function
  []|[_] ->
    Logs.warn (fun m -> m "%a: insufficient samples" pp_run_bench id);
    List.map (fun _ -> `Missing) counters
| samples  -> match selector "iterations" with
    None ->
      Logs.warn (fun m -> m "%a: missing `iterations'" pp_run_bench id);
      List.map (fun _ -> `Missing) counters
  | Some i ->
      let xs = of_list_f (fun s -> s.(i)) samples in
      counters |> List.map @@ fun c ->
        match selector c with
          None   -> `Missing
        | Some j ->
            let ys = of_list_f (fun s -> s.(j)) samples in
            `Estimate (estimator xs ys)

module SMap = Map.Make (String)

let selector counters =
  let aux (m, i) (name, _) = (SMap.add name i m, succ i) in
  let (m, _) = List.fold_left aux (SMap.empty, 0) counters in
  (fun counter -> SMap.find_opt counter m)

let ctr_units ctrss =
  let aux m = function
    (_   , None)   -> m
  | (name, Some x) -> match SMap.find_opt name m with
      Some v when v = x -> m | _ -> SMap.add name x m in
  let m = List.fold_left (List.fold_left aux) SMap.empty ctrss in
  (fun ctr -> SMap.find_opt ctr m)

type bench = {
  name    : Benchmarks.name
; runs    : int list
; samples : int list
; attrs   : Attr.t list
; ctrs    : [ `Estimate of Estimate.e | `Missing ] list list
}

let def_ctrs = ["time"; "min"; "maj"]

let report ?(estimator = tse) ?(counters = def_ctrs) runs =
  let open Benchmarks in
  let f_suite suite runs =
    let runs  = List.sort (fun r1 r2 -> compare r1.time r2.time) runs
                |> enumerate ~from:1
    and units = List.map (fun res -> res.counters) runs |> ctr_units in
    let rmap  = List.map (fun (id, res) -> (id, (res.time, res.note))) runs
                |> IMap.of_list
    and benches = runs |>
      List.(map (fun (id, res) ->
        let s = selector res.counters in
        res.benchmarks |> map (fun (name, attr, samples) ->
          let b = estimates ~estimator ~counters (id, name) s samples in
          (name, (id, samples, attr, b))))) |>
      List.concat |> partition |>
      List.(map (fun (name, runs) ->
        { name
        ; runs    = map (fun (id, _, _, _) -> id) runs
        ; samples = map (fun (_, ss, _, _) -> length ss) runs
        ; attrs   = map (fun (_, _, at, _) -> at) runs
        ; ctrs    = map (fun (_, _, _, cs) -> cs) runs |> transpose })) in
    (suite, rmap, units, benches) in
  (counters, runs |> List.map (fun x -> x.suite, x) |> partition
                  |> List.map (fun (suite, runs) -> f_suite suite runs))

let zeroish x = abs_float x < epsilon_float

let tz_offset_s = Ptime_clock.current_tz_offset_s () |> Option.get ~none:0
let pp_time_f fmt ppf f = match Ptime.of_float_s f with
  Some p -> fmt ppf p
| None   -> Fmt.pf ppf "<INVALID TIME: %f>" f
let pp_time_human_f = pp_time_f (Ptime.pp_human ~tz_offset_s ())
let pp_time_rfc3339_f = pp_time_f (Ptime.pp_rfc3339 ~tz_offset_s ())

let pp_f_hum ppf x =
  let (fmt: _ format6) =
    let a = abs_float x in
    if zeroish a then "%.0f"  else
    if a < 1.    then "%.02e" else
    if a < 10.   then "%.02f" else
    if a < 100.  then "%.01f" else "%.0f" in
  Fmt.pf ppf fmt x

let pp_SI ~unit ppf x =
  let a = abs_float x in
  let (x, p) =
    if a = 0.   then (x        , "" ) else
    if a < 1e-9 then (x *. 1e12, "p") else
    if a < 1e-6 then (x *. 1e9 , "n") else
    if a < 1e-3 then (x *. 1e6 , "μ") else
    if a < 1e0  then (x *. 1e3 , "m") else (x, "") in
  Fmt.pf ppf "%a %s%s" pp_f_hum x p unit

let pp_counter ~unit () ppf x =
  let a = abs_float x in
  match unit with
    Some unit -> pp_SI ~unit ppf x
  | None      -> if a < 1e8 then pp_f_hum ppf x else Fmt.pf ppf "%.03e" x

module To_JSON = struct

  open Unmark_json

  let arr xs =
    G.(arr (List.fold_right (fun x xs -> el x ++ xs) xs empty))

  let obj xs =
    G.(obj (List.fold_right (fun (n, x) xs -> mem n x ++ xs) xs empty))

  let time_f f = G.string @@ Fmt.strf "%a" pp_time_rfc3339_f f

  let stretch nruns runids results =
    let rec f (id, r as e) k x = match compare x id with
        -1 -> `Missing :: f e k (succ x)
      |  0 -> r :: k (succ x)
      |  _ -> assert false in
    let tagged = List.combine runids results in
    List.fold_right f tagged (fun x -> replicate (nruns - x) `Missing) 1

  let of_bench nruns counters b =
    let counters =
      List.map2 (fun counter results ->
        let items =
          results |> stretch nruns b.runs |> List.map (function
              `Missing -> [G.null; G.null; G.null; G.null]
            | `Estimate { b; r2; bounds = lazy (b0, b1); _ } ->
                [ G.float b; G.float b0; G.float b1; G.float r2 ]
          ) |> transpose |> List.map arr
            |> List.combine ["v"; "min"; "max"; "r2"] |> obj in
        (counter, items)
      ) counters b.ctrs in
    ("name", G.string (qual_name b.name)) :: counters |> obj

  let of_suite ~counters (suite, runs, _units, benches) =
    obj [ "suite", G.string suite
        ; "runs" ,
          IMap.bindings runs |> List.map (fun (_, (time, note)) ->
              obj [ "time", time_f time; "note", G.string note ])
          |> arr
        ; "benchmarks",
          List.map (of_bench (IMap.cardinal runs) counters) benches |> arr ]

  let output_json chan (counters, suites) =
    suites |> List.iter @@ fun suite ->
      G.output chan (of_suite ~counters suite);
      output_string chan "\n"
end

type autobool = [`Yes | `No | `Auto]
let auto x = function `Auto -> if x then `Yes else `No | v -> v
let av ?(auto = false) = function `Yes -> true | `No -> false | `Auto -> auto

module To_Notty = struct

  type props = {
    compact   : autobool
  ; intervals : bool
  ; show_r2   : autobool
  ; samples   : bool
  ; no_meta   : bool
  }
  let props compact intervals show_r2 samples no_meta =
    { compact; intervals; show_r2; samples; no_meta }
  let def_p = props `Auto false `Auto false false

  open Notty
  open Notty.Infix

  let string ?(a = A.empty) s = I.string a s

  let vcat_r xs =
    List.map (max_by I.width xs |> I.hsnap ~align:`Right) xs |> I.vcat

  let complete f xss =
    let rec go i j n xs = match n, xs with
      0, []    -> []
    | n, []    -> f i j :: go i (j + 1) (n - 1) []
    | n, x::xs -> x :: go i (j + 1) (n - 1) xs in
    let m = max_by List.length xss in
    List.mapi (fun i -> go i 0 m) xss

  let snap ~align w h x =
    let halign = match align with
      `NW | `W | `SW -> `Left
    | `N  | `C | `S  -> `Middle
    | `NE | `E | `SE -> `Right
    and valign = match align with
      `NW | `N | `NE -> `Top
    | `W  | `C | `E  -> `Middle
    | `SW | `S | `SE -> `Bottom in
    x |> I.hsnap ~align:halign w |> I.vsnap ~align:valign h

  let table ?spacing:((sw, sh) = (0, 0)) xss =
    let open List in
    let xss = complete (fun _ _ -> I.empty, `C) xss in
    let hs  = map (max_by (I.height % fst)) xss in
    let ws  = map (max_by (I.width % fst)) (transpose xss) in
    let cell h w (x, align) =
      x |> snap ~align w h |> I.vpad 0 sh |> I.hpad 0 sw in
    map2 (fun h row ->
      map2 (fun w x -> cell h w x) ws row |> I.hcat
    ) hs xss |> I.vcat

  let groups ~by xs =
    let rec f = function
      [] -> []
    | (([], _), x) :: xs -> `B x :: f xs
    | ((p::ps, n), x) :: xs -> g p [((ps, n), x)] xs
    and g p0 acc = function
      ((p::ps, n), x) :: xs when p0 = p -> g p0 (((ps, n), x) :: acc) xs
    | xs -> `G (p0, List.rev acc |> f) :: f xs in
    List.map (fun x -> by x, x) xs |> f

  let show_estimate ~unit ~v x =
    let attr = match v with
      `Good | `Undef -> A.empty
    | `Meh  | `Bad   -> A.(st underline) in
    I.strf "%a" (pp_counter ~unit () |> I.pp_attr attr) x
  let show_bound ~unit = I.strf "%a" (pp_counter ~unit ())
  let show_r2 = I.strf "%.02f"

  let show_name ?(group = false) name indent =
    let a = A.(if group then empty else st bold) in
    I.string a name |> I.hpad (2 * indent) 3

  let headings counter = ["⟵", `C; counter, `C; "⟶", `C; "R²", `C]

  let indicators ~p ~unit = function
    `Missing -> [None; None; None; None]
  | `Estimate ({ b; r2; bounds; _ } as e) ->
      let v = validity e in
      let bound prj =
        if not p.intervals then None else
          let b = Lazy.force bounds |> prj in
          if zeroish b then None else Some (show_bound ~unit b) in
      [ bound fst
      ; Some (show_estimate ~unit ~v b)
      ; bound snd
      ; match (p.show_r2, v) with
          (`Auto, `Bad) | (`Yes, _) -> Some (show_r2 r2) | _ -> None ]

  let indicator_grid ~p ~counters ~units bench_ctr_run =
    let columns =
      List.(map2_t @@ fun ctr benches_ctr ->
        let unit = units ctr in
        let bench runs =
          let cat xs = Option.(vcat_r %% cat ~none:(I.void 0 1) xs) in
          runs |> map (indicators ~p ~unit) |> transpose |> map cat
        and cat = Option.cat ~none:I.empty in
        benches_ctr |> map bench |> transpose |> map cat |> combine (headings ctr)
      ) counters (transpose bench_ctr_run)
      |> List.concat
      |> map_f (function (h, Some x) -> Some (h, x) | _ -> None) in
    List.(map fst columns, map snd columns |> transpose)

  let cons ?(cond = true) x xs = if cond then x :: xs else xs

  let show_benches ~p:({ samples; _ } as p: props) ~counters ~units benches =
    let compact = av p.compact
    and meta    = not p.no_meta &&
      List.(exists (fun b -> exists (not % Attr.is_empty) b.attrs)) benches
    and indicators =
      List.map (fun b -> b.ctrs) benches |>
        indicator_grid ~p ~counters ~units in
    match indicators with
      ([], _)         -> I.string A.(st bold) "¯\\_(ツ)_/¯"
    | (headers, rows) ->
      let headers' =
        cons (I.empty, `C) @@
        cons ~cond:(not compact) (I.empty, `C) @@
        cons ~cond:meta (I.empty, `C) @@
        cons ~cond:samples (string "n", `C) @@
        List.map (fun (n, a) -> string n |> I.vpad 0 1, a) headers
      and rows' =
        let rec flatten i = function
          []                -> []
        | `G (g, es) :: xs  ->
            [show_name ~group:true g i, `NW] :: flatten (i + 1) es @ flatten i xs
        | `B (b, row) :: xs ->
            let ints  = vcat_r % List.map (I.strf "%d")
            and attrs = vcat_r % List.map (I.strf "%a" Attr.pp) in
            ( cons (show_name (snd b.name) i, `NW) @@
              cons ~cond:(not compact) (ints b.runs, `E) @@
              cons ~cond:meta (attrs b.attrs, `W) @@
              cons ~cond:samples (ints b.samples, `E) @@
              List.map (fun i -> (i, `E)) row ) :: flatten i xs in
        combine_t benches rows |>
        groups ~by:(fun (b, _) -> b.name) |> flatten 0 in
      table (headers' :: rows') ~spacing:(3, if compact then 0 else 1)

  let show_run ?id time note =
    I.hcat @@ intersperse ~slip:(string " — ") @@
      ( match id with Some i -> [I.strf "%d" i] | _ -> [] ) @
      [ I.strf "%a" pp_time_human_f time ] @
      ( match note with "" -> [] | _ -> [I.string A.(st italic) note] )

  let show_report ?(p = def_p) (counters, suites) =
    let suite (name, runs, units, benches) =
      let header =
        let name = I.string A.(st bold ++ st underline) name in
        match IMap.bindings runs with
          [(_, (time, note))] ->
            name <|> string " — " <|> show_run time note
        | runs ->
            name <->
            (runs |> List.map (fun (id, (time, note)) -> show_run ~id time note)
                  |> I.vcat |> I.pad ~l:2 ~t:1)
      and body =
        let compact = p.compact |> auto (IMap.cardinal runs < 2) in
        show_benches ~counters ~units benches ~p:{p with compact} in
      (header <-> I.void 0 2 <-> I.hpad 2 0 body) |> I.vpad 1 1 in
    List.map suite suites |> I.vcat
end

let input_all chan =
  let rec go bf b = match input chan b 0 4096 with
    0 -> Buffer.contents bf
  | n -> Buffer.add_subbytes bf b 0 n; go bf b in
  go (Buffer.create 4096) (Bytes.create 4096)

let rec read_runs_v acc = function
  "" -> Ok (List.rev acc)
| s  -> match Benchmarks.Run.of_json s with
    Ok (r, s) -> read_runs_v (r::acc) s
  | Error e   -> Error e

let or_die = function
  Ok x           -> x
| Error (`Msg e) -> Logs.err (fun m -> m "%s\n%!" e); exit 1

let run_main ?q ~props ~counters ~estimator ~json ic =
  let runs = input_all ic |> read_runs_v [] |> or_die in
  let runs = List.map (Benchmarks.Run.filter ?q) runs in
  let rep  = report ~counters ~estimator runs in
  match json with
    false -> To_Notty.show_report ~p:props rep
             |> Notty_unix.eol |> Notty_unix.output_image
  | true  -> To_JSON.output_json stdout rep; flush stdout

open Cmdliner

let err_msg msg = Error (`Msg msg)

let estimator = Arg.conv (
  (function
    "TSE"|"tse" -> Ok tse
  | "OLS"|"ols" -> Ok ols
  | _           -> err_msg "unknown estimator."),
  (fun ppf e -> Fmt.string ppf @@
    if e == tse then "TSE" else if e == ols then "OLS" else "?"))

let autobool = Arg.conv (
  (function
    "true" |"on"  -> Ok `Yes
  | "false"|"off" -> Ok `No
  | "auto"        -> Ok `Auto
  | _ -> err_msg "can be one of `true', `on', `false', `off', or `auto'."),
  (fun ppf x -> Fmt.string ppf @@
    match x with `Yes -> "true" | `No -> "false" | `Auto -> "auto"))

let ($$) f a = Term.(const f $ a)

let s_est = "ESTIMATORS"

let info =
  Term.info "unmark" ~exits:Term.default_exits
  ~version:"%%VERSION%%"
  ~doc:"Unmark results viewer"
  ~man:[

  `S Manpage.s_description
; `P "$(mname) is used to summarize raw measurements produced by $(b,Unmark). \
      The output can be reduced or expanded by filtering the benchmarks and \
      by selecting which counters to display."
; `P "$(mname) expects a stream of JSON documents on stdin, and will display \
      a human-readable report on stdout. This can be overriden to produce a \
      machine-readable JSON document instead."

; `S Manpage.s_options

; `S "FILTERING"

; `P "Filtering restricts analysis to a subset of benchmarks. Specifying \
      multiple queries selects the benchmarks that match any of them."
; `P "For more information, see the filtering documentation in \
      $(i,Unmark.Benchmarks) [1]."
; `Pre "[1] - https://pqwy.github.io/unmark/doc/Unmark.Benchmarks.html#filtering"


; `S s_est

; `P "$(mname) supports two methods of estimating the \"true\" value of a \
      counter from the sequence of measurements: Ordinary Least Squares (OLS), \
      which is faster, and Theil-Sen Estimator (TSE), which is more robust."

; `P "$(mname) defaults to TSE. Consult the documentation for \
      $(i,Unmark.Estimate) [1] for more information."

; `P "If $(i,--r2) is `auto', estimates are highlited when \
      $(i,Unmark.Estimate.validity) is $(i,`Bad)  or $(i,`Meh), and \
      $(i,R^2) values are shown when validity is $(i,`Bad). \
      Consult the documentation for $(i,validity) [2] for more information."

; `Pre "[1] - https://pqwy.github.io/unmark/doc/Unmark.Estimate.html#estimators\n\
        [2] - https://pqwy.github.io/unmark/doc/Unmark.Estimate.html#VALvalidity"
]

let t_render_props =
  let open Arg in
  let compact = value @@ opt autobool `Auto @@ info ["compact"]
  and intervals = value @@ flag @@ info ["i"; "intervals"]
      ~docs:s_est
      ~doc:"Compute the 0.99 confidence intervals for the estimates."
  and show_r2 = value @@ opt autobool `Auto @@ info ["r2"]
      ~docs:s_est
      ~doc:"Show $(i,R^2) values. One of `true', `false', or `auto'."
  and samples = value @@ flag @@ info ["s"; "samples"]
      ~docs:s_est ~doc:"Show sample counts."
  and no_meta = value @@ flag @@ info ["m"; "no-meta"]
      ~docs:Manpage.s_options ~doc:"Hide benchmark metadata." in
  Term.(To_Notty.props $$ compact $ intervals $ show_r2 $ samples $ no_meta)

let term =
  let open Arg in
  let filter = value @@ opt_all string [] @@ info ["f"; "filter"]
      ~docs:Manpage.s_options ~docv:"QUERY"
      ~doc:"Show only the benchmarks matching $(docv). Can be repeated. \
            See FILTERING."
  and json = value @@ flag @@ info ["json"]
      ~docs:Manpage.s_options ~doc:"Output the report as JSON."
  and counters = value @@ opt (list string) def_ctrs @@ info ["c"; "counters"]
      ~docs:Manpage.s_options ~docv:"COUNTERS"
      ~doc:"The list of counters to display."
  and est = value @@ opt estimator tse @@ info ["e"; "estimator"]
      ~docs:"ESTIMATORS" ~docv:"ESTIMATOR"
      ~doc:"Estimator to use. One of `OLS' or `TSE'." in
  Term.((fun () props q json counters estimator ->
          run_main ~q ~props ~counters ~estimator ~json stdin)
    $$ (Logs.set_level ~all:true $$ Logs_cli.level ())
    $ t_render_props $ filter $ json $ counters $ est )

let () =
  Fmt_tty.setup_std_outputs ();
  Logs.set_reporter (Logs_fmt.reporter ());
  Term.eval (term, info) |> Term.exit
