(* Copyright (c) 2018 David Kaloper Meršinjak. All rights reserved.
   See LICENSE.md *)

open Unmark
open Rresult

let (%) f g x = f (g x)

module Option = struct
  let (>>|) a f = match a with Some x -> Some (f x) | _ -> None
  let (%%) f a = a >>| f
  let get ~none = function Some x -> x | _ -> none
end

let rec sequence = function
  []    -> Ok []
| r::rs -> r >>= fun x -> sequence rs >>| fun xs -> x::xs

let rec cat_some =
  function [] -> [] | Some x::xs -> x::cat_some xs | _::xs -> cat_some xs

let pp_err ppf (`Msg m) = Fmt.string ppf m

let log_error = function
  Ok x    -> Ok x
| Error e -> Logs.err (fun k -> k "%a" pp_err e); Error ()

let or_die = function Ok x -> x | _ -> exit 1

let call_msg f a =
  Bos.OS.U.call f a |> R.error_to_msg ~pp_error:Bos.OS.U.pp_error

let openfile ~flags ?(mode = 0o660) path =
  call_msg (fun path -> Unix.openfile (Bos.Pat.to_string path) flags mode) path
  |> R.reword_error (fun (`Msg m) -> R.msgf "open `%a': %s" Bos.Pat.pp path m)

let open_out ?(flags = []) path =
  openfile ~flags:([Unix.O_WRONLY] @ flags) path >>| Unix.out_channel_of_descr

let style ss = List.fold_right Fmt.styled ss
let bold = style [`Bold]

let pp_out ppf = function
  `STDOUT    -> Fmt.string ppf "STDOUT"
| `STDERR    -> Fmt.string ppf "STDERR"
| `File path -> Bos.Pat.pp ppf path

let write_buffer buf = function
  `STDOUT | `STDERR as out ->
    let oc = match out with `STDOUT -> stdout | _ -> stderr in
    Buffer.output_buffer oc buf; flush oc; Ok ()
| `File path ->
    open_out ~flags:Unix.[O_CREAT; O_APPEND] path
    >>| (fun oc -> Buffer.output_buffer oc buf; close_out oc)

let pipe_buffer buf cmd =
  Bos.OS.Cmd.(Buffer.contents buf |> in_string |> run_in cmd)

let run ?(warmup=true) ?out ?pipe b_run =
  let open Option in
  Logs.info (fun m -> m "Output to %a." pp_out %% out |> ignore);
  Logs.info (fun m -> m "Piping to %a." Bos.Cmd.pp %% pipe |> ignore);
  if warmup then Measurement.warmup ();
  let run = b_run () in
  let b = Buffer.create 4096 in
  Benchmarks.Run.add_json b run; Buffer.add_char b '\n';
  let r1 = (log_error % write_buffer b %% out |> get ~none:(Ok ()))
  and r2 = (log_error % pipe_buffer b %% pipe |> get ~none:(Ok ())) in
  sequence [r1; r2] |> or_die |> ignore

let pp_attr ppf a =
  if not (Attr.is_empty a) then Fmt.pf ppf "(%a)" Attr.pp a

let pp_benches ppf benches =
  let pp_fs ppf xs = xs |> cat_some |> Fmt.list (fun _ f -> f ppf ()) ppf in
  let bench p a _ =
    Some (fun _ () -> Fmt.pf ppf "%s  %a" (snd p) pp_attr a)
  and group p t = match Benchmarks.eval t with
    [] -> None
  | fs -> Some (fun _ () -> Fmt.pf ppf "%s/@;  @[<v>%a@]" (snd p) pp_fs fs)
  and suite = Fmt.pf ppf "@[<v>%a@]" pp_fs in
  Benchmarks.fold ~bench ~group ~suite benches

let pp_counters ppf p =
  let open Measurement in
  let pp_ctr ppf ctr =
    Fmt.(pf ppf "%a — %s" (bold string)) (Probe.name ctr) (Probe.desc ctr) in
  Fmt.(pf ppf "@[<v>%a@]" (list pp_ctr))
         (core_counters @ Probe.counters p |> List.tl)

let run_info ?(ppf = Format.std_formatter) ~probe ~suite benches =
  Fmt.pf ppf "Suite: %s\n\n%!" suite;
  Fmt.pf ppf "Benchmarks:@,  %a\n\n%!" pp_benches benches;
  Fmt.pf ppf "Counters:@,  %a\n\n%!" pp_counters probe

open Cmdliner

let ($$) f a = Term.(const f $ a)

let s_bench_opt = "BENCHMARKING OPTIONS"
let s_out_opt = "OUTPUT OPTIONS"

let info name =
  Term.info ~version:"%%VERSION%%"
  ~exits:Term.(exit_info 1 ~doc:"on output error." :: default_exits)
  ~doc:(Fmt.strf "$(b,%s) benchmarks" name)
  Filename.(Sys.executable_name |> basename |> remove_extension)
  ~man:[

  `S Manpage.s_description

; `P (Fmt.strf "$(mname) runs the $(b,%s) benchmark suite." name)
; `P "$(mname) contains a set of compiled-in benchmarks. It can run them \
      and perform measurements.  The raw results of running the benchmarks \
      - a sequence of individual measurements - are formatted as JSON text. \
      This text can be written to a file, or piped to a separate \
      program for processing."
; `P "$(mname) defaults to piping the results to the executable $(b,unmark), \
      default result viewer installed with the $(b,Unmark) library."
; `P "Pass $(b,--info) to list the benchmarks."
; `P "Pass $(b,-v) to see the progress."

; `S s_out_opt

; `S s_bench_opt

; `S Manpage.s_options

; `S "FILTERING"

; `P "Filtering restricts the run to a subset of benchmarks. Specifying \
      multiple queries selects the benchmarks that match any of them."
; `P "For more information, see the filtering documentation in \
      $(i,Unmark.Benchmarks) [1]."
; `Pre "[1] - https://pqwy.github.io/unmark/doc/Unmark.Benchmarks.html#filtering"

; `S Manpage.s_examples

; `P "Run the benchmarks:"
; `Pre "    $(mname)"
; `P "Show which benchmarks can be run, and which counters are collected:"
; `Pre "    $(mname) --info"
; `P "Do a very quick (and unstable) run of everything:"
; `Pre "    $(mname) --no-warmup --min-time 0.01"
; `P "Run the group $(i,g) for an extended period, showing progress:"
; `Pre "    $(mname) --filter g --min-time 5 -v"
; `P "Run everything, and silently append raw measurements to a file:"
; `Pre "    $(mname) --note \"now with stuff\" --pipe --output results.json"
; `P "Run $(i,g), and tell $(b,unmark) which counters to display:"
; `Pre "    $(mname) -- --counters time,cycles"
; `P "Pipe to $(b,jq) to print the number of samples per benchmark:"
; `Pre "    $(mname) --pipe jq -- -c '.benchmarks[]|[.name,(.samples|length)]'"
]

let def_pipe = Bos.Cmd.v "unmark"

let c_cmd = Arg.conv Bos.(OS.Cmd.must_exist % Cmd.v, Cmd.pp)
let c_out = Arg.conv ((function
    "STDOUT" -> Ok `STDOUT
  | "STDERR" -> Ok `STDERR
  | path     -> Ok (`File (Bos.Pat.v path))
  ), pp_out)

let t_outputs =
  let open Arg in
  let pipe = value @@ opt (some c_cmd) None ~vopt:(Some def_pipe) @@
      info ["p"; "pipe"] ~docv:"PROGRAM" ~docs:s_out_opt
      ~doc:"Pipe the results to $(docv). \
            Enabled by default, unless $(b,--out) is given."
  and args = value @@ pos_all string [] @@ info [] ~docv:"ARG"
      ~doc:"Additional arguments are passed to the $(b,--pipe) program."
  and out = value @@ opt (some c_out) None ~vopt:(Some `STDOUT) @@
      info ["o"; "output"] ~docv:"FILE" ~docs:s_out_opt
      ~doc:"Append the results to $(docv). $(docv) can be a path, \
            `STDOUT', or `STDERR'. When this option is set, $(b,--pipe) \
            defaults to disabled." in
  Term.((fun cmd args out ->
    let cmd = match (cmd, out) with None, None -> Some def_pipe | _ -> cmd in
    (Option.(cmd >>| fun c -> Bos.Cmd.(c %% of_list args)), out))
  $$ pipe $ args $ out)

let t_log_level = Logs.set_level ~all:true $$ Logs_cli.level ()

let t_b_run (def_t, def_s, def_flt) =
  let open Arg in
  let min_t = value @@ opt float def_t @@ info ["t"; "min-time"]
      ~docv:"SECONDS" ~docs:s_bench_opt
      ~doc:"Minimal running time per benchmark."
  and min_s = value @@ opt int def_s @@ info ["s"; "min-samples"]
      ~docv:"SAMPLES" ~docs:s_bench_opt
      ~doc:"Minimal number of samples per benchmark."
  and q = value @@ opt_all string def_flt ~vopt:"" @@ info ["f"; "filter"]
      ~docv:"QUERY" ~docs:s_bench_opt
      ~doc:"Run only the benchmarks matching $(docv). Can be repeated. \
            See FILTERING."
  and note = value @@ opt (some string) None @@ info ["n"; "note"]
      ~docv:"TEXT" ~docs:s_out_opt
      ~doc:"Annotate the results with $(docv)." in
  Term.((fun min_t min_s q note ~probe ~suite bs ->
    Benchmarks.run ~probe ~min_t ~min_s ~q ?note ~suite bs)
  $$ min_t $ min_s $ q $ note)

let t bm_def ~probe ~suite ~arg f =
  let open Arg in
  let w = value @@ flag @@ info ["w"; "no-warmup"] ~docs:s_bench_opt
      ~doc:"Don't try to force the CPU into full-power mode."
  and nfo = value @@ flag @@ info ["i"; "info"]
      ~doc:"List the available benchmarks and exit." in
  Term.((fun p () b_run (pipe, out) w -> function
    true  -> run_info ~probe ~suite (f p)
  | false -> run ~warmup:(not w) ?pipe ?out
              (fun () -> b_run ~probe ~suite (f p)))
  $$ arg $ t_log_level $ (t_b_run bm_def) $ t_outputs $ w $ nfo)

let probe = Measurement.Probe.gc_counters
let min_t, min_s = (1., 10)

let main_ext ?(probe = probe) ?(min_t = min_t) ?(min_s = min_s)
             ?(def_filter = []) ~arg suite f =
  let term = t (min_t, min_s, def_filter) ~probe ~suite ~arg f in
  Term.eval (term, info suite) |> Term.exit

let main ?probe ?min_t ?min_s ?def_filter suite benches =
  main_ext ?probe ?min_t ?min_s ?def_filter suite
            ~arg:(Term.const ()) (fun () -> benches)

let () =
  Fmt_tty.setup_std_outputs ();
  Logs.set_reporter (Logs_fmt.reporter ())
