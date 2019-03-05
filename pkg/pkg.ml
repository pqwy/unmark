#!/usr/bin/env ocaml
#use "topfind"
#require "topkg"
#require "ocb-stubblr.topkg"
open Topkg

let cli    = Conf.with_pkg ~default:true "cli"
let viewer = Conf.with_pkg ~default:true "viewer"
let python = Conf.with_pkg ~default:true "python"
let papi   = Conf.with_pkg ~default:true "papi"

let build = Pkg.build ~cmd:Ocb_stubblr_topkg.cmd ()

let () =
  Pkg.describe "unmark" ~build @@ fun c ->
    let (cli, viewer, python, papi) =
      Conf.(value c cli, value c viewer, value c python, value c papi) in
    let moves = [
      Pkg.clib  "src/libunmark_stubs.clib";
      Pkg.mllib "src/unmark.mllib" ~api:["Unmark"];
      Pkg.mllib "src-cli/unmark_cli.mllib" ~cond:cli;
      Pkg.mllib "src-papi/unmark_papi.mllib" ~cond:papi;
      Pkg.bin   "src-bin/unmark_bin" ~cond:viewer ~dst:"unmark";

      Pkg.test  "test/sanity";

      Pkg.test ~run:false "test/test";
      Pkg.test ~run:false "test/basics";
      Pkg.test ~run:false "test/est";
      Pkg.test ~run:false "test/naming";
      Pkg.test ~run:false "test/deps";
      Pkg.test ~run:false "test/alloc";
      Pkg.test ~run:false "test/arrays";
    ]
    and pysrc, pydst = "src-python/unmark/", "python/unmark/" in
    let skip f = not Fpath.(is_dir_path f || has_ext "py" f)
    and cons f ms = Pkg.share f ~cond:python ~dst:pydst ~built:false :: ms in
    OS.File.fold ~skip cons [] [pysrc] >>| fun pys -> moves @ pys
