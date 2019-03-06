open Notty.Infix
open Notty

type i = attr -> image

let lift i _ = i
let (!) i = i A.empty
let attr a1 i a2 = i A.(a2 ++ a1)

let shape i = let i = !i in I.(width i, height i)
let width i = I.width !i
let height i = I.height !i

let empty = lift I.empty
let void w h _ = I.void w h

let (<|>) i1 i2 a = i1 a <|> i2 a
let (<->) i1 i2 a = i1 a <-> i2 a
let (</>) i1 i2 a = i1 a </> i2 a

let string ?attr:(a1 = A.empty) s a2 = I.string A.(a2 ++ a1) s
(* let strf ?attr ?w fmt = I.kstrf ?attr ?w (fun i -> assert false) fmt *)
(* let _ = Format. *)

let hcat fs a = List.map (fun f -> f a) fs |> I.hcat
let vcat fs a = List.map (fun f -> f a) fs |> I.vcat

let pad ?l ?r ?t ?b f a = I.pad ?l ?r ?t ?b (f a)
let hpad l r f a = I.hpad l r (f a)
let vpad t b f a = I.vpad t b (f a)

let crop ?l ?r ?t ?b f a = I.crop ?l ?r ?t ?b (f a)
let hcrop l r f a = I.hcrop l r (f a)
let vcrop t b f a = I.vcrop t b (f a)

let hsnap ?align n f a = I.hsnap ?align n (f a)
let vsnap ?align n f a = I.vsnap ?align n (f a)

let pp ppf i = Render.pp Cap.ansi ppf !i
