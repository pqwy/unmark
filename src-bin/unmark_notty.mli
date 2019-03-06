open Notty

type i

val lift : image -> i
val (!) : i -> image
val attr : attr -> i -> i

val shape : i -> int * int
val width : i -> int
val height : i -> int

val empty : i
val void : int -> int -> i

val (<|>) : i -> i -> i
val (<->) : i -> i -> i
val (</>) : i -> i -> i

val string : ?attr:attr -> string -> i
val strf : ?attr:attr -> ?w:int -> ('a, Format.formatter, unit, i) format4 -> 'a

val hcat : i list -> i
val vcat : i list -> i

val pad : ?l:int -> ?r:int -> ?t:int -> ?b:int -> i -> i
val hpad : int -> int -> i -> i
val vpad : int -> int -> i -> i

val crop : ?l:int -> ?r:int -> ?t:int -> ?b:int -> i -> i
val hcrop : int -> int -> i -> i
val vcrop : int -> int -> i -> i

val hsnap : ?align:[`Left | `Middle | `Right] -> int -> i -> i
val vsnap : ?align:[`Bottom | `Middle | `Top] -> int -> i -> i

val pp : Format.formatter -> i -> unit
