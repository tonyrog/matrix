%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%
%%% @end
%%% Created :  1 Nov 2017 by Tony Rogvall <tony@rogvall.se>

-ifndef(__MATRIX_HRL__).
-define(__MATRIX_HRL__,true).

-define(int8_t,       0).
-define(int16_t,      1).
-define(int32_t,      2).
-define(int64_t,      3).
-define(int128_t,     4).
-define(float32_t,    5).
-define(float64_t,    6).
-define(float128_t,   7).
-define(complex64_t,  8).
-define(complex128_t, 9).

-define(int8(X),     (X):8/native-unsigned-integer).
-define(int16(X),    (X):16/native-unsigned-integer).
-define(int32(X),    (X):32/native-unsigned-integer).
-define(int64(X),    (X):64/native-unsigned-integer).
-define(int128(X),   (X):128/native-unsigned-integer).
-define(float32(X),  (X):32/native-float).
-define(float64(X),  (X):64/native-float).
-define(float128(X), (0):128/native-unsigned-integer).
-define(complex64(R,I), ?float32(R), ?float32(I)).
-define(complex128(R,I), ?float64(R), ?float64(I)).

-type unsigned() :: non_neg_integer().
-type matrix_type() :: int8|int16|int32|int64|int128|
		       float32|float64|float128|
		       complex64|complex128.
-type encoded_type() :: 0..9.
		       
-type complex() :: {float(),float()}.
-type scalar() :: integer()|float()|complex().
-type resource() :: reference() | {unsigned(),binary()} | binary().

-record(matrix,
	{
	 type :: encoded_type(),
	 resource :: resource(),   %% native encode raw matrix data
	 n :: unsigned(),          %% rows
	 m :: unsigned(),          %% columns
	 nstep :: integer(),       %% #elements in n direction
	 mstep :: integer(),       %% #elements in m direction
	 offset = 0 :: unsigned(), %% offset to first element
	 rowmajor :: boolean()     %% stored row-by-row else col-by-col
	}).

-record(matrix_t,
	{
	 type     :: encoded_type(),
	 resource :: resource()
	}).

-type matrix() :: #matrix_t{} | #matrix{}.

-define(is_matrix(X), (is_record((X),matrix) orelse is_record((X),matrix_t))).

-define(is_complex(X), (is_number(erlang:element(1,(X))) andalso is_number(erlang:element(2,(X))))).
-define(is_scalar(X), (is_number((X)) orelse ?is_complex(X))).

-define(matrix_cdata_vec(M, Type, Data),
	#matrix { type=(Type),
		  resource=(Data),
		  n=1, m=(M), nstep=0, mstep=1, rowmajor=true }).

-define(matrix_vec_const3f(X,Y,Z),
	?matrix_cdata_vec(3, ?float32_t, 
			  <<?float32(X),?float32(Y),?float32(Z)>>)).
-define(matrix_vec_const4f(X,Y,Z,W),
	?matrix_cdata_vec(4, ?float32_t, 
			  <<?float32(X),?float32(Y),
			    ?float32(Z),?float32(W)>>)).

-endif.



