%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%
%%% @end
%%% Created :  1 Nov 2017 by Tony Rogvall <tony@rogvall.se>

-ifndef(__MATRIX_HRL__).
-define(__MATRIX_HRL__,true).

-define(int8,       0).
-define(int16,      1).
-define(int32,      2).
-define(int64,      3).
-define(float32,    4).
-define(float64,    5).
-define(complex64,  6).
-define(complex128, 7).

-type unsigned() :: non_neg_integer().
-type matrix_type() :: int8|int16|int32|int64|
		       float32|float64|
		       complex64|complex128.
-type encoded_type() :: 0..7.
		       
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

-endif.



