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
		       
-type complex() :: {float(),float()}.
-type scalar() :: integer()|float()|complex().

-record(matrix,
	{
	  n :: unsigned(),          %% rows
	  m :: unsigned(),          %% columns
	  type :: 0..7,             %% encoded element type
	  ptr = 0 :: unsigned(),    %% 0 is binary, not 0 is resource binary
	  offset = 0 :: unsigned(), %% offset to first element
	  stride :: unsigned(),     %% number of elements per (padded) row
	  rowmajor :: boolean(),    %% stored row-by-row else col-by-col
	  data :: binary()          %% native encode raw matrix data
	}).

-type matrix() :: #matrix{}.

-define(is_complex_matrix(X),
	((X#matrix.type >= ?complex64) andalso (X#matrix.type =< ?complex128))).

-define(is_float_matrix(X),
	((X#matrix.type >= ?float32) andalso (X#matrix.type =< ?float64))).

-define(is_int_matrix(X),
	((X#matrix.type >= ?int8) andalso (X#matrix.type =< ?int64))).

-define(is_complex(X), (is_number(element(1,(X))) andalso is_number(element(2,(X))))).

-define(is_scalar(X), (is_number((X)) orelse ?is_complex(X))).

-endif.



