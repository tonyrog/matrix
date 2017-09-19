%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    binary matrix version
%%%
%%% @end

-module(matrix).

%% -compile(native).
%% -on_load(init/0).
-export([new/4]).
-export([new_/4]).
-export([from_list/1, from_list/2, from_list/3, from_list/4]).
-export([to_list/1]).
-export([normal/1, uniform/1, zero/1, one/1, identity/1]).
-export([normal/2, uniform/2, zero/2, one/2, identity/2]).
-export([normal/3, uniform/3, zero/3, one/3, identity/3]).
-export([constant/4]).
-export([add/2,subtract/2,negate/1]).
-export([multiply/2, scale/2, square/1, pow/2]).
-export([size/1, type/1]).
-export([element/3]).
-export([sigmoid/1]).
-export([sigmoid_prime/1]).
-export([rectifier/1]).
-export([softplus/1]).
-export([transpose/1]).
-export([print/1, print/2, format/1, format/2]).
-export([row/2, column/2, submatrix/5]).
-export([convolve/4, convolve/6]).
-export([rconvolve/4, rconvolve/6]).
-export([max/3, max/5, l2/3, l2/5]).
-export([is_integer_matrix/1]).
-export([filter/3, filter/5]).
-export([fold_elems/8]).
-export([rfold_elems/8]).
-compile(export_all).

-compile({no_auto_import,[size/1]}).

-define(int8,    0).
-define(int16,   1).
-define(int32,   2).
-define(int64,   3).
-define(float32, 4).
-define(float64, 5).

-type unsigned() :: non_neg_integer().
-type matrix_type() :: float32|float64|int64|int32|int16|int8.

-record(matrix,
	{
	  n :: unsigned(),          %% rows
	  m :: unsigned(),          %% columns
	  type :: 0..5,             %% encoded element type
	  ptr = 0 :: unsigned(),    %% 0 is binary, not 0 is resource binary
	  offset = 0 :: unsigned(), %% offset to first element
	  stride :: unsigned(),     %% number of elements per (padded) row
	  data :: binary()          %% native encode raw matrix data
	}).

-type matrix() :: #matrix{}.

init() ->
    Nif = filename:join(code:priv_dir(matrix), "matrix_drv"),
    erlang:load_nif(Nif, 0).

-spec new(N::unsigned(), M::unsigned(), T::matrix_type(), Es::iolist()) ->
		 matrix().
new(N,M,T,Es) when is_integer(N), N>0,
		   is_integer(M), N>0,
		   is_atom(T),
		   is_list(Es) ->
    Type = encode_type(T),
    new_(N,M,Type,Es).

%% FIXME: new_ is normally a nif, but we may use the library without nifs,
%% to allow for interop betweeen we should calculate the stride in 
%% number of elements per row and align row data to som vector size
new_(N,M,Type,Es) ->
    Stride = M,
    #matrix { n=N, m=M, type=Type, stride=Stride, data=iolist_to_binary(Es)}.

%% create matrix from list of lists
from_list(Rows) ->
    N = length(Rows),
    M = lists:max([length(R) || R <- Rows]),
    from_list(Rows, N, M).

from_list(Rows,Type) ->
    N = length(Rows),
    M = lists:max([length(R) || R <- Rows]),
    from_list(Rows,N,M,Type).

from_list(Rows, N, M) ->
    from_list_(Rows, N, M, type_lists(Rows)).

from_list(Rows, N, M, Type) ->
    from_list_(Rows, N, M, encode_type(Type)).

from_list_(Rows, N, M, T) ->
    Es = from_list_(Rows, 0, N, M, T),
    new_(N, M, T, Es).

from_list_([], N, N, _M, _T) ->
    [];
from_list_([], I, N, M, T) when I < N ->
    Pad = lists:duplicate(M, number_to_bin(T, 0)),
    [Pad | from_list_([], I+1, N, M, T)];
from_list_([R|Rs], I, N, M, T) ->
    L = length(R),
    Pad = lists:duplicate(M - L, number_to_bin(T, 0)),
    [ [ number_to_bin(T,E) || E <- R ], Pad |
      from_list_(Rs, I+1, N, M, T)].

%% scan elements and find a matrix type
type_lists(Rows) ->
    type_lists(Rows, -1).

type_lists(_, T) when T >= ?float32 ->
    T;
type_lists([R|Rs], T) ->
    T1 = type_list(R, T),
    type_lists(Rs, T1);
type_lists([], T) ->
    T.

type_list(_, T) when T >= ?float32 ->
    T;
type_list([E|Es], T) ->
    if is_integer(E) ->
	    if E >= -16#80, E < 16#80 ->
		    type_list(Es, max(?int8,T));
	       E >= -16#8000, E < 16#8000 ->
		    type_list(Es, max(?int16,T));
	       E >= -16#80000000, E < 16#80000000 ->
		    type_list(Es, max(?int32,T));
	       true ->
		    type_list(Es, max(?int64,T))
	    end;
       is_float(E) ->
	    type_list(Es, ?float64)
    end;
type_list([], T) ->
    T.

%%
%% Produce a list of lists representation of a matrix
%%
-spec to_list(X::matrix()) -> [[number()]].

to_list(#matrix{n=N,m=M,offset=Offset,stride=Stride,type=T,data=Bin0}) ->
    Offset1 = Offset*element_bytes(T),
    <<_:Offset1/binary,Bin/binary>> = Bin0,
    As = elements_to_list(T,Bin),
    to_list_(N, M, Stride, As).

to_list_(0, _M, _Stride, _As) ->
    [];
to_list_(I, M, Stride, As) ->
    {Row0,As1} = split_row(Stride, As),
    {Row,_}  = lists:split(M, Row0),
    [Row | to_list_(I-1,M,Stride,As1)].

-spec normal({N::unsigned(), M::unsigned()}) -> matrix().
normal({N,M}) ->
    normal(N,M,float64).

-spec normal({N::unsigned(), M::unsigned()},T::matrix_type()) -> matrix().
normal({N,M},T) ->
    normal(N,M,T).

-spec normal(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
normal(N,M,T) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Es = [normal_bin(0.0,1.0,Type) || _ <- lists:seq(1,N*M)],
    new_(N,M,Type,Es).

-spec uniform({N::unsigned(), M::unsigned()}) -> matrix().
uniform({N,M}) ->
    uniform(N,M,float64).

-spec uniform({N::unsigned(), M::unsigned()},T::matrix_type()) -> matrix().
uniform({N,M},T) ->
    uniform(N,M,T).

-spec uniform(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
uniform(N,M,T) when is_integer(N), N >= 1,
		    is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Es = [uniform_bin(Type) || _ <- lists:seq(1,N*M)],
    new_(N,M,Type,Es).

-spec zero({N::unsigned(), M::unsigned()}) -> matrix().
zero({N,M}) -> zero(N,M,float64).

-spec zero({N::unsigned(), M::unsigned()}, T::matrix_type()) -> matrix().
zero({N,M},T) -> zero(N,M,T).

-spec zero(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
zero(N,M,T) when is_integer(N), N >= 1,
		 is_integer(M), M >= 1 ->
    constant(N,M,T,0).

-spec one({N::unsigned(), M::unsigned()}) -> matrix().
one({N,M}) -> one(N,M,float64).

-spec one({N::unsigned(), M::unsigned()}, T::matrix_type()) -> matrix().
one({N,M},T) -> one(N,M,T).

-spec one(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
one(N,M,T) when is_integer(N), N >= 1,
		 is_integer(M), M >= 1 ->
    constant(N,M,T,1).


-spec constant(N::unsigned(), M::unsigned(), T::matrix_type(), C::number()) ->
		      matrix().
constant(N,M,T,C) when is_integer(N), N >= 1,
		       is_integer(M), M >= 1,
		       is_number(C) ->
    Type = encode_type(T),
    Z = number_to_bin(Type, C),
    BinList = lists:duplicate(N*M, Z),
    new_(N,M,Type,BinList).


-spec identity({N::unsigned(), M::unsigned()}) -> matrix().
identity({N,M}) ->
    identity(N,M,float64).

-spec identity({N::unsigned(),M::unsigned()},T::matrix_type()) -> matrix().
identity({N,M},T) ->
    identity(N,M,T).

-spec identity(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
identity(N,M,T) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Bv = {number_to_bin(Type, 0),number_to_bin(Type, 1)},
    Data = [[element(X+1,Bv) ||
		<<X:1>> <= <<(1 bsl ((M-1)-I)):M>>] ||
	       I <- lists:seq(0, N-1)],
    new_(N,M,Type,Data).

-spec size(M::matrix()) -> {unsigned(), unsigned()}.
size(#matrix{n=N,m=M}) ->
    {N,M}.

type(#matrix{type=T}) ->
    case T of
	?int8 -> int8;
	?int16 -> int16;
	?int32 -> int32;
	?int64 -> int64;
	?float32 -> float32;
	?float64 -> float64
    end.

-spec element(I::unsigned(),J::unsigned(),X::matrix()) -> number().
element(I,J,#matrix{n=N,m=M,offset=Offs,stride=Stride,type=T,data=Bin}) when
      is_integer(I), I > 0, I =< N, 
      is_integer(J), J > 0, J =< M ->
    P = Offs + (I-1)*Stride+J-1,
    element_(P, T, Bin).

%% P is element position not byte position
element_(P, T, Bin) ->
    case T of
	?float64 -> 
	    <<_:P/binary-unit:64,X:64/native-float,_/binary>> = Bin, X;
	?float32 -> 
	    <<_:P/binary-unit:32,X:32/native-float,_/binary>> = Bin, X;
	?int64 ->
	    <<_:P/binary-unit:64,X:64/native-signed-integer,_/binary>> = Bin, X;
	?int32 ->
	    <<_:P/binary-unit:32,X:32/native-signed-integer,_/binary>> = Bin, X;
	?int16 ->
	    <<_:P/binary-unit:16,X:16/native-signed-integer,_/binary>> = Bin, X;
	?int8 ->
	    <<_:P/binary-unit:8,X:8/native-signed-integer,_/binary>> = Bin, X
    end.

map(F, #matrix{type=T,data=Bin}) ->
    case T of
	?float64 ->
	    [F(X) || <<X:64/native-float>> <= Bin ];
	?float32 ->
	    [F(X) || <<X:32/native-float>> <= Bin ];
	?int64 ->
	    [F(X) || <<X:64/native-signed-integer>> <= Bin ];
	?int32 ->
	    [F(X) || <<X:32/native-signed-integer>> <= Bin ];
	?int16 ->
	    [F(X) || <<X:16/native-signed-integer>> <= Bin ];
	?int8 ->
	    [F(X) || <<X:8/native-signed-integer>> <= Bin ]
    end.

elements_to_list(T, Bin) ->
    case T of
	?float64 ->
	    [X || <<X:64/native-float>> <= Bin ];
	?float32 ->
	    [X || <<X:32/native-float>> <= Bin ];
	?int64 ->
	    [X || <<X:64/native-signed-integer>> <= Bin ];
	?int32 ->
	    [X || <<X:32/native-signed-integer>> <= Bin ];
	?int16 ->
	    [X || <<X:16/native-signed-integer>> <= Bin ];
	?int8 ->
	    [X || <<X:8/native-signed-integer>> <= Bin ]
    end.

%%
%%
%% Fold F with accumulator A over the matrix 
-spec foldr(fun((number(),term()) -> term()), term(), matrix()) -> term().

foldr(F, A, #matrix{n=N,m=M,offset=Offs,stride=Stride,type=T,data=Bin}) ->
    P = Offs + (N-1)*Stride + M - 1,
    foldr_(F,A,N,M,M,T,Stride,P,Bin).

foldr_(_F,A,1,0,_M,_T,_S,_P,_Bin) ->
    A;
foldr_(F,A,N,0,M,T,S,P,Bin) ->
    P1 = P + M - S,
    foldr_(F,A,N-1,M,M,T,S,P1,Bin);
foldr_(F,A,N,J,M,T,S,P,Bin) ->
    E = element_(P,T,Bin),
    foldr_(F,F(E,A),N,J-1,M,T,S,P-1,Bin).
%%
%%
%%
-spec foldl(fun((number(),term()) -> term()), term(), matrix()) -> term().

foldl(F, A, #matrix{n=N,m=M,offset=Offs,stride=Stride,type=T,data=Bin}) ->
    P = Offs,
    foldl_(F,A,N,M,M,T,Stride,P,Bin).

foldl_(_F,A,1,0,_M,_T,_S,_P,_Bin) ->
    A;
foldl_(F,A,N,0,M,T,S,P,Bin) ->
    P1 = P - M + S,
    foldr_(F,A,N-1,M,M,T,S,P1,Bin);
foldl_(F,A,N,J,M,T,S,P,Bin) ->
    E = element_(P,T,Bin),
    foldl_(F,F(E,A),N,J-1,M,T,S,P+1,Bin).

%%
%%
%%
-spec zipfoldr(fun((number(),number(), term()) -> term()), term(),
	       matrix(), matrix()) -> term().
		      
zipfoldr(F, A,
	 #matrix{n=N,m=M,offset=Offs1,stride=Stride1,type=T1,data=Bin1}, 
	 #matrix{n=N,m=M,offset=Offs2,stride=Stride2,type=T2,data=Bin2}) ->
    P1 = Offs1 + (N-1)*Stride1 + M - 1,
    P2 = Offs2 + (N-1)*Stride2 + M - 1,
    zipfoldr_(F,A,N,M,M,
	      T1,Stride1,P1,Bin1,
	      T2,Stride2,P2,Bin2).

zipfoldr_(_F, A, 1, 0, _M,
	  _T1,_S1,_P1,_Bin1,
	  _T2,_S2,_P2,_Bin2) ->
    A;
zipfoldr_(F, A, N, 0, M,
	  T1,S1,P1,Bin1,
	  T2,S2,P2,Bin2) ->
    P11 = P1 + M - S1,
    P21 = P2 + M - S2,
    zipfoldr_(F,A,N-1,M,M,
	      T1,S1,P11,Bin1,
	      T2,S2,P21,Bin2);
zipfoldr_(F,A,N,J,M,
	  T1,S1,P1,Bin1,
	  T2,S2,P2,Bin2) ->
    E1 = element_(P1,T1,Bin1),
    E2 = element_(P2,T2,Bin2),
    zipfoldr_(F, F(E1,E2,A),N,J-1,M,
	      T1,S1,P1-1,Bin1,
	      T2,S2,P2-1,Bin2).

%%
%% Add two matrices
%%
-spec add(A::matrix(), B::matrix()) -> matrix().

add(X=#matrix{n=N,m=M,type=T1},
    Y=#matrix{n=N,m=M,type=T2}) ->
    Type = type_combine(T1,T2),
    Es = zipfoldr(
	   fun(Xi,Yi,Acc) ->
		   [number_to_bin(Type,Xi+Yi)|Acc]
	   end, [], X, Y),
    new_(N,M,Type,Es).

%%
%% Subtract two matrices
%%
-spec subtract(A::matrix(), B::matrix()) -> matrix().

subtract(X=#matrix{n=N,m=M,type=T1}, 
	 Y=#matrix{n=N,m=M,type=T2}) ->
    Type = type_combine(T1,T2),
    Es = zipfoldr(
	   fun(Xi,Yi,Acc) ->
		   [number_to_bin(Type,Xi-Yi)|Acc]
	   end, [], X, Y),
    new_(N,M,Type,Es).

%%
%% Negate a matrix
%%
-spec negate(A::matrix()) -> matrix().
negate(X=#matrix{n=N,m=M,type=T}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,-Xi)|Acc]
	   end, [], X),
    new_(N,M,T,Es).

%%
%% Scale a matrix by a scalar number
%%
-spec scale(F::number(), A::matrix()) -> matrix().
scale(F, X=#matrix{n=N,m=M,type=T}) when is_number(F) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,F*Xi)|Acc]
	   end, [], X),
    new_(N,M,T,Es).

%%
%% Multiply elementwise and add everything
%%
-spec mulsum(A::matrix(), B::matrix()) -> number().

mulsum(X,Y) ->
    zipfoldr(fun(Xi,Yi,Sum) -> Xi*Yi+Sum end, 0, X, Y).

%%
%% Calculate X^2 
%%
-spec square(X::matrix()) -> matrix().
square(X) ->
    multiply(X, X).

%%
%% Calculate X^n
%%
-spec pow(X::matrix(), N::unsigned()) -> matrix().

pow(X,0) ->
    identity(size(X),type(X));
pow(X,N) when is_integer(N), N>0 ->
    pow_(X, N, identity(size(X),type(X))).

pow_(A,1,P) ->
    multiply(A,P);
pow_(A,B,P) ->
    B1 = B bsr 1,
    A1 = square(A),
    if B - B1 =:= B1 ->
	    pow_(A1, B1, P);
       true ->
	    pow_(A1, B1, multiply(A,P))
    end.

%%
%% Multiply two matrices
%%
-spec multiply(X::matrix(), Y::matrix()) -> matrix().

multiply(#matrix{n=Nx,m=Mx,offset=Offs1,stride=Stride1,type=T1,data=Bin1},
	 #matrix{n=Ny,m=My,offset=Offs2,stride=Stride2,type=T2,data=Bin2}) when
      Mx =:= Ny ->
    P1  = Offs1 + (Nx-1)*Stride1,  %% last row in X
    P2  = Offs2 + My-1,            %% last column in Y
    T = type_combine(T1,T2),
    Es = mult_(Nx,My,Mx,My,T,Bin1,P1,Stride1,T1,  Bin2,P2,Stride2,T2, []),
    new_(Nx,My,T,Es).

mult_(1,0,_Mx,_My,_T,_Bin1,_P1,_S1,_T1, _Bin2,_P2,_S2,_T2,Acc) ->
    Acc;
mult_(N,0,Mx,My,T,Bin1,P1,S1,T1,  Bin2,P2,S2,T2, Acc) ->
    mult_(N-1,My,Mx,My,T,Bin1,P1-S1,S1,T1, Bin2,P2+My,S2,T2, Acc);
mult_(N,J,Mx,My,T,Bin1,P1,S1,T1,  Bin2,P2,S2,T2,  Acc) ->
    C = dot_(Bin1,P1,1,T1,  Bin2,P2,S2,T2,  Mx,0),
    %% io:format("j=~w,sum=~w\n", [J,C]),
    mult_(N,J-1,Mx,My,T,Bin1,P1,S1,T1, Bin2,P2-1,S2,T2,
	  [number_to_bin(T,C)|Acc]).

dot_(_Bin1,_P1,_S1,_T1, _Bin2,_P2,_S2,_T2, 0,Sum) ->
    Sum;
dot_(Bin1,P1,S1,T1, Bin2,P2,S2,T2, K,Sum) ->
    %% io:format("p1=~w, p2=~w\n", [P1,P2]),
    E1 = element_(P1,T1,Bin1),
    E2 = element_(P2,T2,Bin2),
    Sum1 = E1*E2+Sum,
    dot_(Bin1,P1+S1,S1,T1, Bin2,P2+S2,S2,T2, K-1,Sum1).

%%
%% Transpose a matrix
%%
-spec transpose(A::matrix()) -> matrix().
transpose(#matrix{n=N,m=M,offset=Offs,stride=Stride,type=Type,data=Bin}) ->
    ES  = element_bytes(Type),
    RS  = ES*Stride,                         %% row bytes
    End = ES*(Offs + (N-1)*Stride + M - 1),  %% end element position
    Es = trans_(M,N,N,Bin,End,End,ES,RS,[]),
    new_(M,N,Type,Es).

trans_(1,0,_N,_Bin,_Pos,_Pos0,_ES,_RS,Acc) ->
    Acc;
trans_(J,0,N,Bin,_Pos,Pos0,ES,RS,Acc) ->
    Pos1 = Pos0-ES,
    %% io:format("pos1=~w\n", [Pos1]),
    trans_(J-1,N,N,Bin,Pos1,Pos1,ES,RS,Acc);
trans_(J,I,N,Bin,Pos,Pos0,ES,RS,Acc) ->
    %% io:format("pos=~w, pos0=~w\n", [Pos,Pos0]),
    <<_:Pos/binary,E:ES/binary,_/binary>> = Bin,
    trans_(J,I-1,N,Bin,Pos-RS,Pos0,ES,RS,[E|Acc]).

%%
%% select a row, return as a matrix with one row
%%
-spec row(I::unsigned(), A::matrix()) -> matrix().
row(I, X=#matrix{m=M}) ->
    submatrix(I, 1, 1, M, X).

%%
%% select a column, return as a matrix with one column
%%
-spec column(J::unsigned(), A::matrix()) -> matrix().
column(J, X=#matrix{n=N}) ->
    submatrix(1, J, N, 1, X).

%%
%% select a portion of a matrix
%%
-spec submatrix(I::unsigned(), J::unsigned(),
		N::unsigned(), M::unsigned(),
		X::matrix()) -> matrix().

submatrix(I, J, N, M, X=#matrix{offset=Offset,stride=Stride}) ->
    Offset1 = Offset + (I-1)*Stride + (J-1),
    X#matrix { n=N, m=M, offset=Offset1}.

%%
%% convolve a NxM matrix over the matrix A (soon: with padding Px, Py and
%% padding value PAD) using Sx and Sy as stride steps.
%%

-spec convolve(F::function(),
	       N::unsigned(),M::unsigned(),
	       Sx::unsigned(), Sy::unsigned(),A::matrix()) ->
		      matrix().

convolve(F,N,M,Sx,Sy,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(1,Nx-N+1,Sx), J <- lists:seq(1,Mx-M+1,Sy)].

-spec convolve(F::fun((integer(),integer()) -> term()),
	       N::unsigned(),M::unsigned(),A::matrix()) -> [term()].

convolve(F,N,M,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(1,Nx-N+1,1), J <- lists:seq(1,Mx-M+1,1)].

-spec rconvolve(F::function(),
	       N::unsigned(),M::unsigned(),
	       Sx::unsigned(), Sy::unsigned(),A::matrix()) ->
		      matrix().

rconvolve(F,N,M,Sx,Sy,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(Nx-N+1,1,-Sx), J <- lists:seq(Mx-M+1,1,-Sy)].

-spec rconvolve(F::fun((integer(),integer()) -> term()),
	       N::unsigned(),M::unsigned(),A::matrix()) -> [term()].

rconvolve(F,N,M,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(Nx-N+1,1,-1), J <- lists:seq(Mx-M+1,1,-1)].

%%
%%
%%
-spec max(N::unsigned(),M::unsigned(),matrix()) -> matrix().

max(N, M, X) ->
    max(N, M, 1, 1, X).

-spec max(N::unsigned(),M::unsigned(),Sx::unsigned(),Sy::unsigned(),
	  matrix()) -> matrix().

max(N, M, Sx, Sy, X=#matrix{n=Nx,m=Mx,type=T})
  when N =< Nx, M =< Mx ->
    Es = convolve(
	   fun(I, J) ->
		   V = fold_elems(
			 fun(E,undefined) -> E;
			    (E,Max) -> max(E,Max)
			 end, undefined, I, J, N, M, X),
		   %% io:format("(~w,~w) = ~w\n", [I,J,V]),
		   number_to_bin(T,V)
	   end, N, M, Sx, Sy, X),
    new_(((Nx-N) div Sx) +1, ((Mx-M) div Sy)+1, T, Es).

%%
%%
%%

-spec l2(N::unsigned(),M::unsigned(),matrix()) -> matrix().
l2(N, M, X) ->
    l2(N, M, 1, 1, X).

%%
%%
%%
-spec l2(N::unsigned(),M::unsigned(),Sx::unsigned(),Sy::unsigned(),
	 matrix()) -> matrix().

l2(N, M, Sx, Sy, X=#matrix{n=Nx,m=Mx,type=T})
  when N =< Nx, M =< Mx ->
    Es = convolve(
	   fun(I, J) ->
		   S = fold_elems(
			 fun(E,undefined) -> E*E;
			    (E,Sum) -> E*E+Sum
			 end, undefined, I, J, N, M, X),
		   number_to_bin(T,math:sqrt(S))
	   end, N, M, Sx, Sy, X),
    new_(((Nx-N) div Sx)+1, ((Mx-M) div Sy)+1, T, Es).

%%
%%
%%
-spec filter(W::matrix(), B::number(), X::matrix()) -> matrix().
filter(W, B, X) ->
    filter(W, B, 1, 1, X).

-spec filter(W::matrix(), B::number(), Sx::unsigned(), Sy::unsigned(),
	     X::matrix()) -> matrix().

filter(W=#matrix{n=Nw,m=Mw}, B, Sx, Sy, X=#matrix{n=Nx,m=Mx,type=T})
  when Nw =< Nx, Mw =< Mx ->
    Es = convolve(
	   fun(I, J) ->
		   A = submatrix(I,J,Nw,Mw,X),
		   number_to_bin(T,mulsum(W,A)+B)
	   end, Nw, Mw, Sx, Sy, X),
    new_(((Nx-Nw) div Sx)+1, ((Mx-Mw) div Sy)+1, T, Es).

%%
%% Scan embedded matrix data
%%
fold_elems(F,Acc,I,J,N,M,#matrix{n=Nx,m=Mx,offset=Offs,stride=Stride,
				 type=T,data=Bin})
  when N=<Nx, Mx=<Mx ->  %% FIXME: add more conditions
    P = Offs + (I-1)*Stride + J-1,
    fold_elems(F,Acc,Bin,P,M,T,Stride,N*M).


fold_elems(F,Acc,Bin,Start,RowLen,Type,RowStride,Total) ->
    fold_elems_(F,Acc,Bin,Start,0,RowLen,Type,RowStride,Total).

fold_elems_(F,Acc,Bin,Start,I,RowLen,T,RowStride,K) when I < RowLen ->
    E = element_(Start+I,T,Bin),
    fold_elems_(F,F(E,Acc),Bin,Start,I+1,RowLen,T,RowStride,K-1);
fold_elems_(_F,Acc,_Bin,_Start,_I,_RowLen,_T,_RowStride,K) when K =< 0 ->
    Acc;
fold_elems_(F,Acc,Bin,Start,_I,RowLen,T,RowStride,K) ->
    fold_elems_(F,Acc,Bin,Start+RowStride,0,RowLen,T,RowStride,K).

%% fold elements in reverse
rfold_elems(F,Acc,Bin,End,RowLen,Type,RowStride,Total) ->
    rfold_elems_(F,Acc,Bin,End,0,RowLen,Type,RowStride,Total).

rfold_elems_(F,Acc,Bin,End,I,RowLen,T,RowStride,K) when I < RowLen ->
    E = element_(End-I,T,Bin),
    rfold_elems_(F,F(E,Acc),Bin,End,I+1,RowLen,
		 T,RowStride,K-1);
rfold_elems_(_F,Acc,_Bin,_End,_I,_RowLen,_T,_RowStride,K) when K =< 0 ->
    Acc;
rfold_elems_(F,Acc,Bin,End,_I,RowLen,T,RowStride,K) ->
    rfold_elems_(F,Acc,Bin,End-RowStride,0,RowLen,T,RowStride,K).

-spec sigmoid(A::matrix()) -> matrix().
sigmoid(X=#matrix{n=N,m=M,type=T}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,sigmoid__(Xi))|Acc]
	   end, [], X),
    new_(N,M,T,Es).

sigmoid__(X) when is_float(X) ->
    1.0/(1.0 + math:exp(-X)).

-spec sigmoid_prime(A::matrix()) -> matrix().
sigmoid_prime(X=#matrix{n=N,m=M,type=T}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,sigmoid_prime__(Xi))|Acc]
	   end, [], X),
    new_(N,M,T,Es).

sigmoid_prime__(X) ->
    Z = sigmoid__(X),
    Z*(1-Z).

-spec rectifier(A::matrix()) -> matrix().
rectifier(X=#matrix{n=N,m=M,type=T}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,rectifier__(Xi))|Acc]
	   end, [], X),
    new_(N,M,T,Es).

rectifier__(X) when X < 0 -> 0;
rectifier__(X) -> X.

-spec softplus(A::matrix()) -> matrix().
softplus(X=#matrix{n=N,m=M,type=T}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,softplus__(Xi))|Acc]
	   end, [], X),
    new_(N,M,T,Es).

softplus__(X) ->
    math:log(1 + math:exp(X)).

print(A) ->
    io:put_chars(format(A)).

print(A,Prec) ->
    io:put_chars(format(A,Prec)).

format(X) ->
    format(X, 0).

format(#matrix{n=N,m=M,offset=Offset,stride=Stride,type=T,data=Bin0},Prec) ->
    Offset1 = Offset*element_bytes(T),
    <<_:Offset1/binary,Bin/binary>> = Bin0,
    Es = elements_to_list(T,Bin),
    Fs = [format_element(E,Prec) || E <- Es],
    W = lists:max([length(F) || F <- Fs]),
    %% left pad
    Fs2 = [lists:duplicate(W-length(F),$\s)++F || F <- Fs],
    format_rows(N,M,Stride,Fs2,"|","|\n"," ","~s",[]).

%% fixme allow integers as weights
format_rows(0,_M,_,_As,_Rb,_Re,_Sep,_Fmt,_Args) ->
    [];
format_rows(I,M,Stride,As,RowStart,RowEnd,Sep,Fmt,Args) ->
    {Row0,As1} = split_row(Stride, As),
    %% io:format("Row0=~p, As1=~p\n", [Row0, As1]),
    {Row,_}  = lists:split(M, Row0),
    %% io:format("Row=~p\n", [Row]),
    Es = [io_lib:format(Fmt,Args++[Xij]) || Xij <- Row],
    [ [RowStart,lists:join(Sep,Es),RowEnd] |
      format_rows(I-1,M,Stride,As1,RowStart,RowEnd,Sep,Fmt,Args)].

%% special split that allows a short tail
split_row(Pos, As) ->
    split_row(Pos, As, []).

split_row(0, As, Acc) -> 
    {lists:reverse(Acc), As};
split_row(I, [A|As], Acc) -> 
    split_row(I-1, As, [A|Acc]);
split_row(_I, [], Acc) ->
    {lists:reverse(Acc), []}.


is_integer_matrix(#matrix{type=T}) -> T < ?float32.
is_float_matrix(#matrix{type=T}) -> T >= ?float32.

format_element(X,_) when is_integer(X) ->
    integer_to_list(X);
format_element(X,0) when is_float(X) ->
    lists:flatten(io_lib_format:fwrite_g(X));
format_element(X,Prec) when is_float(X) ->
    lists:flatten(io_lib:format("~.*f", [Prec,X])).

normal_bin(M,S,T) ->
    V = normal_(M,S),
    case T of
	float64 -> <<V:64/native-float>>;
	float32 -> <<V:32/native-float>>
    end.

%% Generate a normal distributed random number
%% S is the standard deviation = sqrt(CoVariance)
normal_(M, S) when is_float(M), is_float(S), S > 0 ->
    X1 = uniform_(),
    X2 = uniform_(),
    M + S*math:sqrt(-2*math:log(X1))*math:cos(2*math:pi()*X2).

uniform_bin(?float64) ->
    F = uniform_(?float64),
    <<F:64/native-float>>;
uniform_bin(?float32) ->
    F = uniform_(?float32),
    <<F:32/native-float>>;
uniform_bin(?int64) ->
    crypto:strong_rand_bytes(8);
uniform_bin(?int32) ->
    crypto:strong_rand_bytes(4);
uniform_bin(?int16) ->
    crypto:strong_rand_bytes(2);
uniform_bin(?int8) ->
    crypto:strong_rand_bytes(1).

%% generate a double precision random number in [0-1)
%% or an integer random number.
uniform_() ->
    uniform_(?float64).

uniform_(?float64) ->
    <<_:4,X:52>> = crypto:strong_rand_bytes(7),
    <<F:64/float>> = <<16#3ff:12,X:52>>,
    F - 1;
uniform_(?float32) ->
    <<_:1,X:23>> = crypto:strong_rand_bytes(3),
    <<F:32/float>> = <<16#7f:9,X:23>>,
    F - 1;
uniform_(?int64) ->
    <<X:64/signed>> = crypto:strong_rand_bytes(8),
    X;
uniform_(?int32) ->
    <<X:32/signed>> = crypto:strong_rand_bytes(4),
    X;
uniform_(?int16) ->
    <<X:16/signed>> = crypto:strong_rand_bytes(2),
    X;
uniform_(?int8) ->
    <<X:8/signed>> = crypto:strong_rand_bytes(1),
    X.

typeof({_N,_M,T,_Bin}) -> T.
typeof({N,M,T1,_Bin1},{N,M,T2,_Bin2}) -> type_combine(T1,T2).
    
%% combine types to the more general one
type_combine(T1,T2) -> erlang:max(T1,T2).

number_to_bin(?float64, X) ->
    <<(float(X)):64/native-float>>;
number_to_bin(?float32, X) ->
    <<(float(X)):32/native-float>>;
number_to_bin(?int64, X) ->
    <<(trunc(X)):64/native-signed-integer>>;
number_to_bin(?int32, X) ->
    <<(trunc(X)):32/native-signed-integer>>;
number_to_bin(?int16, X) ->
    <<(trunc(X)):16/native-signed-integer>>;
number_to_bin(?int8, X) ->
    <<(trunc(X)):8/native-signed-integer>>.

element_bits(?float64) -> 64;
element_bits(?float32) -> 32;
element_bits(?int64) -> 64;
element_bits(?int32) -> 32;
element_bits(?int16) -> 16;
element_bits(?int8) -> 8.

element_bytes(?float64) -> 8;
element_bytes(?float32) -> 4;
element_bytes(?int64) -> 8;
element_bytes(?int32) -> 4;
element_bytes(?int16) -> 2;
element_bytes(?int8) -> 1.

encode_type(int8) -> ?int8;
encode_type(int16) -> ?int16;
encode_type(int32) -> ?int32;
encode_type(int64) -> ?int64;
encode_type(float32) -> ?float32;
encode_type(float64) -> ?float64.
