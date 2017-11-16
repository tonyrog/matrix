%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    binary matrix version
%%% @end

-module(matrix).

%% -compile(native).
-on_load(init/0).
-export([create/4, create/5]).
-export([element/3]).
-export([copy/1, copy/2, copy/4]).
-export([fill/2]).
-export([from_list/1, from_list/2, from_list/3, from_list/4]).
-export([to_list/1]).
-export([normal/1, uniform/1, zero/1, one/1, identity/1]).
-export([normal/2, uniform/2, zero/2, one/2, identity/2]).
-export([normal/3, uniform/3, zero/3, one/3, identity/3]).
-export([constant/3, constant/4]).
-export([add/2,add/3]).
-export([subtract/2,subtract/3]).
-export([multiply/2,multiply/3]).
-export([kmultiply/3]).
-export([mulsum/2]).
-export([expsum/1]).
-export([sum/1]).
-export([times/2,times/3]).
-export([ktimes/3]).
-export([scale/2, scale/3]).
-export([exp/1]).
-export([square/1]).
-export([pow/2]).
-export([negate/1, negate/2]).
-export([size/1]).
-export([type/1]).
-export([is_integer_matrix/1]).
-export([is_float_matrix/1]).
-export([is_complex_matrix/1]).

-export([sigmoid/1, sigmoid_prime/2]).
-export([relu/1, relu_prime/2]).
-export([leaky_relu/1, leaky_relu_prime/2]).
-export([linear/1, linear_prime/2]).
-export([tanh/1, tanh_prime/2]).
-export([softplus/1,softplus_prime/2]).
-export([softmax/1,softmax_prime/2]).
-export([transpose/1]).
-export([transpose_data/1, transpose_data/2]).
-export([print/1, print/2, format/1, format/2]).
-export([row/2, column/2, submatrix/5]).
-export([argmax/1, argmax/2]).
-export([min/2, max/2]).
-export([min/1, max/1]).
-export([topk/2]).
-export([convolve/4, convolve/6]).
-export([rconvolve/4, rconvolve/6]).
-export([maxpool/3, maxpool/5, maxpool/6]).
-export([l2pool/3, l2pool/5, l2pool/6]).
-export([filter/2, filter/4, filter/5]).

%% internal nifs
-export([add_/2,add_/3]).
-export([multiply_/2, multiply_/3]).
-export([kmultiply_/3, kmultiply_/4]).
-export([ktimes_/3, ktimes_/4]).
-export([negate_/1, negate_/2]).
-export([sigmoid_/1]).
-export([sigmoid_prime_/2]).
-export([l2pool_/5, l2pool_/6]).
-export([maxpool_/5, maxpool_/6]).
-export([max_/2, min_/2]).
-export([filter_/4, filter_/5]).
-export([apply1_/2, apply1_/3]).

%% internal use
-export([encode_type/1]).
-export([setelement_/4]).
-export([element_/3]).
-export([type_combine/2]).
-export([elem_to_bin/2]).

%% debug
-export([foldl_row/4, foldr_row/4]).
-export([foldl_column/4, foldr_column/4]).
-export([foldl_rows/5, foldr_rows/5]).
-export([foldl_matrix/3, foldr_matrix/3]).

-export_type([matrix/0]).

%% performance counter
-export([dump/0, dump/1]).
-export([clear_counters/0]).
-define(count_op(OP,A,B), count_op((OP),(A),(B))).
%% -define(count_op(OP,A,B), ok).

%% maximum numbr of elements for add/sub/times/negate...
%% -define(MAX_NM, (256*4096)).
-define(MAX_ADD_NM, (10*10)).
-define(MAX_MUL_NM, (10*10)).

-compile({no_auto_import,[size/1]}).
-compile({no_auto_import,[max/2]}).
-compile({no_auto_import,[min/2]}).

-include("matrix.hrl").

-define(nif_stub(),
	erlang:nif_error({nif_not_loaded,module,?MODULE,line,?LINE})).

init() ->
    Nif = filename:join(code:priv_dir(matrix), "matrix_drv"),
    erlang:load_nif(Nif, 0).

-spec create(N::unsigned(), M::unsigned(), T::matrix_type(), Data::iolist()) ->
		 matrix().

create(N,M,T,Data) ->
    create(N,M,T,true,Data).

create(N,M,Type,RowMajor,Data) when is_atom(Type) ->
    create(N,M,encode_type(Type),RowMajor,Data);
create(N,M,T,RowMajor,Data) when is_integer(N), N>0,
				 is_integer(M), M>0,
				 is_integer(T),
				 (is_list(Data) orelse is_binary(Data)) ->
    create_(N,M,T,RowMajor,Data).

create_(N,M,T,RowMajor,Data) ->
    matrix_ref:create(N,M,T,RowMajor,Data).

-spec copy(Src::matrix()) ->  matrix().
copy(_Src) ->
    ?nif_stub().

-spec copy(Src::matrix(), Dst::matrix()) ->
		  matrix().
copy(_Src, _Dst) ->
    ?nif_stub().

-spec copy(Src::matrix(), Dst::matrix(), 
	   RepeatHorizontal::unsigned(),
	   RepeatVertical::unsigned()) ->
		  matrix().

%% DESTRUCTIVE
copy(_Src, _Dst, _Repeat_h, _Rpeat_v) ->
    ?nif_stub().


-spec fill(Src::matrix(), Dst::matrix()) ->
		  matrix().

%% DESTRUCTIVE
fill(_Src, _Dst) ->
    ?nif_stub().

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
    create(N, M, T, true, Es).

from_list_([], N, N, _M, _T) ->
    [];
from_list_([], I, N, M, T) when I < N ->
    Pad = lists:duplicate(M, elem_to_bin(T, 0)),
    [Pad | from_list_([], I+1, N, M, T)];
from_list_([R|Rs], I, N, M, T) ->
    L = length(R),
    Pad = lists:duplicate(M - L, elem_to_bin(T, 0)),
    [ [ elem_to_bin(T,E) || E <- R ], Pad |
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
type_list([E|_Es], _T) when ?is_complex(E) ->
    ?complex128;
type_list([E|Es], T) ->
    if is_integer(E) ->
	    if E >= -16#80, E < 16#80 ->
		    type_list(Es, erlang:max(?int8,T));
	       E >= -16#8000, E < 16#8000 ->
		    type_list(Es, erlang:max(?int16,T));
	       E >= -16#80000000, E < 16#80000000 ->
		    type_list(Es, erlang:max(?int32,T));
	       true ->
		    type_list(Es, erlang:max(?int64,T))
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

to_list(A=#matrix{n=N,rowmajor=true}) ->
    [foldr_row(I, fun(Xij,Acc) -> [Xij|Acc] end, [], A) ||
	I <- lists:seq(1,N)];
to_list(A=#matrix{m=M,rowmajor=false}) ->
    [foldr_row(I, fun(Xij,Acc) -> [Xij|Acc] end, [], A) ||
	I <- lists:seq(1,M)].

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
    A = create(N,M,Type,true,[]),
    apply1_(normal, A, A).

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
    A = create(N,M,Type,true,[]),
    apply1_(uniform, A, A).

-spec zero({N::unsigned(), M::unsigned()}) -> matrix().
zero({N,M}) -> zero(N,M,float64).

-spec zero({N::unsigned(), M::unsigned()}, T::matrix_type()) -> matrix().
zero({N,M},T) -> zero(N,M,T).

-spec zero(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
zero(N,M,Type) when is_integer(N), N >= 1,
		    is_integer(M), M >= 1 ->
    constant_(N, M, Type, 0).

-spec one({N::unsigned(), M::unsigned()}) -> matrix().
one({N,M}) -> one(N,M,float64).

-spec one({N::unsigned(), M::unsigned()}, T::matrix_type()) -> matrix().
one({N,M},T) -> one(N,M,T).

-spec one(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
one(N,M,Type) when is_integer(N), N >= 1,
		 is_integer(M), M >= 1 ->
    constant_(N, M, Type, 1).

-spec constant(N::unsigned(), M::unsigned(),C::scalar()) -> matrix().
constant({N,M},Type,C) ->
    constant(N,M,Type,C);
constant(N,M,C) when is_integer(C) ->
    constant(N,M,int32,C);
constant(N,M,C) when is_float(C) ->
    constant(N,M,float32,C);
constant(N,M,C) when ?is_complex(C) ->
    constant(N,M,complex64,C).

-spec constant(N::unsigned(), M::unsigned(), T::matrix_type(), C::scalar()) ->
		      matrix().
constant(N,M,Type,C) when is_integer(N), N >= 1,
			  is_integer(M), M >= 1,
			  ?is_scalar(C) ->
    constant_(N,M,Type,C).

%% special constant for testing!
constant_(N,M,Type,C) ->
    T = encode_type(Type),
    Bin = elem_to_bin(T,C),
    #matrix { n=N, m=M, type=T, stride=0, rowmajor=true, data=Bin }.

-spec identity({N::unsigned(), M::unsigned()}) -> matrix().
identity({N,M}) ->
    identity(N,M,float64).

-spec identity({N::unsigned(),M::unsigned()},T::matrix_type()) -> matrix().
identity({N,M},T) ->
    identity(N,M,T).

-spec identity(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
identity(N,M,Type) when is_integer(N), N >= 1,
			is_integer(M), M >= 1 ->
    T = encode_type(Type),
    identity_(N,M,T).

-spec identity_(N::unsigned(), M::unsigned(), T::integer()) -> matrix().

identity_(_N,_M,_T) ->
    ?nif_stub().     


-spec apply1_(A::matrix(), Op::atom()) -> matrix().
apply1_(_A, _Op) ->
    ?nif_stub(). 

-spec apply1_(A::matrix(), Dst::matrix(), Op::atom()) -> matrix().
apply1_(_A, _Dst, _Op) ->
    ?nif_stub(). 

%% return dimension in row major order
-spec size(M::matrix()) -> {unsigned(), unsigned()}.
size(#matrix{rowmajor=true,n=N,m=M}) ->
    {N,M};
size(#matrix{rowmajor=false,n=N,m=M}) ->
    {M,N}.

type(#matrix{type=T}) ->
    case T of
	?int8 -> int8;
	?int16 -> int16;
	?int32 -> int32;
	?int64 -> int64;
	?float32 -> float32;
	?float64 -> float64;
	?complex64 -> complex64;
	?complex128 -> complex128
    end.
-spec is_integer_matrix(X::matrix()) -> boolean().
is_integer_matrix(X) -> ?is_int_matrix(X).

-spec is_float_matrix(X::matrix()) -> boolean().
is_float_matrix(X) -> ?is_float_matrix(X).

-spec is_complex_matrix(X::matrix()) -> boolean().
is_complex_matrix(X) -> ?is_complex_matrix(X).

-spec element(I::unsigned(),J::unsigned(),X::matrix()) -> scalar().

element(I,J,A) ->
    element_(I,J,A).

element_(I,J,A) ->
    matrix_ref:element(I,J,A).

-spec setelement_(I::unsigned(),J::unsigned(),X::matrix(),V::scalar()) -> 
			matrix().
setelement_(_I,_J,_A,_V) ->
    ?nif_stub().


foldl_matrix(F,A,X=#matrix{rowmajor=true,n=N}) ->
    foldl_rows(1,N,F,A,X);
foldl_matrix(F,A,X=#matrix{rowmajor=false,m=M}) ->
    foldl_rows(1,M,F,A,X).

foldr_matrix(F,A,X=#matrix{rowmajor=true,n=N}) ->
    foldr_rows(1,N,F,A,X);
foldr_matrix(F,A,X=#matrix{rowmajor=false,m=M}) ->
    foldr_rows(1,M,F,A,X).

foldl_rows(I,N,_F,A,_X) when I > N -> A;
foldl_rows(I,N,F,A,X) ->
    A1 = foldl_row(I,F,A,X),
    foldl_rows(I+1,N,F,A1,X).

foldr_rows(I,N,_F,A,_X) when I > N -> A;
foldr_rows(I,N,F,A,X) ->
    A1 = foldr_row(I,F,A,X),
    foldr_rows(I+1,N,F,A1,X).

%% fold left over row
foldl_row(I,F,A,#matrix{rowmajor=true,n=_N,m=M,offset=O,
		       stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,M);
       true ->
	    P = O + (I-1)*S,
	    fold_elems_(F,A,D,P,T,1,M)
    end;
foldl_row(I,F,A,#matrix{rowmajor=false,n=N,m=_M,offset=O,
		       stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,N);
       true ->
	    P = O + (I-1),
	    fold_elems_(F,A,D,P,T,S,N)
    end.


%% fold left over column
foldl_column(J,F,A,#matrix{rowmajor=false,n=_N,m=M,offset=O,
			   stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,M);
       true ->
	    P = O + (J-1)*S,
	    fold_elems_(F,A,D,P,T,1,M)
    end;
foldl_column(J,F,A,#matrix{rowmajor=true,n=N,m=_M,offset=O,
			   stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,N);
       true ->
	    P = O + (J-1),
	    fold_elems_(F,A,D,P,T,S,N)
    end.

%% fold right over rows
foldr_row(I,F,A,#matrix{rowmajor=true,n=_N,m=M,offset=O,
			stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,M);
       true ->
	    P = O + (I-1)*S + M-1,
	    fold_elems_(F,A,D,P,T,-1,M)
    end;
foldr_row(I,F,A,#matrix{rowmajor=false,n=N,m=_M,offset=O,
			stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	        fold_elems_(F,A,D,P,T,0,N);
       true ->
	    P = O + (N-1)*S + (I-1),
	    fold_elems_(F,A,D,P,T,-S,N)
    end.

foldr_column(J,F,A,#matrix{rowmajor=false,n=_N,m=M,offset=O,
			   stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,M);
       true ->
	    P = O + (J-1)*S + M-1,
	    fold_elems_(F,A,D,P,T,-1,M)
    end;
foldr_column(J,F,A,#matrix{rowmajor=true,n=N,m=_M,offset=O,
			   stride=S,type=T,data=D}) ->
    if S =:= 0 ->
	    P = O,
	    fold_elems_(F,A,D,P,T,0,N);
       true ->
	    P = O + (N-1)*S + (J-1),
	    fold_elems_(F,A,D,P,T,-S,N)
    end.

fold_elems_(_F,A,_D,_P,_T,_S,0) -> A;
fold_elems_(F,A,D,P,T,S,I) -> 
    E = matrix_ref:decode_element_at(P,T,D),
    A1 = F(E,A),
    fold_elems_(F,A1,D,P+S,T,S,I-1).

%%
%% Add two matrices
%%
-spec add(A::matrix(), B::matrix()) -> matrix();
	 (A::matrix(), B::scalar()) -> matrix();
	 (A::scalar(), B::matrix()) -> matrix().

add(A,B) ->
    ?count_op(add,A,B),
    add_(A,B).

add_(A,B) ->
    matrix_ref:add(A,B).

%% destructive add
-spec add(A::matrix(), B::matrix(), Dst::matrix()) -> matrix();
	 (A::matrix(), B::scalar(), Dst::matrix()) -> matrix();
	 (A::scalar(), B::matrix(), Dst::matrix()) -> matrix().

add(A, B, Dst) ->
    add_(A, B, Dst).

add_(_A, _B, _Dst) ->
    ?nif_stub().

%%
%% Subtract two matrices
%%

-spec subtract(A::matrix(), B::matrix()) -> matrix();
	      (A::matrix(), B::scalar()) -> matrix();
	      (A::scalar(), B::matrix()) -> matrix().

subtract(A, B) ->
    ?count_op(subtract,A,B),
    subtract_(A, B).

%% destructive subtract
-spec subtract(A::matrix(), B::matrix(), Dst::matrix()) -> matrix();
	      (A::matrix(), B::scalar(), Dst::matrix()) -> matrix();
	      (A::scalar(), B::matrix(), Dst::matrix()) -> matrix().

subtract(A, B, Dst) ->
    subtract_(A,B,Dst).

subtract_(_A,_B,_Dst) ->
    ?nif_stub().

subtract_(A, B) ->
    matrix_ref:subtract(A, B).

%%
%% Multiply two matrices element wise
%%
-spec times(A::matrix(), B::matrix()) -> matrix();
	   (A::matrix(), B::scalar()) -> matrix();
	   (A::scalar(), B::matrix()) -> matrix().

times(A,B) ->
    ?count_op(times,A,B),
    times_(A,B).

times_(A,B) ->
    matrix_ref:times(A,B).

-spec times(A::matrix(), B::matrix(), Dst::matrix()) -> matrix();
	   (A::matrix(), B::scalar(), Dst::matrix()) -> matrix();
	   (A::scalar(), B::matrix(), Dst::matrix()) -> matrix().

times(A,B,Dst) ->
    times_(A,B,Dst).

times_(_X,_Y,_Dst) ->
    ?nif_stub().

%%
%% Negate a matrix
%%
-spec negate(A::matrix()) -> matrix().
negate(X) ->
    negate_(X).

negate_(X) ->
    matrix_ref:negate(X).

-spec negate(X::matrix(),Dst::matrix()) -> matrix().
negate(X, Dst) ->
    negate_(X, Dst).

negate_(_X, _Dst) ->
    ?nif_stub().

%%
%% Scale a matrix by a scalar number
%%
-spec scale(F::scalar(), X::matrix()) -> matrix().

scale(F, X=#matrix{}) when ?is_scalar(F) ->
    times(F, X).

-spec scale(F::number(), X::matrix(), Dst::matrix()) -> matrix().
scale(F, X=#matrix{}, Dst=#matrix{}) when ?is_scalar(F) ->
    times(F, X, Dst).
    
%% expontiation of all elements
-spec exp(X::matrix()) -> matrix().
exp(X) ->
    exp_(X).

exp_(X) ->
    matrix_ref:exp(X).
%%
%% Multiply elementwise and add everything
%%
-spec mulsum(X::matrix(), Y::matrix()) -> number().
mulsum(X,Y) ->
    mulsum_(X,Y).

mulsum_(X,Y) ->
    matrix_ref:mulsum(X,Y).

%% sum all elements in matrix
-spec sum(X::matrix()) -> number().
sum(X) ->
    sum_(X).

sum_(X) ->
    matrix_ref:sum(X).

%% expsum
%% sum all elements after exponetiation
-spec expsum(X::matrix()) -> number().

expsum(X) ->
    sum(exp(X)).
    
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

multiply(A, B) ->
    ?count_op(multiply,A,B),
    multiply_(A,B).

multiply_(A, B) ->
    matrix_ref:multiply(A,B).

-spec multiply(X::matrix(), Y::matrix(), RowMajor::boolean) ->  matrix() ;
	      (X::matrix(), Y::matrix(), Dst::matrix()) -> matrix().

multiply(A, B, Arg) ->
    multiply_(A, B, Arg).

multiply_(A, B, RowMajor) when is_boolean(RowMajor) ->
    matrix_ref:multiply(A,B,RowMajor);
multiply_(_A, _B, _Dst) ->
    ?nif_stub().

%% get the top K elements from A as a integer matrix Kwith indices
-spec topk(A::matrix(), K::unsigned()) -> matrix().
topk(A, K) ->
    topk_(A,K).

topk_(A,K) ->
    matrix_ref:topk(A, K).

%%
%% Multiply two matrices but only in rows from K
%%
-spec kmultiply(X::matrix(), Y::matrix(), K::matrix()) -> matrix().

kmultiply(A, B, undefined) ->
    multiply(A, B);
kmultiply(A, B, K) ->
    ?count_op({kmultiply,K},A,B),
    kmultiply_(A, B, K).

-spec kmultiply_(X::matrix(), Y::matrix(), K::matrix()) -> matrix().
kmultiply_(A, B, K) ->
    matrix_ref:kmultiply(A, B, K).

-spec kmultiply_(X::matrix(), Y::matrix(), K::matrix(), C::matrix())-> matrix().
kmultiply_(_A, _B, _K, _C) ->
    ?nif_stub().

%%
%% Multiply two matrices element wise
%% (X o Y) o K
%%
-spec ktimes(X::matrix(), Y::matrix(), K::matrix()) -> matrix().

ktimes(A, B, undefined) ->
    times(A, B);
ktimes(A, B, K) ->
    ?count_op({ktimes,K},A,B),
    ktimes_(A, B, K).

-spec ktimes_(X::matrix(), Y::matrix(), K::matrix()) -> matrix().
ktimes_(A, B, K) ->
    matrix_ref:ktimes(A, B, K).

-spec ktimes_(X::matrix(), Y::matrix(), K::matrix(), C::matrix()) -> matrix().
ktimes_(_A, _B, _K, _C) ->
    ?nif_stub().
    
%%
%% Transpose a matrix
%% if rowmajor then n = number of rows, m = number of columns
%% if !rowmajor then n = number of columns, m = number of rows!
%%
-spec transpose(A::matrix()) -> matrix().
transpose(X = #matrix{rowmajor=RowMajor}) ->
    X#matrix { rowmajor=not RowMajor }.

-spec transpose_data(Src::matrix()) -> matrix().
transpose_data(Src) ->
    matrix_ref:transpose_data(Src).

-spec transpose_data(Src::matrix(),Dst::matrix()) -> matrix().
transpose_data(_Src, _Dst) ->
    ?nif_stub().

%%
%% select a row, return as a matrix with one row
%%
-spec row(I::unsigned(), A::matrix()) -> matrix().
row(I, X=#matrix{rowmajor=true,n=N,m=M}) when I >= 1, I =< N ->
    submatrix(I, 1, 1, M, X);
row(I, X=#matrix{rowmajor=false,n=N,m=M}) when I >= 1, I =< M ->
    submatrix(I, 1, 1, N, X).

%%
%% select a column, return as a matrix with one column
%%
-spec column(J::unsigned(), A::matrix()) -> matrix().
column(J, X=#matrix{rowmajor=true,n=N,m=M}) when J >= 1, J =< M ->
    submatrix(1, J, N, 1, X);
column(J, X=#matrix{rowmajor=false,n=N,m=M}) when J >= 1, J =< N ->
    submatrix(1, J, M, 1, X).

%%
%% select a portion of a matrix
%%
-spec submatrix(I::unsigned(), J::unsigned(),
		N::unsigned(), M::unsigned(),
		X::matrix()) -> matrix().

%% fixme range check!
submatrix(I, J, N, M, X=#matrix{rowmajor=true,offset=Offset,stride=Stride}) ->
    Offset1 = if Stride =:= 0 ->
		      Offset;
		 true ->
		      Offset + (I-1)*Stride + (J-1)
	      end,
    X#matrix { n=N, m=M, offset=Offset1};
submatrix(I, J, N, M, X=#matrix{rowmajor=false,offset=Offset,stride=Stride}) ->
    Offset1 = if Stride =:= 0 ->
		      Offset;
		 true ->
		      Offset + (J-1)*Stride + (I-1)
	      end,
    X#matrix { n=M, m=N, offset=Offset1}.

%%
%% convolve a NxM matrix over the matrix A (soon: with padding Px, Py and
%% padding value PAD) using Sn and Sm as stride steps.
%%

-spec convolve(F::function(),
	       N::unsigned(),M::unsigned(),
	       Sn::unsigned(), Sm::unsigned(),A::matrix()) ->
		      matrix().

convolve(F,N,M,Sn,Sm,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(1,Nx-N+1,Sn), J <- lists:seq(1,Mx-M+1,Sm)].

-spec convolve(F::fun((integer(),integer()) -> term()),
	       N::unsigned(),M::unsigned(),A::matrix()) -> [term()].

convolve(F,N,M,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(1,Nx-N+1,1), J <- lists:seq(1,Mx-M+1,1)].

-spec rconvolve(F::function(),
	       N::unsigned(),M::unsigned(),
	       Sn::unsigned(), Sm::unsigned(),A::matrix()) ->
		      matrix().

rconvolve(F,N,M,Sn,Sm,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(Nx-N+1,1,-Sn), J <- lists:seq(Mx-M+1,1,-Sm)].

-spec rconvolve(F::fun((integer(),integer()) -> term()),
	       N::unsigned(),M::unsigned(),A::matrix()) -> [term()].

rconvolve(F,N,M,#matrix{n=Nx,m=Mx}) when N =< Nx, M =< Mx ->
    [ F(I,J) || I <- lists:seq(Nx-N+1,1,-1), J <- lists:seq(Mx-M+1,1,-1)].

%%
%%
%%
-spec maxpool(N::unsigned(),M::unsigned(),A::matrix()) -> matrix().

maxpool(N, M, A) ->
    maxpool(N, M, 1, 1, A).

-spec maxpool(N::unsigned(),M::unsigned(),
	      Sn::unsigned(),Sm::unsigned(),A::matrix()) -> matrix().

maxpool(N, M, Sn, Sm, A) ->
    maxpool_(N, M, Sn, Sm, A).

maxpool_(N, M, Sn, Sm, A) ->
    matrix_ref:maxpool(N, M, Sn, Sm, A).

-spec maxpool(N::unsigned(),M::unsigned(),Sn::unsigned(),Sm::unsigned(),
	      A::matrix(),Dst::matrix()) ->
		     matrix().

maxpool(N, M, Sn, Sm, A, Dst) ->
    maxpool_(N, M, Sn, Sm, A, Dst).

maxpool_(_N, _M, _Sn, _Sm, _A, _Dst) ->
    ?nif_stub().

%% l2pool
-spec l2pool(N::unsigned(),M::unsigned(),A::matrix()) -> matrix().
l2pool(N, M, A) ->
    l2pool(N, M, 1, 1, A).

%% l2pool 
-spec l2pool(N::unsigned(),M::unsigned(),
	     Sn::unsigned(),Sm::unsigned(),A::matrix()) -> matrix().

l2pool(N, M, Sn, Sm, A) ->
    l2pool_(N, M, Sn, Sm, A).

l2pool_(N, M, Sn, Sm, A) ->
    matrix_ref:l2pool(N, M, Sn, Sm, A).

-spec l2pool(A::matrix(),N::unsigned(),M::unsigned(),
	     Sn::unsigned(),Sm::unsigned(),Dst::matrix()) -> matrix().

l2pool(N, M, Sn, Sm, A, Dst) ->
    l2pool_(N, M, Sn, Sm, A, Dst).

l2pool_(_N, _M, _Sn, _Sm, _A, _Dst) ->
    ?nif_stub().

%%
%%
%%
-spec filter(W::matrix(), X::matrix()) -> matrix().
filter(W, X) ->
    filter(W, 1, 1, X).

-spec filter(W::matrix(), Sn::unsigned(), Sm::unsigned(), A::matrix()) -> 
		    matrix().

filter(W, Sn, Sm, A) ->
    filter_(W, Sn, Sm, A).

filter_(W, Sn, Sm, A) ->
    matrix_ref:filter(W, Sn, Sm, A).

-spec filter(W::matrix(), Sn::unsigned(), Sm::unsigned(), A::matrix(),
	     Dst::matrix()) -> matrix().

filter(W, Sn, Sm, A, Dst) ->
    filter_(W, Sn, Sm, A, Dst).

filter_(_W, _Sn, _Sm, _A, _Dst) ->
    ?nif_stub().

%% argmax
-spec argmax(A::matrix(),Axis::0|1) -> [integer()].

argmax(A) ->
    Ai = argmax(A,0),    %% vector of indices for max columns
    to_list(Ai).

argmax(A,I) ->
    argmax_(A,I).

argmax_(A,I) ->
    matrix_ref:argmax(A,I).

-spec max(A::matrix()) -> scalar().
max(A) ->
    max_(A).

max_(A) ->
    matrix_ref:max(A).

-spec max(A::matrix(), Axis::0|1) -> matrix().
max(A, Axis) ->
    max_(A, Axis).

max_(A, Axis) ->
    matrix_ref:max(A, Axis).

-spec min(A::matrix()) -> scalar().
min(A) ->
    min_(A).

min_(A) ->
    matrix_ref:min(A).

-spec min(A::matrix(), Axis::0|1) -> matrix().
min(A, Axis) ->
    min_(A, Axis).

min_(A, Axis) ->
    matrix_ref:min(A, Axis).

-spec sigmoid(A::matrix()) -> matrix().
sigmoid(X) ->
    sigmoid_(X).

sigmoid_(X) ->
    matrix_ref:sigmoid(X).

-spec sigmoid_prime(A::matrix(),Out::matrix()) -> matrix().
%% Out = sigmoid(A)!!!  sigmoid_prime = Out*(1-Out)
sigmoid_prime(X,Out) ->
    sigmoid_prime_(X,Out).

sigmoid_prime_(A,Out) ->
    matrix_ref:sigmoid_prime(A,Out).

-spec relu(A::matrix()) -> matrix().
relu(A) ->
    apply1_(relu, A).

-spec relu_prime(A::matrix(),Out::matrix()) -> matrix().
relu_prime(A,_Out) ->
    apply1_(relu_prime, A).

-spec leaky_relu(A::matrix()) -> matrix().
leaky_relu(A) ->
    apply1_(leaky_relu, A).

-spec leaky_relu_prime(A::matrix(),Out::matrix()) -> matrix().
leaky_relu_prime(A,_Out) ->
    apply1_(leaky_relu_prime, A).

-spec linear(A::matrix()) -> matrix().
linear(A) -> 
    A.

-spec linear_prime(A::matrix(),Out::matrix()) -> matrix().
linear_prime(A,_Out) ->
    one(size(A), type(A)).

-spec softplus(A::matrix()) -> matrix().
softplus(A) ->
    apply1_(softplus, A).

-spec softplus_prime(A::matrix(),Out::matrix()) -> matrix().
softplus_prime(A,_Out) ->
    apply1_(softplus_prime, A).

-spec softmax(A::matrix()) -> matrix().
softmax(A) ->
    scale(apply1_(exp,A), 1/expsum(A)).

-spec softmax_prime(A::matrix(),Out::matrix()) -> matrix().
softmax_prime(A,_Out) ->
    apply1_(softmax_prime, A).


-spec tanh(A::matrix()) -> matrix().
tanh(A) ->
    apply1_(tanh, A).

-spec tanh_prime(A::matrix(),Out::matrix()) -> matrix().
%% grad'(x) = 
tanh_prime(_A,Out) ->
    apply1_(tanh_prime1, Out).

print(A) ->
    io:put_chars(format(A)).

print(A,Prec) ->
    io:put_chars(format(A,Prec)).

format(X) ->
    format(X, 2).

format(X,Prec) ->
    Rows = to_list(X),
    FRows = [ [format_element(E,Prec) || E <- Row] || Row <- Rows],
    ColWs = column_width(FRows),
    format_rows(FRows,ColWs,"|","|\n"," ","~s",[]).

format_rows([],_ColWs,_Rb,_Re,_Sep,_Fmt,_Args) ->
    [];
format_rows([Row|Rows],ColWs,RowStart,RowEnd,Sep,Fmt,Args) ->
    Es = lists:zipwith(
	   fun(F,W) ->
		   lists:duplicate(W-length(F),$\s)++F
	   end, Row, ColWs),
    [ [RowStart,lists:join(Sep,Es),RowEnd] |
      format_rows(Rows,ColWs,RowStart,RowEnd,Sep,Fmt,Args)].

%% calculate column width (max length over columns )
column_width([[]|_]) ->
    [];
column_width(Rows) ->
    Column = [length(hd(Row)) || Row <- Rows],
    [ lists:max(Column) | column_width([tl(Row) || Row <- Rows]) ].


format_element({R,I},Prec) when is_float(R),is_float(I) ->
    if I == 0 ->
	    format_element(R,Prec)++" ";
       I < 0 ->
	    format_element(R,Prec)++"-"++format_element(-I,Prec)++"i";
       true ->
	    format_element(R,Prec)++"+"++format_element(I,Prec)++"i"
    end;
format_element(X,_) when is_integer(X) ->
    integer_to_list(X);
format_element(X,0) when is_float(X) ->
    lists:flatten(io_lib_format:fwrite_g(X));
format_element(X,Prec) when is_float(X) ->
    lists:flatten(io_lib:format("~.*f", [Prec,X])).

%% combine types to the more general one
type_combine(T1,T2) -> erlang:max(T1,T2).

elem_to_bin(?complex128, {R,I}) ->
    <<(float(R)):64/native-float,(float(I)):64/native-float>>;
elem_to_bin(?complex128, R) when is_number(R) ->
    <<(float(R)):64/native-float,0.0:64/native-float>>;
elem_to_bin(?complex64, {R,I}) ->
    <<(float(R)):32/native-float,(float(I)):32/native-float>>;
elem_to_bin(?complex64, R) ->
    <<(float(R)):32/native-float,0.0:32/native-float>>;
elem_to_bin(?float64, X) ->
    <<(float(X)):64/native-float>>;
elem_to_bin(?float32, X) ->
    <<(float(X)):32/native-float>>;
elem_to_bin(?int64, X) ->
    <<(trunc(X)):64/native-signed-integer>>;
elem_to_bin(?int32, X) ->
    <<(trunc(X)):32/native-signed-integer>>;
elem_to_bin(?int16, X) ->
    <<(trunc(X)):16/native-signed-integer>>;
elem_to_bin(?int8, X) ->
    <<(trunc(X)):8/native-signed-integer>>.


%% simulate float128 binary
-ifdef(not_used).
float128_to_binary(X) ->
    <<S:1,E:11,F:52>> = <<(float(X)):64/float>>, %% first 64 bit
    E1 = E-1023,
    <<Xi:128>> = <<S:1,(E1+16383):15,F:52,0:60>>,  %% endian
    <<Xi:128/native>>.
-endif.


element_bytes(?complex128) -> 16;
element_bytes(?complex64) -> 8;
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
encode_type(float64) -> ?float64;
encode_type(complex64) -> ?complex64;
encode_type(complex128) -> ?complex128.

%% performance counters

count_op(multiply,
	 #matrix{rowmajor=R1,n=N1,m=M1,type=T1},
	 #matrix{rowmajor=R2,n=N2,m=M2,type=T2}) ->
    K1 = {R1,N1,M1},
    K2 = {R2,N2,M2},
    if T1 =:= T2, R1, R2 ->
	    count({{multiply,simd},K1,K2},1);
       T1 =:= T2, R1, not R2 -> %% extra fast?
	    count({{multiply,simd},K1,K2},1);
       T1 =:= T2, not R1, R2 ->
	    count({{multiply,plain},K1,K2},1);
       T1 =:= T2, not R1, not R2 ->
	    count({{multiply,simd},K1,K2},1);
       true ->
	    count({{multiply,slow},K1,K2},1)
    end;
count_op(add,
	 #matrix{rowmajor=R1,n=N1,m=M1,type=T1},
	 #matrix{rowmajor=R2,n=N2,m=M2,type=T2}) ->
    K1 = {R1,N1,M1},
    K2 = {R2,N2,M2},
    if T1 =:= T2, R1 =:= R2 ->
	    count({{add,simd},K1,K2},1);
       T1 =:= T2 ->
	    count({{add,plain},K1,K2},1);
       true ->
	    count({{add,slow},K1,K2},1)
    end;
count_op(add,
	 #matrix{rowmajor=R1,n=N1,m=M1},
	 Yc) when ?is_scalar(Yc) ->
    K1 = {R1,N1,M1},
    count({{add,const},K1},1);
count_op(add,  Xc, #matrix{rowmajor=R2,n=N2,m=M2}) when ?is_scalar(Xc) ->
    K2 = {R2,N2,M2},
    count({{add,const},K2},1);
count_op(subtract,
	 #matrix{rowmajor=R1,n=N1,m=M1,type=T1},
	 #matrix{rowmajor=R2,n=N2,m=M2,type=T2}) ->
    K1 = {R1,N1,M1},
    K2 = {R2,N2,M2},
    if T1 =:= T2, R1 =:= R2 ->
	    count({{subtract,simd},K1,K2},1);
       T1 =:= T2 ->
	    count({{subtract,plain},K1,K2},1);
       true ->
	    count({{subtract,slow},K1,K2},1)
    end;
count_op(subtract,#matrix{rowmajor=R1,n=N1,m=M1},Yc) when ?is_scalar(Yc) ->
    K1 = {R1,N1,M1},
    count({{subtract,const},K1},1);
count_op(subtract, Xc, #matrix{rowmajor=R2,n=N2,m=M2}) when ?is_scalar(Xc) ->
    K2 = {R2,N2,M2},
    count({{subtract,const},K2},1);
count_op(times,
	 #matrix{rowmajor=R1,n=N1,m=M1,type=T1},
	 #matrix{rowmajor=R2,n=N2,m=M2,type=T2}) ->
    K1 = {R1,N1,M1},
    K2 = {R2,N2,M2},
    if T1 =:= T2, R1 =:= R2 ->
	    count({{times,simd},K1,K2},1);
       T1 =:= T2 ->
	    count({{times,plain},K1,K2},1);
       true ->
	    count({{times,slow},K1,K2},1)
    end;
count_op(times,#matrix{rowmajor=R1,n=N1,m=M1},Yc) when ?is_scalar(Yc) ->
    K1 = {R1,N1,M1},
    count({{times,const},K1},1);
count_op(times, Xc, #matrix{rowmajor=R2,n=N2,m=M2}) when ?is_scalar(Xc) ->
    K2 = {R2,N2,M2},
    count({{times,const},K2},1);
count_op({Op,#matrix{rowmajor=R,n=N,m=M}},
	 #matrix{rowmajor=R1,n=N1,m=M1},
	 #matrix{rowmajor=R2,n=N2,m=M2}) ->
    K1 = {R1,N1,M1},
    K2 = {R2,N2,M2},
    count({{Op,{R,N,M}},K1,K2}, 1);
count_op(Op,
	 #matrix{rowmajor=R1,n=N1,m=M1},
	 #matrix{rowmajor=R2,n=N2,m=M2}) ->
    K1 = {R1,N1,M1},
    K2 = {R2,N2,M2},
    count({{Op,plain},K1,K2}, 1).

dump() ->
    dump(all).

dump(MatchKey) ->
    lists:foreach(
      fun
	  ({ {{Key,SubKey},K1,K2}, Count, Caller }) 
	    when Key =:= MatchKey; MatchKey =:= all ->
	      print_counter(Key,SubKey,K1,K2,Count,Caller);
	  ({ {{Key,SubKey},K1}, Count, Caller }) 
	    when Key =:= MatchKey; MatchKey =:= all ->
	      print_counter(Key,SubKey,K1,Count,Caller);
	  (_) ->
	      ok
      end, lists:sort(get_counters())).

get_counters() ->
    [{Key,Value,Caller} || {{'$counter',Key,Caller},Value} <- get()].

%% clear all $counters
clear_counters() ->
    lists:foreach(
      fun({Key={'$counter',_Key,_Caller},_Value}) ->
	      erase(Key);
	 (_) -> ok
      end, get()).

%% overall counter  name R(4x5) C(5x4)
print_counter(Key,SubKey,K1,K2,Value,Caller) ->
    io:put_chars([atom_to_list(Key),"/",format_subkey(SubKey)," ",
		  format_dim(K1)," ",format_dim(K2),
		  " = ", integer_to_list(Value),
		  " @ ",format_caller(Caller),
		  "\n"]).

print_counter(Key,SubKey,K1,Value,Caller) ->
    io:put_chars([atom_to_list(Key),"/",format_subkey(SubKey)," ",
		  format_dim(K1)," ",
		  " = ", integer_to_list(Value),
		  " @ ",format_caller(Caller),
		  "\n"]).

format_subkey(Dim) when is_tuple(Dim) ->
    format_dim(Dim);
format_subkey(SubKey) when is_atom(SubKey) ->
    atom_to_list(SubKey).


%% format dimension and rowmajor
format_dim({true,N,M}) -> ["R(",integer_to_list(N),"x",integer_to_list(M),")"];
format_dim({false,M,N}) -> ["C(",integer_to_list(N),"x",integer_to_list(M),")"].

format_caller({M,F,A,Ln}) ->
    [atom_to_list(M),":",atom_to_list(F),"/",integer_to_list(A),":",
     integer_to_list(Ln),":"].
			  
count(Key, Value) ->
    Caller  = get_counter_caller(),
    Key1 = {'$counter',Key,Caller},
    case get(Key1) of
	undefined ->
	    put(Key1, Value);
	Value0 ->
	    put(Key1, Value0+Value)
    end.

%% slow but give us info where the counter is counted from
get_counter_caller() ->
    Es = try erlang:error(fake) catch error:_ -> erlang:get_stacktrace() end,
    [_Here,_MatrixCount,_MatrixOp,{M,F,A,[_,{line,Ln}]}|_] = Es,
    {M,F,A,Ln}.
