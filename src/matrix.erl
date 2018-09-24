%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    nif matrix api
%%% @end

-module(matrix).

%% -compile(native).
-on_load(init/0).
-export([create/4, create/5]).
-export([element/2, element/3]).
-export([setelement/4]).
-export([copy/1, copy/2, copy/4]).
-export([fill/2]).
-export([from_list/1, from_list/2, from_list/3, from_list/4]).
-export([to_list/1]).
-export([normal/1, uniform/1, zero/1, one/1, identity/1]).
-export([normal/2, uniform/2, zero/2, one/2, identity/2]).
-export([normal/3, uniform/3, zero/3, one/3, identity/3]).
-export([rep/3, rep/4]).
-export([constant/2, constant/3, constant/4]).

-export([cdata/2, cdata/3]).
-export([add/2,add/3]).
-export([subtract/2,subtract/3]).
-export([multiply/2,multiply/3]).
-export([kmultiply/3]).
-export([mulsum/2]).
-export([expsum/1]).
-export([sum/1, sum/2]).
-export([times/2,times/3]).
-export([ktimes/3]).
-export([scale/2, scale/3]).
-export([exp/1]).
-export([square/1]).
-export([pow/2]).
-export([negate/1, negate/2]).
-export([reciprocal/1, reciprocal/2]).
-export([size/1]).
-export([type/1]).
-export([signature/1]).
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
-export([transpose/1,transpose/2]).
-export([transpose_data/1, transpose_data/2]).
-export([print/1, print/2, format/1, format/2]).
-export([row/2, row/4]).
-export([column/2, column/4]).
-export([submatrix/5]).
-export([argmax/1, argmax/2, argmax/3]).
-export([argmin/1, argmin/2, argmin/3]).
-export([min/1, min/2, min/3]).
-export([max/1, max/2, max/3]).
-export([topk/2]).
-export([convolve/4, convolve/6]).
-export([rconvolve/4, rconvolve/6]).
-export([maxpool/3, maxpool/5, maxpool/6]).
-export([l2pool/3, l2pool/5, l2pool/6]).
-export([filter/2, filter/4, filter/5]).
-export([swap/4, swap/6]).
-export([sort/2, sort/3, sort/4]).

-export([invert/1, det/1]).

%% internal nifs
-export([create_/5, identity_/3]).
-export([kmultiply_/4, ktimes_/4]).

%% internal use
-export([encode_type/1]).
-export([setelement_/4]).
-export([element_/3]).
-export([element_/2]).
-export([type_combine/2]).
-export([elem_to_bin/2]).
-export([element_reciprocal/1, element_negate/1]).
-export([element_divide/2, element_multiply/2]).
%% debug
-export([foldl_row/4, foldr_row/4]).
-export([foldl_column/4, foldr_column/4]).
-export([foldl_matrix/3, foldr_matrix/3]).
-export([invert_l/1, invert_u/1]).

-export_type([matrix/0]).

%% maximum numbr of elements for add/sub/times/negate...
%% -define(MAX_NM, (256*4096)).
-define(MAX_ADD_NM, (10*10)).
-define(MAX_MUL_NM, (10*10)).
-define(EPS, 2.2204460492503131e-16).        %% float32 smallest step
-type compare_option() :: abs.
-type sort_option() :: abs|ascend|descend.


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

create_(_N,_M,_T,_RowMajor,_Data) ->
    ?nif_stub().

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
type_list([E|Es], T) ->
    type_list(Es, erlang:max(arg_type(E),T));
type_list([], T) ->
    T.

%%
%% Produce a list of lists representation of a matrix
%%
-spec to_list(X::matrix()) -> [[number()]].

to_list(A) ->
    {N,_} = size(A),
    [foldr_row(I, fun(Xij,Acc) -> [Xij|Acc] end, [], A) ||
	I <- lists:seq(1,N)].

-spec normal({N::unsigned(), M::unsigned()}) -> matrix();
	    ({N::unsigned(), M::unsigned(),T::matrix_type()}) -> matrix().
normal({N,M}) ->
    normal(N,M,float64);
normal({N,M,T}) ->
    normal(N,M,T).

-spec normal({N::unsigned(), M::unsigned()},T::matrix_type()) -> matrix().
normal({N,M},T) ->
    normal(N,M,T).

-spec normal(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
normal(N,M,T) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    Type = encode_type(T),
    A = create(N,M,Type,true,[]),
    apply1(normal, A, A).

-spec uniform({N::unsigned(), M::unsigned()}) -> matrix();
	     ({N::unsigned(), M::unsigned(),T::matrix_type()}) -> matrix().
uniform({N,M}) ->
    uniform(N,M,float64);
uniform({N,M,T}) ->
    uniform(N,M,T).

-spec uniform({N::unsigned(), M::unsigned()},T::matrix_type()) -> matrix().
uniform({N,M},T) ->
    uniform(N,M,T).

-spec uniform(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
uniform(N,M,T) when is_integer(N), N >= 1,
		    is_integer(M), M >= 1 ->
    Type = encode_type(T),
    A = create(N,M,Type,true,[]),
    apply1(uniform, A, A).

-spec zero({N::unsigned(), M::unsigned()}) -> matrix();
	  ({N::unsigned(), M::unsigned(), T::matrix_type()}) -> matrix().
zero({N,M}) ->
    zero(N,M,float64);
zero({N,M,T}) ->
    zero(N,M,T).

-spec zero({N::unsigned(), M::unsigned()}, T::matrix_type()) -> matrix().
zero({N,M},T) -> zero(N,M,T).

-spec zero(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
zero(N,M,Type) when is_integer(N), N >= 1,
		    is_integer(M), M >= 1 ->
    constant_(N, M, Type, 0).

-spec one({N::unsigned(), M::unsigned()}) -> matrix();
	 ({N::unsigned(), M::unsigned(), T::matrix_type()}) -> matrix().
one({N,M}) -> 
    one(N,M,float64);
one({N,M,T}) -> 
    one(N,M,T).

-spec one({N::unsigned(), M::unsigned()}, T::matrix_type()) -> matrix().
one({N,M},T) -> one(N,M,T).

-spec one(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
one(N,M,Type) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    constant_(N, M, Type, 1).

-spec constant({N::unsigned(),M::unsigned()},C::scalar()) -> matrix();
	      ({N::unsigned(),M::unsigned(),T::matrix_type()},C::scalar()) ->
		      matrix().
constant({N,M},C) ->
    constant_(N,M,arg_type(C),C);
constant({N,M,T},C) ->
    constant_(N,M,T,C).

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

constant_(N,M,Type,C) ->
    T = encode_type(Type),
    Bin = elem_to_bin(T,C),
    #matrix { type=T, n=N, m=M, nstep=0, mstep=0, rowmajor=true, resource=Bin }.

-spec rep(N::unsigned(), M::unsigned(), scalar()) -> matrix().
rep(N, M, C) when is_integer(C) ->
    rep(N,M,int32,C);
rep(N, M, C) when is_float(C) ->
    rep(N,M,int32,C);
rep(N, M, C) when ?is_complex(C) ->
    rep(N,M,complex64,C);
rep(N, 1, A=#matrix{n=1,rowmajor=true}) ->
    A#matrix { n=N, nstep=0 };
rep(1, M, A=#matrix{m=1,rowmajor=false}) ->
    A#matrix { m=M, mstep=0 };
rep(N, M, A) when ?is_matrix(A) ->
    rep_(N, M, A).

rep_(N, M, A) ->
    {Na,Ma,Ta} = signature(A),
    Dst = create(N*Na,M*Ma,Ta,[]),
    copy(A, Dst, N, M).

rep(N, M, Type, C) when is_atom(Type), ?is_scalar(C) ->
    T = encode_type(Type),
    Bin = elem_to_bin(T,C),
    #matrix { type=T, n=N, m=M, nstep=0, mstep=0, rowmajor=true, resource=Bin }.


-spec cdata(N::unsigned(), Data::[scalar()]) -> matrix().
cdata(N,X=#matrix{n=1,m=_M}) when N > 0 ->
    X#matrix { n=N, nstep=0 };
cdata(M,X=#matrix{m=1,n=_N}) when M > 0 ->
    X#matrix { m=M, mstep=0 };
cdata(N,CData) when is_integer(N), N>=0, is_list(CData) ->
    T = type_list(CData, -1),
    cdata_(N, T, CData).

cdata(N,Type,CData) when is_integer(N), N>=0, is_list(CData) ->
    cdata_(N, encode_type(Type), CData).

cdata_(N,T,CData) ->
    M = length(CData),
    Bin = list_to_binary([elem_to_bin(T,E) || E <- CData]),
    #matrix { type=T, n=N, m=M, nstep=0, mstep=1, rowmajor=true, resource=Bin }.

-spec identity({N::unsigned(), M::unsigned()}) -> matrix();
	      ({N::unsigned(), M::unsigned(), T::matrix_type()}) -> matrix().
identity({N,M}) ->
    identity(N,M,float64);
identity({N,M,T}) ->
    identity(N,M,T).

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

-spec apply1(A::matrix(), Op::atom()) -> matrix().
apply1(_A, _Op) ->
    ?nif_stub().

-spec apply1(A::matrix(), Dst::matrix(), Op::atom()) -> matrix().
apply1(_A, _Dst, _Op) ->
    ?nif_stub().

%% return dimension in row major order
-spec size(A::matrix()) -> {unsigned(), unsigned()}.
size(_A) ->
    ?nif_stub().

-spec type(A::matrix()) -> matrix_type().
type(#matrix_t{type=T}) -> decode_type(T);
type(#matrix{type=T}) -> decode_type(T).

-spec signature(A::matrix()) -> {unsigned(),unsigned(),matrix_type()}.
signature(#matrix{n=N,m=M,type=T,rowmajor=true}) ->
    {N,M,decode_type(T)};
signature(#matrix{n=N,m=M,type=T,rowmajor=false}) ->
    {M,N,decode_type(T)};
signature(A=#matrix_t{type=T}) ->
    {N,M} = size(A),
    {N,M,decode_type(T)}.

-spec is_integer_matrix(X::matrix()) -> boolean().
is_integer_matrix(#matrix_t{type=T}) -> 
    (T >= ?int8) andalso (T =< ?int64);
is_integer_matrix(#matrix{type=T}) -> 
    (T >= ?int8) andalso (T =< ?int64).

-spec is_float_matrix(X::matrix()) -> boolean().
is_float_matrix(#matrix_t{type=T}) ->
    (T >= ?float32) andalso (T =< ?float64);
is_float_matrix(#matrix{type=T}) ->
    (T >= ?float32) andalso (T =< ?float64).

-spec is_complex_matrix(X::matrix()) -> boolean().
is_complex_matrix(#matrix_t{type=T}) ->
    (T >= ?complex64) andalso (T =< ?complex128);
is_complex_matrix(#matrix{type=T}) ->
    (T >= ?complex64) andalso (T =< ?complex128).


%% element(I,A) -> A[I], row(I) or column(I) depening on rowmajor
%% element([[r1,r2,..,rm]],A) -> [[A[r1][1], A[r2][2], ... A[rm][m]]]
%% element([[c1],[c2],..,[cn]],A) -> [[A[1][c1]], [A[2][c2]], ... [A[n][cn]]]
%%
-spec element(I::unsigned(),J::unsigned(),X::matrix()) -> scalar().

element(I,J,A) ->
    element_(I,J,A).

element_(_I,_J,_A) ->
    ?nif_stub().

%% pick diagonal with:
%% Index = matrix:from_list([lists:seq(1,N)],int32),
%% Diag = element(Index, A),

-spec element(I::matrix(),A::matrix()) -> matrix();
	     ({I::unsigned(),J::unsigned()},A::matrix()) -> scalar().

element({I,J},A) when is_integer(I), is_integer(J) -> element(I,J,A);
element(I,A) -> element_(I,A).

element_(_I,_A) ->
    ?nif_stub().

%% DESTRUCTIVE
-spec setelement(I::unsigned(),J::unsigned(),X::matrix(),V::scalar()) ->
			matrix().
setelement(I,J,X,V) ->
    setelement_(I,J,X,V).

-spec setelement_(I::unsigned(),J::unsigned(),X::matrix(),V::scalar()) ->
			matrix().
setelement_(_I,_J,_X,_V) ->
    ?nif_stub().

foldl_matrix(F,A,X) ->
    {N,M} = size(X),
    foldl_rows(1,N,M,F,A,X).

foldl_rows(I,N,_M,_F,A,_X) when I > N -> A;
foldl_rows(I,N,M,F,A,X) ->
    A1 = foldl_row(I,F,A,X),
    foldl_rows(I+1,N,M,F,A1,X).

foldl_row(I,F,A,X) ->
    {_N,M} = size(X),
    foldl_row_(I,1,M,F,A,X).

foldl_row_(_I,J,M,_F,A,_X) when J > M -> A;
foldl_row_(I,J,M,F,A,X) -> 
    E = element(I,J,X),
    A1 = F(E,A),
    foldl_row_(I,J+1,M,F,A1,X).


foldr_matrix(F,A,X) ->
    {N,_M} = size(X),
    foldr_rows(1,N,F,A,X).

foldr_rows(I,N,_F,A,_X) when I > N -> A;
foldr_rows(I,N,F,A,X) ->
    A1 = foldr_row(I,F,A,X),
    foldr_rows(I+1,N,F,A1,X).

foldr_row(I,F,A,X) ->
    {_N,M} = size(X),
    foldr_row_(I,M,F,A,X).

foldr_row_(_I,0,_F,A,_X) -> A;
foldr_row_(I,J,F,A,X) -> 
    E = element(I,J,X),
    A1 = F(E,A),
    foldr_row_(I,J-1,F,A1,X).

foldl_column(J,F,A,X) ->
    {N,_M} = size(X),
    foldl_column_(1,J,N,F,A,X).

foldl_column_(I,_J,N,_F,A,_X) when I > N -> A;
foldl_column_(I,J,N,F,A,X) -> 
    E = element(I,J,X),
    A1 = F(E,A),
    foldl_column_(I+1,J,N,F,A1,X).

foldr_column(J,F,A,X) ->
    {N,_M} = size(X),
    foldr_column_(N,J,F,A,X).

foldr_column_(0,_J,_F,A,_X) -> A;
foldr_column_(I,J,F,A,X) -> 
    E = element(I,J,X),
    A1 = F(E,A),
    foldr_column_(I-1,J,F,A1,X).

%%
%% Add two matrices
%%
-spec add(A::matrix(), B::matrix()) -> matrix();
	 (A::matrix(), B::scalar()) -> matrix();
	 (A::scalar(), B::matrix()) -> matrix().

add(_A,_B) ->
    ?nif_stub().

%% destructive add
-spec add(A::matrix(), B::matrix(), Dst::matrix()) -> matrix();
	 (A::matrix(), B::scalar(), Dst::matrix()) -> matrix();
	 (A::scalar(), B::matrix(), Dst::matrix()) -> matrix().

add(_A, _B, _Dst) ->
    ?nif_stub().

%%
%% Subtract two matrices
%%

-spec subtract(A::matrix(), B::matrix()) -> matrix();
	      (A::matrix(), B::scalar()) -> matrix();
	      (A::scalar(), B::matrix()) -> matrix().

subtract(_A, _B) ->
    ?nif_stub().

%% destructive subtract
-spec subtract(A::matrix(), B::matrix(), Dst::matrix()) -> matrix();
	      (A::matrix(), B::scalar(), Dst::matrix()) -> matrix();
	      (A::scalar(), B::matrix(), Dst::matrix()) -> matrix().

subtract(_A,_B,_Dst) ->
    ?nif_stub().

%%
%% Multiply two matrices element wise
%%
-spec times(A::matrix(), B::matrix()) -> matrix();
	   (A::matrix(), B::scalar()) -> matrix();
	   (A::scalar(), B::matrix()) -> matrix().

times(_A,_B) ->
    ?nif_stub().

-spec times(A::matrix(), B::matrix(), Dst::matrix()) -> matrix();
	   (A::matrix(), B::scalar(), Dst::matrix()) -> matrix();
	   (A::scalar(), B::matrix(), Dst::matrix()) -> matrix().

times(_X,_Y,_Dst) ->
    ?nif_stub().

%%
%% Negate a matrix
%%
-spec negate(A::matrix()) -> matrix().
negate(_A) ->
    ?nif_stub().

-spec negate(A::matrix(),Dst::matrix()) -> matrix().
negate(_A, _Dst) ->
    ?nif_stub().

%%
%% Invert a matrix element wise
%%
-spec reciprocal(A::matrix()) -> matrix().
reciprocal(_A) ->
    ?nif_stub().

-spec reciprocal(A::matrix(),Dst::matrix()) -> matrix().
reciprocal(_A, _Dst) ->
    ?nif_stub().

%%
%% Scale a matrix by a scalar number
%%
-spec scale(F::scalar(), X::matrix()) -> matrix().

scale(F, X) when ?is_scalar(F) ->
    times(F, X).

-spec scale(F::number(), X::matrix(), Dst::matrix()) -> matrix().
scale(F, X, Dst) when ?is_scalar(F) ->
    times(F, X, Dst).

%% expontiation of all elements
-spec exp(X::matrix()) -> matrix().
exp(X) ->
    apply1(exp, X).

%%
%% Multiply elementwise and add everything
%%
-spec mulsum(X::matrix(), Y::matrix()) -> number().
mulsum(_X,_Y) ->
    ?nif_stub().

%% sum all elements in matrix
-spec sum(X::matrix()) -> number().
sum(_X) ->
    ?nif_stub().

-spec sum(A::matrix(), Axis::0..2) -> matrix().
sum(_A, _Axis) ->
    ?nif_stub().

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
multiply(_A, _B) ->
    ?nif_stub().

-spec multiply(X::matrix(), Y::matrix(), RowMajor::boolean) ->  matrix() ;
	      (X::matrix(), Y::matrix(), Dst::matrix()) -> matrix().

multiply(_A, _B, _Arg) ->
    ?nif_stub().

%% get the top K elements from A as a integer matrix Kwith indices
-spec topk(A::matrix(), K::unsigned()) -> matrix().
topk(_A,_K) ->
    ?nif_stub().    

%%
%% Multiply two matrices but only in rows from K
%%
-spec kmultiply(A::matrix(), B::matrix(), K::matrix()) -> matrix().

kmultiply(A, B, undefined) ->
    multiply(A, B);
kmultiply(A, B, K) ->
    kmultiply_(A, B, K).

-spec kmultiply_(A::matrix(), B::matrix(), K::matrix()) -> matrix().
kmultiply_(_A, _B, _K) ->
    ?nif_stub().

-spec kmultiply_(A::matrix(), B::matrix(), K::matrix(), C::matrix())-> matrix().
kmultiply_(_A, _B, _K, _C) ->
    ?nif_stub().

%%
%% Multiply two matrices element wise
%% (X o Y) o K
%%
-spec ktimes(A::matrix(), B::matrix(), K::matrix()) -> matrix().

ktimes(A, B, undefined) ->
    times(A, B);
ktimes(A, B, K) ->
    ktimes_(A, B, K).

-spec ktimes_(A::matrix(), B::matrix(), K::matrix()) -> matrix().
ktimes_(_A, _B, _K) ->
    ?nif_stub().

-spec ktimes_(X::matrix(), Y::matrix(), K::matrix(), C::matrix()) -> matrix().
ktimes_(_A, _B, _K, _C) ->
    ?nif_stub().

%% Transpose a matrix (as a submatrix to A)
-spec transpose(A::matrix()) -> matrix().
transpose(_A) ->
    ?nif_stub().

%% Transpose a matrix (into C)
-spec transpose(A::matrix(),C::matrix()) -> matrix().
transpose(_A,_C) ->
    ?nif_stub().

-spec transpose_data(Src::matrix()) -> matrix().
transpose_data(_Src) ->
    ?nif_stub().

-spec transpose_data(Src::matrix(),Dst::matrix()) -> matrix().
transpose_data(_Src, _Dst) ->
    ?nif_stub().

%%
%% select a row, return as a matrix with one row
%%
-spec row(I::unsigned(), A::matrix()) -> matrix().
row(I, A) ->
    {_N,M} = size(A),
    submatrix(I, 1, 1, M, A).

%% select part of a row as row vector
-spec row(I::unsigned(),J1::unsigned(),J2::unsigned(),A::matrix()) -> 
		 matrix().

row(I,J1,J2,A) when J1 =< J2 ->
    submatrix(I, J1, 1, J2-J1+1, A).

%%
%% select a column, return as a matrix with one column
%%
-spec column(J::unsigned(), A::matrix()) -> matrix().
column(J, A) ->
    {N,_M} = size(A),
    submatrix(1, J, N, 1, A).

-spec column(J::unsigned(),I1::unsigned(),I2::unsigned(),A::matrix()) -> 
		    matrix().
column(J,I1,I2,A) when I1 =< I2 ->
    submatrix(I1,J,I2-I1+1,1,A).

%%
%% select a portion of a matrix
%%
-spec submatrix(I::unsigned(), J::unsigned(),
		N::unsigned(), M::unsigned(),
		X::matrix()) -> matrix().

submatrix(_I,_J,_N,_M,_A) ->
    ?nif_stub().

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

maxpool(_N, _M, _Sn, _Sm, _A) ->
    ?nif_stub().

-spec maxpool(N::unsigned(),M::unsigned(),Sn::unsigned(),Sm::unsigned(),
	      A::matrix(),Dst::matrix()) ->
		     matrix().

maxpool(_N, _M, _Sn, _Sm, _A, _Dst) ->
    ?nif_stub().

%% l2pool
-spec l2pool(N::unsigned(),M::unsigned(),A::matrix()) -> matrix().
l2pool(N, M, A) ->
    l2pool(N, M, 1, 1, A).

%% l2pool
-spec l2pool(N::unsigned(),M::unsigned(),
	     Sn::unsigned(),Sm::unsigned(),A::matrix()) -> matrix().

l2pool(_N, _M, _Sn, _Sm, _A) ->
    ?nif_stub().

-spec l2pool(A::matrix(),N::unsigned(),M::unsigned(),
	     Sn::unsigned(),Sm::unsigned(),Dst::matrix()) -> matrix().

l2pool(_N, _M, _Sn, _Sm, _A, _Dst) ->
    ?nif_stub().

%%
%%
%%
-spec filter(W::matrix(), X::matrix()) -> matrix().
filter(W, X) ->
    filter(W, 1, 1, X).

-spec filter(W::matrix(), Sn::unsigned(), Sm::unsigned(), A::matrix()) ->
		    matrix().

filter(_W, _Sn, _Sm, _A) ->
    ?nif_stub().

-spec filter(W::matrix(), Sn::unsigned(), Sm::unsigned(), A::matrix(),
	     Dst::matrix()) -> matrix().

filter(_W, _Sn, _Sm, _A, _Dst) ->
    ?nif_stub().

-spec swap(K::integer(), I::integer(), A::matrix(), Axis::1..2) -> matrix().
%% swap row K and I if Axis = 1
%% swap column K and I if Axis = 2
swap(_K, _I, _A, _Axis) ->
    ?nif_stub().

-spec swap(K::integer(), I::integer(), J::integer(), L::integer(),
	   A::matrix(), Axis::1..2) -> matrix().
%% swap row K and I if Axis = 1  column from J to L
%% swap column K and L if Axis = 2  row from J to L
swap(_K, _I, _J, _L, _A, _Axis) ->
    ?nif_stub().

-spec sort(A::matrix(), K::integer()) -> 
		  matrix();
	  (A::matrix(), Elems::matrix()) -> 
		  matrix().
sort(A,K) -> 
    sort(A, K, 1, []).

-spec sort(A::matrix(), K::integer(), Axis::1..2) -> 
		  matrix();
	  (A::matrix(), Elems::matrix(), Axis::1..2) -> 
		  matrix().

sort(A, K, Axis) ->
    sort(A, K, Axis, []).

-spec sort(A::matrix(), K::integer(), Axis::1..2, Opts::[sort_option()]) -> 
		  matrix();
	  (A::matrix(), Elems::matrix(), Axis::1..2, Opts::[sort_option()]) -> 
		  matrix().

%% Sort columns according to row K if Axis = 1
%% Sort rows according to column K if Axis = 2
%% or sort according to row matrix Elems if Axis = 1
%% or sort according to column matrix Elems if Axis = 2

sort(_A, _K, _Axis, _Opts) ->
    ?nif_stub().    

%% pickout max element with (example)
%% Index = argmax(A,1),
%% Max = element(Index, A),

%% argmax
-spec argmax(A::matrix()) -> {I::integer(),J::integer()}.
argmax(A) ->
    argmax(A, 0, []).

-spec argmax(A::matrix(),Axis::0..2) -> matrix().
argmax(A,I) ->
    argmax(A,I,[]).

-spec argmax(A::matrix(),Axis::0..2,Opts::[compare_option()]) -> matrix().
argmax(_A,_I,_Opts) ->
    ?nif_stub().

%% argmin
-spec argmin(A::matrix()) -> {I::integer(),J::integer()}.
argmin(A) ->
    argmin(A,0,[]).

-spec argmin(A::matrix(),Axis::0..2) -> matrix().
argmin(A,I) ->
    argmin(A,I,[]).

-spec argmin(A::matrix(),Axis::0..2,Opts::[compare_option()]) -> matrix().
argmin(_A,_I,_Opts) ->
    ?nif_stub().

-spec max(A::matrix()) -> scalar().
max(A) ->
    max(A,0,[]).

-spec max(A::matrix(), Axis::0..2) -> matrix().
max(A, Axis) -> 
    max(A,Axis,[]).

-spec max(A::matrix(), Axis::0..2,Opts::[compare_option()]) -> matrix().
max(_A, _Axis, _Opts) ->
    ?nif_stub().

-spec min(A::matrix()) -> scalar().
min(A) ->
    min(A, 0, []).

-spec min(A::matrix(), Axis::0..2) -> matrix().
min(A, Axis) ->
    min(A, Axis, []).

-spec min(A::matrix(), Axis::0..2,Opts::[compare_option()]) -> matrix().
min(_A, _Axis, _Opts) ->
    ?nif_stub().

-spec sigmoid(A::matrix()) -> matrix().
sigmoid(_X) ->
    ?nif_stub().

-spec sigmoid_prime(A::matrix(),Out::matrix()) -> matrix().
%% Out = sigmoid(A)!!!  sigmoid_prime = Out*(1-Out)
sigmoid_prime(_A,_Out) ->
    ?nif_stub().

-spec relu(A::matrix()) -> matrix().
relu(A) ->
    apply1(relu, A).

-spec relu_prime(A::matrix(),Out::matrix()) -> matrix().
relu_prime(A,_Out) ->
    apply1(relu_prime, A).

-spec leaky_relu(A::matrix()) -> matrix().
leaky_relu(A) ->
    apply1(leaky_relu, A).

-spec leaky_relu_prime(A::matrix(),Out::matrix()) -> matrix().
leaky_relu_prime(A,_Out) ->
    apply1(leaky_relu_prime, A).

-spec linear(A::matrix()) -> matrix().
linear(A) ->
    A.

-spec linear_prime(A::matrix(),Out::matrix()) -> matrix().
linear_prime(A,_Out) ->
    one(size(A), type(A)).

-spec softplus(A::matrix()) -> matrix().
softplus(A) ->
    apply1(softplus, A).

-spec softplus_prime(A::matrix(),Out::matrix()) -> matrix().
softplus_prime(A,_Out) ->
    apply1(softplus_prime, A).

-spec softmax(A::matrix()) -> matrix().
softmax(A) ->
    A1 = subtract(A, max(A)),
    scale(element_reciprocal(expsum(A1)), apply1(exp,A1)).

-spec softmax_prime(A::matrix(),Out::matrix()) -> matrix().
softmax_prime(A,_Out) ->
    apply1(softmax_prime, A).


-spec tanh(A::matrix()) -> matrix().
tanh(A) ->
    apply1(tanh, A).

-spec tanh_prime(A::matrix(),Out::matrix()) -> matrix().
%% grad'(x) =
tanh_prime(_A,Out) ->
    apply1(tanh_prime1, Out).

-spec invert(A::matrix()) -> matrix() | false.

invert(A) ->
    {L,U,P,Pn} = matrix_lu:decompose(A),
    {N,_} = size(U),
    D = det_u(N,element_one(A),U),
    Abs = element_abs(D),
    if Abs >= ?EPS -> 
	    if Pn =:= 0 ->
		    multiply(invert_u(U), invert_l(L));
	       true ->
		    multiply(multiply(invert_u(U), invert_l(L)), P)
	    end;
       true ->
	    false
    end.

invert_u(U) ->
    U1 = transpose(U),
    U2 = invert_l(U1),
    transpose(U2).

invert_l(L) ->
    Signature = {N,_,_} = signature(L),
    ID = identity(Signature),
    invert_l_rows(1,N,L,ID).

%% row by row
invert_l_rows(I,N,A,X) when I =< N ->
    Xii = element(I,I,A),
    X1  = setelement(I,I,X,element_reciprocal(Xii)),
    X2  = invert_l_row(I, I-1, Xii, A, X1),
    invert_l_rows(I+1,N,A,X2);
invert_l_rows(_I,_N,_A,X) ->
    X.

invert_l_row(_I,0,_Xii,_A,X) ->
    X;
invert_l_row(I,J,Xii,A,X) ->
    K = I-1,
    Aik = row(I,J,K,A),     %% A[I][J]..A[I][K]
    print(Aik),
    Xkj = column(J,J,K,X),  %% X[J][J]..X[K][J]
    print(Xkj),
    S = mulsum(Aik,transpose(Xkj)),
    %% S = sum_l(I, J, I-1, A, X, element_zero(A)),
    Sneg = element_negate(S),
    X1 = setelement(I,J,X,element_divide(Sneg,Xii)),  %% destructive!
    invert_l_row(I,J-1,Xii,A,X1).

sum_l(_I,J,K,_A,_X,Sum)  when K < J->
    Sum;
sum_l(I,J,K,A,X,Sum) ->
    Aik = element(I,K,A),
    Xkj = element(K,J,X),
    sum_l(I,J,K-1,A,X,element_add(element_multiply(Aik,Xkj),Sum)).
    
-spec det(A::matrix()) -> scalar().

-define(pow_sign(N), ((((N)+1) band 1)*2 - 1)).

det(A) ->
    {_L,U,_P,Pn} = matrix_lu:decompose(A),
    {N,_,T} = signature(U),
    D = det_u(N,elem_one(T),U),       %% multiply diagonal
    element_multiply(D,?pow_sign(Pn)).   %% with -1^Pn
    
det_u(0,D,_U) -> D;
det_u(I,D,U) -> det_u(I-1,element_multiply(D,element(I,I,U)),U).

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

%% get argument type
arg_type(X) ->
    if is_integer(X) ->
	    if X >= -16#80, X < 16#80 -> ?int8;
	       X >= -16#8000, X < 16#8000 -> ?int16;
	       X >= -16#80000000, X < 16#80000000 -> ?int32;
	       true -> ?int64
	    end;
       is_float(X) -> ?float64;
       ?is_complex(X) -> ?complex128
    end.

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

encode_type(int8) -> ?int8;
encode_type(int16) -> ?int16;
encode_type(int32) -> ?int32;
encode_type(int64) -> ?int64;
encode_type(float32) -> ?float32;
encode_type(float64) -> ?float64;
encode_type(complex64) -> ?complex64;
encode_type(complex128) -> ?complex128.

decode_type(T) ->
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


element_divide(A,B) when is_integer(A), is_integer(B), A rem B =:= 0 ->
    A div B;
element_divide(A,B) when is_number(A), is_number(B) ->
    A / B;
element_divide({A1,B1},{A2,B2}) ->
    D = A2*A2 + B2*B2,
    {(A1*A2 + B1*B2) / D, -((A1*B2 + A2*B1) / D)}.

element_reciprocal(A) when is_number(A) -> 1/A;
element_reciprocal({A2,B2}) ->
    D = A2*A2 + B2*B2,
    {(A2) / D, -((B2) / D)}.
    
element_negate(A) when is_number(A) -> -A;
element_negate({A,B}) -> {-A,-B}.

element_multiply(A,B) when is_number(A), is_number(B) -> A*B;
element_multiply({A1,B1},{A2,B2}) ->
    {A1*A2-B1*B2,A1*B2+A2*B1};
element_multiply({A1,B1},C) when is_number(C) ->
    {A1*C, B1*C};
element_multiply(C, {A2,B2}) when is_number(C) ->
    {C*A2,C*B2}.

element_add(A,B) when is_number(A), is_number(B) -> A+B;
element_add({A1,B1},{A2,B2}) ->
    {A1+A2,B1+B2};
element_add({A1,B1},C) when is_number(C) ->
    {A1+C, B1};
element_add(C, {A2,B2}) when is_number(C) ->
    {C+A2,B2}.

element_subtract(A,B) when is_number(A), is_number(B) -> A-B;
element_subtract({A1,B1},{A2,B2}) ->
    {A1-A2,B1-B2};
element_subtract({A1,B1},C) when is_number(C) ->
    {A1-C, B1};
element_subtract(C, {A2,B2}) when is_number(C) ->
    {C-A2,B2}.

element_abs(C) when is_number(C) -> abs(C);
element_abs({A,B}) -> math:sqrt(A*A+B*B).

element_zero(#matrix_t{type=T}) -> elem_zero(T);
element_zero(#matrix{type=T}) -> elem_zero(T).

element_one(#matrix_t{type=T}) -> elem_one(T);
element_one(#matrix{type=T}) -> elem_one(T).

elem_zero(T) ->
    case T of
	?int8 -> 0;
	?int16 -> 0;
	?int32 -> 0;
	?int64 -> 0;
	?float32 -> 0.0;
	?float64 -> 0.0;
	?complex64 -> {0.0, 0.0};
	?complex128 -> {0.0, 0.0}
    end.

elem_one(T) ->
    case T of
	?int8 -> 1;
	?int16 -> 1;
	?int32 -> 1;
	?int64 -> 1;
	?float32 -> 1.0;
	?float64 -> 1.0;
	?complex64 -> {1.0, 0.0};
	?complex128 -> {1.0, 0.0}
    end.
