-module(vecf).

-export([new/1, new/3, new/4,
	 zero/0,
	 one/0, 
	 uniform/0, 
	 element/2,setelement/3,
	 is_vector/1
	]).

-export([add/2,
	 subtract/2,
	 multiply/2, 
	 divide/2, 
	 invert/1, invert/2,
	 negate/1,
	 dot/2,
	 cross/2,
	 normalize/1,
	 len/1, len2/1,
	 manhattan/2,
	 distance/2, distance2/2,
	 min/2, min/1, minarg/1,
	 max/2, max/1, maxarg/1
	]).
-export([add/1]).
-export([to_tuple/1, to_list/1]).
-export([print/1,print/2,print/3]).
-export([format/1,format/2,format/3]).

-include_lib("matrix/include/matrix.hrl").

-type vec4f() :: matrix:matrix(). %% 1x4 (?ERAY_TYPE)
-export_type([vec4f/0]).
-define(ELEM_TYPE, float32).
-define(BIN_ELEM(X), ?float32(X)).
-define(EPS, 1.0e-8).  %% small number

-spec new(X::scalar(),Y::scalar(),Z::scalar(),W::scalar()) -> vec4f().
new(X,Y,Z,W) -> 
    matrix:create(1, 4, ?ELEM_TYPE,
		  <<?BIN_ELEM(X),
		    ?BIN_ELEM(Y),
		    ?BIN_ELEM(Z),
		    ?BIN_ELEM(W)>>).

-spec new(X::scalar(),Y::scalar(),Z::scalar()) -> vec4f().
new(X,Y,Z) -> 
    matrix:create(1, 4, ?ELEM_TYPE,
		  <<?BIN_ELEM(X),
		    ?BIN_ELEM(Y),
		    ?BIN_ELEM(Z),
		    ?BIN_ELEM(0)>>).

-spec new(A::scalar()) -> vec4f();
	 (A::tuple()) -> vec4f();
	 (A::[scalar]) -> vec4f().
new(A) when is_number(A) ->
    new(A,A,A);
%%new({X,Y,Z,W}) -> new(X,Y,Z,W);
%%new([X,Y,Z,W]) -> new(X,Y,Z,W).
new({X,Y,Z})   -> new(X,Y,Z);
new([X,Y,Z])   -> new(X,Y,Z).


-spec zero() -> vec4f().
zero() -> new(0,0,0).

-spec one() -> vec4f().
one()  -> new(1,1,1).

-spec uniform() -> vec4f().
uniform() -> 
    matrix:uniform(1, 4, ?ELEM_TYPE).

-spec element(I::integer(), Vec::vec4f()) -> scalar().
element(I, Vec) ->
    matrix:element(1, I, Vec).

%% DESTRUCTIVE
-spec setelement(I::integer(), Vec::vec4f(), E::scalar()) -> vec4f().
setelement(I, Vec, E) ->
    matrix:setelement(1, I, Vec, E).

-spec is_vector(A::term()) -> boolean().
is_vector(A) ->
    ?is_matrix(A).

-spec to_tuple(Vec::vec4f()) -> tuple().
to_tuple(Vec) ->
    [Es] = matrix:to_list(Vec),
    list_to_tuple(Es).

-spec to_list(Vec::vec4f()) -> [scalar()].
to_list(A) ->
    [Es] = matrix:to_list(A),
    Es.

print(A) ->
    matrix:print(A).
print(A,Prec) ->
    matrix:print(A,Prec).
print(A,Prec,BENS) ->
    matrix:print(A,Prec,BENS).

format(A) ->
    matrix:format(A).
format(A,Prec) ->
    matrix:format(A,Prec).
format(A,Prec,BENS) ->
    matrix:format(A,Prec,BENS).

add([]) -> zero();
add([A]) -> A;
add([A|As]) -> add(A,add(As)).
    
add(A, B) -> matrix:add(A, B).

subtract(A, B) -> matrix:subtract(A, B).

multiply(A, B) -> matrix:times(A, B).

divide(A, B) -> matrix:divide(A, B).

invert(A) ->
    invert(A, ?EPS).

invert(A,Eps) ->
    [Ax,Ay,Az|_] = to_list(A),
    new(if abs(Ax) < Eps -> 0; true -> 1/Ax end,
	if abs(Ay) < Eps -> 0; true -> 1/Ay end,
	if abs(Az) < Eps -> 0; true -> 1/Az end).
%% matrix:reciprocal(A).

negate(A) -> matrix:negate(A).

%% fixme: accelerated code exist in matrix but is not accessible, yet?
%% or ktimes?
dot(A, B) -> 
    A_B = matrix:multiply(A, matrix:transpose(B)),
    matrix:element(1,1,A_B).

%% fixme: 
cross(A, B) ->
    [[Ax,Ay,Az|_]] = matrix:to_list(A),
    [[Bx,By,Bz|_]] = matrix:to_list(B),
    new(Ay*Bz-Az*By, Az*Bx-Ax*Bz, Ax*By-Ay*Bx).

%% ( ai / len(A) )
normalize(A) ->
    N = len(A),
    matrix:scale(1/N, A).

%% SQRT SUM ai^2
len(A) ->
    math:sqrt(len2(A)).

len2(A) ->
%%     matrix:sum(matrix:times(A,A)). %% assume w=0
    [Ax,Ay,Az|_] = to_list(A),
    Ax*Ax + Ay*Ay + Az*Az.

distance2(A,B) ->
    len2(matrix:subtract(B, A)).

distance(A,B) ->
    math:sqrt(distance2(A,B)).

%% SUM | bi - ai |
manhattan(A,B) ->
    D = matrix:subtract(A, B),  
    matrix:abs(D).

minarg(A) -> {_,J} = matrix:argmin(A), J.

min(A,B) -> min_(A,B).
min_(A,B) -> matrix:minimum(A,B).

maxarg(A) -> {_,J} = matrix:argmax(A), J.

max(A,B) -> max_(A,B).
max_(A,B) -> matrix:maximum(A,B).

min([A|As]) -> min_list(As,A);
min(At) when is_tuple(At) -> min_tuple(At, 2, erlang:element(1,At)).
    
min_list([B|Bs],A) -> min_list(Bs,min_(B,A));
min_list([],A) -> A.

min_tuple(At,I,A) when I =< tuple_size(At) -> 
    min_tuple(At,I+1,min_(erlang:element(I,At),A));
min_tuple(_At,_I,A) -> A.

max([A|As]) -> max_list_(As,A);
max(At) when is_tuple(At) -> max_tuple_(At, 2, erlang:element(1,At)).

max_list_([B|Bs],A) -> max_list_(Bs,max_(B,A));
max_list_([],A) -> A.

max_tuple_(At,I,A) when I =< tuple_size(At) -> 
    max_tuple_(At,I+1,max_(erlang:element(I,At),A));
max_tuple_(_At,_I,A) -> A.
