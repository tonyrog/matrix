%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2018, Tony Rogvall
%%% @doc
%%%    basic decompose matrix
%%% @end
%%% Created :  3 Sep 2018 by Tony Rogvall <tony@rogvall.se>

-module(matrix_lu).
-export([decompose/1, decompose/2]).
%% debug
-export([test/0]).

-include("matrix.hrl").

-define(MIN_FLOAT, 2.2250738585072014e-308). %% min non-negative float
-define(MAX_FLOAT, 1.7976931348623157e+308). %% max positive float
-define(EPS, 2.2204460492503131e-16).        %% float32 smallest step

test() ->
    A = matrix:uniform(4,4,float32),
    io:format("A=\n"), matrix:print(A),
    {L,U,P,Pn} = decompose(A),
    io:format("L=\n"), matrix:print(L),
    io:format("U=\n"), matrix:print(U),
    io:format("P=\n"), matrix:print(P),
    P1 = matrix:transpose(P),
    io:format("P'=\n"), matrix:print(P1),
    io:format("Pn=~w\n",[Pn]),

    LU = matrix:multiply(L,U),
    PA = matrix:multiply(P,A),
    io:format("PA=\n"), matrix:print(PA),
    io:format("LU=\n"), matrix:print(LU),
    io:format("LUP'=\n"), matrix:print(matrix:multiply(LU,P1)),
    {L,U,P,Pn}.


%% Decompose square matrix A into L*U where 
%% L is lower triangular and U is upper triangular
%% return L,U,P,Pn  where P is the permutation matrix
%% Pn is number of raw swaps used to calculate LU
-spec decompose(A::matrix()) ->
		       {L::matrix(),U::matrix(),P::matrix(),Pn::integer()}.

decompose(A) ->
    Signature = {N,N,_T} = matrix:signature(A),
    L0 = matrix:identity(Signature),
    P0 = matrix:identity(Signature),
    U0 = matrix:copy(A),
    decompose_(L0,U0,P0,0,1,N).
    
decompose(A,U0) ->
    Signature = {N,N,_T} = matrix:signature(A),
    L0 = matrix:identity(Signature),
    P0 = matrix:identity(Signature),
    matrix:copy(A,U0),
    decompose_(L0,U0,P0,0,1,N).

decompose_(L,U,P,Pn,I,N) when I>N ->
    {L,U,P,Pn};
decompose_(L,U,P,Pn,I,N) ->
    case find_none_zero_element(U,I,N) of
	false -> 
	    false;
	{Aii,I} ->
	    %% FIXME: Ri=submatrix(I,I,1,N-I+1,U)
	    {L1,U1} = eliminate(Aii,matrix:row(I,U),L,U,I+1,I,N),
	    decompose_(L1,U1,P,Pn,I+1,N);
	{Aii,I1} ->
	    L1 = L,
	    U1 = matrix:swap(I,I1,U,1),
	    P1 = matrix:swap(I,I1,P,1),
	    %% FIXME: Ri=submatrix(I,I,1,N-I+1,U)
	    {L2,U2} = eliminate(Aii,matrix:row(I,U1),L1,U1,I+1,I,N),
	    decompose_(L2,U2,P1,Pn+1,I+1,N)
    end.

eliminate(_Aii,_Rii,L,U,I,_J,N) when I > N ->
    {L,U};
eliminate(Aii,Rii,L,U,I,J,N) ->
    Aij = matrix:element(I,J,U),
    %% Try to keep integer factors if possible
    Factor = if is_integer(Aii), is_integer(Aij), Aij rem Aii =:= 0 ->
		     -(Aij div Aii);
		true -> %% fixme complex!
		     -(Aij / Aii)
	     end,
    Ui = matrix:row(I,U), %% fixme: Ui=submatrix(I,J,1,N-J+1,U)?
    matrix:add(Ui, matrix:times(Rii,Factor), Ui),
    matrix:setelement(I,J,L,-Factor),
    eliminate(Aii,Rii,L,U,I+1,J,N).

%% find the value in column I row I..N with the largest absolute value
find_none_zero_element(U,I,N) ->
    UC = matrix:submatrix(I,I,N-I+1,1,U),
    {I1,_J} = matrix:argmax(UC,0,[abs]),
    II = I+I1-1,
    Aij = matrix:element(II,I,U),
    if abs(Aij) =< ?EPS -> false;
       true -> {Aij,II}
    end.
