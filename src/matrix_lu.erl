%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2018, Tony Rogvall
%%% @doc
%%%    basic decompose matrix
%%% @end
%%% Created :  3 Sep 2018 by Tony Rogvall <tony@rogvall.se>

-module(matrix_lu).
-export([decompose/1, decompose/2]).

-include("../include/matrix.hrl").

-define(MIN_FLOAT, 2.2250738585072014e-308). %% min non-negative float
-define(MAX_FLOAT, 1.7976931348623157e+308). %% max positive float
-define(EPS, 2.2204460492503131e-16).        %% float32 smallest step


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
    Signature = {M,M,_T} = matrix:signature(A),
    L0 = matrix:identity(Signature),
    P0 = matrix:identity(Signature),
    matrix:copy(A,U0),
    decompose_(L0,U0,P0,0,1,M).

decompose_(L,U,P,Pn,K,M) when K>=M ->
    {L,U,P,Pn};
decompose_(L,U,P,Pn,K,M) ->
    case select_pivot(U,K,M) of
	{Ukk,K} ->
	    %% FIXME: Ri=submatrix(K,K,1,M-K+1,U)
	    {L1,U1} = eliminate(Ukk,matrix:row(K,U),L,U,K+1,K,M),
	    decompose_(L1,U1,P,Pn,K+1,M);
	{Ukk,I} ->
	    U1 = matrix:swap(K,I,K,M,U,1),
	    L1 = matrix:swap(K,I,1,K-1,L,1),
	    P1 = matrix:swap(K,I,P,1),
	    %% FIXME: Ri=submatrix(K,K,1,M-I+1,U)
	    {L2,U2} = eliminate(Ukk,matrix:row(K,U1),L1,U1,K+1,K,M),
	    decompose_(L2,U2,P1,Pn+1,K+1,M)
    end.

eliminate(_Aii,_Rii,L,U,I,_J,N) when I > N ->
    {L,U};
eliminate(Aii,Rii,L,U,I,J,N) ->
    Aij = matrix:element(I,J,U),
    F = matrix:element_divide(Aij, Aii),
    Fneg = matrix:element_negate(F),
    Ui = matrix:row(I,U), %% fixme: Ui=submatrix(I,J,1,N-J+1,U)?
    matrix:add(Ui, matrix:times(Rii,Fneg), Ui),
    matrix:setelement(I,J,L,F),
    eliminate(Aii,Rii,L,U,I+1,J,N).

%% find the value in column I row I..N with the largest absolute value
select_pivot(U,K,N) ->
    UC = matrix:submatrix(K,K,N-K+1,1,U),
    {I0,_J} = matrix:argmax(UC,0,[abs]),
    I = K+I0-1,
    Uik = matrix:element(I,K,U),
    {Uik,I}.
