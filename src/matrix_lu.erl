%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2018, Tony Rogvall
%%% @doc
%%%    basic decompose matrix
%%% @end
%%% Created :  3 Sep 2018 by Tony Rogvall <tony@rogvall.se>

-module(matrix_lu).
-export([decompose/1, decompose/2]).

-export([itest/0,itest/1,ftest/0,ftest/1,test/1]). %% debug test
-export([test_matrix/2]).   %% debug test

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
    decompose1(matrix:copy(A), matrix:signature(A)).
decompose(A,U0) ->
    decompose1(matrix:copy(A,U0), matrix:signature(A)).

decompose1(A0,Sig={N,N,T}) ->
    L0 = matrix:identity(Sig),
    P0 = matrix:identity(Sig),
    U0 = A0,
    case matrix:is_integer_matrix(A0) of
	true ->
	    idecompose_(L0,U0,P0,0,1,N);
	false ->
	    case matrix:is_float_matrix(A0) of
		true -> decompose_(L0,U0,P0,0,1,N);
		false -> error({not_supported_matrix_type,{N,N,T}})
	    end
    end;
decompose1(_A,{N,M}) ->
    error({not_a_square_matrix,{N,M}}).


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

eliminate(_Akk,_Rii,L,U,I,_J,N) when I > N ->
    {L,U};
eliminate(Akk,Rii,L,U,I,J,N) ->
    Aij = matrix:element(I,J,U),
    F = matrix:element_divide(Aij, Akk),
    Fneg = matrix:element_negate(F),
    %% io:format("Akk=~p,Aij=~p,F=~p\n",[Akk,Aij,F]),
    Ui = matrix:row(I,U),
    matrix:add(Ui, matrix:times(Rii,Fneg), Ui),
    matrix:setelement(I,J,L,F),
    eliminate(Akk,Rii,L,U,I+1,J,N).

%% find the value in column I row I..N with the largest absolute value
select_pivot(U,K,N) ->
    UC = matrix:submatrix(K,K,N-K+1,1,U),
    {I0,_J} = matrix:argmax(UC,0,[abs]),
    I = K+I0-1,
    Uik = matrix:element(I,K,U),
    {Uik,I}.

%% 
idecompose_(L,U,P,Pn,K,M) when K>=M ->
    {L,U,P,Pn};
idecompose_(L,U,P,Pn,K,M) ->
    case iselect_pivot(U,K,M) of
	{Ukk,K} ->
	    %% FIXME: Ri=submatrix(K,K,1,M-K+1,U)
	    {L1,U1} = ieliminate(Ukk,
				 matrix:row(K,U),
				 matrix:row(K,L),
				 L,U,K+1,K,M),
	    idecompose_(L1,U1,P,Pn,K+1,M);
	{Ukk,I} ->
	    U1 = matrix:swap(K,I,K,M,U,1),
	    L1 = matrix:swap(K,I,1,K-1,L,1),
	    P1 = matrix:swap(K,I,P,1),
	    %% FIXME: Ri=submatrix(K,K,1,M-I+1,U)
	    {L2,U2} = ieliminate(Ukk,
				 matrix:row(K,U1),
				 matrix:row(K,L1),
				 L1,U1,K+1,K,M),
	    idecompose_(L2,U2,P1,Pn+1,K+1,M)
    end.

ieliminate(_Akk,_Rk,_Qk,L,U,I,_J,N) when I > N ->
    {L,U};
ieliminate(Akk,Rk,Qk,L,U,I,J,N) ->
    Aij = matrix:element(I,J,U),
    Lcm = lcm(Akk,Aij),
    Fi = Lcm div Aij,
    Fk = Lcm div Akk,
    io:format("Akk=~p,Aij=~p,Lcm=~p Fi=~p Fk=~p\n",[Akk,Aij,Lcm,Fi,Fk]),
    Ui = matrix:row(I,U),
    matrix:times(Rk, Fk, Rk),  %% scale pivot row
    matrix:times(Ui,-Fi, Ui),
    matrix:add(Ui,Rk,Ui),

    Li = matrix:row(I,L),
    matrix:times(Qk,Fk,Qk),
    matrix:times(Li,-Fi,Li),
    matrix:add(Li,Qk,Li),

    Akk1 = Akk*Fk,
    io:format("U\n"),matrix:print(U),
    io:format("L\n"),matrix:print(L),
    ieliminate(Akk1,Rk,Qk,L,U,I+1,J,N).

%% find the value in column I row I..N with the smallest absolute value
iselect_pivot(U,K,N) ->
    UC = matrix:submatrix(K,K,N-K+1,1,U),
    {I0,_J} = matrix:argmin(UC,0,[abs]),
    I = K+I0-1,
    Uik = matrix:element(I,K,U),
    {Uik,I}.

gcd(R, Q) when is_integer(R), is_integer(Q) ->
    R1 = abs(R),
    Q1 = abs(Q),
    if Q1 < R1 -> gcd_(Q1,R1);
       true -> gcd_(R1,Q1)
    end.

gcd_(0, Q) -> Q;
gcd_(R, Q) ->
    gcd_(Q rem R, R).

%%
%% Least common multiple of (R,Q)
%%
lcm(0, _Q) -> 0;
lcm(_R, 0) -> 0;
lcm(R, Q) ->
    (Q div gcd(R, Q)) * R.

%% small test
itest() -> itest(3).
itest(N) ->
    A = test_matrix(N,int32),
    test(A).

ftest() -> ftest(3).
ftest(N) ->
    A = test_matrix(N,float32),
    test(A).

test(A) ->
    io:format("A=\n~s\n",[matrix:format(A)]),
    {L,U,P,Pn} = decompose(A),
    io:format("L=\n~s\n",[matrix:format(L)]),
    io:format("U=\n~s\n",[matrix:format(U)]),
    io:format("P=\n~s\n",[matrix:format(P)]),
    io:format("Pn=~p\n",[Pn]),
    LU = matrix:multiply(L,U),
    io:format("LU=\n~s\n", [matrix:format(LU)]),
    PA = matrix:multiply(P,A),
    io:format("AP=\n~s\n", [matrix:format(PA)]).

test_matrix(3,Type) ->
    matrix:from_list([[1,2,3],[7,9,8],[2,2,2]], Type);
test_matrix(4,Type) ->
    matrix:from_list([[1,2,3,4],[7,9,8,6],[2,2,2,2],[9,1,2,5]], Type).
