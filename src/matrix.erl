%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    flat tuple matrix version
%%% @end
%%% Created : 22 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(matrix).

%% -compile(native).

-export([normal/1, uniform/1, zero/1, identity/1]).
-export([normal/2, uniform/2, zero/2, identity/2]).
-export([add/2]).
-export([subtract/2]).
-export([multiply/2, scale/2]).
-export([size/1]).
-export([element/3]).
-export([sigmoid/1]).
-export([sigmoid_prime/1]).
-export([rectifier/1]).
-export([softplus/1]).
-export([transpose/1]).
-export([print/1, format/1, format/2]).
-export([row/2, column/2, sub_matrix/5]).
-export([convolve/4, convolve/6]).
-export([max/3, max/5, l2/3, l2/5]).
-export([filter/3, filter/5]).
-export([fold_elems/7]).
-export([bench/1]).

-type unsigned() :: non_neg_integer().
-type matrix() :: {unsigned(), unsigned(), tuple(number())}.

-spec normal({N::unsigned(), M::unsigned()}) -> matrix().
normal({N,M}) ->
    normal(N,M).

-spec normal(N::unsigned(), M::unsigned()) -> matrix().
normal(N,M) when is_integer(N), N >= 1,
		 is_integer(M), M >= 1 ->
    {N,M,list_to_tuple(deep_random:normal_vector(N*M))}.

-spec uniform({N::unsigned(), M::unsigned()}) -> matrix().
uniform({N,M}) ->
    uniform(N,M).

-spec uniform(N::unsigned(), M::unsigned()) -> matrix().
uniform(N,M) when is_integer(N), N >= 1,
		  is_integer(M), M >= 1 ->
    {N,M,list_to_tuple(deep_random:uniform_vector(N*M))}.

-spec zero({N::unsigned(), M::unsigned()}) -> matrix().
zero({N,M}) ->
    zero(N,M).

-spec zero(N::unsigned(), M::unsigned()) -> matrix().
zero(N,M) when is_integer(N), N >= 1,
	       is_integer(M), M >= 1 ->
    {N,M,erlang:make_tuple(N*M, 0.0)}.

-spec identity({N::unsigned(), M::unsigned()}) -> matrix().
identity({N,M}) ->
    identity(N,M).

-spec identity(N::unsigned(), M::unsigned()) -> matrix().
identity(N,M) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    {N,M,list_to_tuple(
	   lists:append([[float(X) || <<X:1>> <= <<(1 bsl ((M-1)-I)):M>>] ||
			    I <- lists:seq(0, N-1)]))}.

-spec size(M::matrix()) -> {unsigned(), unsigned()}.
size({N,M,_}) ->
    {N,M}.

-spec element(I::unsigned(),J::unsigned(),X::matrix()) -> number().
element(I,J,{_N,M,Xt}) ->
    element((I-1)*M+J, Xt).
     
-spec add(A::matrix(), B::matrix()) -> matrix().

add({2,2,{X11,X12, X21,X22}},
    {2,2,{Y11,Y12, Y21,Y22}}) ->
    {2,2,{X11+Y11, X12+Y12, X21+Y21, X22+Y22}};
add({3,3,{X11,X12,X13, X21,X22,X23, X31,X32,X33}},
    {3,3,{Y11,Y12,Y13, Y21,Y22,Y23, Y31,Y32,Y33}}) ->
    {3,3,
     {X11+Y11, X12+Y12, X13+Y13,
      X21+Y21, X22+Y22, X23+Y23,
      X31+Y31, X32+Y32, X33+Y33}};
add({N,M,X},{N,M,Y}) ->
    {N,M,list_to_tuple(add_(X,Y,N*M,[]))}.

add_(_Xt,_Yt,0,Acc) ->
    Acc;
add_(Xt,Yt,I,Acc) ->
    add_(Xt,Yt,I-1,[element(I,Xt)+element(I,Yt)|Acc]).

%% multiply elementwise and add everythin
-spec mulsum(A::matrix(), B::matrix()) -> matrix().
mulsum({N,M,X},{N,M,Y}) ->
    mulsum_(X,Y,N*M,0).

mulsum_(_Xt,_Yt,0,Sum) ->
    Sum;
mulsum_(Xt,Yt,I,Sum) ->
    mulsum_(Xt,Yt,I-1,element(I,Xt)*element(I,Yt)+Sum).

-spec subtract(A::matrix(), B::matrix()) -> matrix().

subtract({N,M,X},{N,M,Y}) ->
    {N,M,list_to_tuple(sub_(X,Y,N*M,[]))}.

sub_(_Xt,_Yt,0,Acc) ->
    Acc;
sub_(Xt,Yt,I,Acc) ->
    sub_(Xt,Yt,I-1,[element(I,Xt)-element(I,Yt)|Acc]).

-spec scale(F::number(), A::matrix()) -> matrix().
scale(F, {N,M,Xt}) when is_number(F) ->
    {N,M,list_to_tuple(scl_(F,Xt,N*M,[]))}.

scl_(_F,_Xt,0,Acc) ->
    Acc;
scl_(F,Xt,I,Acc) ->
    scl_(F,Xt,I-1,[F*element(I,Xt)|Acc]).

-spec multiply(X::matrix(), Y::matrix()) -> matrix().

multiply({2,2,{X11,X12,
	       X21,X22}},
	 {2,2,{Y11,Y12,
	       Y21,Y22}}) ->
    {2,2,{X11*Y11+X12*Y21, X11*Y12+X12*Y22,
	  X21*Y11+X22*Y21, X21*Y12+X22*Y22}};
multiply({3,3,{X11,X12,X13,
	       X21,X22,X23,
	       X31,X32,X33}},
	 {3,3,{Y11,Y12,Y13,
	       Y21,Y22,Y23,
	       Y31,Y32,Y33}}) ->
    {3,3,
     {X11*Y11+X12*Y21+X13*Y31,X11*Y12+X12*Y22+X13*Y32,X11*Y13+X12*Y23+X13*Y33,
      X21*Y11+X22*Y21+X23*Y31,X21*Y12+X22*Y22+X23*Y32,X21*Y13+X22*Y23+X23*Y33,
      X31*Y11+X32*Y21+X33*Y31,X31*Y12+X32*Y22+X33*Y32,X31*Y13+X32*Y23+X33*Y33}};
multiply({N,M,Xt}, {M,N,Yt}) ->
    {N,N,list_to_tuple(mult_(Xt,(N-1)*M+1,M, Yt,N,N, N*N, []))}.

mult_(_Xt,_I,_M, _Yt,_J,_N,0,Acc) ->
    Acc;
mult_(Xt,I,M, Yt,0,N,K,Acc) ->
    mult_(Xt,I-M, M, Yt,N,N, K, Acc);
mult_(Xt,I,M, Yt,J,N,K,Acc) ->
    C = dot_(Xt,I,Yt,J,N,M,0),
    mult_(Xt,I,M, Yt,J-1,N,K-1,[C|Acc]).

dot_(_Xt,_I,_Yt,_J,_N,0,Sum) ->
    Sum;
dot_(Xt,I,Yt,J,N,K,Sum) ->
    dot_(Xt,I+1,Yt,J+N,N,K-1,element(I,Xt)*element(J,Yt)+Sum).

-spec transpose(A::matrix()) -> matrix().
transpose({N,M,Xt}) ->
    NM = N*M,
    {M,N,list_to_tuple(trans_(Xt,NM,NM,N,M,[]))}.

trans_(Xt,I,J,N,M,Acc) when I =< 0 ->
    if I =:= -(M-1) -> Acc;
       true ->
	    J1 = J-1,
	    trans_(Xt,J1,J1,N,M,Acc)
    end;
trans_(Xt,I,J,N,M,Acc) ->
    trans_(Xt,I-M,J,N,M,[element(I,Xt)|Acc]).

%% select a row, return as a matrix with one row
-spec row(I::unsigned(), A::matrix()) -> matrix().
row(I, X={_N,M,_Xt}) ->
    sub_matrix(I, 1, 1, M, X).

%% select a column, return as a matrix with one column
-spec column(J::unsigned(), A::matrix()) -> matrix().
column(J, X={N,_M,_Xt}) ->
    sub_matrix(1, J, N, 1, X).

%%
%%
%%
-spec sub_matrix(I::unsigned(), J::unsigned(), 
		 N::unsigned(), M::unsigned(), 
		 X::matrix()) -> matrix().

sub_matrix(I, J, N, M, {_Nx,Mx,Xt}) ->
    Es = rfold_elems_(fun(E,Acc) -> [E|Acc] end,
		     [], Xt, ((I-1)+(N-1))*Mx+1+((J-1)+(M-1)), 0, M, Mx, N*M),
    {N,M,list_to_tuple(Es)}.
    

%% convolve a NxM matrix over the matrix A (soon: with padding Px, Py and
%% padding value PAD) using Sx and Sy as stride steps.

-spec convolve(F::function(),
	       N::unsigned(),M::unsigned(),
	       Sx::unsigned(), Sy::unsigned(),A::matrix()) ->
		      matrix().

convolve(F,N,M,Sx,Sy,X={Nx,Mx,_Xt}) when N =< Nx, M =< Mx ->
    convolve_(F,[],1,1,Sx,Sy,N,M,X).

-spec convolve(F::function(),
	       N::unsigned(),M::unsigned(),A::matrix()) ->
		      matrix().

convolve(F,N,M,X={Nx,Mx,_Xt}) when N =< Nx, M =< Mx ->
    convolve_(F,[],1,1,1,1,N,M,X).

convolve_(F,Acc,I,J,Sx,Sy,N,M,X={Nx,Mx,Xt}) when (J-1)+(M-1) < Mx ->
    E = F(I, J),
    convolve_(F,[E|Acc],I,J+Sx,Sx,Sy,N,M,X={Nx,Mx,Xt});
convolve_(F,Acc,I,_J,Sx,Sy,N,M,X={Nx,_Mx,_Xt}) when (I-1)+N < Nx ->
    convolve_(F,Acc,I+Sy,1,Sx,Sy,N,M,X);
convolve_(_F,Acc,_I,_J,_Sx,_Sy,_N,_M,_X) ->
    Acc.

max(N, M, X) ->
    max(N, M, 1, 1, X).

max(N, M, Sx, Sy, X={Nx,Mx,Xt}) when N =< Nx, M =< Mx ->
    Es = convolve(
	   fun(I, J) ->
		   I1 = (Nx-I)+1 - (N-1),  %% down->up
		   J1 = (Mx-J)+1 - (M-1),  %% right->left
		   P = (I1-1)*Mx + J1,
		   fold_elems_(
		     fun(E,undefined) -> E;
			(E,Max) -> max(E,Max)
		     end, undefined, Xt, P, 0, M, Mx, N*M)
	   end, N, M, Sx, Sy, X),
    {((Nx-N) div Sx) +1, ((Mx-M) div Sy)+1, list_to_tuple(Es)}.

l2(N, M, X) ->
    l2(N, M, 1, 1, X).

l2(N, M, Sx, Sy, X={Nx,Mx,Xt}) when N =< Nx, M =< Mx ->
    Es = convolve(
	   fun(I, J) ->
		   I1 = (Nx-I)+1 - (N-1),  %% down->up
		   J1 = (Mx-J)+1 - (M-1),  %% right->left
		   P = (I1-1)*Mx + J1,
		   S = fold_elems_(
			 fun(E,undefined) -> E*E;
			    (E,Sum) -> E*E+Sum
			 end, undefined, Xt, P, 0, M, Mx, N*M),
		   math:sqrt(S)
	   end, N, M, Sx, Sy, X),
    {((Nx-N) div Sx)+1, ((Mx-M) div Sy)+1, list_to_tuple(Es)}.

filter(W, B, X) ->
    filter(W, B, 1, 1, X).

filter(W={Nw,Mw,_Wt}, B, Sx, Sy, X={Nx,Mx,_Xt}) when Nw =< Nx, Mw =< Mx ->
    Es = convolve(
	   fun(I, J) ->
		   I1 = (Nx-I)+1 - (Nw-1),
		   J1 = (Mx-J)+1 - (Mw-1),
		   A = sub_matrix(I1,J1,Mw,Mw,X),
		   mulsum(W,A)+B
	   end, Nw, Mw, Sx, Sy, X),
    {((Nx-Nw) div Sx)+1, ((Mx-Mw) div Sy)+1, list_to_tuple(Es)}.

%%
%% Scan embedded matrix data
%%
fold_elems(F,Acc,Xt,Start,RowLen,RowStride,Total) ->
    fold_elems_(F,Acc,Xt,Start,0,RowLen,RowStride,Total).

fold_elems_(F,Acc,Xt,Start,I,RowLen,RowStride,K) when I < RowLen ->
    fold_elems_(F,F(element(Start+I,Xt),Acc),Xt,Start,I+1,RowLen,RowStride,K-1);
fold_elems_(_F,Acc,_Xt,_Start,_I,_RowLen,_RowStride,K) when K =< 0 ->
    Acc;
fold_elems_(F,Acc,Xt,Start,_I,RowLen,RowStride,K) ->
    fold_elems_(F,Acc,Xt,Start+RowStride,0,RowLen,RowStride,K).

%% fold elements in reverse
rfold_elems(F,Acc,Xt,End,RowLen,RowStride,Total) ->
    rfold_elems_(F,Acc,Xt,End,0,RowLen,RowStride,Total).

rfold_elems_(F,Acc,Xt,End,I,RowLen,RowStride,K) when I < RowLen ->
    rfold_elems_(F,F(element(End-I,Xt),Acc),Xt,End,I+1,RowLen,
		 RowStride,K-1);
rfold_elems_(_F,Acc,_Xt,_End,_I,_RowLen,_RowStride,K) when K =< 0 ->
    Acc;
rfold_elems_(F,Acc,Xt,End,_I,RowLen,RowStride,K) ->
    rfold_elems_(F,Acc,Xt,End-RowStride,0,RowLen,RowStride,K).


-spec sigmoid(A::matrix()) -> matrix().
sigmoid({N,M,Xt}) ->
    {N,M,list_to_tuple(sigm_(Xt,N*M,[]))}.

sigm_(_Xt,0,Acc) ->
    Acc;
sigm_(Xt,I,Acc) ->
    sigm_(Xt,I-1,[sigmoid__(element(I,Xt))|Acc]).

sigmoid__(X) when is_float(X) ->
    1.0/(1.0 + math:exp(-X)).

-spec sigmoid_prime(A::matrix()) -> matrix().
sigmoid_prime({N,M,Xt}) ->
    {N,M,list_to_tuple(sigmp_(Xt,N*M,[]))}.

sigmp_(_Xt,0,Acc) ->
    Acc;
sigmp_(Xt,I,Acc) ->
    sigmp_(Xt,I-1,[sigmoid_prime__(element(I,Xt))|Acc]).

sigmoid_prime__(X) ->
    Z = sigmoid__(X),
    Z*(1-Z).

-spec rectifier(A::matrix()) -> matrix().
rectifier({N,M,Xt}) ->
    {N,M,list_to_tuple(rectifier_(Xt,N*M,[]))}.

rectifier_(_Xt,0,Acc) ->
    Acc;
rectifier_(Xt,I,Acc) ->
    rectifier_(Xt,I-1,[rectifier__(element(I,Xt))|Acc]).

rectifier__(X) when X < 0 -> 0;
rectifier__(X) -> X.

-spec softplus(A::matrix()) -> matrix().
softplus({N,M,Xt}) ->
    {N,M,list_to_tuple(softplus_(Xt,N*M,[]))}.

softplus_(_Xt,0,Acc) ->
    Acc;
softplus_(Xt,I,Acc) ->
    softplus_(Xt,I-1,[softplus__(element(I,Xt))|Acc]).

softplus__(X) ->
    math:log(1 + math:exp(X)).


print(A) ->
    io:put_chars(format(A)).

format({_N,M,Xt}) ->
    Es = tuple_to_list(Xt),
    Fs = [format_element(E,0) || E <- Es],
    W = lists:max([length(F) || F <- Fs]),
    %% left pad
    Fs2 = [lists:duplicate(W-length(F),$\s)++F || F <- Fs],
    format_rows(M,Fs2, "|", "|\n", " ", "~s",[]).

format({_N,M,Xt},W) ->
    Es = tuple_to_list(Xt),
    Fs = [format_element(E,W) || E <- Es],
    W = lists:max([length(F) || F <- Fs]),
    %% left pad
    Fs2 = [lists:duplicate(W-length(F),$\s)++F || F <- Fs],
    format_rows(M,Fs2,"|","|\n"," ","~s",[]).

%% fixme allow integers as weights
format_rows(_M,[],_Rb,_Re,_Sep,_Fmt,_Args) ->
    [];
format_rows(M,As,RowStart,RowEnd,Sep,Fmt,Args) ->
    {Row,As1} = lists:split(M, As),
    Es = [io_lib:format(Fmt,Args++[Xij]) || Xij <- Row],
    [ [RowStart,lists:join(Sep,Es),RowEnd] |
      format_rows(M,As1,RowStart,RowEnd,Sep,Fmt,Args)].

is_integer_matrix({_N,_M,Xt}) ->
    lists:all(fun erlang:is_integer/1, tuple_to_list(Xt)).


format_element(X,_) when is_integer(X) ->
    integer_to_list(X);
format_element(X,0) when is_float(X) ->
    lists:flatten(io_lib_format:fwrite_g(X));
format_element(X,P) when is_float(X) ->
    lists:flatten(io_lib:format("~.*f", [P,X])).

		        
bench(N) ->
    spawn(
      fun() ->
	      A = uniform(N,N),
	      B = uniform(N,N),
	      L = 1000,
	      T0 = erlang:system_time(milli_seconds),
	      bench_loop(L,A,B,undefined),
	      T1 = erlang:system_time(milli_seconds),
	      io:format("~s: mult~wx~w/s = ~.2f\n",
			[?MODULE, N, N, 1000*(L / (T1-T0))])
      end).

bench_loop(0, _, _, _) ->
    ok;
bench_loop(I, A, B, _) ->
    C = multiply(A, B),
    bench_loop(I-1,A,B,C).
