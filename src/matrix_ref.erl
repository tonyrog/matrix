%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    reference implementation
%%% @end
%%% Created :  1 Nov 2017 by Tony Rogvall <tony@rogvall.se>

-module(matrix_ref).

%% reference for testing
-export([create/5]).
-export([element/3]).
-export([add/2]).
-export([subtract/2]).
-export([times/2]).
-export([ktimes/3]).
-export([multiply/2,multiply/3]).
-export([rmultiply/2,rmultiply/3]).
-export([kmultiply/3]).
-export([scale/2]).
-export([exp/1]).
-export([negate/1]).
-export([mulsum/2]).
-export([sum/1]).
-export([transpose_data/1]).
-export([sigmoid/1, sigmoid_prime/2]).
-export([softplus/1, softplus_prime/2]).
-export([softmax/1, softmax_prime/2]).
-export([tanh/1, tanh_prime/2]).
-export([relu/1, relu_prime/2]).
-export([argmax/2]).
-export([min/1, min/2]).
-export([max/1, max/2]).
-export([l2pool/5]).
-export([maxpool/5]).
-export([filter/2, filter/4]).
%%
-export([transpose_list/1]).
-export([topk/2]).
-export([stopk/2]).
-export([decode_element_at/3]).

%% complex
-export([complex/1]).
-export([complex_add/2]).
-export([complex_subtract/2]).
-export([complex_multiply/2]).
-export([complex_divide/2]).
-export([complex_negate/1]).
-export([complex_conjugate/1]).
-export([complex_abs/1]).
-export([complex_exp/1]).
-export([complex_sinh/1]).
-export([complex_cosh/1]).
-export([complex_tanh/1]).

-import(matrix, [elem_to_bin/2, type_combine/2]).
-compile({no_auto_import,[max/2]}).
-compile({no_auto_import,[min/2]}).

-include("matrix.hrl").

-define(badargif(Cond),
	case Cond of
	    true  -> erlang:error(badarg);
	    false -> ok
	end).

create(N,M,Type,RowMajor,Data) when is_atom(Type) ->
    create(N,M,matrix:encode_type(Type),RowMajor,Data);
create(N,M,T,RowMajor,Data) ->
    Stride = M,
    #matrix { n=N, m=M, type=T, stride=Stride, rowmajor=RowMajor,
	      data=iolist_to_binary(Data)}.


-spec element(I::unsigned(),J::unsigned(),X::matrix()) -> number().
%% element I,J in row/column order (i.e rowmajor)
element(I,J,#matrix{rowmajor=true,n=N,m=M,offset=O,stride=S,type=T,data=D}) 
  when
      is_integer(I), I > 0, I =< N, 
      is_integer(J), J > 0, J =< M ->
    P = O + (I-1)*S+J-1,
    decode_element_at(P, T, D);
element(I,J,#matrix{rowmajor=false,n=N,m=M,offset=O,stride=S,type=T,data=D})
  when
      is_integer(I), I > 0, I =< M, 
      is_integer(J), J > 0, J =< N ->
    P = O + (J-1)*S+I-1,
    decode_element_at(P, T, D).

%% P is element position not byte position
decode_element_at(P, T, Bin) ->
    case T of
	?complex128 ->
	    <<_:P/binary-unit:128,R:64/native-float,I:64/native-float,
	      _/binary>> = Bin, {R,I};
	?complex64 ->
	    <<_:P/binary-unit:64,R:32/native-float,I:32/native-float,
	      _/binary>> = Bin, {R,I};
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

-ifdef(not_used).
native_binary_to_float32(<<F:32/native>>) ->
    binary_to_float32(<<F:32>>).
native_binary_to_float64(<<F:64/native>>) ->
    binary_to_float64(<<F:64>>).
native_binary_to_float128(<<F:128/native>>) ->
    binary_to_float128(<<F:128>>).

binary_to_float32(<<S:1,E:8,F:23>>) ->
    to_float(S,E,F,23,(1 bsl (8-1))-1).

binary_to_float64(<<S:1,E:11,F:52>>) ->
    to_float(S,E,F,52,(1 bsl (11-1))-1).

binary_to_float128(<<S:1,E:15,F:112>>) ->
    to_float(S,E,F,112,(1 bsl (15-1))-1).

to_float(S,E,F,Fn,Bn) ->
    F1 = (1 + F/(1 bsl Fn)),
    E1 = E-Bn,
    EF = if E1 < 0 -> F1/(1 bsl -E1);
	    E1 =:= 0 -> F1;
	    true -> F1*(1 bsl E1)
	 end,
    if S=:= 1 -> -EF;
       true -> EF
    end.
-endif.

-spec add(X::matrix(), Y::matrix()) -> matrix().

add(X=#matrix{rowmajor=R,n=N,m=M},Y=#matrix{rowmajor=R,n=N,m=M}) ->
    add_(X,Y);
add(X=#matrix{n=N,m=M},Y=#matrix{n=M,m=N}) ->
    add_(X,Y);
add(X=#matrix{}, Yc) when ?is_scalar(Yc) ->
    Y = matrix:constant(matrix:size(X), matrix:type(X), Yc),
    add_(X, Y);
add(Xc, Y=#matrix{}) when ?is_scalar(Xc) ->
    X = matrix:constant(matrix:size(Y), matrix:type(Y), Xc),
    add_(X, Y).
    
add_(X,Y) ->
    T = type_combine(X#matrix.type,Y#matrix.type),
    {N,M} = matrix:size(X),
    Data = map_elems(fun(Xij,Yij) ->
			     elem_to_bin(T,scalar_add(Xij,Yij))
		     end, X, Y),
    matrix:create(N,M,T,true,Data).

-spec subtract(A::matrix(), B::matrix()) -> matrix().

subtract(X=#matrix{rowmajor=R,n=N,m=M},Y=#matrix{rowmajor=R,n=N,m=M}) ->
    subtract_(X,Y);
subtract(X=#matrix{n=N,m=M},Y=#matrix{n=M,m=N}) ->
    subtract_(X,Y);
subtract(X=#matrix{},Yc) when ?is_scalar(Yc) ->
    Y = matrix:constant(matrix:size(X), matrix:type(X), Yc),
    subtract_(X,Y);
subtract(Xc, Y=#matrix{}) when ?is_scalar(Xc) ->
    X = matrix:constant(matrix:size(Y), matrix:type(Y), Xc),
    subtract_(X, Y).
    
subtract_(X,Y) ->
    T = type_combine(X#matrix.type,Y#matrix.type),
    {N,M} = matrix:size(X),
    Data = map_elems(fun(Xij,Yij) ->
			     elem_to_bin(T,scalar_subtract(Xij,Yij))
		     end, X, Y),
    matrix:create(N,M,T,true,Data).

-spec times(A::matrix(), B::matrix()) -> matrix().

times(X=#matrix{rowmajor=R,n=N,m=M},Y=#matrix{rowmajor=R,n=N,m=M}) ->
    times_(X,Y);
times(X=#matrix{n=N,m=M},Y=#matrix{n=M,m=N}) ->
    times_(X,Y);
times(X=#matrix{}, Yc) when ?is_scalar(Yc) ->
    Y = matrix:constant(matrix:size(X), matrix:type(X), Yc),
    times_(X, Y);
times(Xc, Y=#matrix{}) when ?is_scalar(Xc) ->
    X = matrix:constant(matrix:size(Y), matrix:type(Y), Xc),
    times_(X, Y).
    
times_(X,Y) ->
    T = type_combine(X#matrix.type,Y#matrix.type),
    {N,M} = matrix:size(X),
    Data = map_elems(fun(Xij,Yij) ->
			     elem_to_bin(T,scalar_multiply(Xij,Yij))
		     end, X, Y),
    matrix:create(N,M,T,true,Data).

-spec negate(A::matrix()) -> matrix().
negate(X) ->
    {N,M} = matrix:size(X),
    T = X#matrix.type,
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_negate(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

%% scale (alias for times/2)
-spec scale(F::scalar(), X::matrix()) -> matrix().

scale(F, X=#matrix{}) when ?is_scalar(F) ->
    times(F, X).

-spec multiply(X::matrix(), Y::matrix()) -> matrix().

multiply(X, Y) ->
    multiply(X, Y, true).

multiply(X=#matrix{type=Tx},Y=#matrix{type=Ty},RowMajor) when
      is_boolean(RowMajor) ->
    {Nx,_Mx} = matrix:size(X),
    {_Ny,My} = matrix:size(Y),
    Xs = matrix:to_list(X),
    Zs = [ [dot(Xi,Yt) || Xi <- Xs] ||
	     Yt <- matrix:to_list(matrix:transpose(Y))],
    T = type_combine(Tx,Ty),
    if RowMajor ->
	    Es = [[elem_to_bin(T,Zij)||Zij<-Zr] || Zr <- transpose_list(Zs)],
	    matrix:create(Nx,My,T,RowMajor,Es);
       true ->
	    Es = [[elem_to_bin(T,Zij)||Zij<-Zr] || Zr <- Zs],
	    matrix:create(My,Nx,T,RowMajor,Es)
    end.

dot([A|As],[B|Bs]) ->
    dot(As,Bs,scalar_multiply(A,B)).

dot([A|As],[B|Bs],Sum) ->
    dot(As,Bs,scalar_add(scalar_multiply(A,B),Sum));
dot([], [], Sum) ->
    Sum.

%% multiply matrix in chunks

%% multiply large matrices
%% FIXME recurse on size of NxM not N!
rmultiply(A, B) ->
    {Na,Ma} = matrix:size(A),
    {Nb,Mb} = matrix:size(B),
    ?badargif(Ma =/= Nb),
    if Ma =< 16 ->
	    matrix:multiply(A,B);
       true ->
	    Ma1 = Ma div 2,
	    Ma2 = Ma - Ma1,
	    Nb1 = Ma1,
	    Nb2 = Ma2,
	    A1 = matrix:submatrix(1,1,Na,Ma1,A),
	    A2 = matrix:submatrix(1,Ma1+1,Na,Ma2,A),
	    B1 = matrix:submatrix(1,1,Nb1,Mb,B),
	    B2 = matrix:submatrix(Nb1+1,1,Nb2,Mb,B),
	    C1 = matrix:multiply(A1,B1),  %% rmultiply, SOON!
	    C2 = matrix:multiply(A2,B2),  %% rmultiply, SOON!
	    matrix:add(C1,C2)
    end.

%% FIXME recurse on size of NxM not N!
rmultiply(A,B,C) ->
    {Na,Ma} = matrix:size(A),
    {Nb,Mb} = matrix:size(B),
    {Nc,Mc} = matrix:size(B),
    ?badargif(Ma =/= Nb),
    ?badargif(Na =/= Nc),
    ?badargif(Mb =/= Mc),
    if Ma =< 16 ->
	    matrix:multiply(A,B,C);
       true ->
	    Ma1 = Ma div 2,
	    Ma2 = Ma - Ma1,
	    Nb1 = Ma1,
	    Nb2 = Ma2,
	    A1 = matrix:submatrix(1,1,Na,Ma1,A),
	    A2 = matrix:submatrix(1,Ma1+1,Na,Ma2,A),
	    B1 = matrix:submatrix(1,1,Nb1,Mb,B),
	    B2 = matrix:submatrix(Nb1+1,1,Nb2,Mb,B),
	    matrix:multiply(A1,B1,C), %% rmultiply, SOON!
	    C2 = matrix:multiply(A2,B2), %% rmultiply, SOON!
	    matrix:add(C,C2,C)
    end.

%%
%% kmultiply multiply only rows Xk with Y where k is taken from K
%% it is assumed here that K is a matrix with a single row of 
%% integer indices.
%%
-spec kmultiply(X::matrix(), Y::matrix(), K::matrix()) -> matrix().

kmultiply(X=#matrix{type=T1}, Y=#matrix{type=T2}, K) ->
    {Nx,Mx} = matrix:size(X),
    {Ny,My} = matrix:size(Y),
    if Mx =/= Ny -> erlang:error(badarg);
       true -> ok
    end,
    T = type_combine(T1,T2),
    Z = matrix:create(Nx,My,T,true,[]), %% result matrix
    matrix:apply1_(zero, Z, Z),  %% set to zero
    [Ks] = matrix:to_list(K),    %% Ks is the index list
    lists:foreach(
      fun(I) ->
	      Xi = matrix:row(I, X),
	      Zi = matrix:row(I, Z),
	      matrix:multiply(Xi,Y,Zi)
      end, Ks),
    Z.

%% ktimes multiply only rows Xi with Yi where i is taken from K
%% it is assumed here that K is a matrix with a single row of 
%% integer indices.

-spec ktimes(X::matrix(), Y::matrix(), K::matrix()) -> matrix().

ktimes(X=#matrix{type=T1}, Y=#matrix{type=T2}, K) ->
    {Nx,Mx} = matrix:size(X),
    {Ny,My} = matrix:size(Y),
    if Nx =/= Ny -> errlang:error(badarg);
       Mx =/= My -> errlang:error(badarg);
       true -> ok
    end,
    T = type_combine(T1,T2),
    Z = matrix:create(Nx,My,T,true,[]), %% result matrix
    matrix:apply1_(zero, Z, Z),         %% set to zero
    [Ks] = matrix:to_list(K),           %% Ks is the index list
    lists:foreach(
      fun(I) ->
	      Xi = matrix:row(I, X),
	      Yi = matrix:row(I, Y),
	      Zi = matrix:row(I, Z),
	      matrix:times(Xi,Yi,Zi)
      end, Ks),
    Z.

-spec mulsum(X::matrix(), Y::matrix()) -> number().

mulsum(X,Y) ->
    T = type_combine(X#matrix.type, Y#matrix.type),
    mulsum(X,Y,scalar_zero(T)).

mulsum(X,Y,Sum) ->
    Es = map_elems(fun(Xij,Yij) ->
			   scalar_multiply(Xij,Yij)
		   end, X, Y),
    lists:foldl(
      fun(Zi,Si) ->
	      lists:foldl(fun(Zij,Sij) -> scalar_add(Zij,Sij) end, Si, Zi)
      end, Sum, Es).

-spec sum(X::matrix()) -> number().
sum(X) ->
    sum_(X, scalar_zero(X#matrix.type)).

sum_(X, Sum) ->
    lists:foldl(
      fun(Zi,Si) ->
	      lists:foldl(fun(Zij,Sij) -> scalar_add(Zij,Sij) end, Si, Zi)
      end, Sum, matrix:to_list(X)).

-spec exp(X::matrix()) -> matrix().
exp(X) ->
    {N,M} = matrix:size(X),
    T = matrix:type(X),
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_exp(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec transpose_data(Src::matrix()) -> matrix().

transpose_data(X=#matrix{n=N,m=M,type=T,rowmajor=RowMajor}) ->
    Zs = matrix:to_list(X),
    Es = [[elem_to_bin(T,Zij)||Zij<-Zr] || Zr <- transpose_list(Zs)],
    matrix:create(M,N,T,RowMajor,Es).


%% argmax
-spec argmax(A::matrix(),Axis::0|1) -> [integer()].

argmax(A,1) ->
    matrix:transpose(argmax(matrix:transpose(A),0));
argmax(A,0) ->
    {_N,M} = matrix:size(A),
    matrix:from_list([argmax_l(matrix:to_list(A),M)],int32).

argmax_l([R|Rs],M) ->
    argmax_l(Rs,2,R,lists:duplicate(M,1)).

argmax_l([Q|Rs],I,ArgVal,ArgMax) ->
    {ArgVal1,ArgMax1} = argmax_r(ArgVal,ArgMax,Q,I,[],[]),
    argmax_l(Rs,I+1,ArgVal1,ArgMax1);
argmax_l([],_I,_ArgValue,ArgMax) ->
    ArgMax.

argmax_r([B|Bs],[X|Xs],[A|As],I,Bs1,Xs1) ->
    if A > B ->
	    argmax_r(Bs,Xs,As,I,[A|Bs1],[I|Xs1]);
       true ->
	    argmax_r(Bs,Xs,As,I,[B|Bs1],[X|Xs1])
    end;
argmax_r([],_Xs,_As,_I,Bs1,Xs1) ->
    {lists:reverse(Bs1),lists:reverse(Xs1)}.

-spec max(A::matrix()) -> scalar().
max(A) ->
    matrix:element(1,1,matrix:max(matrix:max(A, 0), 1)).

max(A,1) ->
    matrix:transpose(max(matrix:transpose(A),0));
max(A,0) ->
    Es = [ [s_max(R)] || R <- matrix:to_list(A)],
    matrix:from_list(Es, matrix:type(A)).

-spec min(A::matrix()) -> scalar().
min(A) ->
    matrix:element(1,1,matrix:min(matrix:min(A, 0), 1)).

min(A,1) ->
    matrix:transpose(min(matrix:transpose(A),0));
min(A,0) ->
    Es = [ [lists:min(R)] || R <- matrix:to_list(A)],
    matrix:from_list(Es, matrix:type(A)).


-spec sigmoid(A::matrix()) -> matrix().
sigmoid(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,sigmoid__(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec sigmoid_prime(A::matrix(),_Out::matrix()) -> matrix().
%% sigmoid_prime(A) == sigmoid(A)*(1-sigmoid(A)) 
%% Out = sigmoid(A)!
sigmoid_prime(_A,Out=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Yij) -> 
			   E = scalar_subtract(Yij,scalar_multiply(Yij,Yij)),
			   elem_to_bin(T,E)
		   end, Out),
    matrix:create(N,M,T,true,Es).

-spec relu(A::matrix()) -> matrix().
relu(X) ->
    {N,M} = matrix:size(X),
    T = X#matrix.type,
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_relu(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec relu_prime(A::matrix(),Out::matrix()) -> matrix().
relu_prime(X,_Out) ->
    {N,M} = matrix:size(X),
    T = X#matrix.type,
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_relu_prime(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec softplus(A::matrix()) -> matrix().
softplus(X) ->
    {N,M} = matrix:size(X),
    T = X#matrix.type,
    Es = map_elems(fun(Xij) -> elem_to_bin(T,softplus__(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec softplus_prime(A::matrix(),Out::matrix()) -> matrix().
softplus_prime(X,_Out) ->
    matrix:sigmoid(X).

-spec softmax(A::matrix()) -> matrix().
softmax(X) ->
    M = matrix:max(X),
    E = matrix:exp(matrix:subtract(X, M)),
    S = matrix:sum(E),
    matrix:scale(1/S, E).

-spec softmax_prime(A::matrix(),Out::matrix()) -> matrix().
softmax_prime(_X,Out) ->
    %% FIXME!!! WTF
    Out.

-spec tanh(A::matrix()) -> matrix().
tanh(A) ->
    {N,M} = matrix:size(A),
    T = A#matrix.type,
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_tanh(Xij)) end, A),
    matrix:create(N,M,T,true,Es).

-spec tanh_prime(A::matrix(),Out::matrix()) -> matrix().
%% tanh_prime(A) == 1 - tanh(A)*tanh(A)
%% Out = tanh(A)
tanh_prime(_A,Out) ->
    matrix:subtract(matrix:one(matrix:size(Out),matrix:type(Out)),
		    matrix:multiply(Out,Out)).

-spec l2pool(A::matrix(),N::unsigned(),M::unsigned(),
	     Sn::unsigned(),Sm::unsigned()) -> matrix().

%% fixme: rowmajor!
l2pool(N,M,Sn,Sm,A=#matrix{n=Nx,m=Mx,type=T})
  when N =< Nx, M =< Mx ->
    Es = matrix:convolve(
	   fun(I, J) ->
		   B = matrix:submatrix(I,J,N,M,A),
		   S = matrix:mulsum(B,B),
		   elem_to_bin(T,math:sqrt(S))
	   end, N, M, Sn, Sm, A),
    matrix:create(((Nx-N) div Sn)+1, ((Mx-M) div Sm)+1, T, true, Es).

-spec maxpool(N::unsigned(),M::unsigned(),Sn::unsigned(),Sm::unsigned(),
	      matrix()) -> matrix().
%% fixme: rowmajor!
maxpool(N,M,Sn,Sm,A=#matrix{n=Nx,m=Mx,type=T})
  when N =< Nx, M =< Mx ->
    Es = matrix:convolve(
	   fun(I, J) ->
		   B = matrix:submatrix(I,J,N,M,A),
		   S = s_max([s_max(R) || R <- matrix:to_list(B)]),
		   elem_to_bin(T,S)
	   end, N, M, Sn, Sm, A),
    matrix:create(((Nx-N) div Sn)+1, ((Mx-M) div Sm)+1, T, true, Es).

%%
%% Note that filter([[S]], A) == scale(S, A)
%%

-spec filter(W::matrix(), A::matrix()) -> 
		    matrix().

filter(W, A) ->
    filter(W, 1, 1, A).

-spec filter(W::matrix(), Sn::unsigned(), Sm::unsigned(), A::matrix()) -> 
		    matrix().

%% fixme: rowmajor!
filter(W=#matrix{n=Wn,m=Wm}, Sn, Sm, X=#matrix{n=Xn,m=Xm,type=T})
  when Wn =< Xn, Wm =< Xm, is_integer(Sn), Sn>0, is_integer(Sm), Sm>0 ->
    Es = matrix:convolve(
	   fun(I, J) ->
		   A = matrix:submatrix(I,J,Wn,Wm,X),
		   elem_to_bin(T,matrix:mulsum(W,A))
	   end, Wn, Wm, Sn, Sm, X),
    matrix:create(((Xn-Wn) div Sn)+1, ((Xm-Wm) div Sm)+1, T, true, Es).


map_elems(F,X) ->
    [[ F(Xij) || Xij <- R] || R <- matrix:to_list(X)].

map_elems(F,X,Y) ->
    lists:zipwith(
      fun(Xi,Yi) ->
	      lists:zipwith(
		fun(Xij,Yij) ->
			F(Xij,Yij)
		end, Xi, Yi)
      end, matrix:to_list(X), matrix:to_list(Y)).


sigmoid__(X) when is_float(X) ->
    1.0/(1.0 + math:exp(-X)).

sigmoid_prime__(X) ->
    Z = sigmoid__(X),
    Z*(1-Z).

softplus__(X) ->
    math:log(1 + math:exp(X)).

%% max scalar over list
s_max([E|Es]) -> s_max_(Es, E).

s_max_([A|Es], Max) ->
    s_max_(Es, scalar_max(A,Max));
s_max_([], Max) -> Max.
    

%% calculate stopk but with the stopk elements marked with 1
%% in a matrix
%% example: 
%%     | -2 |       
%% top(| 5  |, 2) = | 0 1 1 0|
%%     | 13 |       
%%     | 1  ]       
%%
topk(_A, 0) -> 
    undefined;
topk(A, K) ->
    {N,1} = matrix:size(A),  %% only one column for now!
    TOP = matrix:zero(1,N,int8),
    [V] = matrix:to_list(matrix:transpose(matrix:column(1, A))),
    Vi = lists:sort(fun({X,_},{Y,_}) -> 
			    scalar_abs(X)>scalar_abs(Y) end,
		    lists:zip(V, lists:seq(1,N))),
    lists:foreach(
      fun({_,J}) ->
	      matrix:setelement_(1,J,TOP,1)
      end, lists:sublist(Vi,K)),
    TOP.

%% stopk calculate the top K values positions for each column in A
%% the result is stored in MxK matrix as rows
%% example: 
%%     | -2 |
%% top(| 5  |, 2) = | 2 |
%%     | 13 |       | 3 |
%%     | 1  ]

-spec stopk(A::matrix(), K::unsigned()) -> matrix().

stopk(A, K) ->
    {N,1} = matrix:size(A),
    matrix:from_list(
      [begin
	   [V] = matrix:to_list(matrix:transpose(matrix:column(J, A))),
	   Vi = lists:sort(fun({X,_},{Y,_}) -> scalar_abs(X)>scalar_abs(Y) end,
			   lists:zip(V, lists:seq(1,N))),
	   [I || {_,I} <- lists:sublist(Vi,K)]
       end || J <- [1]], int32).

scalar_zero(?complex128) -> {0.0,0.0};
scalar_zero(?complex64) -> {0.0,0.0};
scalar_zero(?float64) -> 0.0;
scalar_zero(?float32) -> 0.0;
scalar_zero(?int64) -> 0;
scalar_zero(?int32) -> 0;
scalar_zero(?int16) -> 0;
scalar_zero(?int8) -> 0.

scalar_add(A,B) when is_number(A), is_number(B) -> A + B;
scalar_add(A,B) when is_number(A), is_tuple(B) -> complex_add({A,0},B);
scalar_add(A,B) when is_tuple(A), is_number(B) -> complex_add(A,{B,0});
scalar_add(A,B) when is_tuple(A), is_tuple(B) -> complex_add(A,B).

scalar_subtract(A,B) when is_number(A), is_number(B) -> A - B;
scalar_subtract(A,B) when is_number(A), is_tuple(B) -> 
    complex_subtract({A,0},B);
scalar_subtract(A,B) when is_tuple(A), is_number(B) -> 
    complex_subtract(A,{B,0});
scalar_subtract(A,B) when is_tuple(A), is_tuple(B) -> 
    complex_subtract(A,B).

scalar_multiply(A,B) when is_number(A), is_number(B) -> A * B;
scalar_multiply(A,B) when is_number(A), is_tuple(B) -> 
    complex_multiply({A,0},B);
scalar_multiply(A,B) when is_tuple(A), is_number(B) -> 
    complex_multiply(A,{B,0});
scalar_multiply(A,B) when is_tuple(A), is_tuple(B) -> 
    complex_multiply(A,B).

scalar_negate(A) when is_number(A) -> -A;
scalar_negate(A) when is_tuple(A)  -> complex_negate(A).

scalar_exp(X) when is_float(X) -> math:exp(X);
scalar_exp(X) when ?is_complex(X) -> complex_exp(X);
scalar_exp(X) when is_integer(X) -> trunc(math:exp(X)).
    

scalar_relu({X,_}) -> {erlang:max(X,0.0), 0.0};
scalar_relu(X) when is_integer(X) -> erlang:max(X,0);
scalar_relu(X) -> erlang:max(X,0.0).

scalar_relu_prime(X) when is_integer(X), X > 0 -> 1;
scalar_relu_prime(X) when is_float(X), X > 0 -> 1.0;
scalar_relu_prime(X) when is_integer(X) -> 0;
scalar_relu_prime(X) when is_float(X) -> 0.0;
scalar_relu_prime({X,_}) when is_number(X), X > 0 -> {1.0, 0.0};
scalar_relu_prime({X,_}) when is_number(X) -> {1.0, 0.0}.

scalar_tanh(X) when is_number(X) -> math:tanh(X);
scalar_tanh(X) when ?is_complex(X) -> complex_tanh(X).
     
scalar_max(A,B) when is_number(A), is_number(B) ->
    erlang:max(A,B);
scalar_max(A,B) ->
    Ax = complex_abs(complex(A)),
    Bx = complex_abs(complex(B)),
    if Ax > Bx -> A;
       true -> B
    end.

scalar_abs(A) when is_number(A) -> abs(A);
scalar_abs(A) when ?is_complex(A) -> complex_abs(A).
    

complex(R) when is_number(R) -> {float(R),0.0};
complex(A={_R,_I}) -> A.

complex_conjugate({R,I}) -> {R,-I}.

complex_add({A,B},{E,F}) -> {A+E,B+F}.

complex_subtract({A,B},{E,F}) -> {A-E,B-F}.

complex_negate({A,B}) -> {-A,-B}.

complex_multiply({A,B},{E,F}) -> {A*E-B*F, B*E+A*F}.

complex_divide({A1,B1},{A2,B2}) ->
    D = A2*A2 + B2*B2,
    {(A1*A2 + B1*B2) / D, -((A1*B2 + A2*B1) / D)}.

complex_abs({A,B}) -> math:sqrt(A*A+B*B).
complex_arg({A,B}) -> math:atan2(B,A).

complex_exp({X,Y}) -> Ex = math:exp(X), {Ex*math:cos(Y),Ex*math:sin(Y)}.

complex_sinh(Z) -> 
    {A,B} = complex_subtract(complex_exp(Z),complex_exp(-Z)),
    {A/2, B/2}.

complex_cosh(Z) ->
    {A,B} = complex_add(complex_exp(Z),complex_exp(-Z)),
    {A/2, B/2}.

%%  sinh(z) = (e^z - e^-z)/2, cosh(z) = (e^z + e^-z)/2
%%  e^z = e^(x+iy) = e^x(cos(y)+isin(y))

complex_tanh(Z) ->
    %% = complex_divide(complex_sinh(Z), complex_cosh(Z)).
    E1 = complex_exp(Z),
    E2 = complex_exp(-Z),
    {A1,B1} = complex_subtract(E1, E2),
    {A2,B2} = complex_add(E1, E2),
    complex_divide({A1/2,B1/2},{A2/2,B2/2}).

%% transpose a list of list matrix representation
-spec transpose_list(As::[[scalar()]]) -> [[scalar()]].
transpose_list([[]|_]) -> 
    [];
transpose_list(As) ->
    [ [hd(A) || A <- As] | transpose_list([tl(A) || A <- As]) ].


