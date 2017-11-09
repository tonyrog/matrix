%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    reference implementation
%%% @end
%%% Created :  1 Nov 2017 by Tony Rogvall <tony@rogvall.se>

-module(matrix_ref).

%% reference for testing
-export([create/5]).
-export([add/2]).
-export([subtract/2]).
-export([times/2]).
-export([ktimes/3]).
-export([multiply/2,multiply/3]).
-export([kmultiply/3]).
-export([scale/2]).
-export([negate/1]).
-export([mulsum/2]).
-export([transpose_data/1]).
-export([sigmoid/1, sigmoid_prime/1]).
-export([softplus/1, softplus_prime/1]).
-export([tanh/1, tanh_prime/1]).
-export([relu/1, relu_prime/1]).
-export([argmax/2]).
-export([l2pool/5]).
-export([maxpool/5]).
-export([filter/2, filter/4]).
%%
-export([transpose_list/1]).
-export([topk/2]).

-import(matrix, [elem_to_bin/2, type_combine/2]).

-include("matrix.hrl").


create(N,M,Type,RowMajor,Data) when is_atom(Type) ->
    create(N,M,matrix:encode_type(Type),RowMajor,Data);
create(N,M,T,RowMajor,Data) ->
    Stride = M,
    #matrix { n=N, m=M, type=T, stride=Stride, rowmajor=RowMajor,
	      data=iolist_to_binary(Data)}.

-spec add(X::matrix(), Y::matrix()) -> matrix().

add(X=#matrix{rowmajor=R,n=N,m=M},Y=#matrix{rowmajor=R,n=N,m=M}) ->
    add_(X,Y);
add(X=#matrix{n=N,m=M},Y=#matrix{n=M,m=N}) ->
    add_(X,Y).
    
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
    subtract_(X,Y).
    
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
    times_(X,Y).
    
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

scale(F, X) ->
    {N,M} = matrix:size(X),
    T = X#matrix.type,
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_multiply(F,Xij)) end, X),
    matrix:create(N,M,T,true,Es).

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

%%
%% kmultiply multiply only rows Xk with Y where k is taken from K
%% it is assumed here that K is a matrix with a single row of 
%% integer indices.
%% FIXME: topk can be computed over multiple columns
%%
-spec kmultiply(K::matrix(), X::matrix(), Y::matrix()) -> matrix().

kmultiply(K, X=#matrix{n=Nx,m=Mx,type=T1}, Y=#matrix{n=Ny,m=My,type=T2}) 
  when Mx =:= Ny ->
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
%% FIXME: topk can be computed over multiple columns

-spec ktimes(K::matrix(), X::matrix(), Y::matrix()) -> matrix().

ktimes(K, X=#matrix{n=Nx,m=Mx,type=T1}, Y=#matrix{n=Ny,m=My,type=T2}) 
  when Nx =:= Ny, Mx =:= My ->
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

mulsum(X,Y) when ?is_complex_matrix(X) orelse ?is_complex_matrix(Y) ->
    mulsum(X,Y,{0.0,0.0});
mulsum(X,Y) ->
    mulsum(X,Y,0).

mulsum(X,Y,Sum) ->
    Es = map_elems(fun(Xij,Yij) ->
			   scalar_multiply(Xij,Yij)
		   end, X, Y),
    lists:foldl(
      fun(Zi,Si) ->
	      lists:foldl(fun(Zij,Sij) -> scalar_add(Zij,Sij) end, Si, Zi)
      end, Sum, Es).

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

-spec sigmoid(A::matrix()) -> matrix().
sigmoid(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,sigmoid__(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec sigmoid_prime(A::matrix()) -> matrix().
sigmoid_prime(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,sigmoid_prime__(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec relu(A::matrix()) -> matrix().
relu(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_relu(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec relu_prime(A::matrix()) -> matrix().
relu_prime(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_relu_prime(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec softplus(A::matrix()) -> matrix().
softplus(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,softplus__(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec softplus_prime(A::matrix()) -> matrix().
softplus_prime(X) ->
    matrix:sigmoid(X).

-spec tanh(A::matrix()) -> matrix().
tanh(A=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_tanh(Xij)) end, A),
    matrix:create(N,M,T,true,Es).

-spec tanh_prime(A::matrix()) -> matrix().
tanh_prime(A) ->
    TanH = matrix:tanh(A),
    matrix:subtract(matrix:one(matrix:size(A),matrix:type(A)),
		    matrix:multiply(TanH,TanH)).

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
		   S = max([max(R) || R <- matrix:to_list(B)]),
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

%% max scalar
max([E|Es]) -> max_(Es, E).

max_([A|Es], Max) ->
    max_(Es, scalar_max(A,Max));
max_([], Max) -> Max.
    
%% topk calculate the top K values positions for each column in A
%% the result is stored in MxK matrix as rows
%% example: 
%%     | -2 1 |
%% top(| 5  2 |, 2) = | 2 3 |
%%     | 13 3 |       | 3 4 |
%%     | 1  4 ]

-spec topk(A::matrix(), K::unsigned()) -> matrix().

topk(A, K) ->
    {N,M} = matrix:size(A),
    matrix:from_list(
      [begin
	   [V] = matrix:to_list(matrix:transpose(matrix:column(J, A))),
	   Vi = lists:sort(fun({X,_},{Y,_}) -> scalar_abs(X)>scalar_abs(Y) end,
			   lists:zip(V, lists:seq(1,N))),
	   [I || {_,I} <- lists:sublist(Vi,K)]
       end || J <- lists:seq(1,M)], int32).

scalar_zero(?complex128) -> {0.0,0.0};
scalar_zero(?complex64) -> {0.0,0.0};
scalar_zero(?float128) -> 0.0;
scalar_zero(?float64) -> 0.0;
scalar_zero(?float32) -> 0.0;
scalar_zero(?int128) -> 0;
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

scalar_relu({X,_}) -> {max(X,0.0), 0.0};
scalar_relu(X) when is_integer(X) -> max(X,0);
scalar_relu(X) -> max(X,0.0).

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


