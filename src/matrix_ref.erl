%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    reference implementations
%%% @end
%%% Created :  1 Nov 2017 by Tony Rogvall <tony@rogvall.se>

-module(matrix_ref).

%% reference for testing
-export([create/5]).
-export([add/2]).
-export([subtract/2]).
-export([times/2]).
-export([multiply/2]).
-export([scale/2]).
-export([negate/1]).
-export([mulsum/2]).
-export([transpose_data/1]).
-export([rectifier/1]).
-export([sigmoid/1]).
-export([sigmoid_prime/1]).
-export([softplus/1]).
-export([argmax/2]).
-export([l2/5]).
-export([max/5]).
-export([filter/5]).

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

multiply(#matrix{n=Nx,m=Mx,offset=Offs1,stride=Stride1,type=T1,data=Bin1},
	 #matrix{n=Ny,m=My,offset=Offs2,stride=Stride2,type=T2,data=Bin2})
  when Mx =:= Ny ->
    P1  = Offs1 + (Nx-1)*Stride1,  %% last row in X
    P2  = Offs2 + My-1,            %% last column in Y
    T = type_combine(T1,T2),
    Es = mult_(Nx,My,Mx,My,T,Bin1,P1,Stride1,T1,  Bin2,P2,Stride2,T2, []),
    matrix:create(Nx,My,T,true,Es).

mult_(1,0,_Mx,_My,_T,_Bin1,_P1,_S1,_T1, _Bin2,_P2,_S2,_T2,Acc) ->
    Acc;
mult_(N,0,Mx,My,T,Bin1,P1,S1,T1,  Bin2,P2,S2,T2, Acc) ->
    mult_(N-1,My,Mx,My,T,Bin1,P1-S1,S1,T1, Bin2,P2+My,S2,T2, Acc);
mult_(N,J,Mx,My,T,Bin1,P1,S1,T1,  Bin2,P2,S2,T2,  Acc) ->
    C = dot_(Bin1,P1,1,T1,  Bin2,P2,S2,T2,  Mx, scalar_zero(T)),
    %% io:format("j=~w,sum=~w\n", [J,C]),
    mult_(N,J-1,Mx,My,T,Bin1,P1,S1,T1, Bin2,P2-1,S2,T2,
	  [elem_to_bin(T,C)|Acc]).

dot_(_Bin1,_P1,_S1,_T1, _Bin2,_P2,_S2,_T2, 0, Sum) ->
    Sum;
dot_(Bin1,P1,S1,T1, Bin2,P2,S2,T2, K,Sum) ->
    %% io:format("p1=~w, p2=~w\n", [P1,P2]),
    E1 = matrix:element_(P1,T1,Bin1),
    E2 = matrix:element_(P2,T2,Bin2),
    Sum1 = scalar_add(scalar_multiply(E1,E2), Sum),
    dot_(Bin1,P1+S1,S1,T1, Bin2,P2+S2,S2,T2, K-1,Sum1).

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

transpose_data(#matrix{n=N,m=M,offset=Offs,stride=Stride,rowmajor=RowMajor,
			   type=Type,data=Bin}) ->
    ES  = matrix:element_bytes(Type),
    RS  = ES*Stride,                         %% row bytes
    End = ES*(Offs + (N-1)*Stride + M - 1),  %% end element position
    Es = trans_(M,N,N,Bin,End,End,ES,RS,[]),
    matrix:create(M,N,Type,RowMajor,Es).

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

-spec rectifier(A::matrix()) -> matrix().
rectifier(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,scalar_rectifier(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec softplus(A::matrix()) -> matrix().
softplus(X=#matrix{n=N,m=M,type=T}) ->
    Es = map_elems(fun(Xij) -> elem_to_bin(T,softplus__(Xij)) end, X),
    matrix:create(N,M,T,true,Es).

-spec l2(N::unsigned(),M::unsigned(),Sx::unsigned(),Sy::unsigned(),
	 matrix()) -> matrix().
l2(N, M, Sx, Sy, X=#matrix{n=Nx,m=Mx,type=T})
  when N =< Nx, M =< Mx ->
    Es = matrix:convolve(
	   fun(I, J) ->
		   A = matrix:submatrix(I,J,N,M,X),
		   Es = map_elems(fun(Xij) -> Xij*Xij end, A),
		   S = sum([sum(R) || R <- Es]),
		   elem_to_bin(T,math:sqrt(S))
	   end, N, M, Sx, Sy, X),
    matrix:create(((Nx-N) div Sx)+1, ((Mx-M) div Sy)+1, T, true, Es).

-spec max(N::unsigned(),M::unsigned(),Sx::unsigned(),Sy::unsigned(),
	  matrix()) -> matrix().

max(N, M, Sx, Sy, X=#matrix{n=Nx,m=Mx,type=T})
  when N =< Nx, M =< Mx ->
    Es = matrix:convolve(
	   fun(I, J) ->
		   A = matrix:submatrix(I,J,N,M,X),
		   S = max([max(R) || R <- matrix:to_list(A)]),
		   elem_to_bin(T,S)
	   end, N, M, Sx, Sy, X),
    matrix:create(((Nx-N) div Sx) +1, ((Mx-M) div Sy)+1, T, true, Es).


filter(W=#matrix{n=Nw,m=Mw}, B, Sx, Sy, X=#matrix{n=Nx,m=Mx,type=T})
  when Nw =< Nx, Mw =< Mx ->
    Es = matrix:convolve(
	   fun(I, J) ->
		   A = matrix:submatrix(I,J,Nw,Mw,X),
		   elem_to_bin(T,mulsum(W,A)+B)
	   end, Nw, Mw, Sx, Sy, X),
    matrix:create(((Nx-Nw) div Sx)+1, ((Mx-Mw) div Sy)+1, T, true, Es).


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

%% sum scalar
sum([E|Es]) -> sum_(Es, E);
sum([]) -> 0.

sum_([A|Es], Sum) ->
    sum_(Es, scalar_add(A,Sum));
sum_([], Sum) -> Sum.

%% max scalar
max([E|Es]) -> max_(Es, E).

max_([A|Es], Max) ->
    max_(Es, scalar_max(A,Max));
max_([], Max) -> Max.
    

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

scalar_rectifier({X,_}) -> {max(X,0.0), 0.0};
scalar_rectifier(X) when is_integer(X) -> max(X,0);
scalar_rectifier(X) -> max(X,0.0).


scalar_max(A,B) when is_number(A), is_number(B) ->
    erlang:max(A,B);
scalar_max(A,B) ->
    Ax = complex_abs(complex(A)),
    Bx = complex_abs(complex(B)),
    if Ax > Bx -> A;
       true -> B
    end.

complex(R) when is_number(R) -> {float(R),0.0};
complex(A={_R,_I}) -> A.


complex_add({A,B},{E,F}) -> {A+E,B+F}.

complex_subtract({A,B},{E,F}) -> {A-E,B-F}.

complex_negate({A,B}) -> {-A,-B}.

complex_multiply({A,B},{E,F}) -> {A*E-B*F, B*E+A*F}.

complex_abs({A,B}) -> math:sqrt(A*A+B*B).

%% complex_arg({A,B}) -> math:atan2(B,A).
