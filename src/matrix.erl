%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    binary matrix version
%%%
%%%    benchmark:        PLAIN   NATIVE   NIF
%%%               32x32
%%%             100x100
%%%
%%% @end

-module(matrix).

%% -compile(native).
%% -on_load(init/0).
-export([normal/1, uniform/1, zero/1, identity/1]).
-export([normal/3, uniform/3, zero/3, identity/3]).
-export([add/2,subtract/2,negate/1]).
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
-export([is_integer_matrix/1]).
-export([filter/3, filter/5]).
-export([fold_elems/7]).
-export([rfold_elems/7]).
-export([bench/1]).
-compile(export_all).

-define(int8,    0).
-define(int16,   1).
-define(int32,   2).
-define(int64,   3).
-define(float32, 4).
-define(float64, 5).

-type unsigned() :: non_neg_integer().
-type matrix_type() :: float32|float64|int64|int32|int16|int8.

-record(matrix,
	{
	  n :: unsigned(),  %% rows
	  m :: unsigned(),  %% columns
	  type :: 0..5,     %% encoded element type
	  ptr = 0 :: unsigned(),    %% 0 is binary, not 0 is resource binary
	  data :: binary()       %% native encode raw matrix data
	}).

-type matrix() :: #matrix{}.

init() ->
    Nif = filename:join(code:priv_dir(matrix), "matrix_drv"),
    erlang:load_nif(Nif, 0).

-spec normal({N::unsigned(), M::unsigned()}) -> matrix().
normal({N,M}) ->
    normal(N,M,float64).

-spec normal(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
normal(N,M,T) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Data = iolist_to_binary([normal_bin(0.0,1.0,Type) || 
				_ <- lists:seq(1,N*M)]),
    #matrix{n=N,m=M,type=Type,data=Data}.

-spec uniform({N::unsigned(), M::unsigned()}) -> matrix().
uniform({N,M}) ->
    uniform(N,M,float64).

-spec uniform(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
uniform(N,M,T) when is_integer(N), N >= 1,
		    is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Data=iolist_to_binary([uniform_bin(Type) || _ <- lists:seq(1,N*M)]),
    #matrix{n=N,m=M,type=Type,data=Data}.

-spec zero({N::unsigned(), M::unsigned()}) -> matrix().
zero({N,M}) ->
    zero(N,M,float64).

-spec zero(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
zero(N,M,T) when is_integer(N), N >= 1,
		 is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Z = number_to_bin(Type, 0),
    BinList = lists:duplicate(N*M, Z),
    #matrix{n=N,m=M,type=Type,data=iolist_to_binary(BinList)}.

-spec identity({N::unsigned(), M::unsigned()}) -> matrix().
identity({N,M}) ->
    identity(N,M,float64).

-spec identity(N::unsigned(), M::unsigned(), T::matrix_type()) -> matrix().
identity(N,M,T) when is_integer(N), N >= 1,
		   is_integer(M), M >= 1 ->
    Type = encode_type(T),
    Bv = {number_to_bin(T, 0),number_to_bin(Type, 1)},
    Data = iolist_to_binary(
	     lists:append([[element(X+1,Bv) ||
			       <<X:1>> <= <<(1 bsl ((M-1)-I)):M>>] ||
			      I <- lists:seq(0, N-1)])),
    #matrix{n=N,m=M,type=Type,data=Data}.

-spec size(M::matrix()) -> {unsigned(), unsigned()}.
size(#matrix{n=N,m=M}) ->
    {N,M}.

-spec element(I::unsigned(),J::unsigned(),X::matrix()) -> number().
element(I,J,#matrix{m=M,type=T,data=Bin}) ->
    Type = encode_type(T),
    P = element_bytes(Type)*((I-1)*M+(J-1)),
    element_(P, Type, Bin).

element_(P, T, Bin) ->
    case T of
	?float64 -> <<_:P/binary,X:64/native-float,_/binary>> = Bin, X;
	?float32 -> <<_:P/binary,X:32/native-float,_/binary>> = Bin, X;
	?int64 -> <<_:P/binary,X:64/native-signed-integer,_/binary>> = Bin, X;
	?int32 -> <<_:P/binary,X:32/native-signed-integer,_/binary>> = Bin, X;
	?int16 -> <<_:P/binary,X:16/native-signed-integer,_/binary>> = Bin, X;
	?int8 -> <<_:P/binary,X:8/native-signed-integer,_/binary>> = Bin, X
    end.

map(F, #matrix{type=T,data=Bin}) ->
    case T of
	?float64 ->
	    [F(X) || <<X:64/native-float>> <= Bin ];
	?float32 ->
	    [F(X) || <<X:32/native-float>> <= Bin ];
	?int64 ->
	    [F(X) || <<X:64/native-signed-integer>> <= Bin ];
	?int32 ->
	    [F(X) || <<X:32/native-signed-integer>> <= Bin ];
	?int16 ->
	    [F(X) || <<X:16/native-signed-integer>> <= Bin ];
	?int8 ->
	    [F(X) || <<X:8/native-signed-integer>> <= Bin ]
    end.

elements_to_list(T, Bin) ->
    case T of
	?float64 ->
	    [X || <<X:64/native-float>> <= Bin ];
	?float32 ->
	    [X || <<X:32/native-float>> <= Bin ];
	?int64 ->
	    [X || <<X:64/native-signed-integer>> <= Bin ];
	?int32 ->
	    [X || <<X:32/native-signed-integer>> <= Bin ];
	?int16 ->
	    [X || <<X:16/native-signed-integer>> <= Bin ];
	?int8 ->
	    [X || <<X:8/native-signed-integer>> <= Bin ]
    end.

zipfoldr(F, A,
	 #matrix{n=N,m=M,type=T1,data=Bin1}, 
	 #matrix{n=N,m=M,type=T2,data=Bin2}) ->
    Size1 = element_bytes(T1),
    P1    = Size1*N*M,
    Size2 = element_bytes(T2),
    P2    = Size2*N*M,
    zipfoldr_(F, A, 
	      T1, P1, Size1, Bin1,
	      T2, P2, Size2, Bin2).

zipfoldr_(_F, A,
	  _T1, 0, _Size1, _Bin1,
	  _T2, 0, _Size2, _Bin2) ->
    A;
zipfoldr_(F, A, 
	 T1, P1, Size1, Bin1,
	 T2, P2, Size2, Bin2) ->
    P11 = P1 - Size1,
    P21 = P2 - Size2,
    E1 = element_(P11,T1,Bin1),
    E2 = element_(P21,T2,Bin2),
    zipfoldr_(F, F(E1,E2,A), 
	     T1, P11, Size1, Bin1,
	      T2, P21, Size2, Bin2).

foldr(F, A, #matrix{n=N,m=M,type=T,data=Bin}) ->
    Size = element_bytes(T),
    P    = Size*N*M,
    foldr_(F,A,T,P,Size,Bin).

foldr_(_F,A,_T,0,_Size,_Bin) ->
    A;
foldr_(F,A,T,P,Size,Bin) ->
    P1 = P - Size,
    E = element_(P1,T,Bin),
    foldr_(F, F(E,A), 
	   T, P1, Size, Bin).

foldl(F, A, #matrix{n=N,m=M,type=T,data=Bin}) ->
    Size = element_bytes(T),
    End  = Size*N*M,
    foldl_(F,A,T,0,Size,End,Bin).

foldl_(_F,A,_T,P,_Size,P,_Bin) ->
    A;
foldl_(F,A,T,P,Size,End,Bin) ->
    E = element_(P,T,Bin),
    foldl_(F, F(E,A), T, P+Size, Size, End, Bin).
     
-spec add(A::matrix(), B::matrix()) -> matrix().

add(X=#matrix{n=N,m=M,type=T1}, 
    Y=#matrix{n=N,m=M,type=T2}) ->
    T = type_combine(T1,T2),
    Es = zipfoldr(
	   fun(Xi,Yi,Acc) ->
		   [number_to_bin(T,Xi+Yi)|Acc]
	   end, [], X, Y),
    #matrix{n=N,m=M,type=T,data=iolist_to_binary(Es)}.

-spec subtract(A::matrix(), B::matrix()) -> matrix().
subtract(X=#matrix{n=N,m=M,type=T1}, 
	 Y=#matrix{n=N,m=M,type=T2}) ->
    T = type_combine(T1,T2),
    Es = zipfoldr(
	   fun(Xi,Yi,Acc) ->
		   [number_to_bin(T,Xi-Yi)|Acc]
	   end, [], X, Y),
    #matrix{n=N,m=M,type=T,data=iolist_to_binary(Es)}.

-spec negate(A::matrix()) -> matrix().
negate(X=#matrix{n=N,m=M,type=T}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,-Xi)|Acc]
	   end, [], X),
    #matrix{n=N,m=M,type=T,data=iolist_to_binary(Es)}.

-spec scale(F::number(), A::matrix()) -> matrix().
scale(F, X=#matrix{n=N,m=M,type=T}) when is_number(F) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,F*Xi)|Acc]
	   end, [], X),
    #matrix{n=N,m=M,type=T,data=iolist_to_binary(Es)}.

%% multiply elementwise and add everythin
-spec mulsum(A::matrix(), B::matrix()) -> matrix().
mulsum(X,Y) ->
    zipfoldr(fun(Xi,Yi,Sum) -> Xi*Yi+Sum end, 0, X, Y).

-spec multiply(X::matrix(), Y::matrix()) -> matrix().

multiply(#matrix{n=N,m=M,type=T1,data=Bin1},
	 #matrix{n=M,m=N,type=T2,data=Bin2}) ->
    ES1 = element_bytes(T1),
    P1  = ES1*(N*M - M),
    RS1 = ES1*M,

    ES2 = element_bytes(T2),
    P2  = ES2*(N-1),
    RS2 = ES2*N,

    T = type_combine(T1,T2),

    Es = mult_(Bin1,P1,ES1,RS1,T1,  Bin2,P2,ES2,RS2,T2, N*N,M,T, []),
    #matrix{n=N,m=N,type=T,data=iolist_to_binary(Es)}.

mult_(_Bin1,_P1,_ES1,_RS1,_T1, _Bin2,_P2,_ES2,_RS2,_T2, 0,_M,_T,Acc) ->
    Acc;
mult_(Bin1,P1,ES1,RS1,T1,  Bin2,P2,ES2,RS2,T2,  K,M,T,Acc) when P2 < 0 ->
    mult_(Bin1,P1-RS1,ES1,RS1,T1, Bin2,RS2-ES2,ES2,RS2,T2,  K,M,T,Acc);
mult_(Bin1,P1,ES1,RS1,T1,  Bin2,P2,ES2,RS2,T2,  K,M,T,Acc) ->
    C = dot_(Bin1,P1,ES1,T1,  Bin2,P2,RS2,T2,  M,0),
    %% io:format("dot_ = ~w\n", [C]),
    mult_(Bin1,P1,ES1,RS1,T1, Bin2,P2-ES2,ES2,RS2,T2, K-1,M,T,
	  [number_to_bin(T,C)|Acc]).

dot_(_Bin1,_P1,_ES1,_T1, _Bin2,_P2,_S2,_T2, 0,Sum) ->
    Sum;
dot_(Bin1,P1,S1,T1, Bin2,P2,S2,T2, K,Sum) ->
    E1 = element_(P1,T1,Bin1),
    E2 = element_(P2,T2,Bin2),
    %% io:format("p1=~w,e1=~w, p2=~w,e2=~w\n", [P1,E1,P2,E2]),
    Sum1 = E1*E2+Sum,
    dot_(Bin1,P1+S1,S1,T1, Bin2,P2+S2,S2,T2, K-1,Sum1).

-spec transpose(A::matrix()) -> matrix().
transpose({matrix,N,M,T,_Ptr,Bin}) ->
    NM = N*M,
    {M,N,T,list_to_tuple(trans_(Bin,NM,NM,N,M,[]))}.

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
row(I, X={matrix,_N,M,_Ptr,_Bin}) ->
    sub_matrix(I, 1, 1, M, X).

%% select a column, return as a matrix with one column
-spec column(J::unsigned(), A::matrix()) -> matrix().
column(J, X={matrix,N,_M,_Ptr,_Bin}) ->
    sub_matrix(1, J, N, 1, X).

%%
%%
%%
-spec sub_matrix(I::unsigned(), J::unsigned(), 
		 N::unsigned(), M::unsigned(), 
		 X::matrix()) -> matrix().

sub_matrix(I, J, N, M, {matrix,_Nx,Mx,_Ptr,Bin}) ->
    Es = rfold_elems_(fun(E,Acc) -> [E|Acc] end,
		      [], Bin, ((I-1)+(N-1))*Mx+1+((J-1)+(M-1)), 0, M, Mx, N*M),
    {N,M,list_to_tuple(Es)}.
    

%% convolve a NxM matrix over the matrix A (soon: with padding Px, Py and
%% padding value PAD) using Sx and Sy as stride steps.

-spec convolve(F::function(),
	       N::unsigned(),M::unsigned(),
	       Sx::unsigned(), Sy::unsigned(),A::matrix()) ->
		      matrix().

convolve(F,N,M,Sx,Sy,X={matrix,Nx,Mx,_Ptr,_Bin}) when N =< Nx, M =< Mx ->
    convolve_(F,[],1,1,Sx,Sy,N,M,X).

-spec convolve(F::function(),
	       N::unsigned(),M::unsigned(),A::matrix()) ->
		      matrix().

convolve(F,N,M,X={matrix,Nx,Mx,_Ptr,_Bin}) when N =< Nx, M =< Mx ->
    convolve_(F,[],1,1,1,1,N,M,X).

convolve_(F,Acc,I,J,Sx,Sy,N,M,X={matrix,_Nx,Mx,_Ptr,_Bin}) 
  when (J-1)+(M-1) < Mx ->
    E = F(I, J),
    convolve_(F,[E|Acc],I,J+Sx,Sx,Sy,N,M,X);
convolve_(F,Acc,I,_J,Sx,Sy,N,M,X={matrix,Nx,_Mx,_Ptr,_Bin}) when (I-1)+N < Nx ->
    convolve_(F,Acc,I+Sy,1,Sx,Sy,N,M,X);
convolve_(_F,Acc,_I,_J,_Sx,_Sy,_N,_M,_X) ->
    Acc.

max(N, M, X) ->
    max(N, M, 1, 1, X).

max(N, M, Sx, Sy, X={matrix,Nx,Mx,_Ptr,Xt}) when N =< Nx, M =< Mx ->
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

l2(N, M, Sx, Sy, X={matrix,Nx,Mx,_Ptr,Xt}) when N =< Nx, M =< Mx ->
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

filter(W={Nw,Mw,_Wt}, B, Sx, Sy, X={matrix,Nx,Mx,_Ptr,_Xt})
  when Nw =< Nx, Mw =< Mx ->
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
sigmoid(X={N,M,T,_Bin}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,sigmoid__(Xi))|Acc]
	   end, [], X),
    {N,M,T,iolist_to_binary(Es)}.

sigmoid__(X) when is_float(X) ->
    1.0/(1.0 + math:exp(-X)).

-spec sigmoid_prime(A::matrix()) -> matrix().
sigmoid_prime(X={N,M,T,_Bin}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,sigmoid_prime__(Xi))|Acc]
	   end, [], X),
    {N,M,T,iolist_to_binary(Es)}.

sigmoid_prime__(X) ->
    Z = sigmoid__(X),
    Z*(1-Z).

-spec rectifier(A::matrix()) -> matrix().
rectifier(X={N,M,T,_Bin}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,rectifier__(Xi))|Acc]
	   end, [], X),
    {N,M,T,iolist_to_binary(Es)}.

rectifier__(X) when X < 0 -> 0;
rectifier__(X) -> X.

-spec softplus(A::matrix()) -> matrix().
softplus(X={N,M,T,_BIn}) ->
    Es = foldr(
	   fun(Xi,Acc) ->
		   [number_to_bin(T,softplus__(Xi))|Acc]
	   end, [], X),
    {N,M,T,iolist_to_binary(Es)}.

softplus__(X) ->
    math:log(1 + math:exp(X)).

print(A) ->
    io:put_chars(format(A)).

format(#matrix{m=M,type=T,data=Bin}) ->
    Es = elements_to_list(T,Bin),
    Fs = [format_element(E,0) || E <- Es],
    W = lists:max([length(F) || F <- Fs]),
    %% left pad
    Fs2 = [lists:duplicate(W-length(F),$\s)++F || F <- Fs],
    format_rows(M,Fs2, "|", "|\n", " ", "~s",[]).

format(#matrix{m=M,type=T,data=Bin},W) ->
    Es = elements_to_list(T,Bin),
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

is_integer_matrix(#matrix{type=T}) -> T < ?float32.
is_float_matrix(#matrix{type=T}) -> T >= ?float32.

format_element(X,_) when is_integer(X) ->
    integer_to_list(X);
format_element(X,0) when is_float(X) ->
    lists:flatten(io_lib_format:fwrite_g(X));
format_element(X,P) when is_float(X) ->
    lists:flatten(io_lib:format("~.*f", [P,X])).

normal_bin(M,S,T) ->
    V = normal_(M,S),
    case T of
	float64 -> <<V:64/native-float>>;
	float32 -> <<V:32/native-float>>
    end.

%% Generate a normal distributed random number
%% S is the standard deviation = sqrt(CoVariance)
normal_(M, S) when is_float(M), is_float(S), S > 0 ->
    X1 = uniform_(),
    X2 = uniform_(),
    M + S*math:sqrt(-2*math:log(X1))*math:cos(2*math:pi()*X2).

uniform_bin(float64) ->
    F = uniform_(float64),
    <<F:64/native-float>>;
uniform_bin(float32) ->
    F = uniform_(float32),
    <<F:32/native-float>>;
uniform_bin(int64) ->
    crypto:strong_rand_bytes(8);
uniform_bin(int32) ->
    crypto:strong_rand_bytes(4);
uniform_bin(int16) ->
    crypto:strong_rand_bytes(2);
uniform_bin(int8) ->
    crypto:strong_rand_bytes(1).

%% generate a double precision random number in [0-1)
%% or an integer random number.
uniform_() ->
    uniform_(float64).

uniform_(float64) ->
    <<_:4,X:52>> = crypto:strong_rand_bytes(7),
    <<F:64/float>> = <<16#3ff:12,X:52>>,
    F - 1;
uniform_(float32) ->
    <<_:1,X:23>> = crypto:strong_rand_bytes(3),
    <<F:32/float>> = <<16#7f:9,X:23>>,
    F - 1;
uniform_(int64) ->
    <<X:64/signed>> = crypto:strong_rand_bytes(8),
    X;
uniform_(int32) ->
    <<X:32/signed>> = crypto:strong_rand_bytes(4),
    X;
uniform_(int16) ->
    <<X:16/signed>> = crypto:strong_rand_bytes(2),
    X;
uniform_(int8) ->
    <<X:8/signed>> = crypto:strong_rand_bytes(1),
    X.

typeof({_N,_M,T,_Bin}) -> T.
typeof({N,M,T1,_Bin1},{N,M,T2,_Bin2}) -> type_combine(T1,T2).
    
%% combine types to the more general one
type_combine(T1,T2) -> erlang:max(T1,T2).

number_to_bin(?float64, X) ->
    <<(float(X)):64/native-float>>;
number_to_bin(?float32, X) ->
    <<(float(X)):32/native-float>>;
number_to_bin(?int64, X) ->
    <<(trunc(X)):64/native-signed-integer>>;
number_to_bin(?int32, X) ->
    <<(trunc(X)):32/native-signed-integer>>;
number_to_bin(?int16, X) ->
    <<(trunc(X)):16/native-signed-integer>>;
number_to_bin(?int8, X) ->
    <<(trunc(X)):8/native-signed-integer>>.

element_bits(?float64) -> 64;
element_bits(?float32) -> 32;
element_bits(?int64) -> 64;
element_bits(?int32) -> 32;
element_bits(?int16) -> 16;
element_bits(?int8) -> 8.

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
encode_type(float64) -> ?float64.

bench(N) ->
    spawn(
      fun() ->
	      A = uniform(N,N,float32),
	      B = uniform(N,N,float32),
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
