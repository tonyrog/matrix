%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    Matrix test
%%% @end
%%% Created : 10 Sep 2017 by Tony Rogvall <tony@rogvall.se>

-module(matrix_test).

-compile(export_all).
-export([test_add/0]).
-export([test_sub/0]).

-export([bench_mul/1, bench_mul/2, bench_mul/3]).
-export([bench_add/1]).
-export([bench_neg/1]).

%% test basic operation

test_add() ->
    test_add(4,3,int8),
    test_add(24,24,int16),
    test_add(128,128,int32),
    ok.

test_add(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:add(A,B),
    Ref = [[2*J || J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_sub() ->
    test_sub(4,4,int8),
    test_sub(24,24,int16),
    test_sub(128,128,int32).

test_sub(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:subtract(A,B),
    Ref = [[0 || _ <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_neg() ->
    test_neg(4,4,int8),
    test_neg(24,24,int16),
    test_neg(128,128,int32).

test_neg(N,M,T) ->
    A = make(N,M,T),
    C = matrix:negate(A),
    Ref = [[-J || J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_zero() ->
    test_zero(4,4,int8),
    test_zero(24,24,int8),
    test_zero(128,128,int8).

test_zero(N,M,T) ->
    A = matrix:zero(N,M,T),
    [ begin
	  E = 0,
	  E = matrix:element(I,J,A)
      end ||
	I <- lists:seq(1,N),
	J <- lists:seq(1,M)],
    ok.

test_identity() ->
    test_identity(4,4,int8),
    test_identity(24,24,int8),
    test_identity(128,128,int8).

test_identity(N,M,T) ->
    A = matrix:identity(N,M,T),
    [ begin
	  E = if I =:= J -> 1; true -> 0 end,
	  E = matrix:element(I,J,A)
      end || 
	I <- lists:seq(1,N),
	J <- lists:seq(1,M)],
    ok.

test_multiply() ->
    test_multiply1(),
    test_multiply2(),
    test_multiply3(),
    test_multiply4(),
    test_tile_multiply().

test_multiply1() ->
    As = [[2]],
    Bs = [[3]],
    A    = matrix:from_list(As,int32),
    B    = matrix:from_list(Bs,int32),
    C    = matrix:multiply(A,B),
    R    = matrix:to_list(C),
    R    = [[6]],
    ok.

test_multiply2() ->
    As = [[1,2],
	  [3,4]],
    Bs = [[5,6],
	  [7,8]],
    A    = matrix:from_list(As,int32),
    B    = matrix:from_list(Bs,int32),
    C    = matrix:multiply(A,B),
    R    = matrix:to_list(C),
    R    = [[19,22],
	    [43,50]],
    ok.

test_multiply3() ->
    As = [[1,2,3],
	  [4,5,6],
	  [7,8,9]],
    Bs = [[10,11,12],
	  [13,14,15],
	  [16,17,18]],
    A    = matrix:from_list(As,int32),
    B    = matrix:from_list(Bs,int32),
    C    = matrix:multiply(A,B),
    R    = matrix:to_list(C),
    R    = [[84,90,96],[201,216,231],[318,342,366]],
    ok.

test_multiply4() ->
    As = [[1,2,3,4],
	  [5,6,7,8],
	  [9,10,11,12],
	  [13,14,15,16]],
    Bs = [[17,18,19,20],
	  [21,22,23,24],
	  [25,26,27,28],
	  [29,30,31,32]],
    A    = matrix:from_list(As,int32),
    B    = matrix:from_list(Bs,int32),
    C    = matrix:multiply(A,B),
    R    = matrix:to_list(C),
    R    = [[250,260,270,280],
	    [618,644,670,696],
	    [986,1028,1070,1112],
	    [1354,1412,1470,1528]],
    ok.

test_tile_multiply() ->
    As = [[1,2],[3,4]],
    A    = matrix:from_list(As,int16),
    Bs = [[5,6],[7,8]],
    B    = matrix:from_list(Bs,int16),
    As2x3 = tile(2,3,As),
    Bs3x2 = tile(3,2,Bs),
    A2x3 = matrix:from_list(As2x3, int16),
    io:format("a2x3=~w\n",[A2x3]),matrix:print(A2x3),
    B3x2 = matrix:from_list(Bs3x2, int16),
    io:format("b3x2=~w\n",[B3x2]),matrix:print(B3x2),
    C2x2 = matrix:multiply(A2x3, B3x2),
    io:format("c2x2=~w\n",[C2x2]),matrix:print(C2x2),
    C    = matrix:submatrix(1,1,2,2,C2x2),
    AB = matrix:multiply(A,B),
    io:format("ab=\n"),matrix:print(AB),
    AB3 = matrix:add(AB, matrix:add(AB,AB)),
    io:format("ab3=\n"),matrix:print(AB3),
    io:format("c=~w\n",[C]),matrix:print(C),
    R = matrix:to_list(AB3),
    R = matrix:to_list(C),
    ok.

%% paralell multiply A and B blockwise in parallel
test_par_multiply() ->
    par_multiply_verify(
      matrix:from_list([[1,2,3,4,5],
			[1,2,3,4,5],
			[1,2,3,4,5],
			[1,2,3,4,5],
			[1,2,3,4,5],
			[1,2,3,4,5],
			[1,2,3,4,5]],int32),
      matrix:from_list([[6,7,8,9,10,11,12,13,14,15],
			[6,7,8,9,10,11,12,13,14,15],
			[6,7,8,9,10,11,12,13,14,15],
			[6,7,8,9,10,11,12,13,14,15],
			[6,7,8,9,10,11,12,13,14,15]], int32)),
    test_par_multiply1(16,16,16,16),
    test_par_multiply1(5,6,6,5),
    ok.
    

test_par_multiply1(An,Am,Bn,Bm) ->
    A = matrix:uniform(An,Am,int32),
    B = matrix:uniform(Bn,Bm,int32),
    par_multiply(A, B).

par_multiply_verify(A, B) ->
    [C11,C12,C21,C22] = par_multiply(A, B),
    C1 = merge_vertical(matrix:to_list(C11),matrix:to_list(C12)),
    C2 = merge_vertical(matrix:to_list(C21),matrix:to_list(C22)),
    Cl  = merge_horizontal(C1,C2),
    C = matrix:multiply(A,B),
    Cl1 = matrix:to_list(C),
    Cl1 = Cl,
    ok.


par_multiply(A, B) ->
    {An,Am} = matrix:size(A),
    {Bn,Bm} = matrix:size(B),
    Am = Bn,  %% Am1+Am2 = Bn1+Bn2
    An1 = An div 2, An2 = An - An1,
    Am1 = Am div 2, Am2 = Am - Am1,
    Bn1 = Bn div 2, Bn2 = Bn - Bn1,
    Bm1 = Bm div 2, Bm2 = Bm - Bm1,

    A11 = matrix:submatrix(1,    1,    An1,Am1,A),
    A12 = matrix:submatrix(1,    Am1+1,An1,Am2,A),
    A21 = matrix:submatrix(An1+1,1,    An2,Am1,A),
    A22 = matrix:submatrix(An1+1,Am1+1,An2,Am2,A),

    B11 = matrix:submatrix(1,    1,    Bn1,Bm1,B),
    B12 = matrix:submatrix(1,    Bm1+1,Bn1,Bm2,B),
    B21 = matrix:submatrix(Bn1+1,1,    Bn2,Bm1,B),
    B22 = matrix:submatrix(Bn1+1,Bm1+1,Bn2,Bm2,B),

    parlists:map(
      fun({X1,Y1,X2,Y2}) ->
	      matrix:add(matrix:multiply(X1,Y1),matrix:multiply(X2,Y2))
      end, [{A11,B11,A12,B21},
	    {A11,B12,A12,B22},
	    {A21,B11,A22,B21},
	    {A21,B12,A22,B22}]).
    

test_transpose() ->
    test_transpose_1(),
    test_transpose_2(),
    test_transpose_3(15,15,int32),
    test_transpose_3(17,17,int32),
    test_transpose_3(10,100,float32),
    test_transpose_3(32,64,float32),
    ok.

test_transpose_1() ->
    As = [[1,2,3],[4,5,6],[7,8,9]],
    A  = matrix:from_list(As,int8),
    R = [[1,4,7],[2,5,8],[3,6,9]],
    At = matrix:transpose(A),
    R = matrix:to_list(At),
    A1 = matrix:transpose(At),
    As = matrix:to_list(A1),
    ok.

test_transpose_2() ->
    As = [[1,2,3,4],[5,6,7,8]],
    A  = matrix:from_list(As,int8),
    R = [[1,5],[2,6],[3,7],[4,8]],
    At = matrix:transpose(A),
    R  = matrix:to_list(At),
    A1 = matrix:transpose(At),
    As = matrix:to_list(A1),
    ok.

test_transpose_3(N,M,T) ->
    A = matrix:uniform(N,M,T),
    R = matrix:to_list(A),
    At = matrix:transpose(A),
    A1 = matrix:transpose(At),
    R = matrix:to_list(A1),
    ok.

test_submatrix() ->
    test_submatrix1(),
    test_submatrix2(),
    ok.

test_submatrix1() ->
    A = matrix:from_list(
	  [[1,2,3, 1,2,3, 1,2],
	   [2,3,1, 2,3,1, 2,3],
	   [3,1,2, 3,1,2, 3,1],
	   [1,2,3, 1,2,3, 1,2],
	   [2,3,1, 2,3,1, 2,3],
	   [3,1,2, 3,1,2, 3,1],
	   [1,2,3, 1,2,3, 1,2],
	   [2,3,1, 2,3,1, 2,3]],int32),
    W = matrix:one(3,3,int32),
    %% pick all submatrices of size 3x3
    matrix:convolve(
      fun(I,J) ->
	      B = matrix:submatrix(I,J,3,3,A),
	      %% io:format("b(~w,~w,3,3) = \n", [I,J]), matrix:print(B),
	      18 = matrix:mulsum(W,B)
      end, 3, 3, 1, 1, A),
    ok.

test_submatrix2() ->
    A = matrix:from_list(
	  [[1,2,3, 1,2,3, 1,2],
	   [2,3,1, 2,3,1, 2,3],
	   [3,1,2, 3,1,2, 3,1],
	   [1,2,3, 1,2,3, 1,2],
	   [2,3,1, 2,3,1, 2,3],
	   [3,1,2, 3,1,2, 3,1],
	   [1,2,3, 1,2,3, 1,2],
	   [2,3,1, 2,3,1, 2,3]],int32),
    W = matrix:identity(3,3,int32),
    %% pick all submatrices of size 3x3
    matrix:convolve(
      fun(I,J) ->
	      %% io:format("submatrix(~w,~w,3,3)\n", [I,J]),
	      B = matrix:submatrix(I,J,3,3,A),
	      %% io:format("b = \n", []), matrix:print(B),
	      B1 = matrix:multiply(W,B),
	      %% io:format("b1 = \n", []), matrix:print(B1),
	      Bs = matrix:to_list(B),
	      B1s = matrix:to_list(B1),
	      Bs = B1s
      end, 3, 3, 1, 1, A),
    ok.

%% replicate marix A (as list) into N*A rows and M*A columns
tile(N,M,A) ->
    As = lists:foldl(fun(_,As) -> merge_vertical(A,As) end,
		     A, lists:seq(2,M)),
    lists:append(lists:duplicate(N, As)).
    
%% merge two matrices (as list) vertical
merge_vertical([A|As],[B|Bs]) ->
    [A++B | merge_vertical(As,Bs)];
merge_vertical([],[]) ->
    [].

%% merge two matrices (as list) vertical, must be compatible
merge_horizontal(As, Bs) ->
    As ++ Bs.

make_list(N,M) ->
    [lists:seq(I,I+M-1) || I <- lists:seq(1,N*M,M)].

make(N,M,T) ->
    Es = make_list(N,M),
    matrix:from_list(Es, T).
    
%%%
%%%% BENCHMARK (MackBook Pro, 13-inch 2012 2.5 GHz Intel Core i5, 4 core)
%%%
%%%  multiply(float32): PLAIN   NATIVE   NIF(-O3)/PAR
%%%              32x32  144     285      66666
%%%            100x100  5                4048
%%%            128x128                   1785
%%%            256x256                   255  / 393
%%%            512x512                   30
%%%          1024x1024                   3
%%%          2048x2048                   0.36
%%%          4096x4096                   0.05
%%%
%%%       add(float32): PLAIN   NATIVE   NIF
%%%              32x32  3048    5714     250000
%%%            100x100   296    518      34482

bench_mul(N) ->
    bench_mul(N,float32,1000).

bench_mul(N,T) ->
    bench_mul(N,T,1000).

bench_mul(N,T,L) ->
    spawn(
      fun() ->
	      A = matrix:uniform(N,N,T),
	      B = matrix:uniform(N,N,T),
	      T0 = erlang:system_time(milli_seconds),
	      bench_mul_loop(L,A,B,undefined),
	      T1 = erlang:system_time(milli_seconds),
	      io:format("~s: mult~wx~w/s = ~.2f time=~.f\n",
			[?MODULE, N, N, 1000*(L / (T1-T0)), (T1-T0)/1000])
      end).

bench_mul_loop(0, _, _, _) ->
    ok;
bench_mul_loop(I, A, B, _) ->
    C = matrix:multiply(A, B),
    bench_mul_loop(I-1,A,B,C).

%% parallell version using parlists
%
bench_pmul(N) ->
    bench_pmul(N,float32,1000).

bench_pmul(N,T) ->
    bench_pmul(N,T,1000).

bench_pmul(N,T,L) ->
    spawn(
      fun() ->
	      A = matrix:uniform(N,N,T),
	      B = matrix:uniform(N,N,T),
	      T0 = erlang:system_time(milli_seconds),
	      bench_pmul_loop(L,A,B,undefined),
	      T1 = erlang:system_time(milli_seconds),
	      io:format("~s: mult~wx~w/s = ~.2f time=~.f\n",
			[?MODULE, N, N, 1000*(L / (T1-T0)), (T1-T0)/1000])
      end).

bench_pmul_loop(0, _, _, _) ->
    ok;
bench_pmul_loop(I, A, B, _) ->
    C = par_multiply(A, B),
    bench_pmul_loop(I-1,A,B,C).


bench_add(N) ->
    spawn(
      fun() ->
	      A = matrix:uniform(N,N,float32),
	      B = matrix:uniform(N,N,float32),
	      L = 1000,
	      T0 = erlang:system_time(milli_seconds),
	      bench_add_loop(L,A,B,undefined),
	      T1 = erlang:system_time(milli_seconds),
	      io:format("~s: mult~wx~w/s = ~.2f\n",
			[?MODULE, N, N, 1000*(L / (T1-T0))])
      end).

bench_add_loop(0, _, _, _) ->
    ok;
bench_add_loop(I, A, B, _) ->
    C = matrix:add(A, B),
    bench_add_loop(I-1,A,B,C).


bench_neg(N) ->
    spawn(
      fun() ->
	      A = matrix:uniform(N,N,float32),
	      L = 1000,
	      T0 = erlang:system_time(milli_seconds),
	      bench_neg_loop(L,A,undefined),
	      T1 = erlang:system_time(milli_seconds),
	      io:format("~s: mult~wx~w/s = ~.2f\n",
			[?MODULE, N, N, 1000*(L / (T1-T0))])
      end).

bench_neg_loop(0, _, _) ->
    ok;
bench_neg_loop(I, A, _) ->
    C = matrix:negate(A),
    bench_neg_loop(I-1,A,C).
