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
-export([all/0]).

-export([bench_multiply/1,bench_multiply/2,bench_multiply/3,bench_multiply/5]).
-export([bench_add/1,bench_add/2,bench_add/3]).
-export([bench_eval/1,bench_eval/2,bench_eval/3]).
-export([bench_subtract/1,bench_subtract/2,bench_subtract/3]).
-export([bench_times/1,bench_times/2,bench_times/3]).
-export([bench_negate/1,bench_negate/2,bench_negate/3]).
-export([bench_transform/1,
	 bench_transform/2,
	 bench_transform/3,
	 bench_transform/4]).

%%-define(verbose(F,A), io:format((F),(A))).
-define(verbose(F,A), ok).
-define(EPS, 1.0E-8).

all() ->
    test_em([test_add,
	     test_sub,
	     test_times,
	     test_divide,
	     test_remainder,
	     test_negate,
	     test_band,
	     test_bor,
	     test_bxor,
	     test_eq,
	     test_lt,
	     test_lte,
	     test_zero,
	     test_identity,
	     test_multiply,
	     test_mul_t,
	     test_add_t,
	     test_sub_t,
	     test_times_t,
	     test_transpose,
	     test_submatrix,
	     test_min,
	     test_max,
	     test_sum,
	     test_argmax,
	     test_det
	     %% test_lu
	    ]).

test_em([F|Fs]) ->
    test_it(F),
    test_em(Fs);
test_em([]) ->
    ok.

test_it(F) when is_atom(F) ->
    io:format("~s\n", [F]),
    try apply(?MODULE, F, []) of
	ok -> ok
    catch
	error:_ -> error
    end;
test_it({F,As}) ->
    io:format("  ~s~s ... ", [F,fmt_args(As)]),
    R = try apply(?MODULE, F, As) of
	    ok -> "OK"
	catch
	    error:_ -> "ERROR"
	end,
    io:format("~s\n", [R]).

fmt_args([]) -> "";
fmt_args(As) ->
    ["(",fmt_args_(As),")"].

fmt_args_([A]) -> 
    [io_lib:format("~p",[A])];
fmt_args_([A|As]) -> 
    [io_lib:format("~p",[A]),","|fmt_args_(As)].
     

%% test basic operation
%% FIXME: extend with uint8,uin16,uint32,uint64, int64,
%%                    (float16,)
%%                    float32, float64
%% and vector sizes 2,3,4,8,16
%% add constant number to matrices of vectors
%% add constant vector to matrices of vectors
%%
test_add() ->
    test_em([{test_add,[4,3,int8]},
	     {test_add,[24,24,int16]},
	     {test_add,[128,128,int32]}]).

test_add(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:add(A,B),
    Ref = [[2*J || J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_sub() ->
    test_em([{test_sub,[4,4,int8]},
	     {test_sub,[24,24,int16]},
	     {test_sub,[128,128,int32]}]).

test_sub(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:subtract(A,B),
    Ref = [[0 || _ <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_times() ->
    test_em([
	     {test_times,[4,4,int16]},
	     {test_times,[24,24,int32]},
	     {test_times,[128,128,int32]},
	     {test_times,[128,128,int64]}
	    ]).

test_times(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:times(A,B),
    Ref = [[J*J || J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.


test_divide() ->
    test_em([
	     {test_divide,[4,4,int16]},
	     {test_divide,[24,24,int32]},
	     {test_divide,[128,128,int32]},
	     {test_divide,[128,128,int64]}
	    ]).

test_divide(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:divide(A,B),
    Ref = [[1 || _J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_remainder() ->
    test_em([
	     {test_remainder,[4,4,int16]},
	     {test_remainder,[24,24,int32]},
	     {test_remainder,[128,128,int32]},
	     {test_remainder,[128,128,int64]}
	    ]).

test_remainder(N,M,T) ->
    A = make(N,M,T),
    B = make(N,M,T),
    C = matrix:remainder(A,B),
    Ref = [[0 || _J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_band() ->
    test_em([
	     {test_band,[4,3,int8]},
	     {test_band,[24,24,int16]},
	     {test_band,[128,128,int32]},
	     {test_band,[256,256,int64]}
	    ]).

test_band(N,M,T) ->
    test_binary(N,M,T,bitwise_and,fun(Aij,Bij) -> (Aij band Bij) end).

test_bor() ->
    test_em([
	     {test_bor,[4,3,int8]},
	     {test_bor,[24,24,int16]},
	     {test_bor,[128,128,int32]},
	     {test_bor,[256,256,int64]}
	    ]).

test_bor(N,M,T) ->
    test_binary(N,M,T,bitwise_or,fun(Aij,Bij) -> (Aij bor Bij) end).

test_bxor() ->
    test_em([
	     {test_bxor,[4,3,int8]},
	     {test_bxor,[24,24,int16]},
	     {test_bxor,[128,128,int32]},
	     {test_bxor,[256,256,int64]}
	    ]).

test_bxor(N,M,T) ->
    test_binary(N,M,T,bitwise_xor,fun(Aij,Bij) -> (Aij bxor Bij) end).


test_eq() ->
    test_em([
	     {test_eq,[4,3,int8]},
	     {test_eq,[24,24,int16]},
	     {test_eq,[128,128,int32]},
	     {test_eq,[256,256,int64]}
	    ]).

test_eq(N,M,T) ->
    test_binary(N,M,T,eq,fun(Aij,Bij) ->
				 if Aij =:= Bij -> -1;
				    true -> 0
				 end
			 end).


test_lt() ->
    test_em([
	     {test_lt,[4,3,int8]},
	     {test_lt,[24,24,int16]},
	     {test_lt,[128,128,int32]},
	     {test_lt,[256,256,int64]}
	    ]).

test_lt(N,M,T) ->
    test_binary(N,M,T,lt,fun(Aij,Bij) ->
				 if Aij < Bij -> -1;
				    true -> 0
				 end
			 end).

test_lte() ->
    test_em([
	     {test_lte,[4,3,int8]},
	     {test_lte,[24,24,int16]},
	     {test_lte,[128,128,int32]},
	     {test_lte,[256,256,int64]}
	    ]).

test_lte(N,M,T) ->
    test_binary(N,M,T,lte,fun(Aij,Bij) ->
				 if Aij =< Bij -> -1;
				    true -> 0
				 end
			 end).

test_negate() ->
    test_em([{test_negate,[4,4,int8]},
	     {test_negate,[24,24,int16]},
	     {test_negate,[128,128,int32]}]).

test_negate(N,M,T) ->
    A = make(N,M,T),
    C = matrix:negate(A),
    Ref = [[-J || J <- lists:seq(I,I+M-1)] || I <- lists:seq(1,N*M,M)],
    Ref = matrix:to_list(C),
    ok.

test_binary(N,M,T,MatrixOp,BinOp) ->
    As = make_random(N,M,T),
    Am = matrix:from_list(As, T),
    Bs = make_random(N,M,T),
    Bm = matrix:from_list(Bs, T),
    Cm = matrix:MatrixOp(Am, Bm),
    Ref = mmap(BinOp, As, Bs),
    Cs = matrix:to_list(Cm),
    %% if N =:= 4, M =:= 3 ->
    %%  matrix:print(Am),
    %%  matrix:print(Bm),
    %%  matrix:print(Cm);
    %% true ->
    %%  ok
    %% end,
    Cs = Ref,
    ok.

test_unary(N,M,T,MatrixOp,UnaryOp) ->
    A = make(N,M,T),
    C = matrix:MatrixOp(A),
    Ref = mmap(UnaryOp, make_list(N,M)),
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
    test_multiply5(),
    test_multiply6(),
    test_tile_multiply().

primes() ->
    [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311].

p_list(Size) when Size > 1, Size =< 64 ->
    lists:sublist(primes(), Size).

s_list(Size) ->
    lists:seq(1,Size).


row_A(T,N,M) ->
    matrix:from_list([[ I*J || J <- p_list(M)] || I <- p_list(N)],T).

col_A(T,N,M) ->
    X=matrix:from_list([[ I*J || J <- p_list(N)] || I <- p_list(M)],T),
    matrix:transpose(X).

row_B(T,N,M) ->
    matrix:from_list([[ I*J || J <- s_list(M)] || I <- s_list(N)],T).

col_B(T,N,M) ->
    X=matrix:from_list([[ I*J || J <- s_list(N)] || I <- s_list(M)],T),
    matrix:transpose(X).

%% target matrix must be a "real" matrix zero is normally constant and
%% can not be written to!
row_C(T,N,M) ->
    matrix:copy(matrix:zero(N,M,T)).

col_C(T,N,M) ->
    matrix:copy(matrix:transpose(matrix:zero(M,N,T))).

ref_mul_A_B(T,An,Am,Bn,Bm) ->
    matrix:to_list(matrix_ref:multiply(row_A(T,An,Am),row_B(T,Bn,Bm))).

ref_add_A_B(T,An,Am,Bn,Bm) ->
    matrix:to_list(matrix_ref:add(row_A(T,An,Am),row_B(T,Bn,Bm))).

ref_sub_A_B(T,An,Am,Bn,Bm) ->
    matrix:to_list(matrix_ref:subtract(row_A(T,An,Am),row_B(T,Bn,Bm))).

ref_times_A_B(T,An,Am,Bn,Bm) ->
    matrix:to_list(matrix_ref:times(row_A(T,An,Am),row_B(T,Bn,Bm))).


%% test transposed multiply
test_mul_t() ->
    test_mul_t2(int32,5,4,4,3),
    test_mul_t2(int32,64,32,32,16),
    test_mul_t3(int32,5,4,4,3),
    test_mul_t3(int32,64,32,32,16),
    ok.

test_mul_t2(T,An,Am,Bn,Bm) ->
    R = ref_mul_A_B(T,An,Am,Bn,Bm),
    io:format("R=~p\n", [R]),
    R = matrix:to_list(matrix:multiply(row_A(T,An,Am), row_B(T,Bn,Bm))),
    io:format("R=~p\n", [R]),
    R = matrix:to_list(matrix:multiply(row_A(T,An,Am), col_B(T,Bn,Bm))),
    io:format("R=~p\n", [R]),
    R = matrix:to_list(matrix:multiply(col_A(T,An,Am), row_B(T,Bn,Bm))),
    io:format("R=~p\n", [R]),
    R = matrix:to_list(matrix:multiply(col_A(T,An,Am), col_B(T,Bn,Bm))),
    io:format("R=~p\n", [R]),
    ok.

test_mul_t3(T,An,Am,Bn,Bm) ->
    R = ref_mul_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:multiply(row_A(T,An,Am),row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(row_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(col_A(T,An,Am), row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(col_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(row_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(row_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(col_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:multiply(col_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    ok.


%% test transposed add
test_add_t() ->
    test_add_t2_(int32,5,4,5,4),
    test_add_t2_(int32,64,32,64,32),
    test_add_t3_(int32,5,4,5,4),
    test_add_t3_(int32,64,32,64,32),
    ok.

test_add_t2_(T,An,Am,Bn,Bm) ->
    R = ref_add_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:add(row_A(T,An,Am), row_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:add(row_A(T,An,Am), col_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:add(col_A(T,An,Am), row_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:add(col_A(T,An,Am), col_B(T,Bn,Bm))),
    ok.

test_add_t3_(T,An,Am,Bn,Bm) ->
    R = ref_add_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:add(row_A(T,An,Am),row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(row_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(col_A(T,An,Am), row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(col_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(row_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(row_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(col_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:add(col_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    ok.


%% test transposed add
test_sub_t() ->
    test_sub_t2_(int32,5,4,5,4),
    test_sub_t2_(int32,64,32,64,32),
    test_sub_t3_(int32,5,4,5,4),
    test_sub_t3_(int32,64,32,64,32),
    ok.

test_sub_t2_(T,An,Am,Bn,Bm) ->
    R = ref_sub_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:subtract(row_A(T,An,Am), row_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:subtract(row_A(T,An,Am), col_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:subtract(col_A(T,An,Am), row_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:subtract(col_A(T,An,Am), col_B(T,Bn,Bm))),
    ok.

test_sub_t3_(T,An,Am,Bn,Bm) ->
    R = ref_sub_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:subtract(row_A(T,An,Am),row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(row_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(col_A(T,An,Am), row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(col_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(row_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(row_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(col_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:subtract(col_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    ok.

%% test transposed add
test_times_t() ->
    test_times_t2_(int32,5,4,5,4),
    test_times_t2_(int32,64,32,64,32),
    test_times_t3_(int32,5,4,5,4),
    test_times_t3_(int32,64,32,64,32),
    ok.

test_times_t2_(T,An,Am,Bn,Bm) ->
    R = ref_times_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:times(row_A(T,An,Am), row_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:times(row_A(T,An,Am), col_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:times(col_A(T,An,Am), row_B(T,Bn,Bm))),
    R = matrix:to_list(matrix:times(col_A(T,An,Am), col_B(T,Bn,Bm))),
    ok.

test_times_t3_(T,An,Am,Bn,Bm) ->
    R = ref_times_A_B(T,An,Am,Bn,Bm),
    R = matrix:to_list(matrix:times(row_A(T,An,Am),row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(row_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(col_A(T,An,Am), row_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(col_A(T,An,Am), col_B(T,Bn,Bm),
				       row_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(row_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(row_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(col_A(T,An,Am), row_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    R = matrix:to_list(matrix:times(col_A(T,An,Am), col_B(T,Bn,Bm),
				       col_C(T,An,Bm))),
    ok.
    

test_multiply1() ->
    As = [[2]],
    Bs = [[3]],
    A    = matrix:from_list(As,int32),
    B    = matrix:from_list(Bs,int32),
    C    = matrix:multiply(A,B),
    [[6]] = matrix:to_list(C),
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

test_multiply5() ->
    A = matrix:from_list(
	  [
	   [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
	   [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
	   [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
	   [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
	   [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5],
	   [0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0],
	   [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5] ]),
    ?verbose("A=\n~s\n", [matrix:format(A)]),
    X = matrix:from_list([[2,4,6,8,10,12,14,16,18,20,22,24]], float32),
    ?verbose("X=\n~s\n", [matrix:format(X)]),
    Y = matrix:multiply(A, matrix:transpose(X)),
    ?verbose("Y=\n~s\n", [matrix:format(Y)]),
    Y0 = matrix:from_list([[42],[36],[42],[36],[42],[36],[42]], float32),
    0 = compare_matrix(Y, Y0),
    ok.

%% test ALL combinations of square matrix multiply
test_multiply6() ->
    C = matrix:create(4, 4, int16, true, <<>>),
    Ct = matrix:create(4, 4, int16, false, <<>>),
    A  = matrix:from_list([[1,2,3,4],
			   [5,6,7,8],
			   [9,10,11,12],
			   [13,14,15,16]], int8),
    B  = matrix:from_list([[4,3,2,1],
			   [8,7,6,4],
			   [12,11,10,9],
			   [16,15,13,13]], int8),
    matrix:multiply(A, B, C),
    %% io:format("A*B = C\n"), matrix:print(C),
    true = validate_multiply(A, B, C),

    Bt = matrix:transpose(B),
    matrix:multiply(A, Bt, C),  
    %% io:format("A*Bt = C\n"), matrix:print(C),
    true = validate_multiply(A, Bt, C),

    At = matrix:transpose(A),
    matrix:multiply(At, B, C),
    %% io:format("At*B = C\n"), matrix:print(C),
    true = validate_multiply(At, B, C),

    matrix:multiply(At, Bt, C),
    %% io:format("At*Bt = C\n"), matrix:print(C),
    true = validate_multiply(At, Bt, C),

    matrix:multiply(A, B, Ct),
    %% io:format("A*B = Ct\n"), matrix:print(Ct),
    true = validate_multiply(A, B, Ct),

    matrix:multiply(A, Bt, Ct),
    %% io:format("A*Bt = Ct\n"), matrix:print(Ct),
    true = validate_multiply(A, Bt, Ct),

    matrix:multiply(At, B, Ct),
    %% io:format("At*B = Ct\n"), matrix:print(Ct),
    true = validate_multiply(At, B, Ct),

    matrix:multiply(At, Bt, Ct),
    %% io:format("At*Bt = Ct\n"), matrix:print(Ct),
    true = validate_multiply(At, Bt, Ct),
    ok.

validate_multiply(A, B, C) ->
    As = matrix:to_list(A),
    Bts = matrix:to_list(matrix:transpose(B)),
    Cs = [[dot(Ai,Bj) || Bj <- Bts] || Ai <- As],
    compare_lists(Cs, matrix:to_list(C)).

dot([A|As], [B|Bs]) -> A*B + dot(As, Bs);
dot([], []) -> 0.

compare_lists([A | As], [B | Bs]) ->
    compare_list(A, B) andalso compare_lists(As, Bs);
compare_lists([], []) ->
    true.

compare_list([A | As], [B | Bs]) ->
    (abs(A - B) < ?EPS) andalso compare_list(As, Bs);
compare_list([], []) -> 
    true.

test_tile_multiply() ->
    As = [[1,2],[3,4]],
    A    = matrix:from_list(As,int16),
    Bs = [[5,6],[7,8]],
    B    = matrix:from_list(Bs,int16),
    As2x3 = tile(2,3,As),
    Bs3x2 = tile(3,2,Bs),
    A2x3 = matrix:from_list(As2x3, int16),
    %% io:format("a2x3=~w\n",[A2x3]),matrix:print(A2x3),
    B3x2 = matrix:from_list(Bs3x2, int16),
    %% io:format("b3x2=~w\n",[B3x2]),matrix:print(B3x2),
    C2x2 = matrix:multiply(A2x3, B3x2),
    %% io:format("c2x2=~w\n",[C2x2]),matrix:print(C2x2),
    C    = matrix:submatrix(1,1,2,2,C2x2),
    AB = matrix:multiply(A,B),
    %% io:format("ab=\n"),matrix:print(AB),
    AB3 = matrix:add(AB, matrix:add(AB,AB)),
    %% io:format("ab3=\n"),matrix:print(AB3),
    %% io:format("c=~w\n",[C]),matrix:print(C),
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


test_min() ->
    test_min0(),
    test_min(4,3,int8),
    test_min(24,24,int16),
    test_min(128,128,int32),
    ok.

test_min0() ->
    A = matrix:from_list([[1,2,3,4,5],
			  [2,3,4,5,1],
			  [3,4,5,1,2],
			  [4,5,1,2,3],
			  [5,1,2,3,4]],int32),
    R = matrix:min(A,0),
    C = matrix:min(A,1),
    [[1,1,1,1,1]] = matrix:to_list(R),
    [[1],[1],[1],[1],[1]] = matrix:to_list(C),
    1 = matrix:min(A),
    1 = matrix:min(R),
    1 = matrix:min(C),
    1 = matrix:min(A,-1),
    ok.

test_min(N,M,T) ->
    A = make(N,M,T),
    R = matrix:min(A,0),     %% min over each column ( row vector )
    C = matrix:min(A,1),     %% min over each row ( column vector )
    Min = matrix:min(A,-1),  %% total minimum
    Min = matrix:min(C),     %% min in column
    Min = matrix:min(R),     %% min in row
    ok.

test_max() ->
    test_max0(),
    test_max(4,3,int8),
    test_max(24,24,int16),
    test_max(128,128,int32),
    ok.

test_max0() ->
    A = matrix:from_list([[1,2,3,4,5],
			  [2,3,4,5,1],
			  [3,4,5,1,2],
			  [4,5,1,2,3],
			  [5,1,2,3,4]],int32),
    R = matrix:max(A,1),
    C = matrix:max(A,2),
    [[5,5,5,5,5]] = matrix:to_list(R),
    [[5],[5],[5],[5],[5]] = matrix:to_list(C),
    5 = matrix:max(A),
    5 = matrix:max(R),
    5 = matrix:max(C),
    5 = matrix:max(A,0),
    ok.

test_max(N,M,T) ->
    A = make(N,M,T),
    R = matrix:max(A,1),     %% max over each column
    C = matrix:max(A,2),     %% max over each row
    Max = matrix:max(A,0),   %% total maximum
    Max = matrix:max(C),     %% min in column
    Max = matrix:max(R),     %% min in row
    ok.

test_sum() ->
    test_sum0(),
    test_sum(4,3,int8),
    test_sum(24,24,int16),
    test_sum(128,128,int32),
    ok.

test_sum0() ->
    A = matrix:from_list([[1,2,3,4,5],
			  [2,3,4,5,1],
			  [3,4,5,1,2],
			  [4,5,1,2,3],
			  [5,1,2,3,4]],int32),
    R = matrix:sum(A,1),
    C = matrix:sum(A,2),
    [[15,15,15,15,15]] = matrix:to_list(R),
    [[15],[15],[15],[15],[15]] = matrix:to_list(C),
    75 = matrix:sum(A),
    75 = matrix:sum(R),
    75 = matrix:sum(C),
    75 = matrix:sum(A,0),
    ok.

test_sum(N,M,T) ->
    A = make(N,M,T),
    R = matrix:sum(A,1),
    C = matrix:sum(A,2),
    Sum = matrix:sum(A,0),   %% total sum
    Sum = matrix:sum(C),     %% min in column
    Sum = matrix:sum(R),     %% min in row
    ok.

test_argmax() ->
    test_argmax0(),
    test_argmax(4,3,int8),
    test_argmax(24,24,int16),
    test_max(128,128,int32),
    ok.

test_argmax0() ->
    A = matrix:from_list([[1,2,3,4,5],
			  [2,3,4,5,1],
			  [3,4,5,1,2],
			  [4,5,1,2,3],
			  [5,1,2,3,4]],int32),
    R = matrix:argmax(A,1),
    C = matrix:argmax(A,2),
    [[5,4,3,2,1]] = matrix:to_list(R),
    [[5],[4],[3],[2],[1]] = matrix:to_list(C),
    %% argmax(A,0) return index to max element
    I = matrix:element(1,1,matrix:argmax(matrix:max(A, 2),1)),
    J = matrix:element(1,1,matrix:argmax(matrix:max(A, 1),2)),
    {I, J} = matrix:argmax(A,0),
    ok.

test_argmax(N,M,T) ->
    A = make(N,M,T),
    R = matrix:argmax(A,1),
    C = matrix:argmax(A,2),
    R1 = [lists:duplicate(M,N)],
    C1 = [[I] || I <- lists:duplicate(N,M)],
    R1 = matrix:to_list(R),
    C1 = matrix:to_list(C),
    ok.

%% test determinant

test_det() ->
    A = matrix:from_list([[0.0,2.0,3.0],
			  [4.0,5.0,0.0],
			  [2.0,0.0,19.0]]),
    Det33 = det33(A),
    Det33 = matrix:det(A),
    ok.

det33(M) ->
    A = matrix:element(1,1,M),
    B = matrix:element(1,2,M),
    C = matrix:element(1,3,M),
    D = matrix:element(2,1,M),
    E = matrix:element(2,2,M),
    F = matrix:element(2,3,M),
    G = matrix:element(3,1,M),
    H = matrix:element(3,2,M),
    I = matrix:element(3,3,M),
    A*E*I + B*F*G + C*D*H - C*E*G - B*D*I - A*F*H.


test_invert() ->
    A = matrix:from_list([[0.0,2.0,3.0],
			  [4.0,5.0,0.0],
			  [2.0,0.0,19.0]]),
    IA = inv33(A),
    io:format("A=\n"),matrix:print(A),
    io:format("A^-1=\n"),matrix:print(IA),
    {L,U,P,_Pn} = matrix_lu:decompose(A),
    io:format("L(A)=\n"),matrix:print(L),    
    io:format("U(A)=\n"),matrix:print(U),
    io:format("P(A)=\n"),matrix:print(P),
    IIA = matrix:invert(A),
    io:format("invert(A)=\n"),
    matrix:print(IIA),
    ok.

inv33(M) ->
    A = matrix:element(1,1,M),
    B = matrix:element(1,2,M),
    C = matrix:element(1,3,M),
    D = matrix:element(2,1,M),
    E = matrix:element(2,2,M),
    F = matrix:element(2,3,M),
    G = matrix:element(3,1,M),
    H = matrix:element(3,2,M),
    I = matrix:element(3,3,M),
    matrix:from_list([[E*I-F*H, -(B*I-C*H), (B*F-C*E)],
		      [-(D*I-F*G),(A*I-C*G),-(A*F-C*D)],
		      [(D*H-E*G),-(A*H-B*G),(E*E-B*D)]]).


%% example from http://www4.ncsu.edu/~kksivara/ma505/handouts/lu-pivot.pdf
test_decomp_44() ->
    A = matrix:from_list([[2,1,1,0],
			  [4,3,3,1],
			  [8,7,9,5],
			  [6,7,9,8]],float32),
    test_decomp(A).

test_decomp_rand_44() ->
    A = matrix:uniform(4,4,float32),
    test_decomp(A).

test_decomp_rand_44_complex() ->
    A = matrix:uniform(4,4,complex64),
    test_decomp(A).

test_decomp(A) ->
    %% io:format("A=\n"), matrix:print(A),
    {L,U,P,Pn} = matrix_lu:decompose(A),
    %% io:format("L=\n"), matrix:print(L),
    %%io:format("U=\n"), matrix:print(U),
    %%io:format("P=\n"), matrix:print(P),
    %%P1 = matrix:transpose(P),
    %%io:format("P'=\n"), matrix:print(P1),
    %%io:format("Pn=~w\n",[Pn]),

    %% LU = matrix:multiply(L,U),
    %% PA = matrix:multiply(P,A),
    %%io:format("PA=\n"), matrix:print(PA),
    %%io:format("LU=\n"), matrix:print(LU),
    %%io:format("P'LU=\n"), matrix:print(matrix:multiply(P1,LU)),
    {L,U,P,Pn}.

data(1,a) ->
    matrix:from_list([[8,9,11,14],
		      [8,0,11,0],
		      [8,10,11,1],
		      [1,0,0,0]],float32);
data(1,l) ->
    matrix:from_list([[1,0,0,0],
		      [1,1,0,0],
		      [1/8,1/8,1,0],
		      [1,-1/9,0,1]],float32);
data(1,u) ->
    matrix:from_list([[8,9,11,14],
		      [0,-9,0,-14],
		      [0,0,-11/8,0],
		      [0,0,0,-131/9]],float32);
data(1,p) ->
    matrix:from_list([[1,0,0,0],
		      [0,1,0,0],
		      [0,0,0,1],
		      [0,0,1,0]], float32);
%% Data-2
data(2,a) ->
    matrix:from_list([[1,2,3,4,5],
		      [5,4,3,2,1],
		      [1,4,3,2,1],
		      [3,2,1,5,4],
		      [4,3,2,1,5]], float32);
data(2,l) ->
    matrix:from_list([[1,0,0,0,0],
		      [5,1,0,0,0],
		      [1,-1/3,1,0,0],
		      [3,2/3,0,1,0],
		      [4,5/6,0,0,1]], float32);
data(2,u) ->
    matrix:from_list([[1,2,3,4,5],
		      [0,-6,-12,-18,-24],
		      [0,0,-4,-8,-12],
		      [0,0,0,5,5],
		      [0,0,0,0,5]], float32);
data(2,p) ->
    matrix:identity(5,5,float32).

test_lu() ->
    test_lu(1),
    test_lu(2).

test_lu(N) ->
    A = data(N,a),
    L = data(N,l),
    U = data(N,u),
    P = data(N,p),
    {L1,U1,P1,_Pn} = matrix_lu:decompose(A),
    %% io:format("L = \n"), matrix:print(L), 
    %% io:format("L1 = \n"), matrix:print(L1), 
    %% io:format("U = \n"), matrix:print(U), 
    %% io:format("U1 = \n"), matrix:print(U1), 
    %% io:format("P = \n"), matrix:print(P), 
    %% io:format("P1 = \n"), matrix:print(P1), 
    %% io:format("A = \n"), matrix:print(A),

    LU = matrix:multiply(L,U),
%%    io:format("LU = \n"), matrix:print(LU),
    A1 = matrix:multiply(P,LU),
%%    io:format("A1 = \n"), matrix:print(A1),     
    LU1 = matrix:multiply(L1,U1),
%%    io:format("LU1 = \n"), matrix:print(LU1),
    A2 = matrix:multiply(matrix:transpose(P1),LU1),
%%    io:format("A2 = \n"), matrix:print(A2),
    0 = compare_matrix(A, A1),
    0 = compare_matrix(A, A2),
    ok.

compare_matrix(A,B) ->
    {An,Am} = matrix:size(A),
    {Bn,Bm} = matrix:size(B),
    if An < Bn -> -1;
       An > Bn -> 1;
       Am < Bm -> -1;
       Am > Bm -> 1;
       true ->
	    case matrix:is_integer_matrix(A) andalso 
		matrix:is_integer_matrix(B) of
		true -> compare_elements(1,An,Am,A,B,0);
		false -> compare_elements(1,An,Am,A,B,?EPS)
	    end
    end.

compare_elements(I,N,M,A,B,Eps) when I =< N ->
    case compare_row(1,M,I,A,B,Eps) of
	0 -> compare_elements(I+1,N,M,A,B,Eps);
	R -> R
    end;
compare_elements(_I,_N,_M,_A,_B,_Eps) ->
    0.

compare_row(J,M,I,A,B,Eps) when J =< M ->
    Aij = matrix:element(I,J,A),
    Bij = matrix:element(I,J,B),
    Delta = Aij-Bij,
    if 	
	abs(Delta) =< Eps -> compare_row(J+1,M,I,A,B,Eps);
	Delta>0 -> 1;
	Delta<0 -> -1
    end;
compare_row(_J,_M,_I,_A,_B,_Eps) ->
    0.

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

rndt(int8) -> rand:uniform(16#80)-1;
rndt(int16) -> rand:uniform(16#8000)-1;
rndt(int32) -> rand:uniform(16#80000000)-1;
rndt(int64) -> rand:uniform(16#8000000000000000)-1;
rndt(int128) -> rand:uniform(16#80000000000000000000000000000000)-1;
rndt(float32) -> rand:uniform();
rndt(float64) -> rand:uniform();
rndt(float128) -> rand:uniform();
rndt(complex64) -> {rand:uniform(),rand:uniform()};
rndt(complex128) -> {rand:uniform(),rand:uniform()}.
    
make_random(N,M,T) ->
    [[rndt(T) || _ <- lists:seq(I,I+M-1)] ||
	I <- lists:seq(1,N*M,M)].

make_list(N,M) ->
    [lists:seq(I,I+M-1) ||
	I <- lists:seq(1,N*M,M)].

make_rev_list(N,M) ->
    lists:reverse([lists:seq(I,I+M-1) ||
		      I <- lists:reverse(lists:seq(1,N*M,M))]).

make(N,M,T) ->
    Es = make_list(N,M),
    matrix:from_list(Es, T).

make_rev(N,M,T) ->
    Es = make_rev_list(N,M),
    matrix:from_list(Es, T).

mmap(Fun, [A|As], [B|Bs]) ->
    [mmap1(Fun,A,B)|mmap(Fun,As,Bs)];
mmap(_Fun, [], []) ->
    [].

mmap1(Fun,[A|As],[B|Bs]) ->
    [Fun(A,B)|mmap1(Fun,As,Bs)];
mmap1(_Fun, [], []) ->
    [].

mmap(Fun, [A|As]) ->
    [mmap1(Fun,A)|mmap(Fun,As)];
mmap(_Fun, []) ->
    [].

mmap1(Fun,[A|As]) ->
    [Fun(A)|mmap1(Fun,As)];
mmap1(_Fun,[]) ->
    [].

bench_multiply(N) -> bench_multiply(N,float32,1000).
bench_multiply(N,T) -> bench_multiply(N,T,1000).
bench_multiply(N,T,L) -> bench_multiply(N,T,L,false, true).
bench_multiply(N,T,L,At,Bt) -> 
    bench(N,fun(A,B) -> matrix:multiply(A,B) end, T, L, At, Bt).

bench_multiply_table() ->
    bench_multiply_table(float32).
bench_multiply_table(T) ->
    bench_table("matrix:multiply/2",
		fun(A,B) -> matrix:multiply(A,B) end,
		fun(N) ->
			case T of
			    complex64 -> N*N*N*2;
			    complex128 -> N*N*N*2;
			    _ -> N*N*N 
			end
		end,
		T
	       ).

bench_inline_multiply(N) -> bench_inline_multiply(N,float32,1000).
bench_inline_multiply(N,T) -> bench_inline_multiply(N,T,1000).
bench_inline_multiply(N,T,L) -> bench_inline_multiply(N,T,L,false, true).
bench_inline_multiply(N,T,L,At,Bt) -> 
    bench_inline(N,fun(A,B,C) -> matrix:multiply(A,B,C) end, T, L, At, Bt).

bench_inline_multiply_table() ->
    bench_inline_multiply_table(float32).
bench_inline_multiply_table(T) ->
    bench_inline_table("matrix:multiply/2",
		       fun(A,B,C) -> matrix:multiply(A,B,C) end,
		       fun(N) ->
			       case T of
				   complex64 -> N*N*N*2;
				   complex128 -> N*N*N*2;
				   _ -> N*N*N 
			       end
		       end,
		       T).

%% multiply transform with vector4 of size N
bench_transform(N) -> bench_transform(N,float32,1000).
bench_transform(N,T) -> bench_transform(N,T,1000).
bench_transform(N,T,L) -> bench_transform(N,T,L,false).
bench_transform(N,T,L,Vt) ->
    bench_transform(N,fun(A,B) -> matrix:multiply(A,B) end,T,L,Vt).

%% parallell version using parlists
%
bench_pmul(N) -> bench_pmul(N,float32,1000).
bench_pmul(N,T) -> bench_pmul(N,T,1000).

bench_pmul(N,T,L) ->
    spawn(
      fun() ->
	      A = matrix:uniform(N,N,T),
	      B = matrix:uniform(N,N,T),
	      T0 = erlang:monotonic_time(),
	      bench_pmul_loop(L,A,B,undefined),
	      T1 = erlang:monotonic_time(),
	      Time = erlang:convert_time_unit(T1 - T0, native, microsecond),
	      Ts = Time/1000000,
	      io:format("~s: mult~wx~w/s = ~.2f time=~.f\n",
			[?MODULE, N, N, L/Ts, Ts])
      end).

bench_pmul_loop(0, _, _, _) ->
    ok;
bench_pmul_loop(I, A, B, _) ->
    C = par_multiply(A, B),
    bench_pmul_loop(I-1,A,B,C).

bench_eval(N) -> bench_eval(N,float32,1000).
bench_eval(N,T) -> bench_eval(N,T,1000).
bench_eval(N,T,L) -> bench(N,fun(A,B) -> matrix:eval2([a,b,add],A,B) end, T, L).
bench_eval_table() ->
    bench_table("matrix:eval2/3",
		fun(A,B) -> matrix:eval2([a,b,add],A,B) end).

bench_add(N) -> bench_add(N,float32,1000).
bench_add(N,T) -> bench_add(N,T,1000).
bench_add(N,T,L) -> bench(N,fun(A,B) -> matrix:add(A,B) end, T, L).
bench_add_table() ->
    bench_table("matrix:add/2",
		fun(A,B) -> matrix:add(A,B) end).


bench_subtract(N) -> bench_subtract(N,float32,1000).
bench_subtract(N,T) -> bench_subtract(N,T,1000).
bench_subtract(N,T,L) -> bench(N,fun(A,B) -> matrix:subtract(A,B) end, T, L).
bench_subtract_table() ->
    bench_table("matrix:subtract/2",
		fun(A,B) -> matrix:subtract(A,B) end).

bench_times(N) -> bench_times(N,float32,1000).
bench_times(N,T) -> bench_times(N,T,1000).
bench_times(N,T,L) -> bench(N,fun(A,B) -> matrix:times(A,B) end, T, L).
bench_times_table() -> 
    bench_table("matrix:times/2", 
		fun(A,B) -> matrix:times(A,B) end).

bench_negate(N) -> bench_negate(N,float32,1000).
bench_negate(N,T) -> bench_negate(N,T,1000).
bench_negate(N,T,L) -> bench(N,fun(A,_) -> matrix:negate(A) end, T, L).
bench_negate_table() -> bench_table("matrix:negate/1", 
				    fun(A,_) -> matrix:negate(A) end).
    
%%
%% Bench loop / table
%%
bench_table(Name,F) ->
    bench_table(Name,F,fun(_N) -> 0 end,float32).

bench_table(Name,F,Flop,T) ->
    io:format("~s / ~w\n\n", [Name,T]),
    io:format("| NxN        | op/s   |\n"),
    io:format("|------------|--------|\n"),
    bench(32,F,Flop,T,131072),
    bench(64,F,Flop,T,  65536),
    bench(128,F,Flop,T, 8192),
    bench(256,F,Flop,T, 256),
    bench(512,F,Flop,T, 64),
    bench(1024,F,Flop,T,8),
    bench(2048,F,Flop,T,4),
    bench(4096,F,Flop,T,2).

bench_inline_table(Name,F) ->
    bench_inline_table(Name,F,fun(_N) -> 0 end,float32).

bench_inline_table(Name,F,Flop,T) ->
    io:format("~s / ~w\n\n", [Name,T]),
    io:format("| NxN        | op/s   |\n"),
    io:format("|------------|--------|\n"),
    bench_inline(32,F,Flop,T,  131072),
    bench_inline(64,F,Flop,T,  65536),
    bench_inline(128,F,Flop,T, 8192),
    bench_inline(256,F,Flop,T, 256),
    bench_inline(512,F,Flop,T, 64),
    bench_inline(1024,F,Flop,T,8),
    bench_inline(2048,F,Flop,T,4),
    bench_inline(4096,F,Flop,T,2).


bench(N,F,T,L) ->
    bench(N,F,fun(_N) -> 0 end,T,L).

bench(N,F,Flop,T,L) ->
    bench(N,F,Flop,T,L,false,false).

bench(N,F,T,L,At,Bt) ->
    bench(N,F,fun(_N) -> 0 end,T,L,At,Bt).

bench(N,F,Flop,T,L,At,Bt) ->
    A0 = matrix:uniform(N,N,T),
    A = if At -> matrix:transpose(A0);
	   true -> A0
	end,
    B0 = matrix:uniform(N,N,T),
    B = if Bt -> matrix:transpose(B0);
	   true -> B0
	end,
    T0 = erlang:monotonic_time(),
    _R = bench_loop(L,F,A,B,undefined),
    T1 = erlang:monotonic_time(),
    Time = erlang:convert_time_unit(T1 - T0, native, microsecond),
    Ts = Time/1000000,
    Fs = format_flops(Flop, N, L, Ts),
    io:format("|   ~wx~w   | ~.2f | ~s |\n", [N, N, (L/Ts),Fs]).

bench_inline(N,F,T,L) ->
    bench_inline(N,F,fun(_N) -> 0 end,T,L).

bench_inline(N,F,Flop,T,L) ->
    bench_inline(N,F,Flop,T,L,false,false).

bench_inline(N,F,T,L,At,Bt) ->
    bench_inline(N,F,fun(_N) -> 0 end,T,L,At,Bt).

bench_inline(N,F,Flop,T,L,At,Bt) ->
    A0 = matrix:uniform(N,N,T),
    A = if At -> matrix:transpose(A0);
	   true -> A0
	end,
    B0 = matrix:uniform(N,N,T),
    B = if Bt -> matrix:transpose(B0);
	   true -> B0
	end,
    C = matrix:copy(matrix:zero(N,N,T)), %% matrix:zero is read only!
    T0 = erlang:monotonic_time(),
    _R = bench_inline_loop(L,F,A,B,C),
    T1 = erlang:monotonic_time(),
    Time = erlang:convert_time_unit(T1 - T0, native, microsecond),
    Ts = Time/1000000,
    Fs = format_flops(Flop, N, L, Ts),
    io:format("|   ~wx~w   | ~.2f | ~s |\n", [N, N, (L/Ts),Fs]).

format_flops(Flop, N, L, Ts) ->
    Flops = (Flop(N)*L) / Ts,
    if Flops == 0 -> "";
       Flops > 1000000000.0 ->
	    io_lib:format("~.2f GFlops", [Flops/1000000000.0]);
       Flops > 1000000.0 ->
	    io_lib:format("~.2f MFlops", [Flops/1000000.0]);
       Flops > 1000.0 ->
	    io_lib:format("~.2f KFlops", [Flops/1000.0]);
       true ->
	    io_lib:format("~.2f Flops", [Flops])
    end.

bench_transform(N,F,T,L,Vt) ->
    Transform = matrix:uniform(4,4,T),
    V = if Vt -> matrix:transpose(matrix:uniform(N,4,T));
	   true -> matrix:uniform(4,N,T)
	end,
    T0 = erlang:monotonic_time(),
    _R = bench_loop(L,F,Transform,V,undefined),
    T1 = erlang:monotonic_time(),
    Time = erlang:convert_time_unit(T1 - T0, native, microsecond),
    Ts = Time/1000000,
    io:format("|   ~w   | ~.2f  |\n", [N, (L/Ts)]).
    
bench_proc(N,F,T,L) ->
    SELF = self(),
    {Pid,Mon} =
	spawn_monitor(
	  fun() ->
		  A = matrix:uniform(N,N,T),
		  B = matrix:uniform(N,N,T),
		  T0 = erlang:monotonic_time(),
		  R = bench_loop(L,F,A,B,undefined),
		  T1 = erlang:monotonic_time(),
		  Time = erlang:convert_time_unit(T1 - T0, native, microsecond),
		  Ts = Time/1000000,
		  SELF ! {self(),R,Ts}
	  end),
    receive
	{'DOWN',Mon,process,Pid,_Reason} ->
	    io:format("|   ~wx~w   | CRASH  |  r=~ps\n",
		      [N,N, _Reason]);
	{Pid,_R,Ts} ->
	    io:format("|   ~wx~w   | ~.2f  |  t=~fs\n",
		      [N, N, (L/Ts), Ts]),
	    receive
		{'DOWN',Mon,process,Pid,normal} ->
		    ok;
		{'DOWN',Mon,process,Pid,_Reason} ->
		    io:format("Reason = ~w\n", [_Reason]),
		    ok
	    end
    end.


%% loop that keeps the latest result
bench_loop(0, _F, _, _, _) ->
    0;
bench_loop(I, F, A, B, _) ->
    C = F(A,B),
    bench_loop(I-1,F,A,B,C).

%% loop that keeps the latest result
bench_inline_loop(0, _F, _, _, _) ->
    0;
bench_inline_loop(I, F, A, B, C) ->
    C1 = F(A,B,C),
    bench_inline_loop(I-1,F,A,B,C1).

