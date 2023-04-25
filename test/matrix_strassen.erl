%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2023, Tony Rogvall
%%% @doc
%%%    Strassen algorithm
%%% @end
%%% Created : 27 Feb 2023 by Tony Rogvall <tony@rogvall.se>

-module(matrix_strassen).

-export([multiply/2, multiply/3]).
-compile(export_all).

multiply(A, B) ->
    {An,An,T} = matrix:signature(A),
    {An,An,T} = matrix:signature(B),
    C = matrix:create(An,An,T,<<>>),
    multiply(A, B, C).

multiply(A, B, C) ->
    {An,An,T} = matrix:signature(A),
    {An,An,T} = matrix:signature(B),
    {An,An,T} = matrix:signature(C),
    true = ((An band (An -1)) =:= 0),   %% power of 2
    multiply_(An, A, B, C).

multiply_(N, A, B) ->
    {An,An,T} = matrix:signature(A),
    multiply_(N, A, B, matrix:create(An,An,T,<<>>)).


%% M1 = (A11+A22)*(B11+B22)
%% M2 = (A21+A22)*B11
%% M3 = A11*(B12-B22)
%% M4 = A22(B21-B11)
%% M5 = (A11+A12)*B22
%% M6 = (A21-A11)*(B11+B12)
%% M7 = (A12-A22)*(B21+B22)
%% C11 = M1+M4-M5+M7
%% C12 = M3+M5
%% C21 = M2+M4
%% C22 = M1-M2+M3+M6

multiply_(N, A, B, C) when N =< 128 ->
    matrix:multiply(A, B, C);
multiply_(N, A, B, C) ->
    N2 = N div 2,
    {A11,A12,A21,A22} = block(N2,A),
    {B11,B12,B21,B22} = block(N2,B),
    {C11,C12,C21,C22} = block(N2,C),
    M1 = multiply_(N2,matrix:add(A11,A22),matrix:add(B11,B22)),
    M2 = multiply_(N2,matrix:add(A21,A22),B11),
    M3 = multiply_(N2,A11,matrix:subtract(B12,B22)),
    M4 = multiply_(N2,A22,matrix:subtract(B21,B11)),
    M5 = multiply_(N2,matrix:add(A11,A12),B22),
    M6 = multiply_(N2,matrix:subtract(A21,A11),matrix:add(B11,B12)),
    M7 = multiply_(N2,matrix:subtract(A12,A22),matrix:add(B21,B22)),
    matrix:add(matrix:subtract(matrix:add(M1,M4),M5),M7,C11),
    matrix:add(M3,M5,C12),
    matrix:add(M2,M4,C21),
    matrix:add(matrix:add(matrix:subtract(M1,M2),M3),M6,C22),
    C.

%% * C11 = (A11+A22)*(B11+B22)
%% * C22 = C11
%% * C21 = (A21+A22)*B11
%% * C22 -= C21
%% * C12 = A11*(B12-B22)
%% * C22 += C12
%% * T = A22*(B21-B11)
%% * C11 += T
%% * C21 += T
%% * T = (A11+A12)*B22
%% * C12 += T
%% * C11 -= T
%% * T = (A21-A11)*(B11+B12)
%% * C22 += T
%% * T = (A12-A22)*(B21+B22)
%% * C11 += T

multiply2_(N, A, B, C) when N =< 128 ->
    matrix:multiply(A, B, C);
multiply2_(N, A, B, C) ->
    N2 = N div 2,
    {A11,A12,A21,A22} = block(N2,A),
    {B11,B12,B21,B22} = block(N2,B),
    {C11,C12,C21,C22} = block(N2,C),
    T1 = matrix:add(A11,A22),
    T2 = matrix:add(B11,B22),
    multiply2_(N2,T1,T2,C11),
    matrix:copy(C11,C22),            %% C22 = C11
    matrix:add(A21,A22,T1),
    multiply2_(N2,T1,B11,C21),       %% C21 = (A21+A22)*B11
    matrix:subtract(C22, C21, C22),  %% C22 -= C21
    matrix:subtract(B12,B22,T1),
    multiply2_(N2,A11,T1,C12),       %% C12 = A11*(B12-B22)
    matrix:add(C22, C12, C22),       %% C22 += C12
    matrix:subtract(B21,B11,T1),
    multiply2_(N2,A22,T1,T2),        %% T2 = A22*(N21-B11)
    matrix:add(C11, T2, C11),        %% C11 += T2
    matrix:add(C21, T2, C21),        %% C21 += T2
    matrix:add(A11,A12,T1),
    multiply2_(N2,T1,B22,T2),        %% T2 = (A11+A12)*B22
    matrix:add(C12, T2, C12),        %% C12 += T2
    matrix:subtract(C11, T2, C11),   %% C11 += T2
    matrix:subtract(A21, A11, T1),
    matrix:add(B11, B12, T2),
    T3 = matrix:copy(T1),
    multiply2_(N2,T1,T2,T3),         %% T3 = (A21-A11)*(B11+B12)
    matrix:add(C22, T3, C22),        %% C12 += T3
    matrix:subtract(A12,A22,T1),
    matrix:add(B21,B22,T2),
    multiply2_(N2,T1,T2,T3),         %% T = (A21-A11)*(B11+B12)
    matrix:add(C11, T3, C11),        %% C11 += T3
    C.


block(N, A) ->
    {matrix:submatrix(1,1,N,N,A),
     matrix:submatrix(1,N+1,N,N,A),    
     matrix:submatrix(N+1,1,N,N,A),
     matrix:submatrix(N+1,N+1,N,N,A)}.




    
    

