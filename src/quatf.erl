%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2013, Tony Rogvall
%%% @doc
%%%     Quaternion used for 3D rotaion purposes
%%% @end
%%% Created :  6 Sep 2013 by Tony Rogvall <tony@rogvall.se>

-module(quatf).

-export([zero/0, one/0]).
-export([new/4, new/3, new/2, new/1]).
-export([negate/1]).
-export([conjugate/1]).
-export([add/2, subtract/2]).
-export([multiply/2]).
-export([divide/2]).
-export([normalize/1]).
-export([matf/1]).
-export([rot/2]).

%% -define(TEST, true).
%% -export([test/0]).

-type vecf() :: vecf:vec4f().
-type matf() :: matf:mat44f().
-type quatf() :: vecf:vec4f().

-export_type([quatf/0]).

-spec zero() -> quatf().
zero() -> vecf:new(0, 0, 0, 0).

-spec one() -> quatf().
one() -> vecf:new(1, 0, 0, 0).

new(W) when is_number(W) ->
    vecf:new(W, 0, 0, 0);
new({X,Y,Z}) when is_number(X), is_number(Y), is_number(Z) ->
    vecf:new(0, X, Y, Z).

-spec new(X::number(),Y::number(),Z::number(),W::number()) -> quatf().
new(W,X,Y,Z) when is_number(X), is_number(Y), is_number(Z), is_number(W) ->
    vecf:new(W,X,Y,Z).

new(X,Y,Z) when is_number(X), is_number(Y), is_number(Z) ->
    vecf:new(0,X,Y,Z).

-spec new(V::vecf(), A::number()) -> quatf().
new(V, A) when is_number(A) ->
    [Xn,Yn,Zn|_] = vecf:to_list(vecf:normalize(V)),
    S2 = math:sin(A/2),
    new(math:cos(A/2), Xn*S2, Yn*S2, Zn*S2).

-spec normalize(Q::quatf()) -> quatf().
normalize(Q) ->
    N2 = norm2_(Q),
    if N2 > 0 ->
	    Li = 1/math:sqrt(N2),
	    vecf:multiply(Li, Q);
       true ->
	    Q
    end.

norm2_(Q) ->
    vecf:len2(Q).

-spec conjugate(Q::quatf()) -> quatf().
conjugate(Q) ->
    [W,X,Y,Z] = vecf:to_list(Q),
    new(W,-X,-Y,-Z).

-spec negate(Q::quatf()) -> quatf().
negate(Q) ->
    vecf:negate(Q).

-spec add(A::quatf(),B::quatf()) -> quatf().
add(Q, W) when is_number(W) ->
    vecf:add(Q, new(W));
add(W, Q) when is_number(W) ->
    vecf:add(new(W),Q);
add(Q1,Q2) ->
    vecf:add(Q1,Q2).

-spec subtract(A::quatf(),B::quatf()) -> quatf().

subtract(Q, W) when is_number(W) ->
    vecf:subtract(Q, new(W));
subtract(W, Q) when is_number(W) ->
    vecf:subtract(new(W),Q);
subtract(Q1,Q2) ->
    vecf:add(Q1,Q2).

-spec multiply(A::quatf(),B::quatf()) -> quatf().

multiply(Q,W) when is_number(W) ->
    vecf:multiply(W, Q);
multiply(W,Q) when is_number(W) ->
    vecf:multiply(Q, W);
multiply(Q1,Q2) ->
    [W1,X1,Y1,Z1] = vecf:to_list(Q1),
    [W2,X2,Y2,Z2] = vecf:to_list(Q2),
    vecf:new(W1*W2 - X1*X2 - Y1*Y2 - Z1*Z2,
	     W1*X2 + X1*W2 + Y1*Z2 - Z1*Y2,
	     W1*Y2 + Y1*W2 + Z1*X2 - X1*Z2,
	     W1*Z2 + Z1*W2 + X1*Y2 - Y1*X2).

-spec divide(A::quatf(),B::quatf()) -> quatf().
divide(Q,W) when is_number(W), W =/= 0 ->
    vecf:multiply(1/W, Q);
divide(Q1,Q2) ->
    N2 = norm2_(Q2),
    [W1,X1,Y1,Z1] = vecf:to_list(Q1),
    [W2,X2,Y2,Z2] = vecf:to_list(Q2),
    vecf:new((W1*W2 + X1*X2 + Y1*Y2 + Z1*Z2) / N2,
	     (X1*W2 - W1*X2 + Y1*Z2 - Z1*Y2) / N2,
	     (Y1*W2 - W1*Y2 + Z1*X2 - X1*Z2) / N2,
	     (Z1*W2 - W1*Z2 + X1*Y2 - Y1*X2) / N2).

%% @doc
%%   Convert quaternion Q to a 4x4 transformation matrix
%% @end
-spec matf(Q::quatf()) -> matf().
matf(Q) ->
    {W,X,Y,Z} = vecf:to_list(Q),
    WW = W*W, XX = X*X, YY = Y*Y, ZZ = Z*Z,
    XY = X*Y, XZ = X*Z, YZ = Y*Z,
    WX = W*X, WY = W*Y, WZ = W*Z,
    matf:from_list([
		    [ WW+XX-YY-ZZ, 2.0*(XY-WZ), 2*(XZ+WY),   0 ],
		    [ 2*(XY+WZ),   WW-XX+YY-ZZ, 2*(YZ-WX),   0 ],
		    [ 2*(XZ-WY),   2*(YZ-WX),   WW-XX-YY+ZZ, 0 ],
		    [ 0,           0,           0,           1 ]]).

rot(P,Q)  ->
    multiply(multiply(Q,new(P)),conjugate(Q)).

-ifdef(TEST).

test() ->
    Q0 = new(1, 2, 3, 4),
    Q1 = new(2, 3, 4, 5),
    Q2 = new(3, 4, 5, 6),
    R = 7.0,

    io:format("q0:      ~w\n", [ Q0 ]),
    io:format("q1:      ~w\n", [ Q1 ]),
    io:format("q2:      ~w\n", [ Q2 ]),
    io:format("r:       ~w\n", [ R ]),
    io:format("\n"),
    io:format("-q0:     ~w\n", [ negate(Q0) ]),
    io:format("~~q0:     ~w\n", [ conjugate(Q0) ]),
    io:format("\n"),
    io:format("r * q0:  ~w\n", [ multiply(R,Q0) ]),
    io:format("r + q0:  ~w\n", [ add(R,Q0) ]),
    io:format("q0 / r:  ~w\n", [ divide(Q0,R) ]),
    io:format("q0 - r:  ~w\n", [ subtract(Q0,R) ]),
    io:format("\n"),
    io:format("q0 + q1: ~w\n", [ add(Q0,Q1) ]),
    io:format("q0 - q1: ~w\n", [ subtract(Q0,Q1) ]),
    io:format("q0 * q1: ~w\n", [ multiply(Q0,Q1) ]),
    io:format("q0 / q1: ~w\n", [ divide(Q0,Q1) ]),
    io:format("\n"),
    io:format("q0 * ~~q0:     ~w\n", [ multiply(Q0,conjugate(Q0)) ]),
    io:format("q0 + q1*q2:   ~w\n", [ add(Q0, multiply(Q1,Q2)) ]),
    io:format("(q0 + q1)*q2: ~w\n", [ multiply(add(Q0,Q1),Q2) ]),
    io:format("q0*q1*q2:     ~w\n", [ multiply(multiply(Q0,Q1),Q2) ]),
    io:format("(q0*q1)*q2:   ~w\n", [ multiply(multiply(Q0,Q1),Q2) ]),
    io:format("q0*(q1*q2):   ~w\n", [ multiply(Q0,multiply(Q1,Q2)) ]),
    io:format("\n"),
    io:format("||q0||:  ~w\n", [ math:sqrt(norm2_(Q0)) ]),
    io:format("\n"),
    io:format("q0*q1 - q1*q0: ~w\n", [ subtract(multiply(Q0,Q1),multiply(Q1,Q0)) ]),
	    
    %% Other base types
    Q5 = new(2),
    Q6 = new(3),
    io:format("q5*q6: ~w\n", [multiply(Q5,Q6)]),
    ok.
-endif.


