%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2013, Tony Rogvall
%%% @doc
%%%     Quaternion used for 3D rotaion purposes
%%%     represented with vecf (x,y,z,w)
%%% @end
%%% Created :  6 Sep 2013 by Tony Rogvall <tony@rogvall.se>

-module(quatf).

-compile({no_auto_import, [length/1]}).
-export([zero/0, one/0]).
-export([new/4, new/3, new/2, new/1]).
-export([length/1, magnitude/1]).
-export([negate/1]).
-export([conjugate/1, invert/1]).
-export([add/2, subtract/2]).
-export([multiply/2]).
-export([divide/2]).
-export([normalize/1]).
-export([to_mat/1, from_mat/1]).
-export([rotate/2, arotate/2]).
-export([to_euler/1]).
-export([from_euler/3, from_euler/1]).

%% -define(TEST, true).
%% -export([test/0]).

-type vecf() :: vecf:vec4f().
-type matf() :: matf:mat44f().
-type quatf() :: vecf:vec4f().

-export_type([quatf/0]).

-spec zero() -> quatf().
zero() -> vecf:new(0, 0, 0, 0).

-spec one() -> quatf().
one() -> vecf:new(0, 0, 0, 1).

new(W) when is_number(W) ->
    vecf:new(0, 0, 0, W);
new({X,Y,Z}) when is_number(X), is_number(Y), is_number(Z) ->
    vecf:new(X, Y, Z, 0).

-spec new(X::number(),Y::number(),Z::number(),W::number()) -> quatf().
new(X,Y,Z,W) when is_number(X), is_number(Y), is_number(Z), is_number(W) ->
    vecf:new(X,Y,Z,W).
-spec new(X::number(),Y::number(),Z::number()) -> quatf().
new(X,Y,Z) when is_number(X), is_number(Y), is_number(Z) ->
    vecf:new(X,Y,Z,0).

-spec new(V::vecf(), A::number()) -> quatf().
new(V, A) when is_number(A) ->
    [Xn,Yn,Zn|_] = vecf:to_list(vecf:normalize(V)),
    S2 = math:sin(A/2),
    new(Xn*S2, Yn*S2, Zn*S2, math:cos(A/2)).

-spec normalize(Q::quatf()) -> quatf().
normalize(Q) ->
    N2 = norm2_(Q),
    if N2 > 0 ->
	    Li = 1/math:sqrt(N2),
	    vecf:multiply(Li, Q);
       true ->
	    Q
    end.

-spec length(Q::quatf()) -> number().
length(Q) ->
    math:sqrt(norm2_(Q)).

-spec magnitude(Q::quatf()) -> number().
magnitude(Q) ->
    length(Q).

%% X*X + Y*Y + Z*Z + W*W
norm2_(Q) ->
    matrix:mulsum(Q, Q).

-spec conjugate(Q::quatf()) -> quatf().
conjugate(Q) ->
    [X,Y,Z,W] = vecf:to_list(Q),
    new(-X,-Y,-Z,W).

-spec invert(Q::quatf()) -> quatf().
%% Q*Q' = 1
%% assume |Q| = 1 (normalize)
invert(Q) ->
    conjugate(Q).  %% divide by norm2_(Q) if not normalized

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
    [X1,Y1,Z1,W1] = vecf:to_list(Q1),
    [X2,Y2,Z2,W2] = vecf:to_list(Q2),
    vecf:new(W1*X2 + X1*W2 + Y1*Z2 - Z1*Y2,
	     W1*Y2 + Y1*W2 + Z1*X2 - X1*Z2,
	     W1*Z2 + Z1*W2 + X1*Y2 - Y1*X2,
	     W1*W2 - X1*X2 - Y1*Y2 - Z1*Z2).

-spec divide(A::quatf(),B::quatf()) -> quatf().
divide(Q,W) when is_number(W), W =/= 0 ->
    vecf:multiply(1/W, Q);
divide(Q1,Q2) ->
    N2 = norm2_(Q2),
    [X1,Y1,Z1,W1] = vecf:to_list(Q1),
    [X2,Y2,Z2,W2] = vecf:to_list(Q2),
    vecf:new((X1*W2 - W1*X2 + Y1*Z2 - Z1*Y2) / N2,
	     (Y1*W2 - W1*Y2 + Z1*X2 - X1*Z2) / N2,
	     (Z1*W2 - W1*Z2 + X1*Y2 - Y1*X2) / N2,
	     (W1*W2 + X1*X2 + Y1*Y2 + Z1*Z2) / N2).

%% @doc
%%   Convert quaternion Q to a 4x4 transformation matrix
%% @end
-spec to_mat(Q::quatf()) -> matf().
to_mat(Q) ->
    [X,Y,Z,W] = vecf:to_list(Q),
    WW = W*W, XX = X*X, YY = Y*Y, ZZ = Z*Z,
    XY = X*Y, XZ = X*Z, YZ = Y*Z,
    WX = W*X, WY = W*Y, WZ = W*Z,
    matf:new(WW+XX-YY-ZZ, 2.0*(XY-WZ), 2*(XZ+WY),
	     2*(XY+WZ),   WW-XX+YY-ZZ, 2*(YZ-WX),
	     2*(XZ-WY),   2*(YZ-WX),   WW-XX-YY+ZZ).

-spec from_mat(M::matf()) -> quatf().
from_mat(M) ->
    [R11,R12,R13,_,
     R21,R22,R23,_,
     R31,R32,R33,_,
     _, _, _, _] = matrix:to_list(M),
    %% Hmm this does give correct solution when q0 = max(q0,q1,q2,q3)
    S = math:sqrt(1.0 + R11 + R22 + R33) / 2.0,
    X = (R32 - R23) / (4.0*S),
    Y = (R13 - R31) / (4.0*S),
    Z = (R21 - R12) / (4.0*S),
    W = S,
    new(X,Y,Z,W).
    

%% Passive rotation of vector P by quaternion Q 
%% Rotate P with respect to coordinate system defined by Q
%% Assume P.w = 0!
-spec rotate(P::vecf(),Q::quatf()) -> vecf().
rotate(P,Q)  ->
    multiply(multiply(Q,P),invert(Q)).

%% Active rotation of vector P by quaternion Q 
%% Coordinate system is rotated with respect to the point
%% Assume P.w = 0!
-spec arotate(P::vecf(),Q::quatf()) -> vecf().
arotate(P,Q)  ->
    multiply(multiply(invert(Q),new(P)),Q).

%% Convert quaternion to euler angles
-spec to_euler(Q::quatf()) -> {Roll::number(),Pitch::number(),Yaw::number()}.
to_euler(Q) ->
    %% [Q0,Q1,Q2,Q2] = [W,X,Y,Z]
    [Q1,Q2,Q3,Q0] = vecf:to_list(Q),
    R11 = Q0*Q0 + Q1*Q1 - Q2*Q2 - Q3*Q3,
    R21 = 2*(Q0*Q3 + Q1*Q2),
    R31 = 2*(Q0*Q2 - Q1*Q3),
    R32 = 2*(Q0*Q1 + Q2*Q3),
    R33 = Q0*Q0 - Q1*Q1 - Q2*Q2 + Q3*Q3,
    Roll  = math:atan2(R32, R33),  %% U
    Pitch = math:asin(R31),        %% V
    Yaw   = math:atan2(R21, R11),  %% W
    {Roll,Pitch,Yaw}.

-spec from_euler({Roll::number(),Pitch::number(),Yaw::number()}) -> quatf().
from_euler({Roll,Pitch,Yaw}) ->
    from_euler(Roll,Pitch,Yaw).
    
-spec from_euler(Roll::number(),Pitch::number(),Yaw::number()) -> quatf().
from_euler(Roll,Pitch,Yaw) ->  %% U,V,W
    C1 = math:cos(Roll/2),
    S1 = math:sin(Roll/2),
    C2 = math:cos(Pitch/2),
    S2 = math:sin(Pitch/2),
    C3 = math:cos(Yaw/2),
    S3 = math:sin(Yaw/2),
    new(S1*C2*C3 - C1*S2*S3,
	C1*S2*C3 + C1*S2*S3,
	C1*C2*S3 - S1*S2*C3,
	C1*C2*C3 + S1*S2*S3).

-ifdef(TEST).

test() ->
    Q0 = new(1, 0, 0, math:pi()),
    Q1 = new(0, 1, 0, math:pi()/2),
    Q2 = new(0, 0, 1, math:pi()/4),
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


