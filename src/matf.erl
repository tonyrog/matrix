%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2013, Tony Rogvall
%%% @doc
%%%    4x4 dimensional matrix operations
%%% @end
%%% Created : 12 Jul 2013 by Tony Rogvall <tony@rogvall.se>

-module(matf).

-export([zero/0, one/0, uniform/0]).
-export([transpose/1]).
-export([transform/2]).

-export([add/2,
	 subtract/2,
	 multiply/2,
	 pow/2,
	 negate/1,
	 sqr/1, 
	 det/1,
	 invert/1, invert33/1,
	 divide/2
	]).

-export([translate/3,
	 scale/3,
	 rotate/4]).
-export([print/1,print/2,print/3]).
-export([format/1,format/2,format/3]).



-include_lib("matrix/include/matrix.hrl").

-type mat44f() :: matrix:matrix(). %% 4x4 (?FTYPE)
-export_type([mat44f/0]).
-type vecf() :: vecf:vec4f().

-define(ELEM_TYPE, float32).
-define(BIN_ELEM(X), ?float32(X)).

-spec zero() -> mat44f().
zero() -> matrix:zero(4, 4, ?ELEM_TYPE).

-spec one() -> mat44f().
one() -> matrix:one(4, 4, ?ELEM_TYPE).

-spec uniform() -> mat44f().
uniform() -> matrix:uniform(4, 4, ?ELEM_TYPE).

-spec translate(Tx::number(),Ty::number(),Tz::number()) -> mat44f().
translate(Tx,Ty,Tz) when is_number(Tx), is_number(Ty), is_number(Tz) ->
    matrix:create(4, 4, ?ELEM_TYPE,
		  <<?BIN_ELEM(1), ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(0),
		    ?BIN_ELEM(0), ?BIN_ELEM(1), ?BIN_ELEM(0), ?BIN_ELEM(0),
		    ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(1), ?BIN_ELEM(0),
		    ?BIN_ELEM(Tx), ?BIN_ELEM(Ty), ?BIN_ELEM(Tz), ?BIN_ELEM(1)>>).
-spec scale(Sx::number(),Sy::number(),Sz::number()) -> mat44f().
scale(Sx,Sy,Sz) when is_number(Sx), is_number(Sy), is_number(Sz) ->
    matrix:create(4, 4, ?ELEM_TYPE,
		  <<?BIN_ELEM(Sx), ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(0),
		    ?BIN_ELEM(0), ?BIN_ELEM(Sy), ?BIN_ELEM(0), ?BIN_ELEM(0),
		    ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(Sz), ?BIN_ELEM(0),
		    ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(1)>>).

%%
%% create a matrix that rotate A0 degrees around the
%% vector {Ux,Uy,UzZ} must be unit vector where Ux^2 + Uy^2 + Uz^2 = 1
%% This is true for the trivial vectors 
%% (1,0,0) (0,1,0) (0,0,1) 
%% 
-spec rotate(Angle::number(),Ux::number(),Uy::number(),Uz::number()) ->
		    mat44f().
rotate(A0,Ux,Uy,Uz) when is_number(Ux), is_number(Uy), is_number(Uz) ->
    A = A0*(math:pi()/180),
    CosA = math:cos(A),
    SinA = math:sin(A),
    NCosA = (1-CosA),
    Uxy    = Ux*Uy,
    Uxz    = Ux*Uz,
    Uyz    = Uy*Uz,
    UxSinA = Ux*SinA,
    UySinA = Uy*SinA,
    UzSinA = Uz*SinA,
    matrix:create(4, 4, ?ELEM_TYPE,
<<?BIN_ELEM(Ux*Ux*NCosA+CosA), ?BIN_ELEM(Uxy*NCosA-UzSinA), ?BIN_ELEM(Uxz*NCosA+UySinA), ?BIN_ELEM(0),
  ?BIN_ELEM(Uxy*NCosA+UzSinA), ?BIN_ELEM(Uy*Uy*NCosA+CosA), ?BIN_ELEM(Uyz*NCosA-UxSinA), ?BIN_ELEM(0),
  ?BIN_ELEM(Uxz*NCosA-UySinA), ?BIN_ELEM(Uyz*NCosA+UxSinA), ?BIN_ELEM(Uz*Uz*NCosA+CosA), ?BIN_ELEM(0),
  ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(0), ?BIN_ELEM(1)>>).

-spec transpose(A::mat44f()) -> mat44f().
transpose(A) -> 
    matrix:transpose(A).
     
-spec add(A::mat44f(),B::mat44f()) -> mat44f().
add(A, B) -> matrix:add(A, B).

-spec subtract(A::mat44f(),B::mat44f()) -> mat44f().

subtract(A, B) -> matrix:subtract(A, B).

-spec multiply(A::mat44f(),B::mat44f()) -> mat44f();
	      (A::number(),B::mat44f()) -> mat44f().
multiply(A, B) when is_number(A) ->
    matrix:times(A, B);
multiply(A, B) -> 
    matrix:multiply(A, B).

-spec transform(Vs::[vecf()], M::mat44f()) -> [vecf()].
transform(Xs, M) when is_list(Xs) ->
    [matrix:multiply(X, M) || X <- Xs];
transform(Xs, M) when ?is_matrix(Xs) ->
    matrix:multipy(Xs, M).

-spec pow(X::mat44f(),Y::float()) -> mat44f().
pow(A, N) -> matrix:pow(A, N).

-spec sqr(A::mat44f()) -> mat44f().
sqr(A) ->
    matrix:square(A).

-spec negate(A::mat44f()) -> mat44f().
negate(A) -> matrix:negate(A).

-spec det(A::mat44f()) -> float().
det(A) -> matrix:det(A).

-spec invert(A::mat44f()) -> mat44f().
invert(A) ->
    [[A11,A12,A13,A14],
     [A21,A22,A23,A24],
     [A31,A32,A33,A34],
     [A41,A42,A43,A44]] = matrix:to_list(A),

    S1 = A11*A22 - A21*A12,
    S2 = A11*A23 - A21*A13,
    S3 = A11*A24 - A21*A14,
    S4 = A12*A23 - A22*A13,
    S5 = A12*A24 - A22*A14,
    S6 = A13*A24 - A23*A14,

    C6 = A33*A44 - A43*A34,
    C5 = A32*A44 - A42*A34,
    C4 = A32*A43 - A42*A33,
    C3 = A31*A44 - A41*A34,
    C2 = A31*A43 - A41*A33,
    C1 = A31*A42 - A41*A32,

    Det = S1*C6 - S2*C5 + S3*C4 + S4*C3 - S5*C2 + S6*C1,
    try 1/Det of 
	Di ->
	    matrix:from_list(
	      [
	       [ Di*( A22*C6 - A23*C5 + A24*C4),
		 Di*(-A12*C6 + A13*C5 - A14*C4),
		 Di*( A42*S6 - A43*S5 + A44*S4),
		 Di*(-A32*S6 + A33*S5 - A44*S4) ],
	       
	       [ Di*(-A21*C6 + A23*C3 - A24*C2),
		 Di*( A11*C6 - A13*C3 + A14*C2),
		 Di*(-A41*S6 + A43*S3 - A44*S2),
		 Di*( A31*S6 - A33*S3 + A34*S2) ],
	       
	       [ Di*( A21*C5 - A22*C3 + A24*C1),
		 Di*(-A11*C5 + A12*C3 - A14*C1),
		 Di*( A41*S5 - A42*S3 + A44*S1),
		 Di*(-A31*S5 + A32*S3 - A34*S1) ],
	       
	       [ Di*(-A21*C4 + A22*C2 - A23*C1),
		 Di*( A11*C4 - A12*C2 + A13*C1),
		 Di*(-A41*S4 + A42*S2 - A43*S1),
		 Di*( A31*S4 - A32*S2 + A33*S1) ] ], ?ELEM_TYPE)
    catch
	error:badarith ->
	    matrix:print(A),
	    error(det_zero)
    end.

%% invert matrix white consider matrix as 3x3 matrix
-spec invert33(A::mat44f()) -> mat44f().
invert33(A) ->
    [[A11,A12,A13|_],
     [A21,A22,A23|_],
     [A31,A32,A33|_],
     _ ] = matrix:to_list(A),

    M11 = A22*A33 - A32*A23,
    M12 = A12*A33 - A32*A13,
    M13 = A12*A23 - A22*A13,
    
    M21 = A21*A33 - A31*A23,
    M22 = A11*A33 - A31*A13,
    M23 = A11*A23 - A21*A13,

    M31 = A21*A32 - A31*A22,
    M32 = A11*A32 - A31*A12,
    M33 = A11*A22 - A21*A12,

    D1 = A11 * M11,
    D2 = A12 * M21,
    D3 = A13 * M31,
    Det = D1 -D2 + D3,

    try 1/Det of 
	Di ->
	    matrix:from_list(
	      [[ Di*M11,-Di*M12, Di*M13, 0],
	       [-Di*M21, Di*M22,-Di*M23, 0],
	       [ Di*M31,-Di*M32, Di*M33, 0],
	       [ 0,      0,      0,      1]],
	      ?ELEM_TYPE)
    catch
	error:badarith ->
	    matrix:print(A),
	    error(det_zero)
    end.
	
-spec divide(A::mat44f(), B::mat44f()) -> mat44f().
divide(A,B) ->
    multiply(A, invert(B)).

print(A) ->
    matrix:print(A).
print(A,Prec) ->
    matrix:print(A,Prec).
print(A,Prec,BENS) ->
    matrix:print(A,Prec,BENS).

format(A) ->
    matrix:format(A).
format(A,Prec) ->
    matrix:format(A,Prec).
format(A,Prec,BENS) ->
    matrix:format(A,Prec,BENS).
