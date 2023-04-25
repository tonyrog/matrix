%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%
%%% @end
%%% Created :  1 Nov 2017 by Tony Rogvall <tony@rogvall.se>

-ifndef(__MATRIX_HRL__).
-define(__MATRIX_HRL__,true).

%% type numbers
%% <<VectorSize:4,ElemSizeExp:3,ScalarType:2>>
%% VectorsSize: 0-15 interpreted as 1-16 where 1 mean scalar only
%% ElemSizeExp: number of bits encoded as two power exponent 2^i+3
%% Scalartype: UINT=2#00, INT=2#01, FLOAT=2#11 FLOAT01=2#10
%%
-define(TYPE_SIZE_BITS,   2).
-define(VECTOR_SIZE_BITS, 4).
-define(ELEM_TYPE_BITS,   5).  %% type+exp

-define(VECTOR_SIZE_MASK, (16#1e0)).
-define(ELEM_SIZE_MASK,   (16#01c)).
-define(SCALAR_TYPE_MASK, (16#003)).
-define(COMP_TYPE_MASK,   (16#01f)).
-define(TYPE_MASK,        (16#1ff)).

-define(make_scalar_type(ElemExpSize,ScalarType),
	(((ElemExpSize) bsl 2) bor (ScalarType))).

-define(make_vector_type2(VectorSize,ElemType),
	((((VectorSize)-1) bsl 5) bor (ElemType))).

-define(make_vector_type(VectorSize,ElemExpSize,ScalarType),
	?make_vector_type2((VectorSize),?make_scalar_type((ElemExpSize),(ScalarType)))).

-define(get_scalar_type(T),       ((T) band ?SCALAR_TYPE_MASK)).
-define(get_comp_type(T),         ((T) band ?COMP_TYPE_MASK)).
-define(get_comp_exp_size(T),     (((T) bsr 2) band 7)).  %% byte exp size
-define(get_comp_exp_bit_size(T), (?get_comp_exp_size((T))+3)).
-define(get_comp_size(T),         (1 bsl ?get_comp_exp_size((T)))).
-define(get_vector_size(T),       (((T) bsr 5)+1)).
-define(get_element_size(T), (?get_vector_size(T)*?get_comp_size(T))).
%% SclarType
-define(UINT,    2#00).
-define(INT,     2#01).
-define(FLOAT01, 2#10).  %% float in range [0.0-1.0]
-define(FLOAT,   2#11).

-define(ELEM_SIZE8,   0).
-define(ELEM_SIZE16,  1).
-define(ELEM_SIZE32,  2).
-define(ELEM_SIZE64,  3).
-define(ELEM_SIZE128, 4).

-define(uint8_t,      ?make_scalar_type(?ELEM_SIZE8,?UINT)).
-define(uint16_t,     ?make_scalar_type(?ELEM_SIZE16,?UINT)).
-define(uint32_t,     ?make_scalar_type(?ELEM_SIZE32,?UINT)).
-define(uint64_t,     ?make_scalar_type(?ELEM_SIZE64,?UINT)).
-define(uint128_t,    ?make_scalar_type(?ELEM_SIZE128,?UINT)).
-define(int8_t,       ?make_scalar_type(?ELEM_SIZE8,?INT)).
-define(int16_t,      ?make_scalar_type(?ELEM_SIZE16,?INT)).
-define(int32_t,      ?make_scalar_type(?ELEM_SIZE32,?INT)).
-define(int64_t,      ?make_scalar_type(?ELEM_SIZE64,?INT)).
-define(int128_t,     ?make_scalar_type(?ELEM_SIZE128,?INT)).
-define(float16_t,    ?make_scalar_type(?ELEM_SIZE16,?FLOAT)).
-define(float32_t,    ?make_scalar_type(?ELEM_SIZE32,?FLOAT)).
-define(float64_t,    ?make_scalar_type(?ELEM_SIZE64,?FLOAT)).

-define(uint8xn_t(N), ?make_vector_type(N,?ELEM_SIZE8,?UINT)).
-define(uint8x2_t,  ?uint8xn_t(2)).
-define(uint8x3_t,  ?uint8xn_t(3)).
-define(uint8x4_t,  ?uint8xn_t(4)).
-define(uint8x8_t,  ?uint8xn_t(8)).
-define(uint8x16_t, ?uint8xn_t(16)).

-define(uint16xn_t(N), ?make_vector_type(N,?ELEM_SIZE16,?UINT)).
-define(uint16x2_t,  ?uint16xn_t(2)).
-define(uint16x3_t,  ?uint16xn_t(3)).
-define(uint16x4_t,  ?uint16xn_t(4)).
-define(uint16x8_t,  ?uint16xn_t(8)).
-define(uint16x16_t, ?uint16xn_t(16)).

-define(uint32xn_t(N), ?make_vector_type(N,?ELEM_SIZE32,?UINT)).
-define(uint32x2_t,  ?uint32xn_t(2)).
-define(uint32x3_t,  ?uint32xn_t(3)).
-define(uint32x4_t,  ?uint32xn_t(4)).
-define(uint32x8_t,  ?uint32xn_t(8)).
-define(uint32x16_t, ?uint32xn_t(16)).

-define(uint64xn_t(N), ?make_vector_type(N,?ELEM_SIZE64,?UINT)).
-define(uint64x2_t,  ?uint64xn_t(2)).
-define(uint64x3_t,  ?uint64xn_t(3)).
-define(uint64x4_t,  ?uint64xn_t(4)).
-define(uint64x8_t,  ?uint64xn_t(8)).
-define(uint64x16_t, ?uint64xn_t(16)).


-define(int8xn_t(N), ?make_vector_type(N,?ELEM_SIZE8,?INT)).
-define(int8x2_t,  ?int8xn_t(2)).
-define(int8x3_t,  ?int8xn_t(3)).
-define(int8x4_t,  ?int8xn_t(4)).
-define(int8x8_t,  ?int8xn_t(8)).
-define(int8x16_t, ?int8xn_t(16)).

-define(int16xn_t(N), ?make_vector_type(N,?ELEM_SIZE16,?INT)).
-define(int16x2_t,  ?int16xn_t(2)).
-define(int16x3_t,  ?int16xn_t(3)).
-define(int16x4_t,  ?int16xn_t(4)).
-define(int16x8_t,  ?int16xn_t(8)).
-define(int16x16_t, ?int16xn_t(16)).

-define(int32xn_t(N), ?make_vector_type(N,?ELEM_SIZE32,?INT)).
-define(int32x2_t,  ?int32xn_t(2)).
-define(int32x3_t,  ?int32xn_t(3)).
-define(int32x4_t,  ?int32xn_t(4)).
-define(int32x8_t,  ?int32xn_t(8)).
-define(int32x16_t, ?int32xn_t(16)).

-define(int64xn_t(N), ?make_vector_type(N,?ELEM_SIZE64,?INT)).
-define(int64x2_t,  ?int64xn_t(2)).
-define(int64x3_t,  ?int64xn_t(3)).
-define(int64x4_t,  ?int64xn_t(4)).
-define(int64x8_t,  ?int64xn_t(8)).
-define(int64x16_t, ?int64xn_t(16)).


-define(float16xn_t(N), ?make_vector_type(N,?ELEM_SIZE16,?FLOAT)).
-define(float16x2_t,  ?float16xn_t(2)).
-define(float16x3_t,  ?float16xn_t(3)).
-define(float16x4_t,  ?float16xn_t(4)).
-define(float16x8_t,  ?float16xn_t(8)).
-define(float16x16_t, ?float16xn_t(16)).

-define(float32xn_t(N), ?make_vector_type(N,?ELEM_SIZE32,?FLOAT)).
-define(float32x2_t,  ?float32xn_t(2)).
-define(float32x3_t,  ?float32xn_t(3)).
-define(float32x4_t,  ?float32xn_t(4)).
-define(float32x8_t,  ?float32xn_t(8)).
-define(float32x16_t, ?float32xn_t(16)).

-define(float64xn_t(N), ?make_vector_type(N,?ELEM_SIZE64,?FLOAT)).
-define(float64x2_t,  ?float64xn_t(2)).
-define(float64x3_t,  ?float64xn_t(3)).
-define(float64x4_t,  ?float64xn_t(4)).
-define(float64x8_t,  ?float64xn_t(8)).
-define(float64x16_t, ?float64xn_t(16)).

-define(uint8(X),     (X):8/native-unsigned-integer).
-define(uint16(X),    (X):16/native-unsigned-integer).
-define(uint32(X),    (X):32/native-unsigned-integer).
-define(uint64(X),    (X):64/native-unsigned-integer).
-define(uint128(X),   (X):128/native-unsigned-integer).
-define(int8(X),      (X):8/native-signed-integer).
-define(int16(X),     (X):16/native-signed-integer).
-define(int32(X),     (X):32/native-signed-integer).
-define(int64(X),     (X):64/native-signed-integer).
-define(int128(X),    (X):128/native-signed-integer).
-define(float16(X),   (X):16/native-float).
-define(float32(X),   (X):32/native-float).
-define(float64(X),   (X):64/native-float).

-type unsigned() :: non_neg_integer().
-type matrix_int_type() :: uint8|uint16|uint32|uint64|uint128.
-type matrix_uint_type() :: uint8|uint16|uint32|uint64|uint128.
-type matrix_float_type() :: float16|float32|float64.

-type matrix_scalar_type() ::
	matrix_int_type() | matrix_uint_type() |
	matrix_float_type().

-type matrix_vector_type() :: 
	int8x2|int8x3|int8x4|int8x8|int8x16|
	int16x2|int16x3|int16x4|int16x8|int16x16|
	int32x2|int32x3|int32x4|int32x8|int32x16|
	int64x2|int64x3|int64x4|int64x8|int64x16|
	uint8x2|uint8x3|uint8x4|uint8x8|uint8x16|
	uint16x2|uint16x3|uint16x4|uint16x8|uint16x16|
	uint32x2|uint32x3|uint32x4|uint32x8|uint32x16|
	uint64x2|uint64x3|uint64x4|uint64x8|uint64x16|
	float16x2|float16x3|float16x4|float16x8|float16x16|
	float32x2|float32x3|float32x4|float32x8|float32x16|
	float64x2|float64x3|float64x4|float64x8|float64x16.

-type matrix_type() :: matrix_scalar_type() | matrix_vector_type().

-type encoded_type() :: 0..?TYPE_MASK.

-type scalar()   :: integer()|float().
-type vector2()  :: {scalar(),scalar()}.
-type vector3()  :: {scalar(),scalar(),scalar()}.
-type vector4()  :: {scalar(),scalar(),scalar(),scalar()}.
-type vector8()  :: {scalar(),scalar(),scalar(),scalar(),
		     scalar(),scalar(),scalar(),scalar()}.
-type vector16() :: {scalar(),scalar(),scalar(),scalar(),
		     scalar(),scalar(),scalar(),scalar(),
		     scalar(),scalar(),scalar(),scalar(),
		     scalar(),scalar(),scalar(),scalar()}.
-type vector()   :: vector2()|vector3()|vector4()|vector8()|vector16().

-type constant() :: scalar() | vector().

-type resource() :: reference() | {unsigned(),binary()} | binary().

-record(matrix,
	{
	 type :: encoded_type(),   %% element type
	 resource :: resource(),   %% native encode raw matrix data
	 n :: unsigned(),          %% rows
	 m :: unsigned(),          %% columns
	 n_stride :: integer(),    %% #bytes in n direction
	 m_stride :: integer(),    %% #bytes in m direction
	 k_stride :: integer(),    %% #bytes per component
	 offset = 0 :: unsigned(), %% byte offset to first element
	 rowmajor :: boolean()     %% stored row-by-row else col-by-col
	}).

-record(matrix_t,
	{
	 type     :: encoded_type(),
	 resource :: resource()
	}).

-type matrix() :: #matrix_t{} | #matrix{}.

-define(is_matrix(X), (is_record((X),matrix) orelse is_record((X),matrix_t))).

%% fixme: check vector elements??? lets crash later for now
-define(is_vector(X), (is_tuple((X)) andalso (tuple_size((X)) > 1) andalso(tuple_size((X)) =< 16))).
-define(is_constant(X), (is_number((X)) orelse ?is_vector(X))).

-define(matrix_cdata_vec(M, Type, Data),
	#matrix { type=(Type) band ?TYPE_MASk,
		  resource=(Data),
		  n=1,
		  m=(M),
		  n_stride=0,
		  m_stride=?get_element_size(Type),
		  k_stride=?get_comp_size(Type),
		  rowmajor=true }).


-endif.



