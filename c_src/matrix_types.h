/*
 * common types 
 */

#ifndef __MATRIX_TYPES_H__
#define __MATRIX_TYPES_H__

#include <stdint.h>

typedef enum {
    FALSE = 0,
    TRUE  = 1
} bool_t;

typedef unsigned char byte_t;
// fixme: configure
typedef float  float32_t;
typedef double float64_t;
typedef struct { int64_t hi; int64_t lo; } int128_t;
typedef struct { uint64_t hi; uint64_t lo; } uint128_t;

#define UNUSED(a) ((void) a)
#define VOIDPTR(x) ((void*)&(x))

#define int8_t_zero    (0)
#define int16_t_zero   (0)
#define int32_t_zero   (0)
#define int64_t_zero   (0)
#define float32_t_zero (0.0)
#define float64_t_zero (0.0)

#if defined(__AVX512F__)
#define VSIZE 64
#elif defined(__AVX2__)
#define VSIZE 32
#elif defined(__AVX__)
#define VSIZE 32
#elif defined(__SSE__)
#define VSIZE 16
#elif defined(__ARM_NEON__)
#define VSIZE 16
#else
#define VSIZE 1
#endif

typedef uint16_t float16_t; // simulated

#if VSIZE == 1
#define VELEMS(t) 1
#define ALIGN sizeof(void*)

typedef uint8_t       vuint8_t;
typedef uint16_t      vuint16_t;
typedef uint32_t      vuint32_t;
typedef uint64_t      vuint64_t;
typedef int8_t       vint8_t;
typedef int16_t      vint16_t;
typedef int32_t      vint32_t;
typedef int64_t      vint64_t;
typedef int128_t     vint128_t;
typedef float16_t    vfloat16_t;
typedef float32_t    vfloat32_t;
typedef float64_t    vfloat64_t;

#define vint8_t_const(a)       (a)
#define vint16_t_const(a)      (a)
#define vint32_t_const(a)      (a)
#define vint64_t_const(a)      (a)
#define vuint8_t_const(a)       (a)
#define vuint16_t_const(a)      (a)
#define vuint32_t_const(a)      (a)
#define vuint64_t_const(a)      (a)
#define vint128_t_const(a)     {(0),(a)}
#define vfloat16_t_const(a)    (a)
#define vfloat32_t_const(a)    (a)
#define vfloat64_t_const(a)    (a)

#else
#define USE_VECTOR 1
#define VELEMS(t) (VSIZE/sizeof(t))
#define ALIGN VSIZE

typedef uint8_t    vuint8_t    __attribute__ ((vector_size (VSIZE)));
typedef uint16_t   vuint16_t   __attribute__ ((vector_size (VSIZE)));
typedef uint32_t   vuint32_t   __attribute__ ((vector_size (VSIZE)));
typedef uint64_t   vuint64_t   __attribute__ ((vector_size (VSIZE)));
typedef uint64_t  vuint128_t  __attribute__ ((vector_size (VSIZE)));
typedef int8_t    vint8_t    __attribute__ ((vector_size (VSIZE)));
typedef int16_t   vint16_t   __attribute__ ((vector_size (VSIZE)));
typedef int32_t   vint32_t   __attribute__ ((vector_size (VSIZE)));
typedef int64_t   vint64_t   __attribute__ ((vector_size (VSIZE)));
typedef int64_t  vint128_t  __attribute__ ((vector_size (VSIZE)));
typedef float16_t vfloat16_t __attribute__ ((vector_size (VSIZE)));
typedef float32_t vfloat32_t __attribute__ ((vector_size (VSIZE)));
typedef float64_t vfloat64_t __attribute__ ((vector_size (VSIZE)));

#endif


#if VSIZE == 16
#define vint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vint32_t_const(a)   {(a),(a),(a),(a)}
#define vint64_t_const(a)   {(a),(a)}

#define vuint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vuint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vuint32_t_const(a)   {(a),(a),(a),(a)}
#define vuint64_t_const(a)   {(a),(a)}

#define vint128_t_const(a)  {{(0),(a)},{(0),(a)}}
#define vfloat16_t_const(a) {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vfloat32_t_const(a) {(a),(a),(a),(a)}
#define vfloat64_t_const(a) {(a),(a)}

#elif VSIZE == 32
#define vint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint32_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vint64_t_const(a)   {(a),(a),(a),(a)}

#define vuint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vuint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vuint32_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vuint64_t_const(a)   {(a),(a),(a),(a)}

#define vint128_t_const(a)  {{(0),(a)},{(0),(a)},{(0),(a)},{(0),(a)}}
#define vfloat16_t_const(a) {(a),(a),(a),(a),(a),(a),(a),(a), \
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vfloat32_t_const(a) {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vfloat64_t_const(a) {(a),(a),(a),(a)}

#elif VSIZE == 64
#error "implement me"
#endif

#define vint8_t_zero    vint8_t_const(0)
#define vint16_t_zero   vint16_t_const(0)
#define vint32_t_zero   vint32_t_const(0)
#define vint64_t_zero   vint64_t_const(0)
#define vint128_t_zero  vint128_t_const(0)
#define vuint8_t_zero    vuint8_t_const(0)
#define vuint16_t_zero   vuint16_t_const(0)
#define vuint32_t_zero   vuint32_t_const(0)
#define vuint64_t_zero   vuint64_t_const(0)
#define vfloat16_t_zero vfloat16_t_const(0.0)
#define vfloat32_t_zero vfloat32_t_const(0.0)
#define vfloat64_t_zero vfloat64_t_const(0.0)

#define is_aligned(x) ((((uintptr_t)(x)) & (ALIGN-1)) == 0)

#define align_ptr(ptr,align)						\
    ((byte_t*)((((uintptr_t)((byte_t*)(ptr)))+((align)-1)) & ~((align)-1)))

// type numbers
// <<VectorSize:4,ElemSizeExp:3,BaseType:2>>
// VectorsSize: 0-15 interpreted as 1-16 where 1 mean scalar only
// ElemSizeExp: number of bits encoded as two power exponent 2^i+3
// ScalarType: UINT=2#00, INT=2#01, FLOAT=2#11 FLOAT01=2#10
// base type
#define UINT     0
#define INT      1
#define FLOAT01  2
#define FLOAT    3

#define VECTOR_SIZE_MASK (0x1e0)
#define ELEM_SIZE_MASK   (0x01c)
#define BASE_TYPE_MASK   (0x003)
#define SCALAR_TYPE_MASK (0x01f)
#define TYPE_MASK        (0x1ff)

#define TYPE_SIZE_BITS 2 
#define VECTOR_SIZE_BITS 4
#define ELEM_TYPE_BITS   5

#define make_scalar_type(scalar_exp_size,base_type) (((scalar_exp_size)<<2)|(base_type))
#define make_vector_type2(vector_size,scalar_type) \
    ((((vector_size)-1) << 5) | (scalar_type))
#define make_vector_type(vector_size,scalar_exp_size,base_type) \
    make_vector_type2((vector_size),make_scalar_type(scalar_exp_size,base_type))

#define get_base_type(t)           ((t) & BASE_TYPE_MASK)
#define get_scalar_type(t)         ((t) & SCALAR_TYPE_MASK)
#define get_scalar_exp_size(t)     (((t) >> 2) & 7)  // byte exp size
#define get_scalar_exp_bit_size(t) (get_scalar_exp_size((t))+3)
#define get_scalar_size(t)         (1 << get_scalar_exp_size((t)))
#define get_vector_size(t)         (((t) >> 5)+1)

#define ELEM_SIZE8   0
#define ELEM_SIZE16  1
#define ELEM_SIZE32  2
#define ELEM_SIZE64  3
#define ELEM_SIZE128 4

#define    UINT8    make_scalar_type(ELEM_SIZE8,UINT)
#define    UINT16   make_scalar_type(ELEM_SIZE16,UINT)
#define    UINT32   make_scalar_type(ELEM_SIZE32,UINT)
#define    UINT64   make_scalar_type(ELEM_SIZE64,UINT)
#define    UINT128  make_scalar_type(ELEM_SIZE128,UINT)
#define    INT8    make_scalar_type(ELEM_SIZE8,INT)
#define    INT16   make_scalar_type(ELEM_SIZE16,INT)
#define    INT32   make_scalar_type(ELEM_SIZE32,INT)
#define    INT64   make_scalar_type(ELEM_SIZE64,INT)
#define    INT128  make_scalar_type(ELEM_SIZE128,INT)
#define    FLOAT16 make_scalar_type(ELEM_SIZE16,FLOAT)
#define    FLOAT32 make_scalar_type(ELEM_SIZE32,FLOAT)
#define    FLOAT64 make_scalar_type(ELEM_SIZE64,FLOAT)

typedef uint32_t  matrix_type_t;

// num scalar types, used when defining tables
#define NUM_TYPES (1 << ELEM_TYPE_BITS)

// type bits
#define UINT8_TYPE     (1 << UINT8)
#define UINT16_TYPE    (1 << UINT16)
#define UINT32_TYPE    (1 << UINT32)
#define UINT64_TYPE    (1 << UINT64)
#define UINT128_TYPE   (1 << UINT128)
#define UINT_TYPES (UINT8_TYPE|UINT16_TYPE|UINT32_TYPE|UINT64_TYPE|UINT128_TYPE)

#define INT8_TYPE     (1 << INT8)
#define INT16_TYPE    (1 << INT16)
#define INT32_TYPE    (1 << INT32)
#define INT64_TYPE    (1 << INT64)
#define INT128_TYPE   (1 << INT128)
#define INT_TYPES (INT8_TYPE|INT16_TYPE|INT32_TYPE|INT64_TYPE|INT128_TYPE)

#define FLOAT32_TYPE  (1 << FLOAT32)
#define FLOAT64_TYPE  (1 << FLOAT64)
#define FLOAT_TYPES (FLOAT32_TYPE|FLOAT64_TYPE)

#define NONE_INTEGER_TYPES (FLOAT_TYPES)
#define ALL_TYPES (UINT_TYPES|INT_TYPES|FLOAT_TYPES)

#define IS_VECTOR(t) (get_vector_size(t) > 1)
#define IS_SCALAR(t) (get_vector_size(t) == 0)

#define IS_INTEGER_TYPE(t) (((t) & ~ELEM_SIZE_MASK) == INT)
#define IS_FLOAT_TYPE(t)   (((t) & ~ELEM_SIZE_MASK) == FLOAT)

// work for scalar types only!
#define IS_TYPE(t, types)  ( ((1<<(t)) & (types)) != 0 )

#define IS_NONE_INTEGER_TYPE(t)   ( ((1<<(t)) & NONE_INTEGER_TYPES) != 0 )

typedef uint32_t matrix_type_flags_t;

// a union to represent all possible scalar data types
#define MAX_COMPONENTS 16
#define ALLOC_COMPONENTS 32  // works for VSIZE=16,32

typedef union {
    uint8_t   u8;
    uint16_t  u16;
    uint32_t  u32;
    uint64_t  u64;
    uint128_t u128;    
    int8_t    i8;
    int16_t   i16;
    int32_t   i32;
    int64_t   i64;
    int128_t  i128;    
    float16_t f16;
    float32_t f32;
    float64_t f64;
    uint8_t   vu8[MAX_COMPONENTS];
    uint16_t  vu16[MAX_COMPONENTS];
    uint32_t  vu32[MAX_COMPONENTS];
    uint64_t  vu64[MAX_COMPONENTS];
    int8_t    vi8[MAX_COMPONENTS];
    int16_t   vi16[MAX_COMPONENTS];
    int32_t   vi32[MAX_COMPONENTS];
    int64_t   vi64[MAX_COMPONENTS];
    float16_t vf16[MAX_COMPONENTS];
    float32_t vf32[MAX_COMPONENTS];
    float64_t vf64[MAX_COMPONENTS];
    byte_t    data[sizeof(float64_t)*MAX_COMPONENTS];
} scalar_t;

typedef union {
    vuint8_t      vu8;
    vuint16_t     vu16;
    vuint32_t     vu32;
    vuint64_t     vu64;
    vuint128_t    vu128;    
    vint8_t       vi8;
    vint16_t      vi16;
    vint32_t      vi32;
    vint64_t      vi64;
    vint128_t     vi128;
    vfloat16_t    vf16;
    vfloat32_t    vf32;
    vfloat64_t    vf64;
} vscalar_t;

typedef vint8_t vector_t;
typedef void (*unary_op_t)(void* src, void* dst);
typedef void (*binary_op_t)(void* src1, void* src2, void* dst);

typedef void (*unary_vop_t)(void* src, void* dst);
typedef void (*binary_vop_t)(void* src1, void* src2, void* dst);

#if defined(__x86_64__) && (VSIZE == 16)
// FIXME use m256_addsub_ps/pd  for VSIZE==32
#if defined(__SSE3__)
#include <pmmintrin.h>
#endif

static inline vfloat32_t addsub_32(vfloat32_t x, vfloat32_t y)
{
    return (vfloat32_t) __builtin_ia32_addsubps(x,y);
}

static inline vfloat64_t addsub_64(vfloat64_t x, vfloat64_t y)
{
    return (vfloat64_t) __builtin_ia32_addsubpd(x,y);
}

#if 0
// return x & y
static inline vint128_t vop128_band(vint128_t x, vint128_t y)
{
    return (vint128_t) _mm_and_si128(x, y);
}

// return ~x & y
static inline vint128_t vop128_bandn(vint128_t x, vint128_t y)
{
    return (vint128_t) _mm_andnot_si128(x, y);
}

// return x | y
static inline vint128_t vop128_bor(vint128_t x, vint128_t y)
{
    return (vint128_t) _mm_or_si128(x, y);
}

// return x ^ y
static inline vint128_t vop128_bxor(vint128_t x, vint128_t y)
{
    return (vint128_t) _mm_xor_si128(x, y);
}

// return ~x
static inline vint128_t vop128_bnot(vint128_t x)
{
    return (vint128_t) _mm_xor_si128(x, -1);
}
#endif


#elif defined(__x86_64__) && (VSIZE == 32)

#if defined(__AVX__)
#include <immintrin.h>
#endif

static inline vfloat32_t addsub_32(vfloat32_t x, vfloat32_t y)
{
    return (vfloat32_t) _mm256_addsub_ps(x,y);
}

static inline vfloat64_t addsub_64(vfloat64_t x, vfloat64_t y)
{
    return (vfloat64_t) _mm256_addsub_pd(x,y);
}

#else

static inline vfloat32_t addsub_32(vfloat32_t x, vfloat32_t y)
{
    const vfloat32_t neg = vfloat32_t_const(-1.0);
    return x + neg*y;
}

static inline vfloat64_t addsub_64(vfloat64_t x, vfloat64_t y)
{
    const vfloat64_t neg = vfloat64_t_const(-1.0);
    return x + neg*y;
}

#endif  // __x86_64


#endif
