/*
 * common types 
 */

#ifndef __MATRIX_TYPES_H__
#define __MATRIX_TYPES_H__

#include <stdint.h>
#include <complex.h>

typedef enum {
    FALSE = 0,
    TRUE  = 1
} bool_t;

typedef unsigned char byte_t;
// fixme: configure
typedef float  float32_t;
typedef double float64_t;
typedef float complex complex64_t;
typedef double complex complex128_t;

#define UNUSED(a) ((void) a)
#define VOIDPTR(x) ((void*)&(x))

#define int8_t_zero    (0)
#define int16_t_zero   (0)
#define int32_t_zero   (0)
#define int64_t_zero   (0)
#define float32_t_zero (0.0)
#define float64_t_zero (0.0)
#define complex64_t_zero CMPLXF(0.0,0.0)
#define complex128_t_zero CMPLX(0.0,0.0)

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

#if VSIZE == 1
#define VELEMS(t) 1
#define ALIGN sizeof(void*)

typedef int8_t       vint8_t;
typedef int16_t      vint16_t;
typedef int32_t      vint32_t;
typedef int64_t      vint64_t;
typedef float32_t    vfloat32_t;
typedef float64_t    vfloat64_t;
typedef complex64_t  vcomplex64_t;
typedef complex128_t vcomplex128_t;

#define vint8_t_const(a)       (a)
#define vint16_t_const(a)      (a)
#define vint32_t_const(a)      (a)
#define vint64_t_const(a)      (a)
#define vfloat32_t_const(a)    (a)
#define vfloat64_t_const(a)    (a)
#define vcomplex64_t_const(a)  (a)
#define vcomplex128_t_const(a) (a)

#else
#define USE_VECTOR 1
#define VELEMS(t) (VSIZE/sizeof(t))
#define ALIGN VSIZE
typedef int8_t    vint8_t    __attribute__ ((vector_size (VSIZE)));
typedef int16_t   vint16_t   __attribute__ ((vector_size (VSIZE)));
typedef int32_t   vint32_t   __attribute__ ((vector_size (VSIZE)));
typedef int64_t   vint64_t   __attribute__ ((vector_size (VSIZE)));
typedef float32_t vfloat32_t __attribute__ ((vector_size (VSIZE)));
typedef float64_t vfloat64_t __attribute__ ((vector_size (VSIZE)));
typedef float32_t vcomplex64_t __attribute__ ((vector_size (VSIZE)));
typedef float64_t vcomplex128_t __attribute__ ((vector_size (VSIZE)));

#endif


#if VSIZE == 16
#define vint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vint32_t_const(a)   {(a),(a),(a),(a)}
#define vint64_t_const(a)   {(a),(a)}
#define vfloat32_t_const(a) {(a),(a),(a),(a)}
#define vfloat64_t_const(a) {(a),(a)}
#define vcomplex64_t_const(a) {crealf((a)),cimagf((a)),crealf((a)),cimagf((a))}
#define vcomplex128_t_const(a) {creal((a)),cimag((a))}
#elif VSIZE == 32
#define vint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint32_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vint64_t_const(a)   {(a),(a),(a),(a)}
#define vfloat32_t_const(a) {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vfloat64_t_const(a) {(a),(a),(a),(a)}
#define vcomplex64_t_const(a) {crealf((a)),cimagf((a)),\
	                       crealf((a)),cimagf((a)),\
                               crealf((a)),cimagf((a)),\
	                       crealf((a)),cimagf((a))}
#define vcomplex128_t_const(a) {creal((a)),cimag((a)),\
	                        creal((a)),cimag((a))}
#elif VSIZE == 64
#error "implement me"
#endif

#define vint8_t_zero    vint8_t_const(0)
#define vint16_t_zero   vint16_t_const(0)
#define vint32_t_zero   vint32_t_const(0)
#define vint64_t_zero   vint64_t_const(0)
#define vfloat32_t_zero vfloat32_t_const(0.0)
#define vfloat64_t_zero vfloat64_t_const(0.0)
#define vcomplex64_t_zero vcomplex64_t_const(CMPLXF(0.0,0.0))
#define vcomplex128_t_zero vcomplex128_t_const(CMPLX(0.0,0.0))

#define is_aligned(x) ((((uintptr_t)(x)) & (ALIGN-1)) == 0)

#define align_ptr(ptr,align)						\
    ((byte_t*)((((uintptr_t)((byte_t*)(ptr)))+((align)-1)) & ~((align)-1)))

typedef enum {
    INT8    = 0,
    INT16   = 1,
    INT32   = 2,
    INT64   = 3,
    FLOAT32 = 4,
    FLOAT64 = 5,
    COMPLEX64 = 6,
    COMPLEX128 = 7,
} matrix_type_t;

#define NUM_TYPES (COMPLEX128+1)

// a union to represent all possible scalar data types
typedef union {
    int8_t        i8;
    int16_t       i16;
    int32_t       i32;
    int64_t       i64;
    float32_t     f32;
    float64_t     f64;
    complex64_t   c64;
    complex128_t  c128;
    vint8_t       vi8;
    vint16_t      vi16;
    vint32_t      vi32;
    vint64_t      vi64;
    vfloat32_t    vf32;
    vfloat64_t    vf64;
    vcomplex64_t  vc64;
    vcomplex128_t vc128;
    byte_t        data[VSIZE];
} scalar_t;


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

//  x = A B C D
//  y = E F G H
//
//  a = A B C D
//  b = F F H H
//  c = E E G G
//  d = B A D C
//
//  a*c = AE BE CG DG
//  b*d = -BF AF -DH CH
// +      AE-BF BE+AF CG-DH DG+CH
//

#ifdef __clang__
static inline vcomplex64_t complex64_mul(vcomplex64_t x, vcomplex64_t y)
{
    vcomplex64_t a, b, c, d;
    vcomplex64_t r1,r2;

    a = x;
#if VSIZE == 16
    b = __builtin_shufflevector(y, y, 1, 1, 3, 3);
    c = __builtin_shufflevector(y, y, 0, 0, 2, 2);
    d = __builtin_shufflevector(x, x, 1, 0, 3, 2);
#elif VSIZE == 32
    b = __builtin_shufflevector(y, y, 1, 1, 3, 3, 5, 5, 7, 7);
    c = __builtin_shufflevector(y, y, 0, 0, 2, 2, 4, 4, 6, 6);
    d = __builtin_shufflevector(x, x, 1, 0, 3, 2, 5, 4, 7, 6);
#endif
    r1 = a*c;
    r2 = b*d;
    return addsub_32(r1,r2);
}
#else // gcc 
static inline vcomplex64_t complex64_mul(vcomplex64_t x, vcomplex64_t y)
{
    vcomplex64_t a, b, c, d;
    vcomplex64_t r1,r2;
#if VSIZE == 16
    vint32_t m1133 = {1,1,3,3};
    vint32_t m0022 = {0,0,2,2};
    vint32_t m1032 = {1,0,3,2};
#elif VSIZE == 32
    vint32_t m1133 = {1,1,3,3,5,5,7,7};
    vint32_t m0022 = {0,0,2,2,4,4,6,6};
    vint32_t m1032 = {1,0,3,2,5,4,7,6};    
#endif
    a = x;
    b = __builtin_shuffle(y, m1133);
    c = __builtin_shuffle(y, m0022);
    d = __builtin_shuffle(x, m1032);
    r1 = a*c;
    r2 = b*d;
    return addsub_32(r1,r2);
}
#endif


//  x = A B
//  y = E F
//
//  a = A B
//  b = F F
//  c = E E
//  d = B A
//
//  a*c =  AE BE
//  b*d = -BF AF
// +      AE-BF BE+AF
//

#ifdef __clang__
static inline vcomplex128_t complex128_mul(vcomplex128_t x,vcomplex128_t y)
{
    vcomplex128_t a, b, c, d;
    vcomplex128_t r1,r2;

    a = x;
#if VSIZE == 16
    b = __builtin_shufflevector(y, y, 1, 1);
    c = __builtin_shufflevector(y, y, 0, 0);
    d = __builtin_shufflevector(x, x, 1, 0);
#elif VSIZE == 32
    b = __builtin_shufflevector(y, y, 1, 1, 3, 3);
    c = __builtin_shufflevector(y, y, 0, 0, 2, 2);
    d = __builtin_shufflevector(x, x, 1, 0, 3, 2);
#endif
    r1 = a*c;
    r2 = b*d;
    return addsub_64(r1,r2);
}
#else
static inline vcomplex128_t complex128_mul(vcomplex128_t x,vcomplex128_t y)
{
    vcomplex128_t a, b, c, d;
    vcomplex128_t r1,r2;
#if VSIZE == 16
    vint64_t m11 = {1,1};
    vint64_t m00 = {0,0};
    vint64_t m10 = {1,0};
#elif VSIZE == 32
    vint64_t m11 = {1,1,3,3};
    vint64_t m00 = {0,0,2,2};
    vint64_t m10 = {1,0,3,2};
#endif
    a = x;
    b = __builtin_shuffle(y, m11);
    c = __builtin_shuffle(y, m00);
    d = __builtin_shuffle(x, m10);
    r1 = a*c;
    r2 = b*d;
    return addsub_64(r1,r2);
}
#endif


static inline vcomplex64_t complex64_add(vcomplex64_t x, vcomplex64_t y)
{
    return x+y;
}

static inline vcomplex64_t complex64_sub(vcomplex64_t x, vcomplex64_t y)
{
    return x-y;
}

static inline complex64_t complex64_velement(vcomplex64_t x, int i)
{
#if VSIZE == 1
    return x;
#else
    return CMPLXF(x[2*i], x[2*i+1]);
#endif
}

#if 0
static inline void complex64_vsetelement(vcomplex64_t* xp, int i, complex64_t v)
{
#if VSIZE == 1
    *xp = v;
#else
    (*xp)[2*i] = crealf(v);
    (*xp)[2*i+1] = cimagf(v);
#endif
}
#endif

static inline vcomplex64_t complex64_neg(vcomplex64_t x)
{
    return -x;
}

static inline vcomplex128_t complex128_add(vcomplex128_t x, vcomplex128_t y)
{
    return x+y;
}

static inline vcomplex128_t complex128_sub(vcomplex128_t x,
					   vcomplex128_t y)
{
    return x-y;
}

static inline vcomplex128_t complex128_neg(vcomplex128_t x)
{
    return -x;
}

static inline complex128_t complex128_velement(vcomplex128_t x, int i)
{
#if VSIZE == 1
    return x;
#else
    return CMPLX(x[2*i], x[2*i+1]);
#endif
}

#if 0
static inline void complex128_vsetelement(vcomplex128_t* xp, int i,
					  complex128_t v)
{
#if VSIZE == 1
    *xp = v;
#else
    (*xp)[2*i]   = creal(v);
    (*xp)[2*i+1] = cimag(v);
#endif
}
#endif

#endif
