//
// Matrix operations
//
#include <stdio.h>
#include <stdint.h>
#include <memory.h>
#include <math.h>
#include <complex.h>
#include <unistd.h>
#include <fcntl.h>

#include "erl_nif.h"

#define USE_GCC_VECTOR
// #define DEBUG

#ifdef DEBUG
#include <stdio.h>
#define DBG(...) printf(__VA_ARGS__)
#else
#define DBG(...)
#endif

typedef enum {
    INT8    = 0,
    INT16   = 1,
    INT32   = 2,
    INT64   = 3,
    // INT128 = 4
    FLOAT32 = 5,
    FLOAT64 = 6,
    // FLOAT128 = 7
    COMPLEX64 = 8,
    COMPLEX128 = 9,
} matrix_type_t;

#define MAX_TYPE_NUMBER COMPLEX128

typedef enum {
    ZERO,
    ONE,
    COPY,
    NEGATE,    
    IDENTITY,
    SIGMOID,
    SIGMOID_PRIME,
    RECTIFIER,
    TANH,
    UNIFORM,
    NORMAL,
} unary_operation_t;

typedef enum {
    PLUS,
    MINUS,
    TIMES,
} binary_operation_t;

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

#ifdef USE_GCC_VECTOR
#define VSIZE 16
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

#define vint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vint32_t_const(a)   {(a),(a),(a),(a)}
#define vint64_t_const(a)   {(a),(a)}
#define vfloat32_t_const(a) {(a),(a),(a),(a)}
#define vfloat64_t_const(a) {(a),(a)}
#define vcomplex64_t_const(a) {crealf((a)),cimagf((a)),crealf((a)),cimagf((a))}
#define vcomplex128_t_const(a) {creal((a)),cimag((a))}

#define vint8_t_zero    vint8_t_const(0)
#define vint16_t_zero   vint16_t_const(0)
#define vint32_t_zero   vint32_t_const(0)
#define vint64_t_zero   vint64_t_const(0)
#define vfloat32_t_zero vfloat32_t_const(0)
#define vfloat64_t_zero vfloat64_t_const(0)
#define vcomplex64_t_zero vcomplex64_t_const(CMPLX(0.0,0.0))
#define vcomplex128_t_zero vcomplex128_t_const(CMPLXF(0.0,0.0))
#else

#define VSIZE     1
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

#define vint8_t_const(a)    (a)
#define vint16_t_const(a)   (a)
#define vint32_t_const(a)   (a)
#define vint64_t_const(a)   (a)
#define vfloat32_t_const(a) (a)
#define vfloat64_t_const(a) (a)
#define vcomplex64_t_const(a) CMPLX((a),(0))
#define vcomplex128_t_const(a) CMPLXF((a),(0))

#define vint8_t_zero    vint8_t_const(0)
#define vint16_t_zero   vint16_t_const(0)
#define vint32_t_zero   vint32_t_const(0)
#define vint64_t_zero   vint64_t_const(0)
#define vfloat32_t_zero vfloat32_t_const(0)
#define vfloat64_t_zero vfloat64_t_const(0)
#define vcomplex64_t_zero vcomplex64_t_const(0)
#define vcomplex128_t_zero vcomplex128_t_const(0)

#endif

#define is_aligned(x) ((((uintptr_t)(x)) & (ALIGN-1)) == 0)

#define align_ptr(ptr,align)						\
    ((byte_t*)((((uintptr_t)((byte_t*)(ptr)))+((align)-1)) & ~((align)-1)))


#define ATOM(name) atm_##name

#define DECL_ATOM(name) \
    ERL_NIF_TERM atm_##name = 0

// require env in context (ugly)
#define LOAD_ATOM(name)			\
    atm_##name = enif_make_atom(env,#name)

#define LOAD_ATOM_STRING(name,string)		\
    atm_##name = enif_make_atom(env,string)

#define CAT_HELPER3(p,x,y) p ## x ## y
#define CAT3(p,x,y) CAT_HELPER3(p,x,y)

#define CAT_HELPER2(x,y) x ## y
#define CAT2(x,y) CAT_HELPER2(x,y)

#define MT_NAME(p,x) CAT3(p,NAME,x)
#define VTYPE        CAT2(v,TYPE)
#define VTYPE_ZERO   CAT3(v,TYPE,_zero)
#define VTYPE_CONST(name) CAT3(v,TYPE,_const)(name)

    
typedef struct {
    unsigned int n;
    unsigned int m;
    matrix_type_t type;
    size_t size;         // allocated memory size
    unsigned int offset; // offset to first element
    unsigned int stride; // stride elements per row
    unsigned int byte_offset; // offset to first element in bytes
    unsigned int byte_stride; // stride bytes per row
    bool_t       rowmajor;    // stored row-by-row
    ErlNifRWLock* rw_lock;    // make sure we can write "safe"
    byte_t* base;        // allocated memory
    byte_t*  data;       // aligned data
} matrix_t;    

static ErlNifResourceType* matrix_r;
static ErlNifTSDKey matrix_k;

static int matrix_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
static int matrix_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data,
		       ERL_NIF_TERM load_info);
static void matrix_unload(ErlNifEnv* env, void* priv_data);

static ERL_NIF_TERM matrix_create(ErlNifEnv* env, int argc, 
				  const ERL_NIF_TERM argv[]); 
static ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, 
			       const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc, 
				    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_times(ErlNifEnv* env, int argc, 
				 const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_multiply(ErlNifEnv* env, int argc, 
				    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_negate(ErlNifEnv* env, int argc, 
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_scale(ErlNifEnv* env, int argc,
				 const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_transpose_data(ErlNifEnv* env, int argc, 
					  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sigmoid(ErlNifEnv* env, int argc, 
				   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_rectifier(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_copy(ErlNifEnv* env, int argc, 
				const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_fill(ErlNifEnv* env, int argc, 
				const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc, 
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_argmax(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);

#if (ERL_NIF_MAJOR_VERSION > 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION >= 12))
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(ERL_NIF_DIRTY_JOB_CPU_BOUND)}
//#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(0)}
#elif (ERL_NIF_MAJOR_VERSION > 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION >= 7))
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(0)}
#else
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr)}
#endif

ErlNifFunc matrix_funcs[] =
{
    NIF_FUNC("create_",       5, matrix_create),
    NIF_FUNC("add_",          2, matrix_add),
    NIF_FUNC("add_",          3, matrix_add),
    NIF_FUNC("subtract_",     2, matrix_subtract),
    NIF_FUNC("subtract_",     3, matrix_subtract),    
    NIF_FUNC("times_",        2, matrix_times),
    NIF_FUNC("times_",        3, matrix_times),
    NIF_FUNC("multiply_",     2, matrix_multiply),
    NIF_FUNC("multiply_",     3, matrix_multiply),
    NIF_FUNC("negate",        1, matrix_negate),
    NIF_FUNC("negate",        2, matrix_negate),
    NIF_FUNC("scale",         2, matrix_scale),
    NIF_FUNC("scale",         3, matrix_scale),
    NIF_FUNC("transpose_data",1, matrix_transpose_data),
    NIF_FUNC("transpose_data",2, matrix_transpose_data),
    NIF_FUNC("sigmoid",       1, matrix_sigmoid),
    NIF_FUNC("sigmoid_prime", 1, matrix_sigmoid_prime),
    NIF_FUNC("rectifier",     1, matrix_rectifier),
    NIF_FUNC("copy",          1, matrix_copy),
    NIF_FUNC("copy",          2, matrix_copy),
    NIF_FUNC("copy",          4, matrix_copy),
    NIF_FUNC("fill",          2, matrix_fill),
    NIF_FUNC("apply1",        3, matrix_apply1),
    NIF_FUNC("argmax",        2, matrix_argmax),
};

size_t element_size_exp_[10] = { 0, 1, 2, 3, 4,  2, 3, 4,  3, 4 };
size_t element_size_[10]     = { 1, 2, 4, 8, 16, 4, 8, 16, 8, 16 };

typedef enum {
    XOR_SHIFT_32,
    XOR_SHIFT_128,
    XOR_SHIFT_64_STAR,
    XOR_SHIFT_1024_STAR
} rand_alg_t;

#define MATRIX_RAND_ALG XOR_SHIFT_32

typedef struct _rand_state_t {
    size_t     size;  // 32 | 64
    rand_alg_t alg;
    uint32_t (*rand_32)(struct _rand_state_t*);
    uint64_t (*rand_64)(struct _rand_state_t*);
    int p;
    uint64_t s[16];
} rand_state_t;

DECL_ATOM(matrix);
DECL_ATOM(sigmoid);
DECL_ATOM(sigmoid_prime);
DECL_ATOM(rectifier);
DECL_ATOM(tanh);
DECL_ATOM(negate);
DECL_ATOM(uniform);
DECL_ATOM(normal);
DECL_ATOM(zero);
DECL_ATOM(one);
DECL_ATOM(identity);
DECL_ATOM(copy);
DECL_ATOM(true);
DECL_ATOM(false);

static size_t element_size(matrix_type_t type)
{
    return element_size_[type];
}

static int is_integer(matrix_type_t type)
{
    return (type >= INT8) && (type <= INT64);
}

static int is_float(matrix_type_t type)
{
    return (type >= FLOAT32) && (type <= FLOAT64);
}

static int is_complex(matrix_type_t type)
{
    return (type >= COMPLEX64) && (type <= COMPLEX128);
}

static matrix_type_t combine_type(matrix_type_t at, matrix_type_t bt)
{
    return (at > bt) ? at : bt;
}

// read and convert/trunc a number to a integer
static int64_t read_int(matrix_type_t type, byte_t* ptr)
{
    switch(type) {
    case INT8:    return (int64_t) *((int8_t*)ptr);
    case INT16:   return (int64_t) *((int16_t*)ptr);
    case INT32:   return (int64_t) *((int32_t*)ptr);
    case INT64:   return (int64_t) *((int64_t*)ptr);
    case FLOAT32: return (int64_t) *((float32_t*)ptr);
    case FLOAT64: return (int64_t) *((float64_t*)ptr);
    case COMPLEX64: return (int64_t) *((float32_t*)ptr);
    case COMPLEX128: return (int64_t) *((float64_t*)ptr);
    default: return 0;  // fixme: fail
    }
}

// convert and write an integer to matrix memory
static void write_int(matrix_type_t type, byte_t* ptr, int64_t v)
{
    switch(type) {
    case INT8:    *((int8_t*)ptr) = (int8_t) v;  break;
    case INT16:   *((int16_t*)ptr) = (int16_t) v;  break;
    case INT32:   *((int32_t*)ptr) = (int32_t) v;  break;
    case INT64:   *((int64_t*)ptr) = (int64_t) v;  break;
    case FLOAT32: *((float32_t*)ptr) = (float32_t) v; break;
    case FLOAT64: *((float64_t*)ptr) = (float64_t) v;  break;
    case COMPLEX64:
	((float32_t*)ptr)[0] = (float32_t) v;
	((float32_t*)ptr)[1] = 0.0;
	break;
    case COMPLEX128:
	((float64_t*)ptr)[0] = (float64_t) v;
	((float64_t*)ptr)[1] = 0.0;
	break;
    default: break;
    }
}

// read and convert/trunc a number to a integer
static float64_t read_float(matrix_type_t type, byte_t* ptr)
{
    switch(type) {
    case INT8:    return (float64_t) *((int8_t*)ptr);
    case INT16:   return (float64_t) *((int16_t*)ptr);
    case INT32:   return (float64_t) *((int32_t*)ptr);
    case INT64:   return (float64_t) *((int64_t*)ptr);
    case FLOAT32: return (float64_t) *((float32_t*)ptr);
    case FLOAT64: return (float64_t) *((float64_t*)ptr);
    case COMPLEX64: return (float64_t) *((float32_t*)ptr);
    case COMPLEX128: return (float64_t) *((float64_t*)ptr);
    default: return 0;  // fixme: fail
    }
}

// convert and write an integer to matrix memory
static void write_float(matrix_type_t type, byte_t* ptr, float64_t v)
{
    switch(type) {
    case INT8:    *((int8_t*)ptr) = (int8_t) v;  break;
    case INT16:   *((int16_t*)ptr) = (int16_t) v;  break;
    case INT32:   *((int32_t*)ptr) = (int32_t) v;  break;
    case INT64:   *((int64_t*)ptr) = (int64_t) v;  break;
    case FLOAT32: *((float32_t*)ptr) = (float32_t) v; break;
    case FLOAT64: *((float64_t*)ptr) = (float64_t) v;  break;
    case COMPLEX64:
	((float32_t*)ptr)[0] = (float32_t) v;
	((float32_t*)ptr)[1] = 0.0;
	break;
    case COMPLEX128:
	((float64_t*)ptr)[0] = (float64_t) v;
	((float64_t*)ptr)[1] = 0.0;
	break;
    default: break;
    }
}

// read and convert/trunc a number to a integer
static complex128_t read_complex(matrix_type_t type, byte_t* ptr)
{
    switch(type) {
    case INT8:    return (complex128_t) *((int8_t*)ptr);
    case INT16:   return (complex128_t) *((int16_t*)ptr);
    case INT32:   return (complex128_t) *((int32_t*)ptr);
    case INT64:   return (complex128_t) *((int64_t*)ptr);
    case FLOAT32: return (complex128_t) *((float32_t*)ptr);
    case FLOAT64: return (complex128_t) *((float64_t*)ptr);
    case COMPLEX64: return (complex128_t) *((complex64_t*)ptr);
    case COMPLEX128: return (complex128_t) *((complex128_t*)ptr);
    default: return 0;  // fixme: fail
    }
}

// convert and write an integer to matrix memory
static void write_complex(matrix_type_t type, byte_t* ptr, complex128_t v)
{
    switch(type) {
    case INT8:    *((int8_t*)ptr) = (int8_t) creal(v);  break;
    case INT16:   *((int16_t*)ptr) = (int16_t) creal(v);  break;
    case INT32:   *((int32_t*)ptr) = (int32_t) creal(v);  break;
    case INT64:   *((int64_t*)ptr) = (int64_t) creal(v);  break;
    case FLOAT32: *((float32_t*)ptr) = (float32_t) creal(v); break;
    case FLOAT64: *((float64_t*)ptr) = (float64_t) creal(v);  break;
    case COMPLEX64: *((complex64_t*)ptr) = (complex64_t) v; break;
    case COMPLEX128: *((complex64_t*)ptr) = v; break;
    default: break;
    }
}

#ifdef USE_GCC_VECTOR
#ifdef __x86_64__

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
#else

static inline vfloat32_t addsub_32(vfloat32_t x, vfloat32_t y)
{
    const vfloat32_t neg = { -1.0f, 1.0f, -1.0f, 1.0f };
    return x + neg*y;
}

static inline vfloat64_t addsub_64(vfloat64_t x, vfloat64_t y)
{
    const vfloat64_t neg = { -1.0f, 1.0f };
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

#if VSIZE != 16
#error "VSIZE = 16 assumed!!! FIXME"
#endif

#ifdef __clang__
static inline vcomplex64_t complex64_multiply(vcomplex64_t x, vcomplex64_t y)
{
    vcomplex64_t a, b, c, d;
    vcomplex64_t r1,r2;

    a = x;
    b = __builtin_shufflevector(y, y, 1, 1, 3, 3);
    c = __builtin_shufflevector(y, y, 0, 0, 2, 2);
    d = __builtin_shufflevector(x, x, 1, 0, 3, 2);
    r1 = a*c;
    r2 = b*d;
    return addsub_32(r1,r2);
}
#else
static inline vcomplex64_t complex64_multiply(vcomplex64_t x, vcomplex64_t y)
{
    vcomplex64_t a, b, c, d;
    vcomplex64_t r1,r2;
    vint32_t m1133 = {1,1,3,3};
    vint32_t m0022 = {0,0,2,2};
    vint32_t m1032 = {1,0,3,2};
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
static inline vcomplex128_t complex128_multiply(vcomplex128_t x,vcomplex128_t y)
{
    vcomplex128_t a, b, c, d;
    vcomplex128_t r1,r2;

    a = x;
    b = __builtin_shufflevector(y, y, 1, 1);
    c = __builtin_shufflevector(y, y, 0, 0);
    d = __builtin_shufflevector(x, x, 1, 0);
    r1 = a*c;
    r2 = b*d;
    return addsub_64(r1,r2);
}
#else
static inline vcomplex128_t complex128_multiply(vcomplex128_t x,vcomplex128_t y)
{
    vcomplex128_t a, b, c, d;
    vcomplex128_t r1,r2;
    vint64_t m11 = {1,1};
    vint64_t m00 = {0,0};
    vint64_t m10 = {1,0};

    a = x;
    b = __builtin_shuffle(y, m11);
    c = __builtin_shuffle(y, m00);
    d = __builtin_shuffle(x, m10);
    r1 = a*c;
    r2 = b*d;
    return addsub_64(r1,r2);
}
#endif

#else  // USE_GCC_VECTOR

// assume vcomplex64_t/vomplex128_t maps to scalar types
static inline vcomplex64_t complex64_multiply(vcomplex64_t x, vcomplex64_t y)
{
    return x*y;
}

static inline vcomplex64_t complex128_multiply(vcomplex128_t x, vcomplex128_t y)
{
    return x*y;
}

#endif  // USE_GCC_VECTOR

static inline vcomplex64_t complex64_add(vcomplex64_t x, vcomplex64_t y)
{
    return x+y;
}

static inline vcomplex64_t complex64_subtract(vcomplex64_t x, vcomplex64_t y)
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

static inline void complex64_vsetelement(vcomplex64_t* xp, int i, complex64_t v)
{
#if VSIZE == 1
    *xp = v;
#else
    (*xp)[2*i] = crealf(v);
    (*xp)[2*i+1] = cimagf(v);
#endif
}

static inline vcomplex64_t complex64_negate(vcomplex64_t x)
{
    return -x;
}

static inline vcomplex128_t complex128_add(vcomplex128_t x, vcomplex128_t y)
{
    return x+y;
}

static inline vcomplex128_t complex128_subtract(vcomplex128_t x, vcomplex128_t y)
{
    return x-y;
}

static inline vcomplex128_t complex128_negate(vcomplex128_t x)
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

#define cop_sigmoid(x)    (1.0/(1.0 + cexp(-(x))))

static inline complex128_t cop_sigmoid_prime(complex128_t x)
{
    complex128_t z = cop_sigmoid(x);
    return z*(1-z);
}

void copy_circular(uint8_t* dst, size_t n, uint8_t* src, size_t m)
{
    if (src == 0)
	return;
    else if (m == 0)
	memset(dst, 0x55, n);
    else {
	size_t i, j=0;
	for (i=0; i<n; i++) {
	    if (j >= m) j=0;
	    dst[i] = src[j++];
	}
    }
}

static uint32_t xorshift32(rand_state_t* state)
{
    uint32_t x = state->s[0];
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state->s[0] = x;
    return x;
}

static uint64_t xorshift128(rand_state_t* state)
{
    uint32_t t = state->s[3];
    t ^= t << 11;
    t ^= t >> 8;
    state->s[3] = state->s[2];
    state->s[2] = state->s[1];
    state->s[1] = state->s[0];
    t ^= state->s[0];
    t ^= state->s[0] >> 19;	
    state->s[0] = t;
    return t;
}

static uint64_t xorshift64star(rand_state_t* state)
{
    uint64_t x = state->s[0];
    x ^= x >> 12; // a
    x ^= x << 25; // b
    x ^= x >> 27; // c
    state->s[0] = x;
    return x * UINT64_C(0x2545F4914F6CDD1D);
}

static uint64_t xorshift1024star(rand_state_t* state)
{
    int p = state->p;
    const uint64_t s0 = state->s[p];
    uint64_t x = state->s[p = (p + 1) & 15];
    x ^= x << 31; // a
    x = x ^ s0 ^ (x >> 11) ^ (s0 >> 30); // b, c
    state->s[p] = x;
    state->p = p;
    return x * UINT64_C(1181783497276652981);
}

uint32_t rand_32_(rand_state_t* sp)
{
    if (sp->size == 32)
	return sp->rand_32(sp);
    else if (sp->size == 64)
	return sp->rand_64(sp);
    return 0;
}

uint64_t rand_64_(rand_state_t* sp)
{
    if (sp->size == 32) {
	uint64_t x;
	x = sp->rand_32(sp);
	x <<= 32;
	x |= sp->rand_32(sp);
	return x;
    }
    else if (sp->size == 64) {
	return sp->rand_64(sp);
    }
    return 0;
}

float32_t uniform_32_(rand_state_t* sp)
{
    uint32_t x;
    union { float32_t xf; uint32_t  xi; } uf;
    if (sp->size == 32)
	x = sp->rand_32(sp);
    else if (sp->size == 64)
	x = sp->rand_64(sp);
    else
	return 0.0;
    uf.xi = (UINT32_C(0x7f)<<23)|(x&UINT64_C(0x7fffff));
    return uf.xf-1;
}

// m + s*sqrtf(-2*logf(x1))*cosf(2*M_PI*x2);
float32_t normal_32_(rand_state_t* sp, float m, float s, float32_t* n2)
{
    float32_t x1, x2, w;

    do {
	x1 = 2.0*uniform_32_(sp) - 1.0;
	x2 = 2.0*uniform_32_(sp) - 1.0;
	w  = x1*x1 + x2*x2;
    } while(w >= 1.0);
    w = sqrtf((-2.0*logf(w))/w);
    if (n2) *n2 = x2*w*s+m;
    return x1*w*s+m;
}

float64_t uniform_64_(rand_state_t* sp)
{
    uint64_t x;
    union { float64_t xf; uint64_t  xi; } uf;
    if (sp->size == 32) {
	x = sp->rand_32(sp);
	x <<= 32;
	x |= sp->rand_32(sp);
    }
    else {
	x = sp->rand_64(sp);
    }
    uf.xi = (UINT64_C(0x3ff)<<52)|(x&UINT64_C(0xfffffffffffff));
    return uf.xf-1;
}

float64_t normal_64_(rand_state_t* sp, float64_t m, float64_t s, float64_t* n2)
{
    float64_t x1, x2, w;

    do {
	x1 = 2.0 * uniform_64_(sp) - 1.0;
	x2 = 2.0 * uniform_64_(sp) - 1.0;
	w  = x1*x1 + x2*x2;
    } while(w >= 1.0);
    w = sqrt((-2.0*log(w))/w);
    if (n2) *n2 = x2*w*s+m;
    return x1*w*s+m;
}

void rand_init(rand_state_t* sp, rand_alg_t a, uint8_t* data, size_t n)
{
    switch(a) {
    case XOR_SHIFT_32:
	sp->alg = a;	
	sp->size = 32;
	sp->rand_32 = xorshift32;
	copy_circular((uint8_t*)&sp->s[0], 1*sizeof(uint64_t), data, n);
	break;
    case XOR_SHIFT_128:
	sp->alg = a;
	sp->size = 64;
	sp->rand_64 = xorshift128;
	copy_circular((uint8_t*)&sp->s[0], 4*sizeof(uint64_t), data, n);
	break;
    case XOR_SHIFT_64_STAR:
	sp->alg = a;	
	sp->size = 64;
	sp->rand_64 = xorshift64star;
	copy_circular((uint8_t*)&sp->s[0], 1*sizeof(uint64_t), data, n);
	break;
    case XOR_SHIFT_1024_STAR:
	sp->alg = a;	
	sp->p = 0;
	sp->size = 64;
	sp->rand_64 = xorshift1024star;
	copy_circular((uint8_t*)&sp->s[0], 16*sizeof(uint64_t), data, n);
	break;
    default:
	break;
    }
}

int rand_bits(uint8_t* data, int n)
{
    int fd = open("/dev/random", O_RDONLY);
    if (fd < 0) return -1;
    if (read(fd, data, n) < n) { close(fd); return -1; }
    close(fd);
    return 0;
}

rand_state_t* get_tsd_rand_state(rand_alg_t a)
{
    rand_state_t* sp;
    if ((sp=enif_tsd_get(matrix_k)) == NULL) {
	uint8_t data[128];
	rand_bits(data, sizeof(data));
	sp = enif_alloc(sizeof(rand_state_t));
	rand_init(sp, a, data, sizeof(data));
	enif_tsd_set(matrix_k, sp);
    }
    else if (sp->alg != a) {
	rand_init(sp, a, NULL, 0);
    }
    return sp;
}

uint32_t rand_32(rand_alg_t a)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return rand_32_(sp);
}

uint64_t rand_64(rand_alg_t a)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return rand_64_(sp);
}

float32_t uniform_32(rand_alg_t a)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return uniform_32_(sp);
}

float64_t uniform_64(rand_alg_t a)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return uniform_64_(sp);
}

complex64_t uniform_c64(rand_alg_t a)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    float32_t x = uniform_32_(sp);
    float32_t r = uniform_32_(sp);
    complex64_t y = CMPLXF(r*cosf(x*M_PI),r*sinf(x*M_PI));
    return y;
}

complex128_t uniform_c128(rand_alg_t a)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    float64_t x = uniform_64_(sp);
    float64_t r = uniform_64_(sp);
    complex128_t y = CMPLX(r*cos(x*M_PI),r*sin(x*M_PI));
    return y;
}

float32_t normal_32(rand_alg_t a, float m, float s)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return normal_32_(sp, m, s, NULL);
}

float64_t normal_64(rand_alg_t a, float64_t m, float64_t s)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return normal_64_(sp, m, s, NULL);
}

complex64_t normal_c64(rand_alg_t a, float32_t m, float32_t s)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    float32_t n1, n2;
    n1 = normal_32_(sp, m, s, &n2);
    complex64_t r = CMPLXF(n1,n2);
    r *= sqrtf(s/2.0);
    return r;
}

complex128_t normal_c128(rand_alg_t a, float64_t m, float64_t s)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    float64_t n1, n2;
    n1 = normal_64_(sp, m, s, &n2);
    complex128_t r = CMPLX(n1,n2);
    r *= sqrt(s/2.0);
    return r;
}

// functional macros of arithmetic operators
// supported vector operators: +, -, *, /, unary minus, ^, |, &, ~, %.
// shift operators: << and >> for integer vectors
// comparison operators: ==, !=, <, <=, >, >=
#define op_plus(x,y)  ((x)+(y))
#define op_minus(x,y) ((x)-(y))
#define op_times(x,y)   ((x)*(y))
#define op_div(x,y)   ((x)/(y))
#define op_rem(x,y)   ((x)%(y))
#define op_bxor(x,y)  ((x)^(y))
#define op_bor(x,y)   ((x)|(y))
#define op_band(x,y)  ((x)&(y))
#define op_negate(x)  (-(x))
#define op_bnot(x)    (~(x))
#define op_bsl(x,y)   ((x)<<(y))
#define op_bsr(x,y)   ((x)>>(y))
#define op_eq(x,y)    ((x)==(y))
#define op_neq(x,y)   ((x)!=(y))
#define op_lt(x,y)    ((x)<(y))
#define op_lte(x,y)   ((x)<=(y))
#define op_gt(x,y)    ((x)>(y))
#define op_gte(x,y)   ((x)>=(y))
// special...
//  vector versions of max and min are possible to 
//  construct by a little bit fiddel, max for example:
//  m = (x > y)
//  r = (x & m) | (y & ~m)
//
#define op_rectify(x) (((x)>0) & (x))
#define op_min(x,y)   (((x)<(y))?(x):(y))
#define op_max(x,y)   (((x)>(y))?(x):(y))
#define op_sigmoid(x)    (1.0/(1.0 + exp(-(x))))

static inline float64_t op_sigmoid_prime(float64_t x)
{
    float64_t z = op_sigmoid(x);
    return z*(1-z);
}

///////////////////////////////////////////////////////////////////////////////
//  matrix_op.i
//  (soon generated) contain all operations
///////////////////////////////////////////////////////////////////////////////

#include "matrix_op.i"

///////////////////////////////////////////////////////////////////////////////
// a more general function for unary operations but a lot slower
///////////////////////////////////////////////////////////////////////////////

static void apply1(int func,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    size_t i, j;

    au *= element_size(at);
    av *= element_size(at);

    cu *= element_size(ct);
    cv *= element_size(ct);    
    
    if (is_float(ct)) {
	for (i=0; i<n; i++) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		float64_t a = read_float(at, ap1);
		float64_t c;
		ap1 += av;
		switch(func) {
		case SIGMOID:       c = op_sigmoid(a); break;
		case SIGMOID_PRIME: c = op_sigmoid_prime(a); break;
		case RECTIFIER:     c = op_max(0,a); break;
		case TANH:          c = tanh(a); break;		
		case NEGATE:        c = -a; break;
		case COPY:          c = a;  break;		    
		case UNIFORM: c = uniform_64(MATRIX_RAND_ALG); break;
		case NORMAL:  c = normal_64(MATRIX_RAND_ALG,0.0,1.0); break;
		case ONE:     c = 1.0; break;
		case ZERO:     c= 0.0; break;
		case IDENTITY: c = (i==j)?1.0:0.0; break;
		default:      c = 0.0; break;
		}
		write_float(ct, cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
    else if (is_complex(ct)) {
	for (i=0; i<n; i++) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		complex128_t a = read_complex(at, ap1);
		complex128_t c;
		ap1 += av;
		switch(func) {
		case SIGMOID:       c = cop_sigmoid(a); break;
		case SIGMOID_PRIME: c = cop_sigmoid_prime(a); break;
		// case RECTIFIER:  c = op_max(0,a); break;
		case TANH:          c = ctanh(a); break;		
		case NEGATE:        c = -a; break;
		case COPY:          c = a; break;		    
		case UNIFORM:  c = uniform_c128(MATRIX_RAND_ALG); break;
		case NORMAL:   c = normal_c128(MATRIX_RAND_ALG,0.0,1.0); break;
		case ONE:      c = CMPLX(1.0,0.0); break;
		case ZERO:     c = CMPLX(0.0,0.0); break;
		case IDENTITY: c = (i==j)?CMPLX(1.0,0.0):CMPLX(0.0,0.0); break;
		default:       c = CMPLX(0.0,0.0); break;
		}
		write_complex(ct, cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }    
    else {
	for (i=0; i<n; i++) {	
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		int64_t a = read_int(at, ap1);
		int64_t c;
		ap1 += av;
		switch(func) {
		case SIGMOID:   c = op_sigmoid(a); break;
		case SIGMOID_PRIME: c = op_sigmoid_prime(a); break;
		case RECTIFIER: c = op_max(0,a); break;
		case TANH:      c = tanh(a); break;		
		case NEGATE:    c = -a; break;
		case COPY:      c = a; break;		    
		case UNIFORM:   c = rand_64(MATRIX_RAND_ALG); break;
		case ONE:       c = 1; break;
		case ZERO:      c = 0; break;
		case IDENTITY:  c = (i==j); break;
		default:        c = 0; break;	    
		}
		write_int(ct, cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
}

// a more general function for unary operations but a lot slower
static void apply2(int func,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    au *= element_size(at);
    av *= element_size(at);    
    bu *= element_size(bt);
    bv *= element_size(bt);    
    cu *= element_size(ct);
    cv *= element_size(ct);    

    if (is_float(ct)) {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		float64_t a = read_float(at, ap1);
		float64_t b = read_float(bt, bp1);
		float64_t c;
		ap1 += av;
		bp1 += bv;
		switch(func) {
		case PLUS:   c = op_plus(a,b); break;
		case MINUS:  c = op_minus(a,b); break;
		case TIMES:  c = op_times(a,b); break;
		default:     c = 0; break;
		}
		write_float(ct, cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    bp += bu;
	    cp += cu;
	}
    }
    else if (is_complex(ct)) {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		complex128_t a = read_complex(at, ap1);
		complex128_t b = read_complex(bt, bp1);
		complex128_t c;
		ap1 += av;
		bp1 += bv;
		switch(func) {
		case PLUS:   c = op_plus(a,b); break;
		case MINUS:  c = op_minus(a,b); break;
		case TIMES:  c = op_times(a,b); break;
		default:     c = 0; break;
		}
		write_complex(ct, cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    bp += bu;
	    cp += cu;
	}
    }
    else {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap1);
		int64_t b = read_int(bt, bp1);
		int64_t c;
		ap1 += av;
		bp1 += bv;
		switch(func) {
		case PLUS:   c = op_plus(a,b); break;
		case MINUS:  c = op_minus(a,b); break;
		case TIMES:  c = op_times(a,b); break;
		default:     c = 0; break;		    
		}
		write_int(ct, cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    bp += bu;
	    cp += cu;
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// add
///////////////////////////////////////////////////////////////////////////////

static void add(bool_t use_vector,
		matrix_type_t at, byte_t* ap, int au, int av,
		matrix_type_t bt, byte_t* bp, int bu, int bv,
		matrix_type_t ct, byte_t* cp, int cu, int cv,
		size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_add(at, ap, au, bp, bu, cp, cu, n, m);
	else
#endif
	    mt_add(at, ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
    }
    else {
	apply2(PLUS, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// subtract
///////////////////////////////////////////////////////////////////////////////

static void subtract(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t bt, byte_t* bp, int bu, int bv,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR	
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_subtract(at, ap, au, bp, bu, cp, cu, n, m);
	else
#endif	    
	    mt_subtract(at, ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
    }
    else {
	apply2(MINUS, at, ap, au, bv, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// times
///////////////////////////////////////////////////////////////////////////////

static void times(bool_t use_vector,
		  matrix_type_t at, byte_t* ap, int au, int av,
		  matrix_type_t bt, byte_t* bp, int bu, int bv, 
		  matrix_type_t ct, byte_t* cp, int cu, int cv,
		  size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_times(at, ap, au, bp, bu, cp, cu, n, m);
	else
#endif
	    mt_times(at, ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
    }
    else {
	apply2(TIMES, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// negate
///////////////////////////////////////////////////////////////////////////////

static void negate(bool_t use_vector,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp))
	    mtv_negate(at, ap, au, cp, cu, n, m);
	else
#endif
	    mt_negate(at, ap, au, av, cp, cu, cv, n, m);
    }
    else {
	apply1(NEGATE, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// scale_i with integer factor
///////////////////////////////////////////////////////////////////////////////

static void scale_i(bool_t use_vector,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m, int64_t factor)
{
    
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp))
	    mtv_scale_i(at, ap, au, cp, cu, n, m, factor);
	else
#endif	
	    mt_scale_i(at, ap, au, av, cp, cu, cv, n, m, factor);
    }
    else {
	au *= element_size(at);
	av *= element_size(at);    
	cu *= element_size(ct);
	cv *= element_size(ct);

	if (is_integer(ct)) {
	    while(n--) {
		byte_t* ap1 = ap;
		byte_t* cp1 = cp;
		size_t m1 = m;
		while(m1--) {
		    int64_t a = read_int(at, ap1);
		    ap1 += av;
		    write_int(ct, cp1, a*factor);
		    cp1 += cv;
		}
		ap += au;
		cp += au;
	    }	
	}
	else if (is_float(ct)) {
	    while(n--) {
		byte_t* ap1 = ap;
		byte_t* cp1 = cp;
		size_t m1 = m;
		while(m1--) {
		    float64_t a = read_float(at, ap1);
		    ap1 += av;
		    write_float(ct, cp1, a*factor);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }	
	}
	else if (is_complex(ct)) {
	    while(n--) {
		byte_t* ap1 = ap;
		byte_t* cp1 = cp;
		size_t m1 = m;
		while(m1--) {
		    complex128_t a = read_complex(at, ap1);
		    ap1 += av;
		    write_complex(ct, cp1, a*factor);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }	
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// scale_f with floating point factor
///////////////////////////////////////////////////////////////////////////////

static void scale_f(bool_t use_vector,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m, float64_t factor)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp))
	    mtv_scale_f(at, ap, au, cp, cu, n, m, factor);
	else
#endif		
	    mt_scale_f(at, ap, au, av, cp, cu, cv, n, m, factor);
    }
    else {
	au *= element_size(at);
	av *= element_size(at);    
	cu *= element_size(ct);
	cv *= element_size(ct);
	
	if (is_complex(ct)) {
	    while(n--) {
		byte_t* ap1 = ap;
		byte_t* cp1 = cp;
		size_t m1 = m;
		while(m1--) {
		    complex64_t a = read_complex(at, ap1);
		    ap1 += av;
		    write_complex(ct, cp1, a*factor);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }
	}
	else { // a is integer or float
	    while(n--) {
		byte_t* ap1 = ap;
		byte_t* cp1 = cp;
		size_t m1 = m;
		while(m1--) {
		    float64_t a = read_float(at, ap1);
		    ap1 += av;
		    write_float(ct, cp1, a*factor);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }
	}
    }
}


static void scale_c(bool_t use_vector,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m, complex128_t factor)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp))
	    mtv_scale_c(at, ap, au, cp, cu, n, m, factor);
	else
#endif
	    mt_scale_c(at, ap, au, av, cp, cu, cv, n, m, factor);
    }
    else {
	au *= element_size(at);
	av *= element_size(at);    
	cu *= element_size(ct);
	cv *= element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		complex128_t a = read_complex(at, ap1);
		ap1 += av;
		write_complex(ct, cp1, a*factor);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// argmax
///////////////////////////////////////////////////////////////////////////////

static void argmax(matrix_type_t at,byte_t* ap, int au, int av,
		   size_t n, size_t m, int32_t* cp)
{
    au *= element_size(at);
    av *= element_size(at);

    if (is_integer(at)) {
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t max_i = 1;
	    int64_t max_v = read_int(at, ap1);
	    
	    ap1 += au;
	    while(n1--) {
		int64_t v = read_int(at, ap1);
		ap1 += au;
		i++;
		if (v > max_v) { max_v = v; max_i = i; }
	    }
	    *cp++ = max_i;
	    ap += av;
	}	
    }
    else if (is_float(at)) {
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t max_i = 1;
	    float64_t max_v = read_float(at, ap1);
	    
	    ap1 += au;
	    while(n1--) {
		float64_t v = read_float(at, ap1);
		ap1 += au;
		i++;
		if (v > max_v) { max_v = v; max_i = i; }
	    }
	    *cp++ = max_i;
	    ap += av;
	}
    }
    else if (is_complex(at)) {
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t max_i = 1;
	    complex128_t max_v = read_complex(at, ap1);
	    
	    ap1 += au;
	    while(n1--) {
		complex128_t v = read_complex(at, ap1);
		ap1 += au;
		i++;
		if (cabs(v) > cabs(max_v)) { max_v = v; max_i = i; }
	    }
	    *cp++ = max_i;
	    ap += av;

	}	
    }    
}

///////////////////////////////////////////////////////////////////////////////
// sigmoid
///////////////////////////////////////////////////////////////////////////////


static void sigmoid(matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid(at, ap, au, av, cp, cu, cv, n, m);
    }
    else {
	apply1(SIGMOID, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// sigmoid_prime
///////////////////////////////////////////////////////////////////////////////

static void sigmoid_prime(matrix_type_t at, byte_t* ap, int au, int av,
			  matrix_type_t ct, byte_t* cp, int cu, int cv,
			  size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid_prime(at, ap, au, av, cp, cu, cv, n, m);
    }
    else {
	apply1(SIGMOID_PRIME, at, ap, au, av, ct, cp, cu, av, n, m);
    }    
}

///////////////////////////////////////////////////////////////////////////////
// rectifier
///////////////////////////////////////////////////////////////////////////////


static void rectifier(matrix_type_t at, byte_t* ap, int au, int av,
		      matrix_type_t ct, byte_t* cp, int cu, int cv,
		      size_t n, size_t m)
{
    if (at == ct) {
	// fixme: vectorized version!
	mt_rectifier(at, ap, au, av, cp, cu, cv, n, m);
    }
    else {
	apply1(RECTIFIER, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}


///////////////////////////////////////////////////////////////////////////////
// multiply
///////////////////////////////////////////////////////////////////////////////

static void multiply(
    bool_t use_vector,
    matrix_type_t at,byte_t* ap,int au,int av,size_t an,size_t am,
    matrix_type_t bt,byte_t* bp,int bu,int bv,size_t bn,size_t bm,
    matrix_type_t ct,byte_t* cp,int cu,int cv)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_multiply(at,ap,au,an,am,bp,bu,bn,bm,cp,cu,cv);
	else
#endif
	    mt_multiply(at,ap,au,av,an,am,bp,bu,bv,bn,bm,cp,cu,cv);	
    }
    else {
	unsigned int i, j, k;
	au *= element_size(at);
	av *= element_size(at);
	bu *= element_size(bt);
	bv *= element_size(bt);
	cu *= element_size(ct);
	cv *= element_size(ct);

	if (is_float(ct)) {
	    for (i=0; i<an; i++) {
		byte_t* cp1 = cp;
		for (j=0; j<bm; j++) {
		    float64_t sum = 0;
		    byte_t* bp1 = bp + j*bv;  // FIXME: bv2!! column pointer
		    byte_t* ap1 = ap;         // row pointer
		    for (k = 0; k < am; k++) {
			float64_t a = read_float(at, ap1);
			float64_t b = read_float(bt, bp1);
			sum += a*b;
			ap1 += av;
			bp1 += bu;
		    }
		    write_float(ct, cp1, sum);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }
	}
	else if (is_complex(ct)) {
	    for (i=0; i<an; i++) {
		byte_t* cp1 = cp;
		for (j=0; j<bm; j++) {
		    complex128_t sum = 0;
		    byte_t* bp1 = bp + j*bv;
		    byte_t* ap1 = ap;
		    for (k = 0; k < am; k++) {
			complex128_t a = read_complex(at, ap1);
			complex128_t b = read_complex(bt, bp1);
			sum += a*b;
			ap1 += av;
			bp1 += bu;
		    }
		    write_complex(ct, cp1, sum);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }
	}
	else {
	    for (i=0; i<an; i++) {
		byte_t* cp1 = cp;
		for (j=0; j<bm; j++) {
		    int64_t sum = 0;
		    byte_t* bp1 = bp + j*bv;
		    byte_t* ap1 = ap;
		    for (k = 0; k < am; k++) {
			int64_t a = read_int(at, ap1);
			int64_t b = read_int(bt, bp1);
			sum += a*b;
			ap1 += av;
			bp1 += bu;
		    }
		    write_int(ct, cp1, sum);
		    cp1 += cv;
		}
		ap += au;
		cp += cu;
	    }
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// multiply_transposed A*Bt = C
///////////////////////////////////////////////////////////////////////////////

static void multiply_t(
    bool_t use_vector,
    matrix_type_t at,byte_t* ap,int au,int av,size_t an,size_t am,
    matrix_type_t bt,byte_t* bp,int bu,int bv,size_t bn,size_t bm,
    matrix_type_t ct,byte_t* cp,int cu,int cv)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_multiply_transposed(at,ap,au,an,am,bp,bu,bn,bm,cp,cu,cv);
	else
#endif
	    mt_multiply_transposed(at,ap,au,av,an,am,bp,bu,bv,bn,bm,cp,cu,cv);
    }
    else {
	byte_t* bp0 = bp;

	au *= element_size(at);
	av *= element_size(at);
	bu *= element_size(bt);
	bv *= element_size(bt);
	cu *= element_size(ct);
	cv *= element_size(ct);

	if (is_float(ct)) {
	    while(an--) {
		byte_t* cp1 = cp;
		size_t n = bn;
		bp = bp0;
		while(n--) {
		    float64_t sum = 0;
		    byte_t* ap1 = ap;     // row pointer		
		    byte_t* bp1 = bp;     // "column" pointer
		    size_t k = bm;
		    while(k--) {
			float64_t a = read_float(at, ap1);
			float64_t b = read_float(bt, bp1);
			sum += a*b;
			ap1 += av;
			bp1 += bv;
		    }
		    write_float(ct, cp1, sum);
		    cp1 += cv;
		    bp  += bu;
		}
		ap += au;
		cp += cu;
	    }
	}
	else if (is_complex(ct)) {
	    while(an--) {
		byte_t* cp1 = cp;
		size_t n = bn;
		bp = bp0;		
		while(n--) {
		    complex128_t sum = CMPLX(0.0,0.0);
		    byte_t* ap1 = ap;
		    byte_t* bp1 = bp;
		    size_t k = bm;
		    while(k--) {
			complex128_t a = read_complex(at, ap1);
			complex128_t b = read_complex(bt, bp1);
			sum += a*b;
			ap1 += av;
			bp1 += bv;
		    }
		    write_complex(ct, cp1, sum);
		    cp1 += cv;
		    bp += bu;
		}
		ap += au;
		cp += cu;
	    }
	}
	else {
	    while(an--) {
		byte_t* cp1 = cp;
		size_t n = bn;
		bp = bp0;		
		while(n--) {
		    int64_t sum = 0;
		    byte_t* ap1 = ap;
		    byte_t* bp1 = bp;
		    size_t k = bm;
		    while(k--) {
			int64_t a = read_int(at, ap1);
			int64_t b = read_int(bt, bp1);
			sum += a*b;
			ap1 += av;
			bp1 += bv;
		    }
		    write_int(ct, cp1, sum);
		    cp1 += cv;
		    bp += bu;
		}
		ap += au;
		cp += cu;
	    }
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
//  copy element by element assume at == ct
///////////////////////////////////////////////////////////////////////////////

static void mt_copy(matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m)
{
    size_t sz = element_size(at);
    UNUSED(ct);
    au *= sz;
    av *= sz;
    cu *= sz;
    cv *= sz;

    while(n--) {
	if (au == cu)
	    memcpy(cp, ap, au);
	else {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		memcpy(cp1, ap1, sz);
		ap1 += av;
		cp1 += cv;
	    }
	}
	ap += au;
	cp += cu;
    }
}

///////////////////////////////////////////////////////////////////////////////
//  copy vector-by-vector assume at == ct
///////////////////////////////////////////////////////////////////////////////

static void mtv_copy(matrix_type_t at, byte_t* ap, int au,
		     matrix_type_t ct, byte_t* cp, int cu,
		     size_t n, size_t m)
{
    size_t sz = element_size(at);
    UNUSED(ct);
    au *= sz;
    cu *= sz;
    
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* cp1 = cp;
	size_t m1 = m;
	while(m1 >= VELEMS(int8_t)) {
	    *(vint8_t*)cp1 = *(vint8_t*)ap1;
	    ap1 += VELEMS(int8_t);
	    cp1 += VELEMS(int8_t);
	    m1  -= VELEMS(int8_t);
	}
	while(m1--) {
	    memcpy(cp1, ap1, sz);
	    ap1 += sz;
	    cp1 += sz;
	}
        ap += au;
        cp += cu;
    }
}

///////////////////////////////////////////////////////////////////////////////
//  simple copy
///////////////////////////////////////////////////////////////////////////////

static void copy1(bool_t use_vector,
		  matrix_type_t at, byte_t* ap, int au, int av,
		  matrix_type_t ct, byte_t* cp, int cu, int cv,
		  size_t n, size_t m)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp))
	    mtv_copy(at, ap, au, ct, cp, cu, n, m);
	else
#endif
	    mt_copy(at, ap, au, av, ct, cp, cu, cv, n, m);
    }
    else {
	apply1(COPY, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
//  simple tranpose
///////////////////////////////////////////////////////////////////////////////

static void transpose1(bool_t use_vector,
		       matrix_type_t at, byte_t* ap, int au, int av,
		       matrix_type_t ct, byte_t* cp, int cu, int cv,
		       size_t n, size_t m)
{
    UNUSED(use_vector);
    if (at == ct) {
	mt_copy(at, ap, au, av, ct, cp, cv, cu, m, n);
    }
    else {
	apply1(COPY, at, ap, au, av, ct, cp, cv, cu, m, n);
    }
}

///////////////////////////////////////////////////////////////////////////////
// copy
// copy matrix data in A with repeatition into matrix C
///////////////////////////////////////////////////////////////////////////////

static void tile(matrix_type_t at,byte_t* ap,int au,int av,
		 size_t an,size_t am,
		 matrix_type_t ct,byte_t* cp,int cu,int cv,
		 size_t cn, size_t cm,
		 unsigned int repeat_h, unsigned int repeat_v)
{
    // each row in A copy with repeat into row in C until C is filled
    size_t  ai  = 0;
    byte_t* ap0 = ap;
    size_t  n   = cn;

    av *= element_size(at);
    au *= element_size(at);
    cv *= element_size(ct);
    cu *= element_size(ct);    
    
    if (at == ct) { // simple copy row with wrap
	size_t sz = element_size(at);
	while(n--) {
 	    byte_t* ap1;
	    byte_t* cp1 = cp;
	    size_t aj = 0;
	    size_t m = cm;
	    unsigned int rv = repeat_v;
	    
	    if (ai >= an) {
		if (repeat_h==1) return;
		if (repeat_h!=0) repeat_h--;
		ai = 0;
		ap = ap0;
	    }
	    ap1 = ap;
	    while(m--) { // copy row
		if (aj >= am) {
		    if (rv==1) break;
		    if (rv!=0) rv--;
		    aj = 0;
		    ap1 = ap;
		}		
		memcpy(cp1, ap1, sz);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    ap += au;
	    cp += cu;
	    ai++;
	}
    }
    else if (is_float(ct)) {
	while(n--) {
 	    byte_t* ap1;
	    byte_t* cp1 = cp;
	    size_t aj = 0;
	    size_t m = cm;
	    unsigned int rv = repeat_v;
	    
	    if (ai >= an) {
		if (repeat_h==1) return;
		if (repeat_h!=0) repeat_h--;
		ai = 0;
		ap = ap0;
	    }

	    ap1 = ap;
	    while(m--) { // copy row
		float64_t value;
		if (aj >= am) {
		    if (rv==1) break;
		    if (rv!=0) rv--;
		    aj = 0;
		    ap1 = ap;
		}
		value = read_float(at, ap1);
		write_float(ct, cp1, value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    ap += au;
	    cp += cu;
	    ai++;
	}
    }
    else if (is_complex(ct)) {
	while(n--) {
 	    byte_t* ap1;
	    byte_t* cp1 = cp;
	    size_t aj = 0;
	    size_t m = cm;
	    unsigned int rv = repeat_v;
	    
	    if (ai >= an) {
		if (repeat_h==1) return;
		if (repeat_h!=0) repeat_h--;
		ai = 0;
		ap = ap0;
	    }

	    ap1 = ap;
	    while(m--) { // copy row
		complex128_t value;
		if (aj >= am) {
		    if (rv==1) break;
		    if (rv!=0) rv--;
		    aj = 0;
		    ap1 = ap;
		}
		value = read_complex(at, ap1);
		write_complex(ct, cp1, value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    ap += au;
	    cp += cu;
	    ai++;
	}
    }    
    else {
	while(n--) {
 	    byte_t* ap1;
	    byte_t* cp1 = cp;
	    size_t aj = 0;
	    size_t m = cm;
	    unsigned int rv = repeat_v;
	    
	    if (ai >= an) {
		if (repeat_h==1) return;
		if (repeat_h!=0) repeat_h--;
		ai = 0;
		ap = ap0;
	    }
	    ap1 = ap;
	    while(m--) { // copy row
		int64_t value;
		if (aj >= am) {
		    if (rv==1) break;
		    if (rv!=0) rv--;
		    aj = 0;
		    ap1 = ap;
		}
		value = read_int(at, ap1);	
		write_int(ct,cp1,value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    ap += au;
	    cp += cu;
	    ai++;
	}
    }
}


///////////////////////////////////////////////////////////////////////////////
// copy_fill
// fill data sequentially with repeat from matrix A into matrix C
// until matrix C is filled.
///////////////////////////////////////////////////////////////////////////////

static void fill(matrix_type_t at,byte_t* ap,int au,int av,
		 size_t an,size_t am,
		 matrix_type_t ct,byte_t* cp,int cu,int cv,
		 size_t cn,size_t cm)
{
    byte_t* ap0 = ap;
    byte_t* ap1 = ap;
    size_t  ai = 0;
    size_t  aj = 0;
    size_t  n = cn;

    av *= element_size(at);
    au *= element_size(at);
    cv *= element_size(ct);
    cu *= element_size(ct);
    
    if (at == ct) {
	size_t sz = element_size(at);
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		memcpy(cp1, ap1, sz);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }
    else if (is_float(ct)) {
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;
	    
	    while(m--) {
		float64_t value;
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_float(at, ap1);
		write_float(ct, cp1, value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }
    else if (is_complex(ct)) {
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;
	    
	    while(m--) {
		complex128_t value;
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_complex(at, ap1);
		write_complex(ct, cp1, value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }    
    else {
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		int64_t value;
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_int(at, ap1);
		write_int(ct,cp1,value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }
}

matrix_t* alloc_matrix_resource(size_t n, size_t m,
				bool_t rowmajor,matrix_type_t type,size_t align)
{
    matrix_t* mp = enif_alloc_resource(matrix_r, sizeof(matrix_t));

    if (mp != NULL) {
	size_t stride = (m*element_size(type)+align-1) & ~(align-1);
	size_t size   = n*stride;
	
	mp->n       = n;
	mp->m       = m;
	mp->type    = type;
	mp->size    = 0;
	mp->byte_offset = 0;
	mp->offset  = 0;
	mp->rowmajor = rowmajor;
	mp->data    = NULL;
	mp->rw_lock = NULL;
	// fixme: maybe make sure that n is also padded to alignment
	// by adding zero pads in the end. This may simplify code
	// that extract columns and code that transpose using vectors
	
	if ((mp->base = enif_alloc(size+align-1)) != NULL) {
	    mp->size   = size;
	    mp->byte_stride = stride;
	    mp->stride = stride / element_size(type);
	    mp->rw_lock = enif_rwlock_create("matrix");
	    mp->data = align_ptr(mp->base,align);
	    memset(mp->base, 0, size+align-1);
	}
	else {
	    enif_release_resource(mp);
	    return NULL;
	}
    }
    return mp;
}

int create_matrix(ErlNifEnv* env, unsigned int n, unsigned int m,
		  bool_t rowmajor, matrix_type_t type, matrix_t* mp,
		  ERL_NIF_TERM* resp)
{
    matrix_t* np;
    if ((np = alloc_matrix_resource(n, m, rowmajor, type, ALIGN)) != NULL) {
	*resp = enif_make_resource_binary(env,np,np->data,np->size);
	*mp = *np;
	enif_release_resource(np);
	return 1;
    }
    return 0;
}


int make_matrix_binary(ErlNifEnv* env, unsigned int n, unsigned int m,
		       matrix_type_t type, size_t align,
		       ERL_NIF_TERM* term, byte_t** data,
		       matrix_t** mpp)
{
    ErlNifBinary bin;
    size_t stride = (m*element_size(type)+align-1) & ~(align-1);

    if (!enif_alloc_binary(n*stride, &bin))
	return 0;
    *term = enif_make_binary(env, &bin);
    *data = bin.data;
    memset(bin.data, 0, bin.size);
    *mpp = 0;
    return 1;
}

static int get_bool(ErlNifEnv* env, ERL_NIF_TERM arg, bool_t* bp)
{
    UNUSED(env);
    if (arg == ATOM(true)) *bp = TRUE;
    else if (arg == ATOM(false)) *bp = FALSE;
    else return 0;
    return 1;
}

static ERL_NIF_TERM make_bool(ErlNifEnv* env, bool_t b)
{
    UNUSED(env);
    return b ? ATOM(true) : ATOM(false);
}

static int get_complex(ErlNifEnv* env, ERL_NIF_TERM arg, complex128_t* cmplx)
{
    int arity;	    
    const ERL_NIF_TERM* elems;
    ErlNifSInt64 ival;
    float64_t    real;
    float64_t    imag;

    if (!enif_get_tuple(env, arg, &arity, &elems) || (arity != 2))
	return 0;

    if (enif_get_int64(env, elems[0], &ival))
	real = ival;
    else if (!enif_get_double(env, elems[0], &real))
	return 0;
	
    if (enif_get_int64(env, elems[1], &ival))
	imag = ival;
    else if (!enif_get_double(env, elems[1], &imag))
	return 0;
    *cmplx = CMPLX(real, imag);
    return 1;
}


// Get matrix argument
// { 'matrix', n, m, type, ptr, offset, stride, row-major, binary-data }
//   Must match #matrix record in the code!
// FIXME:
//   if matrix has ptr=0, then binary data may not
//   be vector aligned. If USE_GCC_VECTOR is true then
//   that data must be rejected. vector functions will most
//   likely crash otherwise.
//   This may happend if matrix is created without nif being
//   loaded and then later that matrix is passwed to a nif.
//

static int get_matrix(ErlNifEnv* env, ERL_NIF_TERM arg, matrix_t* mp)
{
    int arity;
    unsigned int type;
    unsigned long ptr;
    const ERL_NIF_TERM* elems;
    
    if (!enif_get_tuple(env, arg, &arity, &elems)) return 0;
    if (arity != 9) return 0;
    if (elems[0] != ATOM(matrix)) return 0;
    if (!enif_get_uint(env, elems[1], &mp->n)) return 0;
    if (!enif_get_uint(env, elems[2], &mp->m)) return 0;
    if (!enif_get_uint(env, elems[3], &type)) return 0;
    if (type > MAX_TYPE_NUMBER) return 0;
    mp->type = type;
    if (!enif_get_ulong(env, elems[4], &ptr)) return 0;
    if (!enif_get_uint(env, elems[5], &mp->offset)) return 0;
    if (!enif_get_uint(env, elems[6], &mp->stride)) return 0;
    if (!get_bool(env, elems[7], &mp->rowmajor)) return 0;
    mp->byte_offset = mp->offset*element_size(type);
    mp->byte_stride = mp->stride*element_size(type);	
    
    if (ptr != 0) {
	matrix_t* rmp;
	if (!enif_get_resource(env, elems[8], matrix_r, (void**)&rmp))
	    return 0;
	if (rmp->stride != mp->stride)
	    return 0;
	// check bounds, we may have a submatrix
	if ((mp->byte_offset + (mp->n-1)*mp->byte_stride +
	     mp->m*element_size(type)) > rmp->size)
	    return 0;
	mp->size = rmp->size;
	mp->base = rmp->base;
	mp->data = rmp->data;
	mp->rw_lock = rmp->rw_lock;
    }
    else {
	ErlNifBinary bin;
	if (!enif_inspect_binary(env, elems[8], &bin)) return 0;
	if (bin.size < (mp->n * mp->stride * element_size(type))) return 0;
	mp->size = 0;
	mp->base = NULL;
	mp->data = bin.data;
	mp->rw_lock = NULL;
    }
    return 1;
}

static ERL_NIF_TERM make_matrix(ErlNifEnv* env,
				unsigned int n, unsigned int m,
				bool_t rowmajor,
				matrix_type_t type,
				matrix_t* mp,
				ERL_NIF_TERM bin)
{
    return enif_make_tuple9(env,
			    ATOM(matrix),
			    enif_make_uint(env, n),
			    enif_make_uint(env, m),
			    enif_make_uint(env, type),
			    enif_make_uint64(env, (uintptr_t)mp),
			    enif_make_uint(env, mp->offset),
			    enif_make_uint(env, mp->stride),
			    make_bool(env, rowmajor),
			    bin);
}

// new_(N, M, Type, RowMajor, Data)
ERL_NIF_TERM matrix_create(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n, m, type;
    ErlNifBinary binary;
    matrix_t c;
    ERL_NIF_TERM bin;
    bool_t rowmajor;
    UNUSED(argc);
    
    if (!enif_get_uint(env, argv[0], &n))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &m))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &type))
	return enif_make_badarg(env);
    if (type > MAX_TYPE_NUMBER)
	return enif_make_badarg(env);
    if (!get_bool(env, argv[3], &rowmajor))
	return enif_make_badarg(env);
    if (!enif_inspect_iolist_as_binary(env, argv[4], &binary))
	return enif_make_badarg(env);
    if ((binary.size != 0) && (n*m*element_size(type) < binary.size))
	return enif_make_badarg(env);
    if (!create_matrix(env,n,m,rowmajor,type,&c,&bin))
	return enif_make_badarg(env);
    if (binary.size == n*m*element_size(type)) {
	if (c.stride == c.m)
	    memcpy(c.data, binary.data, c.size);
	else {
	    byte_t* ap = c.data;
	    byte_t* bp = binary.data;
	    size_t  bu = c.m*element_size(type);
	    size_t  au = c.byte_stride;
	    size_t i;
	    for (i=0; i<n; i++) {
		memcpy(ap, bp, bu);
		ap += au;
		bp += bu;
	    }
	}
    }
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// matrix apply1(func,A,B) -> C, check all rowmajor variants
// with and without accelerations
static void m_apply1(
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t ct, byte_t* cp, int cu, int cv,
		 size_t n, size_t m),
    matrix_t* ap, matrix_t* cp)
{
    if (cp->rowmajor == ap->rowmajor)
	func(TRUE,
	     ap->type, ap->data+ap->byte_offset, ap->stride, 1,
	     cp->type, cp->data+cp->byte_offset, cp->stride, 1, cp->n, cp->m);
    else
	func(FALSE,
	     ap->type, ap->data+ap->byte_offset, 1, ap->stride,
	     cp->type, cp->data+cp->byte_offset, cp->stride, 1, cp->n, cp->m);
}


// matrix apply2(func,A,B) -> C, check all rowmajor variants
// with and without accelerations

static void m_apply2(
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t bt, byte_t* bp, int bu, int bv,
		 matrix_type_t ct, byte_t* cp, int cu, int cv,
		 size_t n, size_t m),
    matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor)
	    func(TRUE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (!ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (ap->rowmajor && !bp->rowmajor)
	    func(FALSE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else
	    func(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
    }
    else {
	if (ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (!ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (ap->rowmajor && !bp->rowmajor)
	    func(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else
	    func(TRUE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
    }
}

// matrix add A+B -> C, check all rowmajor variant
// with and without accelerations

static void m_add(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor)
	    add(TRUE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (!ap->rowmajor && bp->rowmajor)
	    add(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (ap->rowmajor && !bp->rowmajor)
	    add(FALSE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else
	    add(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
    }
    else {
	if (ap->rowmajor && bp->rowmajor)
	    add(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (!ap->rowmajor && bp->rowmajor)
	    add(FALSE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, 1, bp->stride,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else if (ap->rowmajor && !bp->rowmajor)
	    add(FALSE,
		ap->type, ap->data+ap->byte_offset, 1, ap->stride,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
	else
	    add(TRUE,
		ap->type, ap->data+ap->byte_offset, ap->stride, 1,
		bp->type, bp->data+bp->byte_offset, bp->stride, 1,
		cp->type, cp->data+cp->byte_offset, cp->stride, 1,
		cp->n, cp->m);
    }
}

// add two matrices
ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.rowmajor == b.rowmajor) && ((a.n != b.n) || (a.m != b.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != b.rowmajor) && ((a.n != b.m) || (a.m != b.n)))
	return enif_make_badarg(env);
    
    if (argc == 2) {
	ERL_NIF_TERM bin;
	matrix_type_t ct;

	ct = combine_type(a.type, b.type);
	
	if (!create_matrix(env,a.n,a.m,a.rowmajor,ct,&c,&bin))
	    return enif_make_badarg(env);

	enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rlock(b.rw_lock);

	m_add(&a, &b, &c);
	
	enif_rwlock_runlock(b.rw_lock);
	enif_rwlock_runlock(a.rw_lock);

	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 3
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);
	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	m_add(&a, &b, &c);

	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_runlock(b.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[2];
    }
}

// subtract two matrices
ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.rowmajor == b.rowmajor) && ((a.n != b.n) || (a.m != b.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != b.rowmajor) && ((a.n != b.m) || (a.m != b.n)))
	return enif_make_badarg(env);
    
    if (argc == 2) {
	ERL_NIF_TERM bin;
	matrix_type_t ct;

	ct = combine_type(a.type, b.type);
	
	if (!create_matrix(env,a.n,a.m,a.rowmajor,ct,&c,&bin))
	    return enif_make_badarg(env);

	enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rlock(b.rw_lock);

	m_apply2(subtract,&a, &b, &c);
	
	enif_rwlock_runlock(b.rw_lock);
	enif_rwlock_runlock(a.rw_lock);

	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 3
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);
	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	m_apply2(subtract, &a, &b, &c);

	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_runlock(b.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[2];
    }
}

// multiply two matrices element wise
ERL_NIF_TERM matrix_times(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.rowmajor == b.rowmajor) && ((a.n != b.n) || (a.m != b.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != b.rowmajor) && ((a.n != b.m) || (a.m != b.n)))
	return enif_make_badarg(env);
    
    if (argc == 2) {
	ERL_NIF_TERM bin;
	matrix_type_t ct;

	ct = combine_type(a.type, b.type);
	
	if (!create_matrix(env,a.n,a.m,a.rowmajor,ct,&c,&bin))
	    return enif_make_badarg(env);

	enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rlock(b.rw_lock);

	m_apply2(times,&a, &b, &c);
	
	enif_rwlock_runlock(b.rw_lock);
	enif_rwlock_runlock(a.rw_lock);

	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 3
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);
	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	m_apply2(times, &a, &b, &c);

	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_runlock(b.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[2];
    }
}

// multiply A*B = C matrices

ERL_NIF_TERM matrix_multiply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;
    size_t n, m;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if (a.rowmajor) {
	if (b.rowmajor) {
	    if (a.m != b.n) return enif_make_badarg(env);
	    n = a.n; m = b.m;
	} else {
	    if (a.m != b.m) return enif_make_badarg(env);
	    n = a.n; m = b.n;
	}
    }
    else {
	if (b.rowmajor) {
	    if (a.n != b.n) return enif_make_badarg(env);
	    n = a.m; m = b.m;
	} else {
	    if (a.n != b.m) return enif_make_badarg(env);
	    n = a.m; m = b.n;
	}
    }
    
    if (argc == 2) {
	matrix_type_t c_t = combine_type(a.type, b.type);
	ERL_NIF_TERM bin;

	if (a.rowmajor && !b.rowmajor) {  // special case?
	    if (!create_matrix(env,m,n,FALSE,c_t,&c,&bin))
		return enif_make_badarg(env);
	}
	else {
	    if (!create_matrix(env,n,m,TRUE,c_t,&c,&bin))
		return enif_make_badarg(env);
	}

	enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rlock(b.rw_lock);

	if (a.rowmajor && b.rowmajor) {
	    multiply(TRUE,
		     a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
		     b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
		     c.type, c.data+c.byte_offset, c.stride, 1);
	} else if (a.rowmajor && !b.rowmajor) {
//	    multiply_t(TRUE,
//		       a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
//		       b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
//		       c.type, c.data+c.byte_offset, c.stride, 1);
	    multiply_t(TRUE,
		       a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
		       b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
		       c.type, c.data+c.byte_offset, 1, c.stride);
	} else if (!a.rowmajor && b.rowmajor) {
	    multiply(FALSE,
		     a.type, a.data+a.byte_offset, 1, a.stride, a.m, a.n,
		     b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
		     c.type, c.data+c.byte_offset, c.stride, 1);
	}
	else { // !a.rowmajor && !b.rowmajor
	    multiply(TRUE,
		     b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
		     a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
		     c.type, c.data+c.byte_offset, 1, c.stride);
	}
	enif_rwlock_runlock(b.rw_lock);
	enif_rwlock_runlock(a.rw_lock);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else { // argc == 3
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if (c.rowmajor && ((c.n != n) || (c.m != m)))
	    return enif_make_badarg(env);
	if (!c.rowmajor && ((c.n != m) || (c.m != n)))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	if (c.rowmajor) {
	    if (a.rowmajor && b.rowmajor) {
		multiply(TRUE,
			 a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
			 b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
			 c.type, c.data+c.byte_offset, c.stride, 1);
	    } else if (a.rowmajor && !b.rowmajor) {
		multiply_t(TRUE,
			   a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
			   b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
			   c.type, c.data+c.byte_offset, c.stride, 1);
	    } else if (!a.rowmajor && b.rowmajor) {
		multiply(FALSE,
			 a.type, a.data+a.byte_offset, 1, a.stride, a.m, a.n,
			 b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
			 c.type, c.data+c.byte_offset, c.stride, 1);
	    }
	    else { // !a.rowmajor && !b.rowmajor
		multiply(FALSE,
			 a.type, a.data+a.byte_offset, 1, a.stride, a.m, a.n,
			 b.type, b.data+b.byte_offset, 1, b.stride, b.m, b.n,
			 c.type, c.data+c.byte_offset, c.stride, 1);
	    }
	}
	else {
	    if (a.rowmajor && b.rowmajor) {
		multiply(TRUE,
			 a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
			 b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
			 c.type, c.data+c.byte_offset, 1, c.stride);
	    } else if (a.rowmajor && !b.rowmajor) {
		multiply_t(TRUE,
			   a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
			   b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
			   c.type, c.data+c.byte_offset, 1, c.stride);
	    } else if (!a.rowmajor && b.rowmajor) {
		multiply(FALSE,
			 a.type, a.data+a.byte_offset, 1, a.stride, a.m, a.n,
			 b.type, b.data+b.byte_offset, b.stride, 1, b.n, b.m,
			 c.type, c.data+c.byte_offset, 1, c.stride);
	    }
	    else { // !a.rowmajor && !b.rowmajor
		multiply(FALSE,
			 a.type, a.data+a.byte_offset, 1, a.stride, a.m, a.n,
			 b.type, b.data+b.byte_offset, 1, b.stride, b.m, b.n,
			 c.type, c.data+c.byte_offset, 1, c.stride);
	    }
	}
	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_runlock(b.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[2];
    }
}


// negate a matrix
ERL_NIF_TERM matrix_negate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    if (argc == 1) {
	matrix_t c;
	ERL_NIF_TERM bin;
	
	if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);
	
	enif_rwlock_rlock(a.rw_lock);

	m_apply1(negate, &a, &c);

	enif_rwlock_runlock(a.rw_lock);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else { // args == 2
	matrix_t c;
	if (!get_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);
	
	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	m_apply1(negate, &a, &c);
	
	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

	return argv[1];
    }
}

// copy a matrix
//   matrix a is tiled into matrix c
//   it is tiled horizontal repeat_m times
//   and vertical repeat_n times, if 0 is specified it repeasts for ever
// 
ERL_NIF_TERM matrix_copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, c;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    if (argc == 1) {
	ERL_NIF_TERM bin;
	if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);
	enif_rwlock_rlock(a.rw_lock);
	m_apply1(copy1, &a, &c);
	enif_rwlock_runlock(a.rw_lock);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else if (argc == 2) {
	if (!get_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);
	// copy into C

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	m_apply1(copy1, &a, &c);
	
	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

	return argv[1];
    }
    else {
	unsigned int repeat_m;
	unsigned int repeat_n;

	if (!get_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);	

	if (!enif_get_uint(env, argv[2], &repeat_m))
	    return enif_make_badarg(env);
	if (!enif_get_uint(env, argv[3], &repeat_n))
	    return enif_make_badarg(env);
    
	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	if (a.rowmajor == c.rowmajor)
	    tile(a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
		 c.type, c.data+c.byte_offset, c.stride, 1, c.n, c.m,
		 repeat_m, repeat_n);
	else
	    tile(a.type, a.data+a.byte_offset, 1, a.stride, a.n, a.m,
		 c.type, c.data+c.byte_offset, c.stride, 1, c.n, c.m,
		 repeat_m, repeat_n);	    
    
	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[1];
    }
}

// copy data row by row from A sequentially into C
ERL_NIF_TERM matrix_fill(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &c))
	return enif_make_badarg(env);

    if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rwlock(c.rw_lock);

    if (a.rowmajor == c.rowmajor)
	fill(a.type, a.data+a.byte_offset, a.stride, 1, a.n, a.m,
		  c.type, c.data+c.byte_offset, c.stride, 1, c.n, c.m);
    else
	fill(a.type, a.data+a.byte_offset, 1, a.stride, a.n, a.m,
	     c.type, c.data+c.byte_offset, c.stride, 1, c.n, c.m);

    enif_rwlock_rwunlock(c.rw_lock);
    if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

    return argv[1];
}

// scale a matrix
//   factor * A =>  A'
//   factor * A =>  C
//
ERL_NIF_TERM matrix_scale(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    ErlNifSInt64 i_scale = 0;
    float64_t    f_scale = 0.0;
    complex128_t c_scale = CMPLX(0.0, 0.0);
    int arg_int=0, arg_float=0, arg_complex=0;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[1], &a))
	return enif_make_badarg(env);

    if (enif_get_int64(env, argv[0], &i_scale))
	arg_int = 1;
    else if (enif_get_double(env, argv[0], &f_scale))
	arg_float = 1;
    else if (get_complex(env, argv[0], &c_scale))
	arg_complex = 1;
    else
	return enif_make_badarg(env);	

    if (argc == 2) {
	matrix_t c;
	ERL_NIF_TERM bin;

	if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);

	enif_rwlock_rlock(a.rw_lock);
	if (arg_int)
	    scale_i(TRUE,
		    a.type, a.data+a.byte_offset, a.stride, 1,
		    c.type, c.data+c.byte_offset, c.stride, 1,
		    a.n, a.m, i_scale);
	else if (arg_float)
	    scale_f(!is_integer(c.type),
		    a.type, a.data+a.byte_offset, a.stride, 1,
		    c.type, c.data+c.byte_offset, c.stride, 1,
		    a.n, a.m, f_scale);
	else if (arg_complex)
	    scale_c(!is_integer(c.type)&&!is_float(c.type),
		    a.type, a.data+a.byte_offset, a.stride, 1,
		    c.type, c.data+c.byte_offset, c.stride, 1,
		    a.n, a.m, c_scale);
	enif_rwlock_runlock(a.rw_lock);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else {
	matrix_t c;
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);
	
	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	if (c.rowmajor == a.rowmajor) {
	    if (arg_int) 
		scale_i(TRUE,
			a.type, a.data+a.byte_offset, a.stride, 1,
			c.type, c.data+c.byte_offset, c.stride, 1,
			c.n, c.m, i_scale);
	    else if (arg_float)
		scale_f(!is_integer(c.type),
			a.type, a.data+a.byte_offset, a.stride, 1,
			c.type, c.data+c.byte_offset, c.stride, 1,
			c.n, c.m, f_scale);
	    else if (arg_complex)
		scale_c(!is_integer(c.type)&&!is_float(c.type),
			a.type, a.data+a.byte_offset, a.stride, 1,
			c.type, c.data+c.byte_offset, c.stride, 1,
			c.n, c.m, c_scale);
	}
	else {
	    if (arg_int)
		scale_i(FALSE,
			a.type, a.data+a.byte_offset, 1, a.stride,
			c.type, c.data+c.byte_offset, c.stride, 1,
			c.n, c.m, i_scale);
	    else if (arg_float)
		scale_f(FALSE,
			a.type, a.data+a.byte_offset, 1, a.stride,
			c.type, c.data+c.byte_offset, c.stride, 1,
			c.n, c.m, f_scale);
	    else if (arg_complex)
		scale_c(FALSE,
			a.type, a.data+a.byte_offset, 1, a.stride,
			c.type, c.data+c.byte_offset, c.stride, 1,
			c.n, c.m, c_scale);
	}

	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

	return argv[2];	
    }
}

// sigmoid a matrix
ERL_NIF_TERM matrix_sigmoid(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	return enif_make_badarg(env);

    enif_rwlock_rlock(a.rw_lock);
    sigmoid(a.type, a.data+a.byte_offset, a.stride, 1,
	    c.type, c.data+c.byte_offset, c.stride, 1,
	    c.n, c.m);
    enif_rwlock_runlock(a.rw_lock);
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// sigmoid a matrix
ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	return enif_make_badarg(env);    
    
    enif_rwlock_rlock(a.rw_lock);
    sigmoid_prime(a.type, a.data+a.byte_offset, a.stride, 1,
		  c.type, c.data+c.byte_offset, c.stride, 1,
		  c.n, c.m);
    enif_rwlock_runlock(a.rw_lock);
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// rectifier a matrix
ERL_NIF_TERM matrix_rectifier(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	return enif_make_badarg(env);

    enif_rwlock_rlock(a.rw_lock);
    rectifier(a.type, a.data+a.byte_offset, a.stride, 1,
	      c.type, c.data+c.byte_offset, c.stride, 1,
	      c.n, c.m);
    enif_rwlock_runlock(a.rw_lock);
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}


// matrix_apply1
ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    unary_operation_t op;
    UNUSED(argc);
    
    if (!enif_is_atom(env, argv[2]))
	return enif_make_badarg(env);
    if (argv[2] == ATOM(sigmoid))            op = SIGMOID;
    else if (argv[2] == ATOM(sigmoid_prime)) op = SIGMOID_PRIME;
    else if (argv[2] == ATOM(rectifier))     op = RECTIFIER;
    else if (argv[2] == ATOM(tanh))          op = TANH;
    else if (argv[2] == ATOM(negate))        op = NEGATE;
    else if (argv[2] == ATOM(copy))          op = COPY;
    else if (argv[2] == ATOM(uniform))       op = UNIFORM;
    else if (argv[2] == ATOM(normal))        op = NORMAL;
    else if (argv[2] == ATOM(zero))          op = ZERO;
    else if (argv[2] == ATOM(one))           op = ONE;
    else if (argv[2] == ATOM(identity))      op = IDENTITY;
    else return enif_make_badarg(env);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &c))
	return enif_make_badarg(env);
    if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	return enif_make_badarg(env);    

    if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rwlock(c.rw_lock);

    if (c.rowmajor == a.rowmajor)
	apply1(op,
	       a.type, a.data+a.byte_offset, a.stride, 1,
	       c.type, c.data+c.byte_offset, c.stride, 1,
	       c.n, c.m);
    else
	apply1(op,
	       a.type, a.data+a.byte_offset, 1, a.stride,
	       c.type, c.data+c.byte_offset, c.stride, 1,
	       c.n, c.m);	

    enif_rwlock_rwunlock(c.rw_lock);
    if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

    return argv[1];	
}

// find argmax in matrix
ERL_NIF_TERM matrix_argmax(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t a;
    unsigned axis;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &axis))
	return enif_make_badarg(env);
    if (axis > 1)
	return enif_make_badarg(env);

    if (a.rowmajor) {
	if (axis == 0) {
	    // argmax for each column is returned (as a row)
	    if (!create_matrix(env,1,a.m,TRUE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    enif_rwlock_rlock(a.rw_lock);
	    argmax(a.type, a.data+a.byte_offset,a.stride,1,a.n,a.m,
		   (int32_t*)c.data);
	    enif_rwlock_runlock(a.rw_lock);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // argmax for each row is returned (as a column)
	    if (!create_matrix(env,1,a.n,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    enif_rwlock_rlock(a.rw_lock);
	    argmax(a.type, a.data+a.byte_offset,1,a.stride,a.m,a.n,
		   (int32_t*)c.data);
	    enif_rwlock_runlock(a.rw_lock);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    // argmax for each column is returned (as a row)
	    if (!create_matrix(env,1,a.n,TRUE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    enif_rwlock_rlock(a.rw_lock);
	    argmax(a.type, a.data+a.byte_offset,1,a.stride,a.m,a.n,
		   (int32_t*)c.data);
	    enif_rwlock_runlock(a.rw_lock);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // argmax for each row is returned (as a column)
	    if (!create_matrix(env,1,a.m,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    enif_rwlock_rlock(a.rw_lock);
	    argmax(a.type, a.data+a.byte_offset,a.stride,1,a.n,a.m,
		   (int32_t*)c.data);
	    enif_rwlock_runlock(a.rw_lock);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}	
    }
}


// transpose data rather then toggle rowmajor
ERL_NIF_TERM matrix_transpose_data(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    ERL_NIF_TERM bin;
    
    if (!get_matrix(env,argv[0],&a))
	return enif_make_badarg(env);

    if (argc == 1) {
	if (!create_matrix(env,a.m,a.n,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);

	enif_rwlock_rlock(a.rw_lock);

	m_apply1(transpose1, &a, &c);
	
	enif_rwlock_runlock(a.rw_lock);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {
	if (!get_matrix(env,argv[1],&c))
	    return enif_make_badarg(env);
	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	m_apply1(transpose1, &a, &c);

	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

	return argv[1];	
    }
}

static void matrix_dtor(ErlNifEnv* env, matrix_t* mat)
{
    UNUSED(env);
    DBG("matrix_dtor: %p\r\n", mat);
    if (mat->rw_lock)
	enif_rwlock_destroy(mat->rw_lock);
    enif_free(mat->base);
}


static int matrix_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    UNUSED(env);
    UNUSED(load_info);
    ErlNifResourceFlags tried;
    
    DBG("matrix_load\r\n");
    LOAD_ATOM(matrix);
    LOAD_ATOM(sigmoid);
    LOAD_ATOM(sigmoid_prime);
    LOAD_ATOM(rectifier);
    LOAD_ATOM(tanh);
    LOAD_ATOM(negate);
    LOAD_ATOM(uniform);
    LOAD_ATOM(normal);
    LOAD_ATOM(zero);
    LOAD_ATOM(one);
    LOAD_ATOM(identity);
    LOAD_ATOM(copy);
    LOAD_ATOM(true);
    LOAD_ATOM(false);

    matrix_r = enif_open_resource_type(env, 0, "matrix",
				       (ErlNifResourceDtor*) matrix_dtor,
				       ERL_NIF_RT_CREATE, &tried);

    enif_tsd_key_create("rand", &matrix_k);
    
    *priv_data = 0;
    return 0;
}

static int matrix_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data, 
			 ERL_NIF_TERM load_info)
{
    UNUSED(env);
    UNUSED(load_info);
    DBG("matrix_upgrade\r\n");
    *priv_data = *old_priv_data;
    return 0;
}

static void matrix_unload(ErlNifEnv* env, void* priv_data)
{
    UNUSED(env);
    UNUSED(priv_data);

    // how to find all thread data?
    enif_tsd_key_destroy(matrix_k);
    
    DBG("matrix_unload\r\n");
}

ERL_NIF_INIT(matrix, matrix_funcs,
	     matrix_load, NULL,
	     matrix_upgrade, matrix_unload)
