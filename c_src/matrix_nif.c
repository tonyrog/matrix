//
// Matrix operations
//
// defined for int8, int16, int32, int64, float32, float64
// both mixed and non mixed operations
// FIXME use vector ops if possible
//
#include <stdio.h>
#include <stdint.h>
#include <memory.h>
#include <math.h>
#include "erl_nif.h"

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
    FLOAT32 = 4,
    FLOAT64 = 5
} matrix_type_t;

typedef enum {
    SIGMOID,
    RECTIFIER,
    TANH,
    NEGATE
} unary_operation_t;

typedef unsigned char byte_t;
typedef float  float32_t;   // fixme: configure
typedef double float64_t;   // fixme: configure

#define USE_GCC_VECTOR

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

#define vint8_t_zero    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
#define vint16_t_zero   {0,0,0,0,0,0,0,0}
#define vint32_t_zero   {0,0,0,0}
#define vint64_t_zero   {0,0}
#define vfloat32_t_zero {0,0,0,0}
#define vfloat64_t_zero {0,0}
#else
#define ALIGN sizeof(void*)
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

#define LOAD_ATOM_STRING(name,string)			\
    atm_##name = enif_make_atom(env,string)

// FIXME: each row MUST be vector aligned!!!
// this means that rows must be padded with zeros
typedef struct {
    unsigned int n;
    unsigned int m;
    matrix_type_t type;
    size_t size;         // allocated memory size
    unsigned int offset; // offset to first element
    unsigned int stride; // stride elements per row
    unsigned int byte_offset; // offset to first element in bytes
    unsigned int byte_stride; // stride bytes per row
    ErlNifRWLock* rw_lock;    // make sure we can write "safe"
    byte_t* base;        // allocated memory
    byte_t*  data;       // aligned data
} matrix_t;    

static ErlNifResourceType* matrix_r;

static int matrix_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
static int matrix_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data,
		       ERL_NIF_TERM load_info);
static void matrix_unload(ErlNifEnv* env, void* priv_data);

static ERL_NIF_TERM matrix_new(ErlNifEnv* env, int argc, 
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
static ERL_NIF_TERM matrix_transpose(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sigmoid(ErlNifEnv* env, int argc, 
				   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc, 
					 const ERL_NIF_TERM argv[]);

#if (ERL_NIF_MAJOR_VERSION > 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION >= 7))
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(0)}
#else
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr)}
#endif

ErlNifFunc matrix_funcs[] =
{
    NIF_FUNC("new_",          4, matrix_new),
    NIF_FUNC("add",           2, matrix_add),
    NIF_FUNC("subtract",      2, matrix_subtract),
    NIF_FUNC("times",         2, matrix_times),
    NIF_FUNC("multiply",      2, matrix_multiply),
    NIF_FUNC("negate",        1, matrix_negate),
    NIF_FUNC("scale",         2, matrix_scale),
    NIF_FUNC("transpose",     1, matrix_transpose),
    NIF_FUNC("sigmoid",       1, matrix_sigmoid),
    NIF_FUNC("sigmoid_prime", 1, matrix_sigmoid_prime),
};

size_t element_size_[6] = { 1, 2, 4, 8, 4, 8 };

DECL_ATOM(matrix);

static size_t element_size(matrix_type_t type)
{
    return element_size_[type];
}

static int element_is_float(matrix_type_t type)
{
    return (type >= FLOAT32);
}

static matrix_type_t combine_type(matrix_type_t at, matrix_type_t bt)
{
    if (at > bt) return at; else return bt;
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
    default: break;
    }
}

// functional macros of arithmetic operators
// supported vector operators: +, -, *, /, unary minus, ^, |, &, ~, %.
// shift operators: << and >> for integer vectors
// comparison operators: ==, !=, <, <=, >, >=
#define plus(x,y)  ((x)+(y))
#define minus(x,y) ((x)-(y))
#define mul(x,y)   ((x)*(y))
#define div(x,y)   ((x)/(y))
#define rem(x,y)   ((x)%(y))
#define bxor(x,y)  ((x)^(y))
#define bor(x,y)   ((x)|(y))
#define band(x,y)  ((x)&(y))
#define unary_minus(x) (-(x))
#define bnot(x)        (~(x))
#define bsl(x,y)   ((x)<<(y))
#define bsr(x,y)   ((x)>>(y))
#define eq(x,y)    ((x)==(y))
#define neq(x,y)   ((x)!=(y))
#define lt(x,y)    ((x)<(y))
#define lte(x,y)   ((x)<=(y))
#define gt(x,y)    ((x)>(y))
#define gte(x,y)   ((x)>=(y))
// special...
//  vector versions of max and min are possible to 
//  construct by a little bit fiddel, max for example:
//  m = (x > y)
//  r = (x & m) | (y & ~m)
//
#define rectify(x) (((x)>0) & (x))
#define min(x,y)   (((x)<(y))?(x):(y))
#define max(x,y)   (((x)>(y))?(x):(y))
#define sigm(x)    (1.0/(1.0 + exp(-(x))))

static inline float64_t sigm_prime(float64_t x)
{
    double z = sigm(x);
    return z*(1-z);
}

#define MT_BINOP(name,fun,type)						\
static void mt_##name##_(type* ap, size_t as, type* bp, size_t bs, type* cp, size_t cs, size_t n, size_t m) \
{									\
    while(n--) {							\
	type* ap1 = ap;							\
	type* bp1 = bp;							\
	type* cp1 = cp;							\
	size_t m1 = m;							\
	while(m1--) {							\
	    type a = *ap1++;						\
	    type b = *bp1++;						\
	    *cp1++ = fun(a,b);						\
	}								\
        ap += as;							\
        bp += bs;							\
        cp += cs;							\
    }									\
}

#define MT_BINOP_SELECT(name)						\
static void mt_##name##_(matrix_type_t type, byte_t* ap, size_t as, byte_t* bp, size_t bs, byte_t* cp, size_t cs, size_t n, size_t m) \
{ \
  switch(type) { \
  case INT8: mt_##name##_int8_((int8_t*)ap,as,(int8_t*)bp,bs,(int8_t*)cp,cs,n,m); break; \
  case INT16: mt_##name##_int16_((int16_t*)ap,as,(int16_t*)bp,bs,(int16_t*)cp,cs,n,m); break; \
  case INT32: mt_##name##_int32_((int32_t*)ap,as,(int32_t*)bp,bs,(int32_t*)cp,cs,n,m); break; \
  case INT64: mt_##name##_int64_((int64_t*)ap,as,(int64_t*)bp,bs,(int64_t*)cp,cs,n,m); break; \
  case FLOAT32: mt_##name##_float32_((float32_t*)ap,as,(float32_t*)bp,bs,(float32_t*)cp,cs,n,m); break; \
  case FLOAT64: mt_##name##_float64_((float64_t*)ap,as,(float64_t*)bp,bs,(float64_t*)cp,cs,n,m); break; \
  default: break;  \
  }  \
}

// declare a binop that expand operators for each instance
// only use if all vectors are VSIZE aligned
#ifdef USE_GCC_VECTOR
#define MT_BINVOP(name,fun,type)					\
static void mtv_##name##_(type* ap, size_t as, type* bp, size_t bs, type* cp, size_t cs, size_t n, size_t m) \
{									\
    while(n--) {							\
	type* ap1 = ap;							\
	type* bp1 = bp;							\
	type* cp1 = cp;							\
	size_t m1 = m;							\
	while(m1 >= VELEMS(type)) {					\
	    v##type a = *(v##type*)ap1;					\
	    v##type b = *(v##type*)bp1;					\
	    ap1 += VELEMS(type);					\
	    bp1 += VELEMS(type);					\
	    *(v##type*)cp1 = fun(a,b);					\
	    cp1 += VELEMS(type);					\
	    m1  -= VELEMS(type);					\
	}								\
	while(m1--) {							\
	    type a = *ap1++;						\
	    type b = *bp1++;						\
	    *cp1++ = fun(a,b);						\
	}								\
        ap += as;							\
        bp += bs;							\
        cp += cs;							\
    }									\
}

#define MT_BINVOP_SELECT(name)						\
static void mtv_##name##_(matrix_type_t type, byte_t* ap, size_t as, byte_t* bp, size_t bs, byte_t* cp, size_t cs, size_t n, size_t m) \
{ \
  switch(type) { \
  case INT8: mtv_##name##_int8_((int8_t*)ap,as,(int8_t*)bp,bs,(int8_t*)cp,cs,n,m); break; \
  case INT16: mtv_##name##_int16_((int16_t*)ap,as,(int16_t*)bp,bs,(int16_t*)cp,cs,n,m); break; \
  case INT32: mtv_##name##_int32_((int32_t*)ap,as,(int32_t*)bp,bs,(int32_t*)cp,cs,n,m); break; \
  case INT64: mtv_##name##_int64_((int64_t*)ap,as,(int64_t*)bp,bs,(int64_t*)cp,cs,n,m); break; \
  case FLOAT32: mtv_##name##_float32_((float32_t*)ap,as,(float32_t*)bp,bs,(float32_t*)cp,cs,n,m); break; \
  case FLOAT64: mtv_##name##_float64_((float64_t*)ap,as,(float64_t*)bp,bs,(float64_t*)cp,cs,n,m); break; \
  default: break;  \
  }  \
}

#endif



#define MT_BINOP_FUN(name,ype)						\
static void mt_##name##_(type (*fun)(type, type),type* ap, size_t as, type* bp, size_t bs, type* cp, size_t cs, size_t n, size_t m) \
{									\
    while(n--) {							\
	type* ap1 = ap;							\
	type* bp1 = bp;							\
	type* cp1 = cp;							\
	size_t m1 = m;							\
	while(m1--) {							\
	    type a = *ap1++;						\
	    type b = *bp1++;						\
	    *cp1++ = fun(a,b);						\
	}								\
        ap += as;							\
        bp += bs;							\
        cp += cs;							\
    }									\
}

#define MT_UNOP(name,fun,type)						\
static void mt_##name##_(type* ap, size_t as, type* cp, size_t cs, size_t n, size_t m) \
{									\
    while(n--) {							\
	type* ap1 = ap;							\
	type* cp1 = cp;							\
	size_t m1 = m;							\
	while(m1--) {							\
	    type a = *ap1++;						\
	    *cp1++ = fun(a);						\
	}								\
        ap += as;							\
        cp += cs;							\
    }									\
}

#define MT_UNOP_SELECT(name) \
static void mt_##name##_(matrix_type_t type, byte_t* ap, size_t as, byte_t* cp, size_t cs, size_t n, size_t m) \
{ \
  switch(type) { \
  case INT8: mt_##name##_int8_((int8_t*)ap,as,(int8_t*)cp,cs,n,m); break; \
  case INT16: mt_##name##_int16_((int16_t*)ap,as,(int16_t*)cp,cs,n,m); break; \
  case INT32: mt_##name##_int32_((int32_t*)ap,as,(int32_t*)cp,cs,n,m); break; \
  case INT64: mt_##name##_int64_((int64_t*)ap,as,(int64_t*)cp,cs,n,m); break; \
  case FLOAT32: mt_##name##_float32_((float32_t*)ap,as,(float32_t*)cp,cs,n,m); break; \
  case FLOAT64: mt_##name##_float64_((float64_t*)ap,as,(float64_t*)cp,cs,n,m); break; \
  default: break;  \
  }  \
}

MT_BINOP(add_int8,plus,int8_t)
MT_BINOP(add_int16,plus,int16_t)
MT_BINOP(add_int32,plus,int32_t)
MT_BINOP(add_int64,plus,int64_t)
MT_BINOP(add_float32,plus,float32_t)
MT_BINOP(add_float64,plus,float64_t)
MT_BINOP_SELECT(add)

#ifdef USE_GCC_VECTOR
MT_BINVOP(add_int8,plus,int8_t)
MT_BINVOP(add_int16,plus,int16_t)
MT_BINVOP(add_int32,plus,int32_t)
MT_BINVOP(add_int64,plus,int64_t)
MT_BINVOP(add_float32,plus,float32_t)
MT_BINVOP(add_float64,plus,float64_t)
MT_BINVOP_SELECT(add)
#endif

MT_BINOP(times_int8,mul,int8_t)
MT_BINOP(times_int16,mul,int16_t)
MT_BINOP(times_int32,mul,int32_t)
MT_BINOP(times_int64,mul,int64_t)
MT_BINOP(times_float32,mul,float32_t)
MT_BINOP(times_float64,mul,float64_t)
MT_BINOP_SELECT(times)

#ifdef USE_GCC_VECTOR
MT_BINVOP(times_int8,mul,int8_t)
MT_BINVOP(times_int16,mul,int16_t)
MT_BINVOP(times_int32,mul,int32_t)
MT_BINVOP(times_int64,mul,int64_t)
MT_BINVOP(times_float32,mul,float32_t)
MT_BINVOP(times_float64,mul,float64_t)
MT_BINVOP_SELECT(times)
#endif

MT_BINOP(subtract_int8,minus,int8_t)
MT_BINOP(subtract_int16,minus,int16_t)
MT_BINOP(subtract_int32,minus,int32_t)
MT_BINOP(subtract_int64,minus,int64_t)
MT_BINOP(subtract_float32,minus,float32_t)
MT_BINOP(subtract_float64,minus,float64_t)
MT_BINOP_SELECT(subtract)

#ifdef USE_GCC_VECTOR
MT_BINVOP(subtract_int8,minus,int8_t)
MT_BINVOP(subtract_int16,minus,int16_t)
MT_BINVOP(subtract_int32,minus,int32_t)
MT_BINVOP(subtract_int64,minus,int64_t)
MT_BINVOP(subtract_float32,minus,float32_t)
MT_BINVOP(subtract_float64,minus,float64_t)
MT_BINVOP_SELECT(subtract)
#endif

MT_UNOP(negate_int8,unary_minus,int8_t)
MT_UNOP(negate_int16,unary_minus,int16_t)
MT_UNOP(negate_int32,unary_minus,int32_t)
MT_UNOP(negate_int64,unary_minus,int64_t)
MT_UNOP(negate_float32,unary_minus,float32_t)
MT_UNOP(negate_float64,unary_minus,float64_t)
MT_UNOP_SELECT(negate)

MT_UNOP(sigmoid_int8,sigm,int8_t)
MT_UNOP(sigmoid_int16,sigm,int16_t)
MT_UNOP(sigmoid_int32,sigm,int32_t)
MT_UNOP(sigmoid_int64,sigm,int64_t)
MT_UNOP(sigmoid_float32,sigm,float32_t)
MT_UNOP(sigmoid_float64,sigm,float64_t)
MT_UNOP_SELECT(sigmoid)

MT_UNOP(sigmoid_prime_int8,sigm_prime,int8_t)
MT_UNOP(sigmoid_prime_int16,sigm_prime,int16_t)
MT_UNOP(sigmoid_prime_int32,sigm_prime,int32_t)
MT_UNOP(sigmoid_prime_int64,sigm_prime,int64_t)
MT_UNOP(sigmoid_prime_float32,sigm_prime,float32_t)
MT_UNOP(sigmoid_prime_float64,sigm_prime,float64_t)
MT_UNOP_SELECT(sigmoid_prime)

#define MT_MULOP(name,type,atype)				      \
    static void mt_##name##_(type* ap,size_t as,size_t an, size_t am,	\
			     type* bp,size_t bs,size_t bn, size_t bm,	\
			     type* cp,size_t cs)			\
{ \
    size_t i, j, k;		    \
    (void) bn;			    \
    for (i=0; i<an; i++) {	    \
        type* cp1 = cp;		    \
	for (j=0; j<bm; j++) {	    \
	    atype sum = 0;	    \
	    type* bp1 = bp + j;		\
	    type* ap1 = ap;		\
	    for (k = 0; k < am; k++) { \
		sum += (*ap1)*(*bp1);	\
		ap1 += 1;		\
		bp1 += bs;		\
	    }				\
	    *cp1++ = sum;		\
	}				\
	ap += as;			\
	cp += cs;			\
    } \
}

#define MT_MULOP_SELECT(name)						\
static void mt_##name##_(matrix_type_t type, byte_t* ap,size_t as,size_t an, size_t am, byte_t* bp,size_t bs,size_t bn, size_t bm, byte_t* cp,size_t cs) \
{									\
    switch(type) {							\
    case INT8: mt_##name##_int8_((int8_t*)ap,as,an,am,(int8_t*)bp,bs,bn,bm,(int8_t*)cp,cs); break; \
    case INT16: mt_##name##_int16_((int16_t*)ap,as,an,am,(int16_t*)bp,bs,bn,bm,(int16_t*)cp,cs); break; \
    case INT32: mt_##name##_int32_((int32_t*)ap,as,an,am,(int32_t*)bp,bs,bn,bm,(int32_t*)cp,cs); break; \
    case INT64: mt_##name##_int64_((int64_t*)ap,as,an,am,(int64_t*)bp,bs,bn,bm,(int64_t*)cp,cs); break; \
    case FLOAT32: mt_##name##_float32_((float32_t*)ap,as,an,am,(float32_t*)bp,bs,bn,bm,(float32_t*)cp,cs); break; \
    case FLOAT64: mt_##name##_float64_((float64_t*)ap,as,an,am,(float64_t*)bp,bs,bn,bm,(float64_t*)cp,cs); break; \
    default: break;							\
    }									\
}

MT_MULOP(multiply_int8,int8_t,int32_t)
MT_MULOP(multiply_int16,int16_t,int32_t)
MT_MULOP(multiply_int32,int32_t,int64_t)
MT_MULOP(multiply_int64,int64_t,int64_t)
MT_MULOP(multiply_float32,float32_t,float64_t)
MT_MULOP(multiply_float64,float64_t,float64_t)
MT_MULOP_SELECT(multiply)

#ifdef USE_GCC_VECTOR

#if 0
// Load column reversed
#define MT_VRLOADCOL(type)					\
v##type mtv_load_column_##type(type* ap, size_t as, size_t n)	\
{								\
    v##type r;							\
    size_t i = VELEMS(type);					\
    while(n--) {						\
        r[--i] = *ap;						\
	ap += as;						\
    }								\
    while(i) {							\
	r[--i] = 0;						\
    }								\
    return r;							\
}
#endif

#define MT_VLOADCOL(type)					\
static inline v##type mtv_load_column_##type(type* ap, size_t as, size_t n) \
{								\
    v##type r;							\
    size_t i = 0;						\
    while(n--) {						\
        r[i++] = *ap;						\
	ap += as;						\
    }								\
    while(i < VELEMS(type)) {					\
	r[i++] = 0;						\
    }								\
    return r;							\
}

MT_VLOADCOL(int8_t)
MT_VLOADCOL(int16_t)
MT_VLOADCOL(int32_t)
MT_VLOADCOL(int64_t)
MT_VLOADCOL(float32_t)
MT_VLOADCOL(float64_t)

#define MT_VMULOP(name,type,atype)					\
static void mtv_##name##_(type* ap,size_t as,size_t an, size_t am, type* bp,size_t bs,size_t bn, size_t bm, type* cp,size_t cs) \
{									\
    while (bm--) {							\
        v##type col[(bn+VELEMS(type)-1)/VELEMS(type)];			\
	type* ap1 = ap;							\
	type* bp1 = bp;							\
	type* cp1 = cp;							\
	size_t n = bn;							\
	size_t j = 0;							\
	while(n >= VELEMS(type)){					\
	    col[j++] = mtv_load_column_##type(bp1,bs,VELEMS(type));	\
	    n -= VELEMS(type);						\
	    bp1 += bs*VELEMS(type);					\
        }								\
	if(n) {								\
	    col[j++] = mtv_load_column_##type(bp1,bs,n);		\
	}								\
	bp++;								\
	n = an;								\
	while(n--) {							\
	    atype sum = 0;						\
	    v##type vsum = v##type##_zero;				\
	    size_t m = am;						\
	    type* ap2 = ap1;						\
	    type* tp = (type*) &col[0];					\
	    size_t i;							\
	    while(m >= VELEMS(type)) {					\
		v##type r = (*(v##type*)tp) * (*(v##type*)ap2);		\
		vsum = vsum + r;					\
		tp += VELEMS(type);					\
		ap2 += VELEMS(type);					\
		m -= VELEMS(type);					\
	    }								\
	    for (i = 0; i < VELEMS(type); i++)				\
		sum += vsum[i];						\
	    while(m--) {						\
		sum += (*tp++ * *ap2++);				\
	    }								\
	    *cp1 = sum;							\
	    cp1 += cs;							\
	    ap1 += as;							\
	}								\
	cp++;								\
    }									\
}

#define MT_VMULOP_SELECT(name)						\
static void mtv_##name##_(matrix_type_t type, byte_t* ap,size_t as,size_t an, size_t am, byte_t* bp,size_t bs,size_t bn, size_t bm, byte_t* cp,size_t cs) \
{									\
    switch(type) {							\
    case INT8: mtv_##name##_int8_((int8_t*)ap,as,an,am,(int8_t*)bp,bs,bn,bm,(int8_t*)cp,cs); break; \
    case INT16: mtv_##name##_int16_((int16_t*)ap,as,an,am,(int16_t*)bp,bs,bn,bm,(int16_t*)cp,cs); break; \
    case INT32: mtv_##name##_int32_((int32_t*)ap,as,an,am,(int32_t*)bp,bs,bn,bm,(int32_t*)cp,cs); break; \
    case INT64: mtv_##name##_int64_((int64_t*)ap,as,an,am,(int64_t*)bp,bs,bn,bm,(int64_t*)cp,cs); break; \
    case FLOAT32: mtv_##name##_float32_((float32_t*)ap,as,an,am,(float32_t*)bp,bs,bn,bm,(float32_t*)cp,cs); break; \
    case FLOAT64: mtv_##name##_float64_((float64_t*)ap,as,an,am,(float64_t*)bp,bs,bn,bm,(float64_t*)cp,cs); break; \
    default: break;							\
    }									\
}

MT_VMULOP(multiply_int8,int8_t,int32_t)
MT_VMULOP(multiply_int16,int16_t,int32_t)
MT_VMULOP(multiply_int32,int32_t,int64_t)
MT_VMULOP(multiply_int64,int64_t,int64_t)
MT_VMULOP(multiply_float32,float32_t,float64_t)
MT_VMULOP(multiply_float64,float64_t,float64_t)
MT_VMULOP_SELECT(multiply)

#endif

static void add(matrix_type_t at, byte_t* ap, size_t as, 
		matrix_type_t bt, byte_t* bp, size_t bs,
		matrix_type_t ct, byte_t* cp, size_t cs,
		size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_add_(at, ap, as, bp, bs, cp, cs, n, m);
	else
#endif
	    mt_add_(at, ap, as, bp, bs, cp, cs, n, m);
    }
    else if (element_is_float(at) || element_is_float(bt) ||
	     element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap1);
		float64_t b = read_float(bt, bp1);
		ap += elem_size_a;
		bp += elem_size_b;
		write_float(ct, cp1, a+b);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap1);
		int64_t b = read_int(bt, bp1);
		ap += elem_size_a;
		bp += elem_size_b;
		write_int(ct, cp1, a+b);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}	
    }
}

static void subtract(matrix_type_t at, byte_t* ap, size_t as, 
		     matrix_type_t bt, byte_t* bp, size_t bs,
		     matrix_type_t ct, byte_t* cp, size_t cs,
		     size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR	
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_subtract_(at, ap, as, bp, bs, cp, cs, n, m);
	else
#endif	    
	    mt_subtract_(at, ap, as, bp, bs, cp, cs, n, m);
    }
    else if (element_is_float(at) || element_is_float(bt) ||
	     element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap1);
		float64_t b = read_float(bt, bp1);
		ap += elem_size_a;
		bp += elem_size_b;
		write_float(ct, cp1, a-b);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap1);
		int64_t b = read_int(bt, bp1);
		ap += elem_size_a;
		bp += elem_size_b;
		write_int(ct, cp1, a-b);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}	
    }
}


static void times(matrix_type_t at, byte_t* ap, size_t as, 
		  matrix_type_t bt, byte_t* bp, size_t bs,
		  matrix_type_t ct, byte_t* cp, size_t cs,
		  size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_times_(at, ap, as, bp, bs, cp, cs, n, m);
	else
#endif
	    mt_times_(at, ap, as, bp, bs, cp, cs, n, m);
    }
    else if (element_is_float(at) || element_is_float(bt) ||
	     element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap1);
		float64_t b = read_float(bt, bp1);
		ap += elem_size_a;
		bp += elem_size_b;
		write_float(ct, cp1, a*b);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap1);
		int64_t b = read_int(bt, bp1);
		ap += elem_size_a;
		bp += elem_size_b;
		write_int(ct, cp1, a*b);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}	
    }
}


// a more general function for unary operations but slower
static void apply1(int func,
		   matrix_type_t at, byte_t* ap, size_t as,
		   matrix_type_t ct, byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    size_t elem_size_a = element_size(at);
    size_t elem_size_c = element_size(ct);

    if (element_is_float(at)) {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		float64_t a = read_float(at, ap1);
		float64_t c;
		ap1 += elem_size_a;
		switch(func) {
		case SIGMOID:   c = sigm(a); break;
		case RECTIFIER: c = max(0,a); break;
		case TANH:      c = tanh(a); break;		
		case NEGATE:    c = -a; break;
		}
		write_float(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
    else {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		int64_t a = read_float(at, ap1);
		int64_t c;
		ap1 += elem_size_a;
		switch(func) {
		case SIGMOID:   c = sigm(a); break;
		case RECTIFIER: c = max(0,a); break;
		case TANH:      c = tanh(a); break;		
		case NEGATE:    c = -a; break;
		}
		write_int(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
}

static void negate(matrix_type_t at, byte_t* ap, size_t as,
		   matrix_type_t ct, byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    if (at == ct) {
	mt_negate_(at, ap, as, cp, cs, n, m);
    }
    else if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap);
		ap += elem_size_a;
		write_float(ct, cp, -a);
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap);
		ap += elem_size_a;
		write_int(ct, cp, -a);
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}	
    }    
}


static void scale_i(matrix_type_t at, byte_t* ap, size_t as,
		    matrix_type_t ct, byte_t* cp, size_t cs,
		    size_t n, size_t m, int64_t factor)
{
    if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap);
		ap += elem_size_a;
		write_float(ct, cp, a*factor);
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap);
		ap += elem_size_a;
		write_int(ct, cp, a*factor);
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}	
    }    
}

static void scale_f(matrix_type_t at, byte_t* ap, size_t as,
		    matrix_type_t ct, byte_t* cp, size_t cs,
		    size_t n, size_t m, float64_t factor)
{
    if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap);
		ap += elem_size_a;
		write_float(ct, cp, a*factor);
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_int(at, ap);
		ap += elem_size_a;
		write_int(ct, cp, a*factor);
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}	
    }    
}


static void sigmoid(matrix_type_t at, byte_t* ap, size_t as,
		    matrix_type_t ct, byte_t* cp, size_t cs,
		    size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid_(at, ap, as, cp, cs, n, m);
    }
    else if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap);
		ap += elem_size_a;
		write_float(ct, cp, sigm(a));
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_int(at, ap);
		ap += elem_size_a;
		write_int(ct, cp, sigm(a));
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}	
    }    
}

static void sigmoid_prime(matrix_type_t at, byte_t* ap, size_t as,
			  matrix_type_t ct, byte_t* cp, size_t cs,
			  size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid_prime_(at, ap, as, cp, cs, n, m);
    }
    else if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap);
		ap += elem_size_a;
		write_float(ct, cp, sigm_prime(a));
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_int(at, ap);
		ap += elem_size_a;
		write_int(ct, cp, sigm_prime(a));
		cp += elem_size_c;
	    }
	    ap1 += as*elem_size_a;
	    cp1 += cs*elem_size_c;
	}	
    }    
}

static void multiply(matrix_type_t at,byte_t* ap,size_t as,size_t an,size_t am,
		     matrix_type_t bt,byte_t* bp,size_t bs,size_t bn,size_t bm,
		     matrix_type_t ct,byte_t* cp,size_t cs)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR	
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_multiply_(at,ap,as,an,am,bp,bs,bn,bm,cp,cs);
	else
#endif	    
	    mt_multiply_(at,ap,as,an,am,bp,bs,bn,bm,cp,cs);

    }
    else if (element_is_float(at) || element_is_float(bt) ||
	     element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);
	unsigned int i, j, k;
	
	for (i=0; i<an; i++) {
	    byte_t* cp1 = cp;
	    for (j=0; j<bm; j++) {
		float64_t sum = 0;
		byte_t* bp1 = bp + j*elem_size_b;  // column pointer
		byte_t* ap1 = ap;                  // row pointer
		for (k = 0; k < am; k++) {
		    float64_t a = read_float(at, ap1);
		    float64_t b = read_float(bt, bp1);
		    sum += a*b;
		    ap1 += elem_size_a;
		    bp1 += bs*elem_size_b;
		}
		write_float(ct, cp1, sum);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);
	unsigned int i, j, k;
	
	for (i=0; i<an; i++) {
	    byte_t* cp1 = cp;
	    for (j=0; j<bm; j++) {
		int64_t sum = 0;
		byte_t* bp1 = bp + j*elem_size_b;  // column pointer
		byte_t* ap1 = ap; // row pointer
		for (k = 0; k < am; k++) {
		    int64_t a = read_int(at, ap1);
		    int64_t b = read_int(bt, bp1);
		    sum += a*b;
		    ap1 += elem_size_a;
		    bp1 += bs*elem_size_b;
		}
		write_int(ct, cp1, sum);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}
    }
}

matrix_t* alloc_matrix_resource(size_t n, size_t m, matrix_type_t type,
				size_t align)
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

int make_matrix_resource(ErlNifEnv* env, unsigned int n, unsigned int m,
			 matrix_type_t type, ERL_NIF_TERM* term, byte_t** data,
			 matrix_t** mpp)
{
    matrix_t* mp;
    
    if ((mp = alloc_matrix_resource(n, m, type, ALIGN)) != NULL) {
	*term = enif_make_resource_binary(env, mp, mp->data, mp->size);
	enif_release_resource(mp);
	*data = mp->data;
	*mpp = mp;
	return 1;
    }
    return 0;
}


// Get matrix argument
// { 'matrix', n, m, type, ptr, offset, stride, binary-data }
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
    if (arity != 8) return 0;
    if (elems[0] != ATOM(matrix)) return 0;
    if (!enif_get_uint(env, elems[1], &mp->n)) return 0;
    if (!enif_get_uint(env, elems[2], &mp->m)) return 0;
    if (!enif_get_uint(env, elems[3], &type)) return 0;
    if (type > FLOAT64) return 0;
    mp->type = type;
    if (!enif_get_ulong(env, elems[4], &ptr)) return 0;
    if (!enif_get_uint(env, elems[5], &mp->offset)) return 0;
    if (!enif_get_uint(env, elems[6], &mp->stride)) return 0;
    mp->byte_offset = mp->offset*element_size(type);
    mp->byte_stride = mp->stride*element_size(type);	
    
    if (ptr != 0) {
	matrix_t* rmp;
	if (!enif_get_resource(env, elems[7], matrix_r, (void**)&rmp))
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
	if (!enif_inspect_binary(env, elems[7], &bin)) return 0;
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
				matrix_type_t type,
				matrix_t* mp,
				ERL_NIF_TERM binary)
{
    unsigned int stride;
    unsigned int offset;

    if (mp != NULL) {
	stride = mp->stride;
	offset = mp->offset;
    }
    else {
	stride = m;  // fixme
	offset = 0;  // fixme
    }
    return enif_make_tuple8(env,
			    ATOM(matrix),
			    enif_make_uint(env, n),
			    enif_make_uint(env, m),
			    enif_make_uint(env, type),
			    enif_make_uint64(env, (uintptr_t)mp),
			    enif_make_uint(env, offset),
			    enif_make_uint(env, stride),
			    binary);
}


ERL_NIF_TERM matrix_new(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n, m, type;
    ErlNifBinary bin;
    matrix_t* mp;
    byte_t* a_data;
    ERL_NIF_TERM a_matrix;
    ERL_NIF_TERM a_bin_term;
    (void) argc;
    
    if (!enif_get_uint(env, argv[0], &n))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &m))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &type))
	return enif_make_badarg(env);
    if (type > FLOAT64)
	return enif_make_badarg(env);
    if (!enif_inspect_iolist_as_binary(env, argv[3], &bin))
	return enif_make_badarg(env);
    if (n*m*element_size(type) < bin.size)
	return enif_make_badarg(env);
    if (!make_matrix_resource(env, n, m, type, &a_bin_term, &a_data, &mp))
	return enif_make_badarg(env);
    if (mp->stride == mp->m)
	memcpy(a_data, bin.data, mp->size);
    else {
	byte_t* b_data = bin.data;
	size_t  b_row_bytes = mp->m*element_size(type);
	size_t  a_row_bytes  = mp->stride*element_size(type);
	size_t i;
	for (i = 0; i < n; i++) {
	    memcpy(a_data, b_data, b_row_bytes);
	    a_data += a_row_bytes;
	    b_data += b_row_bytes;
	}
    }
    a_matrix = make_matrix(env, n, m, type, mp, a_bin_term);
    return a_matrix;
    
}

// add two matrices
ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.n != b.n) || (a.m != b.m))
	return enif_make_badarg(env);

    c_t = combine_type(a.type, b.type);
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);
    
    enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rlock(b.rw_lock);
    add(a.type, a.data+a.byte_offset, a.stride,
	b.type, b.data+b.byte_offset, b.stride,
	c_t, c_data, cp->stride, a.n, a.m);
    enif_rwlock_runlock(b.rw_lock);    
    enif_rwlock_runlock(a.rw_lock);


    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}

// subtract two matrices
ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.n != b.n) || (a.m != b.m))
	return enif_make_badarg(env);

    c_t = combine_type(a.type, b.type);
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);

    enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rlock(b.rw_lock);    
    subtract(a.type, a.data+a.byte_offset, a.stride,
	     b.type, b.data+b.byte_offset, b.stride,
	     c_t, c_data, cp->stride, a.n, a.m);
    enif_rwlock_runlock(b.rw_lock);    
    enif_rwlock_runlock(a.rw_lock);
    
    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}

// multiply two matrices element wise
ERL_NIF_TERM matrix_times(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.n != b.n) || (a.m != b.m))
	return enif_make_badarg(env);

    c_t = combine_type(a.type, b.type);
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);
    
    enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rlock(b.rw_lock);
    times(a.type, a.data+a.byte_offset, a.stride,
	  b.type, b.data+b.byte_offset, b.stride,
	  c_t, c_data, cp->stride, a.n, a.m);
    enif_rwlock_runlock(b.rw_lock);    
    enif_rwlock_runlock(a.rw_lock);

    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}

// multiply two matrices
ERL_NIF_TERM matrix_multiply(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if (a.m != b.n)
	return enif_make_badarg(env);

    c_t = combine_type(a.type, b.type);
    if (!make_matrix_resource(env, a.n, b.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);

    enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rlock(b.rw_lock);
    multiply(a.type, a.data+a.byte_offset, a.stride, a.n, a.m,
	     b.type, b.data+b.byte_offset, b.stride, b.n, b.m,
	     c_t, c_data+cp->byte_offset, cp->stride);
    enif_rwlock_runlock(b.rw_lock);    
    enif_rwlock_runlock(a.rw_lock);
    
    c_matrix = make_matrix(env, a.n, b.m, c_t, cp, c_bin_term);
    return c_matrix;
}


// negate a matrix
ERL_NIF_TERM matrix_negate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;    
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    c_t = a.type;
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);
    enif_rwlock_rlock(a.rw_lock);
#if 1
    negate(a.type, a.data+a.byte_offset, a.stride,
	   c_t, c_data+cp->byte_offset, cp->stride, a.n, a.m);
#else
    apply1(NEGATE,
	   a.type, a.data+a.byte_offset, a.stride,
	   c_t, c_data+cp->byte_offset, cp->stride, a.n, a.m);
#endif
    enif_rwlock_runlock(a.rw_lock);
    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}

// scale a matrix
ERL_NIF_TERM matrix_scale(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    ErlNifSInt64 i_scale;
    double       f_scale;
    int is_int;
    (void) argc;
    
    if (!get_matrix(env, argv[1], &a))
	return enif_make_badarg(env);
    is_int = 1;
    if (!enif_get_int64(env, argv[0], &i_scale)) {
	if (!enif_get_double(env, argv[0], &f_scale))
	    return enif_make_badarg(env);
	is_int = 0;
    }
    c_t = a.type;
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);
    enif_rwlock_rlock(a.rw_lock);
    if (is_int) 
	scale_i(a.type, a.data+a.byte_offset, a.stride,
		c_t, c_data+cp->byte_offset, cp->stride, a.n, a.m, i_scale);
    else
	scale_f(a.type, a.data+a.byte_offset, a.stride,
		c_t, c_data+cp->byte_offset, cp->stride, a.n, a.m, f_scale);
    enif_rwlock_runlock(a.rw_lock);
    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}

// sigmoid a matrix
ERL_NIF_TERM matrix_sigmoid(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;    
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    c_t = a.type;
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);
    enif_rwlock_rlock(a.rw_lock);
    sigmoid(a.type, a.data+a.byte_offset, a.stride,
	    c_t, c_data+cp->byte_offset, cp->stride,
	    a.n, a.m);
    enif_rwlock_runlock(a.rw_lock);    
    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}

// sigmoid a matrix
ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[])
{
    matrix_t a;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;    
    ERL_NIF_TERM c_matrix;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    c_t = a.type;
    if (!make_matrix_resource(env, a.n, a.m, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);
    enif_rwlock_rlock(a.rw_lock);
    sigmoid_prime(a.type, a.data+a.byte_offset, a.stride,
		  c_t, c_data+cp->byte_offset, cp->stride,
		  a.n, a.m);
    enif_rwlock_runlock(a.rw_lock);    
    c_matrix = make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    return c_matrix;
}


// transpose a matrix
ERL_NIF_TERM matrix_transpose(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    size_t elem_size;
    byte_t* c_data;
    matrix_type_t c_t;
    ERL_NIF_TERM c_bin_term;    
    ERL_NIF_TERM c_matrix;    
    size_t i, j;
    matrix_t* cp;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    c_t = a.type;
    if (!make_matrix_resource(env, a.m, a.n, c_t, &c_bin_term, &c_data, &cp))
	return enif_make_badarg(env);

    enif_rwlock_rlock(a.rw_lock);
    elem_size = element_size(a.type);
    
    for (j = 0; j < a.m; j++) {
	byte_t* a_ptr = a.data + a.byte_offset + j*elem_size; // column pointer
	byte_t* c_ptr = c_data + cp->byte_offset;             // row pointer
	
	for (i = 0; i < a.n; i++) {
	    memcpy(c_ptr, a_ptr, elem_size);
	    c_ptr += elem_size;           // next element
	    a_ptr += a.byte_stride;       // next column
	}
	c_data += cp->byte_stride; // next row
    }
    enif_rwlock_runlock(a.rw_lock);    
    c_matrix = make_matrix(env, a.m, a.n, a.type, cp, c_bin_term);
    return c_matrix;
}

static void matrix_dtor(ErlNifEnv* env, matrix_t* mat)
{
    (void) env;
    DBG("matrix_dtor: %p", matj);
    if (mat->rw_lock)
	enif_rwlock_destroy(mat->rw_lock);
    enif_free(mat->base);
}


static int matrix_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    (void) env;
    (void) load_info;
    ErlNifResourceFlags tried;
    
    DBG("matrix_load\r\n");
    LOAD_ATOM(matrix);

    matrix_r = enif_open_resource_type(env, 0, "matrix",
				       (ErlNifResourceDtor*) matrix_dtor,
				       ERL_NIF_RT_CREATE, &tried);
    *priv_data = 0;
    return 0;
}

static int matrix_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data, 
			 ERL_NIF_TERM load_info)
{
    (void) env;
    (void) load_info;
    DBG("matrix_upgrade\r\n");
    *priv_data = *old_priv_data;
    return 0;
}

static void matrix_unload(ErlNifEnv* env, void* priv_data)
{
    (void) env;
    (void) priv_data;
    DBG("matrix_unload\r\n");
}

ERL_NIF_INIT(matrix, matrix_funcs,
	     matrix_load, NULL,
	     matrix_upgrade, matrix_unload)
