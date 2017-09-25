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
    SIGMOID_PRIME,
    RECTIFIER,
    TANH,
    NEGATE
} unary_operation_t;

typedef enum {
    PLUS,
    MINUS,
    TIMES
} binary_operation_t;

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

#define vint8_t_const(a)    {(a),(a),(a),(a),(a),(a),(a),(a),\
	                     (a),(a),(a),(a),(a),(a),(a),(a)}
#define vint16_t_const(a)   {(a),(a),(a),(a),(a),(a),(a),(a)}
#define vint32_t_const(a)   {(a),(a),(a),(a)}
#define vint64_t_const(a)   {(a),(a)}
#define vfloat32_t_const(a) {(a),(a),(a),(a)}
#define vfloat64_t_const(a) {(a),(a)}

#define vint8_t_zero    vint8_t_const(0)
#define vint16_t_zero   vint16_t_const(0)
#define vint32_t_zero   vint32_t_const(0)
#define vint64_t_zero   vint64_t_const(0)
#define vfloat32_t_zero vfloat32_t_const(0)
#define vfloat64_t_zero vfloat64_t_const(0)
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

#if (ERL_NIF_MAJOR_VERSION > 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION >= 12))
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(ERL_NIF_DIRTY_JOB_CPU_BOUND)}
#elif (ERL_NIF_MAJOR_VERSION > 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION >= 7))
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(0)}
#else
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr)}
#endif

ErlNifFunc matrix_funcs[] =
{
    NIF_FUNC("new_",          4, matrix_new),
    NIF_FUNC("add",           2, matrix_add),
    NIF_FUNC("add",           3, matrix_add),
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
    double z = op_sigmoid(x);
    return z*(1-z);
}

/////////////////////////////////////////////////////////////////////////////
//   ADD
/////////////////////////////////////////////////////////////////////////////

// add: int8 x int8 -> int8
#define PROCEDURE      mt_add_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_plus((a),(b))
#include "mt_binary_op.i"

// add: int16 x int16 -> int16
#define PROCEDURE      mt_add_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_plus((a),(b))
#include "mt_binary_op.i"

// add: int32 x int32 -> int32
#define PROCEDURE      mt_add_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_plus((a),(b))
#include "mt_binary_op.i"

// add: int64 x int64 -> int64
#define PROCEDURE      mt_add_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_plus((a),(b))
#include "mt_binary_op.i"

// add: float32 x float32 -> float32
#define PROCEDURE      mt_add_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_plus((a),(b))
#include "mt_binary_op.i"

// add: float64 x float64 -> float64
#define PROCEDURE      mt_add_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_plus((a),(b))
#include "mt_binary_op.i"

//----------------------------------------------------------------------------
//  add(int8/int16/int32/int64/float32/float64)
//----------------------------------------------------------------------------
#define SELECT mt_add
#define NAME add
#include "mt_binary_op_select.i"


#ifdef USE_GCC_VECTOR

// addv: int8 x int8 -> int8
#define PROCEDURE      mtv_add_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_plus((a),(b))
#define OPERATION(a,b) op_plus((a),(b))
#include "mtv_binary_op.i"

// addv: int16 x int16 -> int16
#define PROCEDURE      mtv_add_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_plus((a),(b))
#define OPERATION(a,b) op_plus((a),(b))
#include "mtv_binary_op.i"

// addv: int32 x int32 -> int32
#define PROCEDURE      mtv_add_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_plus((a),(b))
#define OPERATION(a,b) op_plus((a),(b))
#include "mtv_binary_op.i"

// addv: int64 x int64 -> int64
#define PROCEDURE      mtv_add_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_plus((a),(b))
#define OPERATION(a,b) op_plus((a),(b))
#include "mtv_binary_op.i"

// addv: float32 x float32 -> float32
#define PROCEDURE      mtv_add_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_plus((a),(b))
#define OPERATION(a,b) op_plus((a),(b))
#include "mtv_binary_op.i"

// addv: float64 x float64 -> float64
#define PROCEDURE      mtv_add_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_plus((a),(b))
#define OPERATION(a,b) op_plus((a),(b))
#include "mtv_binary_op.i"

#define SELECT mtv_add
#define NAME add
#include "mtv_binary_op_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   SUBTRACT
/////////////////////////////////////////////////////////////////////////////

// subtract: int8 x int8 -> int8
#define PROCEDURE           mt_subtract_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_minus((a),(b))
#include "mt_binary_op.i"

// subtract: int16 x int16 -> int16
#define PROCEDURE           mt_subtract_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_minus((a),(b))
#include "mt_binary_op.i"

// subtract: int32 x int32 -> int32
#define PROCEDURE           mt_subtract_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_minus((a),(b))
#include "mt_binary_op.i"

// subtract: int64 x int64 -> int64
#define PROCEDURE           mt_subtract_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_minus((a),(b))
#include "mt_binary_op.i"

// subtract: float32 x float32 -> float32
#define PROCEDURE           mt_subtract_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_minus((a),(b))
#include "mt_binary_op.i"

// subtract: float64 x float64 -> float64
#define PROCEDURE           mt_subtract_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_minus((a),(b))
#include "mt_binary_op.i"

//----------------------------------------------------------------------------
//  subtract(int8/int16/int32/int64/float32/float64)
//----------------------------------------------------------------------------
#define SELECT mt_subtract
#define NAME subtract
#include "mt_binary_op_select.i"


#ifdef USE_GCC_VECTOR

// subtractv: int8 x int8 -> int8
#define PROCEDURE           mtv_subtract_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_minus((a),(b))
#define OPERATION(a,b) op_minus((a),(b))
#include "mtv_binary_op.i"

// subtractv: int16 x int16 -> int16
#define PROCEDURE           mtv_subtract_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_minus((a),(b))
#define OPERATION(a,b) op_minus((a),(b))
#include "mtv_binary_op.i"

// subtractv: int32 x int32 -> int32
#define PROCEDURE           mtv_subtract_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_minus((a),(b))
#define OPERATION(a,b) op_minus((a),(b))
#include "mtv_binary_op.i"

// subtractv: int64 x int64 -> int64
#define PROCEDURE           mtv_subtract_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_minus((a),(b))
#define OPERATION(a,b) op_minus((a),(b))
#include "mtv_binary_op.i"

// subtractv: float32 x float32 -> float32
#define PROCEDURE           mtv_subtract_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_minus((a),(b))
#define OPERATION(a,b) op_minus((a),(b))
#include "mtv_binary_op.i"

// subtractv: float64 x float64 -> float64
#define PROCEDURE           mtv_subtract_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_minus((a),(b))
#define OPERATION(a,b) op_minus((a),(b))
#include "mtv_binary_op.i"

#define SELECT mtv_subtract
#define NAME subtract
#include "mtv_binary_op_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   TIMES
/////////////////////////////////////////////////////////////////////////////

// times: int8 x int8 -> int8
#define PROCEDURE      mt_times_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#include "mt_binary_op.i"

// times: int16 x int16 -> int16
#define PROCEDURE      mt_times_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#include "mt_binary_op.i"

// times: int32 x int32 -> int32
#define PROCEDURE      mt_times_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#include "mt_binary_op.i"

// times: int64 x int64 -> int64
#define PROCEDURE      mt_times_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#include "mt_binary_op.i"

// times: float32 x float32 -> float32
#define PROCEDURE      mt_times_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#include "mt_binary_op.i"

// times: float64 x float64 -> float64
#define PROCEDURE      mt_times_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#include "mt_binary_op.i"

//----------------------------------------------------------------------------
//  times(int8/int16/int32/int64/float32/float64)
//----------------------------------------------------------------------------
#define SELECT mt_times
#define NAME times
#include "mt_binary_op_select.i"


#ifdef USE_GCC_VECTOR

// timesv: int8 x int8 -> int8
#define PROCEDURE      mtv_times_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_times((a),(b))
#define OPERATION(a,b) op_times((a),(b))
#include "mtv_binary_op.i"

// timesv: int16 x int16 -> int16
#define PROCEDURE           mtv_times_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_times((a),(b))
#define OPERATION(a,b) op_times((a),(b))
#include "mtv_binary_op.i"

// timesv: int32 x int32 -> int32
#define PROCEDURE           mtv_times_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_times((a),(b))
#define OPERATION(a,b) op_times((a),(b))
#include "mtv_binary_op.i"

// timesv: int64 x int64 -> int64
#define PROCEDURE           mtv_times_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_times((a),(b))
#define OPERATION(a,b) op_times((a),(b))
#include "mtv_binary_op.i"

// timesv: float32 x float32 -> float32
#define PROCEDURE           mtv_times_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_times((a),(b))
#define OPERATION(a,b) op_times((a),(b))
#include "mtv_binary_op.i"

// timesv: float64 x float64 -> float64
#define PROCEDURE           mtv_times_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_times((a),(b))
#define OPERATION(a,b) op_times((a),(b))
#include "mtv_binary_op.i"

#define SELECT mtv_times
#define NAME times
#include "mtv_binary_op_select.i"

#endif


/////////////////////////////////////////////////////////////////////////////
//   NEGATE
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_negate_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_negate((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_negate_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_negate((a))
#include "mt_unary_op.i"


#define PROCEDURE           mt_negate_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_negate((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_negate_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_negate((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_negate_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_negate((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_negate_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_negate((a))
#include "mt_unary_op.i"

#define SELECT mt_negate
#define NAME negate
#define PARAMS_DECL
#define LOCALS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

#ifdef USE_GCC_VECTOR
#define PROCEDURE      mtv_negate_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_negate((a))
#define OPERATION(a)   op_negate((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_negate((a))
#define OPERATION(a)   op_negate((a))
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_negate_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_negate((a))
#define OPERATION(a)   op_negate((a))
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_negate_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_negate((a))
#define OPERATION(a)   op_negate((a))
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_negate_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_negate((a))
#define OPERATION(a)   op_negate((a))
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_negate_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_negate((a))
#define OPERATION(a)   op_negate((a))
#include "mtv_unary_op.i"

#define SELECT mtv_negate
#define NAME negate
#define PARAMS_DECL
#define PARAMS
#include "mtv_unary_op_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   SCALE * int64
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_scale_int64_int8
#define TYPE           int8_t
#define PARAMS_DECL    ,int64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_int64_int16
#define TYPE           int16_t
#define PARAMS_DECL    ,int64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_int64_int32
#define TYPE           int32_t
#define PARAMS_DECL    ,int64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_int64_int64
#define TYPE           int64_t
#define PARAMS_DECL    ,int64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_int64_float32
#define TYPE           float32_t
#define PARAMS_DECL    ,int64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_int64_float64
#define TYPE           float64_t
#define PARAMS_DECL    ,int64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define SELECT mt_scale_i
#define NAME scale_int64
#define PARAMS_DECL ,int64_t arg
#define PARAMS      ,arg
#include "mt_unary_op_select.i"

// vector version
#ifdef USE_GCC_VECTOR

#define PROCEDURE           mtv_scale_int64_int8
#define TYPE           int8_t
#define PARAMS_DECL    ,int64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_int64_int16
#define TYPE           int16_t
#define PARAMS_DECL    ,int64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_int64_int32
#define TYPE           int32_t
#define PARAMS_DECL    ,int64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_int64_int64
#define TYPE           int64_t
#define PARAMS_DECL    ,int64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_int64_float32
#define TYPE           float32_t
#define PARAMS_DECL    ,int64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_int64_float64
#define TYPE           float64_t
#define PARAMS_DECL    ,int64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define SELECT mtv_scale_i
#define NAME scale_int64
#define PARAMS_DECL ,int64_t arg
#define PARAMS      ,arg
#include "mtv_unary_op_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   SCALE * float64
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_scale_float64_int8
#define TYPE           int8_t
#define PARAMS_DECL    ,float64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_float64_int16
#define TYPE           int16_t
#define PARAMS_DECL    ,float64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_float64_int32
#define TYPE           int32_t
#define PARAMS_DECL    ,float64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_float64_int64
#define TYPE           int64_t
#define PARAMS_DECL    ,float64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_float64_float32
#define TYPE           float32_t
#define PARAMS_DECL    ,float64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define PROCEDURE           mt_scale_float64_float64
#define TYPE           float64_t
#define PARAMS_DECL    ,float64_t factor
#define LOCALS_DECL
#define OPERATION(a)   op_times((a),factor)
#include "mt_unary_op.i"

#define SELECT mt_scale_f
#define NAME scale_float64
#define PARAMS_DECL ,float64_t arg
#define PARAMS      ,arg
#include "mt_unary_op_select.i"


// vector version
#ifdef USE_GCC_VECTOR

#define PROCEDURE           mtv_scale_float64_int8
#define TYPE           int8_t
#define PARAMS_DECL    ,float64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_float64_int16
#define TYPE           int16_t
#define PARAMS_DECL    ,float64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_float64_int32
#define TYPE           int32_t
#define PARAMS_DECL    ,float64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_float64_int64
#define TYPE           int64_t
#define PARAMS_DECL    ,float64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_float64_float32
#define TYPE           float32_t
#define PARAMS_DECL    ,float64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define PROCEDURE           mtv_scale_float64_float64
#define TYPE           float64_t
#define PARAMS_DECL    ,float64_t arg
#define LOCALS_DECL    TYPE sarg = arg; VTYPE varg = VTYPE_CONST(sarg);
#define VOPERATION(a)  op_times((a),varg)
#define OPERATION(a)   op_times((a),sarg)
#include "mtv_unary_op.i"

#define SELECT mtv_scale_f
#define NAME scale_float64
#define PARAMS_DECL ,float64_t arg
#define PARAMS      ,arg
#include "mtv_unary_op_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   SIGMOID
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_sigmoid_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"


#define PROCEDURE      mt_sigmoid_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define SELECT mt_sigmoid
#define NAME sigmoid
#define PARAMS_DECL
#define LOCALS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

/////////////////////////////////////////////////////////////////////////////
//   SIGMOID_PRIME
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_sigmoid_prime_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_prime_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"


#define PROCEDURE           mt_sigmoid_prime_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_prime_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_prime_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE           mt_sigmoid_prime_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define SELECT mt_sigmoid_prime
#define NAME sigmoid_prime
#define PARAMS_DECL
#define LOCALS_DECL
#define PARAMS
#include "mt_unary_op_select.i"


/////////////////////////////////////////////////////////////////////////////
//   MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_multiply_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mt_mulop.i"

#define PROCEDURE           mt_multiply_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mt_mulop.i"

#define PROCEDURE           mt_multiply_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mt_mulop.i"

#define PROCEDURE           mt_multiply_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mt_mulop.i"

#define PROCEDURE           mt_multiply_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mt_mulop.i"

#define PROCEDURE           mt_multiply_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mt_mulop.i"

#define SELECT mt_multiply
#define NAME multiply
#include "mt_mulop_select.i"

#ifdef USE_GCC_VECTOR

#define PROCEDURE           mtv_multiply_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mtv_mulop.i"

#define PROCEDURE           mtv_multiply_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mtv_mulop.i"

#define PROCEDURE           mtv_multiply_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mtv_mulop.i"

#define PROCEDURE           mtv_multiply_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mtv_mulop.i"

#define PROCEDURE           mtv_multiply_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mtv_mulop.i"

#define PROCEDURE           mtv_multiply_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_times((a),(b))
#define OPERATION2(a,b) op_plus((a),(b))
#include "mtv_mulop.i"

#define SELECT mtv_multiply
#define NAME multiply
#include "mtv_mulop_select.i"

#endif

// a more general function for unary operations but a lot slower
static void apply1(int func,
		   matrix_type_t at, byte_t* ap, size_t as,
		   matrix_type_t ct, byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    size_t elem_size_a = element_size(at);
    size_t elem_size_c = element_size(ct);

    if (element_is_float(at) || element_is_float(ct)) {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		float64_t a = read_float(at, ap1);
		float64_t c;
		ap1 += elem_size_a;
		switch(func) {
		case SIGMOID:   c = op_sigmoid(a); break;
		case SIGMOID_PRIME: c = op_sigmoid_prime(a); break;
		case RECTIFIER: c = op_max(0,a); break;
		case TANH:      c = tanh(a); break;		
		case NEGATE:    c = -a; break;
		default:        c = 0; break;
		}
		write_float(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}
    }
    else {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		int64_t a = read_int(at, ap1);
		int64_t c;
		ap1 += elem_size_a;
		switch(func) {
		case SIGMOID:   c = op_sigmoid(a); break;
		case SIGMOID_PRIME: c = op_sigmoid_prime(a); break;
		case RECTIFIER: c = op_max(0,a); break;
		case TANH:      c = tanh(a); break;		
		case NEGATE:    c = -a; break;
		default:        c = 0; break;		    
		}
		write_int(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}
    }
}

// a more general function for unary operations but a lot slower
static void apply2(int func,
		   matrix_type_t at, byte_t* ap, size_t as,
		   matrix_type_t bt, byte_t* bp, size_t bs,
		   matrix_type_t ct, byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    size_t elem_size_a = element_size(at);
    size_t elem_size_b = element_size(bt);
    size_t elem_size_c = element_size(ct);

    if (element_is_float(at) || element_is_float(bt) || element_is_float(ct)) {
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {	    
		float64_t a = read_float(at, ap1);
		float64_t b = read_float(bt, bp1);
		float64_t c;
		ap1 += elem_size_a;
		bp1 += elem_size_b;
		switch(func) {
		case PLUS:   c = op_plus(a,b); break;
		case MINUS:  c = op_minus(a,b); break;
		case TIMES:  c = op_times(a,b); break;
		default:     c = 0; break;
		}
		write_float(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
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
		ap1 += elem_size_a;
		bp1 += elem_size_b;
		switch(func) {
		case PLUS:   c = op_plus(a,b); break;
		case MINUS:  c = op_minus(a,b); break;
		case TIMES:  c = op_times(a,b); break;
		default:     c = 0; break;		    
		}
		write_int(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    bp += bs*elem_size_b;
	    cp += cs*elem_size_c;
	}
    }
}



static void add(matrix_type_t at, byte_t* ap, size_t as, 
		matrix_type_t bt, byte_t* bp, size_t bs,
		matrix_type_t ct, byte_t* cp, size_t cs,
		size_t n, size_t m)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_add(at, ap, as, bp, bs, cp, cs, n, m);
	else
#endif
	    mt_add(at, ap, as, bp, bs, cp, cs, n, m);
    }
    else {
	apply2(PLUS, at, ap, as, bt, bp, bs, ct, cp, cs, n, m);
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
	    mtv_subtract(at, ap, as, bp, bs, cp, cs, n, m);
	else
#endif	    
	    mt_subtract(at, ap, as, bp, bs, cp, cs, n, m);
    }
    else {
	apply2(MINUS, at, ap, as, bt, bp, bs, ct, cp, cs, n, m);
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
	    mtv_times(at, ap, as, bp, bs, cp, cs, n, m);
	else
#endif
	    mt_times(at, ap, as, bp, bs, cp, cs, n, m);
    }
    else {
	apply2(TIMES, at, ap, as, bt, bp, bs, ct, cp, cs, n, m);
    }
}

static void negate(matrix_type_t at, byte_t* ap, size_t as,
		   matrix_type_t ct, byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(cp))
	    mtv_negate(at, ap, as, cp, cs, n, m);
	else
#endif
	    mt_negate(at, ap, as, cp, cs, n, m);
    }
    else {
	apply1(NEGATE, at, ap, as, ct, cp, cs, n, m);
    }
}

static void scale_i(matrix_type_t at, byte_t* ap, size_t as,
		    matrix_type_t ct, byte_t* cp, size_t cs,
		    size_t n, size_t m, int64_t factor)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(cp))
	    mtv_scale_i(at, ap, as, cp, cs, n, m, factor);
	else
#endif	
	    mt_scale_i(at, ap, as, cp, cs, n, m, factor);
    }
    else if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap1);
		ap1 += elem_size_a;
		write_float(ct, cp1, a*factor);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
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
		int64_t a = read_int(at, ap1);
		ap1 += elem_size_a;
		write_int(ct, cp1, a*factor);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}	
    }    
}

static void scale_f(matrix_type_t at, byte_t* ap, size_t as,
		    matrix_type_t ct, byte_t* cp, size_t cs,
		    size_t n, size_t m, float64_t factor)
{
    if (at == ct) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(cp))
	    mtv_scale_f(at, ap, as, cp, cs, n, m, factor);
	else
#endif		
	    mt_scale_f(at, ap, as, cp, cs, n, m, factor);
    }
    else if (element_is_float(at)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_float(at, ap1);
		ap1 += elem_size_a;
		write_float(ct, cp1, a*factor);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
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
		int64_t a = read_int(at, ap1);
		ap1 += elem_size_a;
		write_int(ct, cp1, a*factor);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}	
    }    
}


static void sigmoid(matrix_type_t at, byte_t* ap, size_t as,
		    matrix_type_t ct, byte_t* cp, size_t cs,
		    size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid(at, ap, as, cp, cs, n, m);
    }
    else {
	apply1(SIGMOID, at, ap, as, ct, cp, cs, n, m);
    }
}


static void sigmoid_prime(matrix_type_t at, byte_t* ap, size_t as,
			  matrix_type_t ct, byte_t* cp, size_t cs,
			  size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid_prime(at, ap, as, cp, cs, n, m);
    }
    else {
	apply1(SIGMOID_PRIME, at, ap, as, ct, cp, cs, n, m);
    }    
}

static void multiply(matrix_type_t at,byte_t* ap,size_t as,size_t an,size_t am,
		     matrix_type_t bt,byte_t* bp,size_t bs,size_t bn,size_t bm,
		     matrix_type_t ct,byte_t* cp,size_t cs)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR	
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_multiply(at,ap,as,an,am,bp,bs,bn,bm,cp,cs);
	else
#endif	    
	    mt_multiply(at,ap,as,an,am,bp,bs,bn,bm,cp,cs);

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
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.n != b.n) || (a.m != b.m))
	return enif_make_badarg(env);
    if (argc == 2) {
	c_t = combine_type(a.type, b.type);
	if (!make_matrix_resource(env,a.n,a.m,c_t,&c_bin_term,&c_data,&cp))
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
    else {  // argc == 3
	matrix_t c;
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if ((a.n != c.n) || (a.m != c.m))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);
	add(a.type, a.data+a.byte_offset, a.stride,
	    b.type, b.data+b.byte_offset, b.stride,
	    c.type, c.data+c.byte_offset, c.stride, c.n, c.m);
	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_runlock(b.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[2];
    }
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

    negate(a.type, a.data+a.byte_offset, a.stride,
	   c_t, c_data+cp->byte_offset, cp->stride, a.n, a.m);

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
