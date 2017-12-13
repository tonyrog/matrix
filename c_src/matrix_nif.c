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

#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#define DBG(...) printf(__VA_ARGS__)
#define BADARG(env) printf("matrix_nif.c: badarg line=%d\r\n", __LINE__), enif_make_badarg((env))
#else
#define DBG(...)
#define BADARG(env) enif_make_badarg((env))
#endif

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

typedef enum {
    ZERO = 0,
    ONE  = 1,
    COPY = 2,
    NEGATE = 3,
    SIGMOID = 4,
    SIGMOID_PRIME = 5,
    SIGMOID_PRIME1 = 6,
    SOFTPLUS = 7,
    SOFTPLUS_PRIME = 8,
    RELU = 9,
    RELU_PRIME = 10,
    LEAKY_RELU = 11,
    LEAKY_RELU_PRIME = 12,
    TANH = 13,
    TANH_PRIME = 14,
    TANH_PRIME1 = 15,
    EXP = 16,
    UNIFORM = 17,
    NORMAL = 18,
} unary_operation_t;

#define NUM_UNARYOP (NORMAL+1)

typedef enum {
    ADD = 0,
    SUB = 1,
    MUL = 2,
} binary_operation_t;

#define NUM_BINOP (MUL+1)

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

#define TYPE_ZERO   CAT2(TYPE,_zero)

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

typedef struct {
    matrix_type_t type;
    unsigned int n;
    unsigned int m;
    int      nstep;         // #elements to next element in n direction
    int      mstep;         // #elements to next element in m direction
    unsigned int offset;    // offset to first element (in number of elements)
    bool_t       rowmajor;  // stored row-by-row
    ErlNifRWLock* rw_lock;  // make sure we can read/write "safe"
    size_t size;            // allocated memory size
    byte_t* base;           // allocated memory
    byte_t* data;           // aligned data
    byte_t* first;          // pointer to first element within data
    uintptr_t ptr;          // resource pointer (may be self)
    scalar_t sdata;         // raw scalar data
} matrix_t;

// Global data (store in env?)
static ErlNifResourceType* matrix_r;
static ErlNifTSDKey matrix_k;

#if 0
static inline size_t matrix_num_rows(matrix_t* ap)
{
    return (ap->rowmajor) ? ap->n : ap->m;
}

static inline size_t matrix_num_columns(matrix_t* ap)
{
    return (ap->rowmajor) ? ap->m : ap->n;
}
#endif

// read lock matrix if lock is defined
static inline void matrix_r_lock(matrix_t* ap)
{
    if (ap->rw_lock != NULL)
	enif_rwlock_rlock(ap->rw_lock);
}

// read unlock matrix if lock is defined
static inline void matrix_r_unlock(matrix_t* ap)
{
    if (ap->rw_lock != NULL)
	enif_rwlock_runlock(ap->rw_lock);
}

// read lock matrix a if a is not the same matrix as c.
// ( since c is going to be locked for write )
static inline void matrix_cr_lock(matrix_t* cp, matrix_t* ap)
{
    if ((cp->rw_lock != ap->rw_lock) && (ap->rw_lock != NULL))
	enif_rwlock_rlock(ap->rw_lock);
}

static inline void matrix_cr_unlock(matrix_t* cp, matrix_t* ap)
{
    if ((cp->rw_lock != ap->rw_lock) && (ap->rw_lock != NULL))
	enif_rwlock_runlock(ap->rw_lock);
}

static inline void matrix_w_lock(matrix_t* cp)
{
    enif_rwlock_rwlock(cp->rw_lock);
}

static inline void matrix_w_unlock(matrix_t* cp)
{
    enif_rwlock_rwunlock(cp->rw_lock);
}

// read, read lock (binary operation)
static inline void matrix_rr_lock(matrix_t* ap, matrix_t* bp)
{
    matrix_r_lock(ap);
    matrix_r_lock(bp);
}

static inline void matrix_rrr_lock(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    matrix_r_lock(ap);
    matrix_r_lock(bp);
    matrix_r_lock(cp);
}

// read, read unlock (binary operation)
static inline void matrix_rr_unlock(matrix_t* ap, matrix_t* bp)
{
    matrix_r_unlock(bp);
    matrix_r_unlock(ap);
}

static inline void matrix_rrr_unlock(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    matrix_r_unlock(cp);
    matrix_r_unlock(bp);
    matrix_r_unlock(ap);
}

// read and write lock (unary operation)
static inline void matrix_rw_lock(matrix_t* ap, matrix_t* cp)
{
    matrix_cr_lock(cp, ap);
    matrix_w_lock(cp);
}

static inline void matrix_rw_unlock(matrix_t* ap, matrix_t* cp)
{
    matrix_w_unlock(cp);
    matrix_cr_unlock(cp, ap);
}

// read, read and write lock (binary operatin)
static inline void matrix_rrw_lock(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    matrix_cr_lock(cp, ap);
    matrix_cr_lock(cp, bp);
    matrix_w_lock(cp);
}

static inline void matrix_rrrw_lock(matrix_t* ap, matrix_t* bp, matrix_t* kp,
				    matrix_t* cp)
{
    matrix_cr_lock(cp, ap);
    matrix_cr_lock(cp, bp);
    matrix_cr_lock(cp, kp);
    matrix_w_lock(cp);
}

// read, read and write unlock (binary operatin)
static inline void matrix_rrw_unlock(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    matrix_w_unlock(cp);
    matrix_cr_unlock(cp, bp);
    matrix_cr_unlock(cp, ap);
}

static inline void matrix_rrrw_unlock(matrix_t* ap, matrix_t* bp, matrix_t* kp,
				      matrix_t* cp)
{
    matrix_w_unlock(cp);
    matrix_cr_unlock(cp, kp);
    matrix_cr_unlock(cp, bp);
    matrix_cr_unlock(cp, ap);
}


static int matrix_load(ErlNifEnv* env, void** priv_data,
		       ERL_NIF_TERM load_info);
static int matrix_upgrade(ErlNifEnv* env, void** priv_data,
			  void** old_priv_data,
		       ERL_NIF_TERM load_info);
static void matrix_unload(ErlNifEnv* env, void* priv_data);

static ERL_NIF_TERM matrix_create(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_element(ErlNifEnv* env, int argc,
				   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_setelement(ErlNifEnv* env, int argc,
				      const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc,
			       const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc,
				    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_times(ErlNifEnv* env, int argc,
				 const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_ktimes(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_multiply(ErlNifEnv* env, int argc,
				    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_kmultiply(ErlNifEnv* env, int argc,
				     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_topk(ErlNifEnv* env, int argc,
				const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_negate(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_mulsum(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sum(ErlNifEnv* env, int argc,
			       const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_l2pool(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_maxpool(ErlNifEnv* env, int argc,
				   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_filter(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_transpose_data(ErlNifEnv* env, int argc,
					  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sigmoid(ErlNifEnv* env, int argc,
				   const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc,
					 const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_relu(ErlNifEnv* env, int argc,
				     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_copy(ErlNifEnv* env, int argc,
				const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_fill(ErlNifEnv* env, int argc,
				const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_identity(ErlNifEnv* env, int argc,
				    const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_argmax(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_max(ErlNifEnv* env, int argc,
			       const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_min(ErlNifEnv* env, int argc,
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
    NIF_FUNC("element_",      2, matrix_element),
    NIF_FUNC("element_",      3, matrix_element),
    NIF_FUNC("setelement_",   4, matrix_setelement),
    NIF_FUNC("add_",          2, matrix_add),
    NIF_FUNC("add_",          3, matrix_add),
    NIF_FUNC("subtract_",     2, matrix_subtract),
    NIF_FUNC("subtract_",     3, matrix_subtract),
    NIF_FUNC("times_",        2, matrix_times),
    NIF_FUNC("times_",        3, matrix_times),
    NIF_FUNC("ktimes_",       3, matrix_ktimes),
    NIF_FUNC("ktimes_",       4, matrix_ktimes),
    NIF_FUNC("multiply_",     2, matrix_multiply),
    NIF_FUNC("multiply_",     3, matrix_multiply),
    NIF_FUNC("kmultiply_",    3, matrix_kmultiply),
    NIF_FUNC("kmultiply_",    4, matrix_kmultiply),
    NIF_FUNC("topk_",         2, matrix_topk),
    NIF_FUNC("negate_",       1, matrix_negate),
    NIF_FUNC("negate_",       2, matrix_negate),
    NIF_FUNC("mulsum_",       2, matrix_mulsum),
    NIF_FUNC("l2pool_",       5, matrix_l2pool),
    NIF_FUNC("l2pool_",       6, matrix_l2pool),
    NIF_FUNC("maxpool_",      5, matrix_maxpool),
    NIF_FUNC("maxpool_",      6, matrix_maxpool),
    NIF_FUNC("filter_",       4, matrix_filter),
    NIF_FUNC("filter_",       5, matrix_filter),
    NIF_FUNC("transpose_data",1, matrix_transpose_data),
    NIF_FUNC("transpose_data",2, matrix_transpose_data),
    NIF_FUNC("sigmoid_",      1, matrix_sigmoid),
    NIF_FUNC("sigmoid_prime_",2, matrix_sigmoid_prime),
    NIF_FUNC("relu",          1, matrix_relu),
    NIF_FUNC("copy",          1, matrix_copy),
    NIF_FUNC("copy",          2, matrix_copy),
    NIF_FUNC("copy",          4, matrix_copy),
    NIF_FUNC("fill",          2, matrix_fill),
    NIF_FUNC("apply1_",       2, matrix_apply1),
    NIF_FUNC("apply1_",       3, matrix_apply1),
    NIF_FUNC("identity_",     3, matrix_identity),
    NIF_FUNC("argmax_",       2, matrix_argmax),
    NIF_FUNC("max_",          1, matrix_max),
    NIF_FUNC("max_",          2, matrix_max),
    NIF_FUNC("min_",          1, matrix_min),
    NIF_FUNC("min_",          2, matrix_min),
    NIF_FUNC("sum_",          1, matrix_sum),
    NIF_FUNC("sum_",          2, matrix_sum),
};

size_t element_size_exp_[NUM_TYPES] = {
    [INT8]  = 0,
    [INT16] = 1,
    [INT32] = 2,
    [INT64] = 3,
    [FLOAT32] = 2,
    [FLOAT64] = 3,
    [COMPLEX64] = 3,
    [COMPLEX128] = 4
};

size_t element_size_[NUM_TYPES] = {
    [INT8] = 1,
    [INT16] = 2,
    [INT32] = 4,
    [INT64] = 8,
    [FLOAT32] = 4,
    [FLOAT64] = 8,
    [COMPLEX64] = 8,
    [COMPLEX128] = 16
};

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
DECL_ATOM(sigmoid_prime1);
DECL_ATOM(relu);
DECL_ATOM(relu_prime);
DECL_ATOM(leaky_relu);
DECL_ATOM(leaky_relu_prime);
DECL_ATOM(tanh);
DECL_ATOM(tanh_prime);
DECL_ATOM(tanh_prime1);
DECL_ATOM(softplus);
DECL_ATOM(softplus_prime);
DECL_ATOM(negate);
DECL_ATOM(uniform);
DECL_ATOM(normal);
DECL_ATOM(zero);
DECL_ATOM(one);
DECL_ATOM(copy);
DECL_ATOM(true);
DECL_ATOM(false);
DECL_ATOM(undefined);
DECL_ATOM(exp);

static size_t element_size(matrix_type_t type)
{
    return element_size_[type];
}

static size_t size_of_array(matrix_type_t type, size_t nelems)
{
    return nelems << element_size_exp_[type];
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

////////////////////////////////////////////////////////////////////////////
//
//  rw_op.i  is generated from ../priv/rw_op.term (with texgen.erl)
//  it generates all functions needed to load value into a general class
//
////////////////////////////////////////////////////////////////////////////

#include "rw_op.i"

#if 0
static int64_t read_int64(matrix_type_t type, byte_t* ptr)
{
    return (read_int64_func[type])(ptr);
}

static float64_t read_float64(matrix_type_t type, byte_t* ptr)
{
    return (read_float64_func[type])(ptr);
}


static complex128_t read_complex128(matrix_type_t type, byte_t* ptr)
{
    return (read_complex128_func[type])(ptr);
}

#endif

static void write_int64(matrix_type_t type, byte_t* ptr, int64_t v)
{
    (write_int64_func[type])(ptr, v);
}


static void write_float64(matrix_type_t type, byte_t* ptr, float64_t v)
{
    (write_float64_func[type])(ptr, v);
}


static void write_complex128(matrix_type_t type, byte_t* ptr, complex128_t v)
{
    (write_complex128_func[type])(ptr, v);
}



// convert scalar to erlang term
static ERL_NIF_TERM read_term(ErlNifEnv* env, matrix_type_t type, byte_t* ptr)
{
    switch(type) {
    case INT8:    return enif_make_int(env, (int) *((int8_t*)ptr));
    case INT16:   return enif_make_int(env, (int) *((int16_t*)ptr));
    case INT32:   return enif_make_int(env, (int) *((int32_t*)ptr));
    case INT64:   return enif_make_int64(env, (ErlNifSInt64) *((int64_t*)ptr));
    case FLOAT32: return enif_make_double(env, *((float32_t*)ptr));
    case FLOAT64: return enif_make_double(env, *((float64_t*)ptr));
    case COMPLEX64:
	return enif_make_tuple2(env,
				enif_make_double(env, ((float32_t*)ptr)[0]),
				enif_make_double(env, ((float32_t*)ptr)[1]));
    case COMPLEX128:
	return enif_make_tuple2(env,
				enif_make_double(env, ((float64_t*)ptr)[0]),
				enif_make_double(env, ((float64_t*)ptr)[1]));
    default:
	return enif_make_badarg(env);
    }
}

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
static inline vcomplex64_t complex64_multiply(vcomplex64_t x, vcomplex64_t y)
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
static inline vcomplex64_t complex64_multiply(vcomplex64_t x, vcomplex64_t y)
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
static inline vcomplex128_t complex128_multiply(vcomplex128_t x,vcomplex128_t y)
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
static inline vcomplex128_t complex128_multiply(vcomplex128_t x,vcomplex128_t y)
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

static inline vcomplex64_t complex64_negate(vcomplex64_t x)
{
    return -x;
}

static inline vcomplex128_t complex128_add(vcomplex128_t x, vcomplex128_t y)
{
    return x+y;
}

static inline vcomplex128_t complex128_subtract(vcomplex128_t x,
						vcomplex128_t y)
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

// use cabs to compare comlex
complex128_t complex128_max(complex128_t a, complex128_t b)
{
    return (cabs(a) > cabs(b)) ? a : b;
}

// use cabs to compare comlex
complex128_t complex128_min(complex128_t a, complex128_t b)
{
    return (cabs(a) < cabs(b)) ? a : b;
}

// use cabs to compare comlex
complex128_t complex128_lt(complex128_t a, complex128_t b)
{
    return (cabs(a) < cabs(b));
}

complex128_t complex128_gt(complex128_t a, complex128_t b)
{
    return (cabs(a) > cabs(b));
}


#define cop_sigmoid(z)    (1.0/(1.0 + cexp(-(z))))

static inline complex128_t cop_sigmoid_prime(complex128_t z)
{
    complex128_t z1 = cop_sigmoid(z);
    return z1*(1-z1);
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
// nullary
#define op_zero() (0)
#define op_one()  (1)

// unary
#define op_neg(x)     (-(x))
#define op_bnot(x)    (~(x))

// binary
#define op_add(x,y)   ((x)+(y))
#define op_sub(x,y)   ((x)-(y))
#define op_mul(x,y)   ((x)*(y))
#define op_div(x,y)   ((x)/(y))
#define op_rem(x,y)   ((x)%(y))
#define op_bxor(x,y)  ((x)^(y))
#define op_bor(x,y)   ((x)|(y))
#define op_band(x,y)  ((x)&(y))
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
//  operations
///////////////////////////////////////////////////////////////////////////////

#include "matrix_add.i"
#include "matrix_sub.i"
#include "matrix_times.i"
#include "matrix_neg.i"
#include "matrix_dot.i"
#include "matrix_multiply.i"
#include "matrix_multiply_t.i"
#include "matrix_kmultiply.i"
#include "matrix_kmultiply_t.i"
#include "matrix_sigmoid.i"
#include "matrix_sigmoid_prime.i"
#include "matrix_sigmoid_prime1.i"
#include "matrix_rectifier.i"

// float unary functions
static float64_t sigmoid_float64(float64_t a)
{
    return op_sigmoid(a);
}

static float64_t sigmoid_prime_float64(float64_t a)
{
    return op_sigmoid_prime(a);
}

static float64_t sigmoid_prime1_float64(float64_t a)
{
    return a*(1.0-a);
}

static float64_t softplus_float64(float64_t a)
{
    return log(1.0+exp(a));
}

static float64_t softplus_prime_float64(float64_t a)
{
    return op_sigmoid(a);
}

static float64_t relu_float64(float64_t a)
{
    return op_max(0.0,a);
}

static float64_t relu_prime_float64(float64_t a)
{
    return (a>0.0)? 1.0 : 0.0;
}

static float64_t leaky_relu_float64(float64_t a)
{
    return (a>0.0)?a:0.1*a;
}

static float64_t leaky_relu_prime_float64(float64_t a)
{
    return (a>0.0)?1.0:0.1;
}

static float64_t exp_float64(float64_t a)
{
    return exp(a);
}

static float64_t tanh_float64(float64_t a)
{
    return tanh(a);
}

static float64_t tanh_prime_float64(float64_t a)
{
    double th = tanh(a);
    return 1.0-th*th;
}

static float64_t tanh_prime1_float64(float64_t a)
{
    return 1.0-a*a;
}

static float64_t negate_float64(float64_t a)
{
    return -a;
}

static float64_t copy_float64(float64_t a)
{
    return a;
}

static float64_t uniform_float64(float64_t a)
{
    UNUSED(a);
    return uniform_64(MATRIX_RAND_ALG);
}

static float64_t normal_float64(float64_t a)
{
    UNUSED(a);
    return normal_64(MATRIX_RAND_ALG,0.0,1.0);
}

static float64_t zero_float64(float64_t a)
{
    UNUSED(a);
    return 0.0;
}

static float64_t one_float64(float64_t a)
{
    UNUSED(a);
    return 1.0;
}

static float64_t (*unaryop_float64[NUM_UNARYOP])(float64_t) = {
    [ZERO] = zero_float64,
    [ONE]  = one_float64,
    [COPY] = copy_float64,
    [NEGATE] = negate_float64,
    [SIGMOID] = sigmoid_float64,
    [SIGMOID_PRIME] = sigmoid_prime_float64,
    [SIGMOID_PRIME1] = sigmoid_prime1_float64,
    [SOFTPLUS] = softplus_float64,
    [SOFTPLUS_PRIME] = softplus_prime_float64,
    [RELU] = relu_float64,
    [RELU_PRIME] = relu_prime_float64,
    [LEAKY_RELU] = leaky_relu_float64,
    [LEAKY_RELU_PRIME] = leaky_relu_prime_float64,
    [TANH] = tanh_float64,
    [TANH_PRIME] = tanh_prime_float64,
    [TANH_PRIME1] = tanh_prime1_float64,
    [EXP] = exp_float64,
    [UNIFORM] = uniform_float64,
    [NORMAL] = normal_float64,
};

// Complex unary functions

static complex128_t sigmoid_complex128(complex128_t a)
{
    return cop_sigmoid(a);
}

static complex128_t sigmoid_prime_complex128(complex128_t a)
{
    return cop_sigmoid_prime(a);
}

static complex128_t sigmoid_prime1_complex128(complex128_t a)
{
    return a*(1.0-a);
}

static complex128_t softplus_complex128(complex128_t a)
{
    return clog(1.0+cexp(a));
}

static complex128_t softplus_prime_complex128(complex128_t a)
{
    return cop_sigmoid(a);
}

static complex128_t relu_complex128(complex128_t a)
{
    float64_t ar = creal(a);
    float64_t ai = cimag(a);
    return CMPLX(ar>0.0?ar:0.0,ai>0.0?ai:0.0);
}

static complex128_t relu_prime_complex128(complex128_t a)
{
    float64_t ar = creal(a);
    float64_t ai = cimag(a);
    return CMPLX(ar>0.0?1.0:0.0,ai>0?1.0:0.0);
}

static complex128_t leaky_relu_complex128(complex128_t a)
{
    float64_t ar = creal(a);
    float64_t ai = cimag(a);
    return CMPLX(ar>0.0?ar:0.1*ar,ai>0.0?ai:0.1*ai);
}

static complex128_t leaky_relu_prime_complex128(complex128_t a)
{
    float64_t ar = creal(a);
    float64_t ai = cimag(a);
    return CMPLX(ar>0.0?1.0:0.1,ai>0.0?1.0:0.1);
}

static complex128_t exp_complex128(complex128_t a)
{
    return cexp(a);
}

static complex128_t tanh_complex128(complex128_t a)
{
    return ctanh(a);
}

static complex128_t tanh_prime_complex128(complex128_t a)
{
    complex128_t th = ctanh(a);
    return 1.0-th*th;
}

static complex128_t tanh_prime1_complex128(complex128_t a)
{
    return 1.0-a*a;
}

static complex128_t negate_complex128(complex128_t a)
{
    return -a;
}

static complex128_t copy_complex128(complex128_t a)
{
    return a;
}

static complex128_t uniform_complex128(complex128_t a)
{
    UNUSED(a);
    return uniform_c128(MATRIX_RAND_ALG);
}

static complex128_t normal_complex128(complex128_t a)
{
    UNUSED(a);
    return normal_c128(MATRIX_RAND_ALG,0.0,1.0);
}

static complex128_t zero_complex128(complex128_t a)
{
    UNUSED(a);
    return CMPLX(0.0,0.0);
}

static complex128_t one_complex128(complex128_t a)
{
    UNUSED(a);
    return CMPLX(1.0,0.0);
}

static complex128_t (*unaryop_complex128[NUM_UNARYOP])(complex128_t) = {
    [ZERO] = zero_complex128,
    [ONE]  = one_complex128,
    [COPY] = copy_complex128,
    [NEGATE] = negate_complex128,
    [SIGMOID] = sigmoid_complex128,
    [SIGMOID_PRIME] = sigmoid_prime_complex128,
    [SIGMOID_PRIME1] = sigmoid_prime1_complex128,
    [SOFTPLUS] = softplus_complex128,
    [SOFTPLUS_PRIME] = softplus_prime_complex128,
    [RELU] = relu_complex128,
    [RELU_PRIME] = relu_prime_complex128,
    [LEAKY_RELU] = leaky_relu_complex128,
    [LEAKY_RELU_PRIME] = leaky_relu_prime_complex128,
    [TANH] = tanh_complex128,
    [TANH_PRIME] = tanh_prime_complex128,
    [TANH_PRIME1] = tanh_prime1_complex128,
    [EXP] = exp_complex128,
    [UNIFORM] = uniform_complex128,
    [NORMAL] = normal_complex128,
};

// integer unary functions (FIXME!!!)
static int64_t sigmoid_int64(int64_t a)
{
    return op_sigmoid(a);
}

static int64_t sigmoid_prime_int64(int64_t a)
{
    return op_sigmoid_prime(a);
}

static int64_t sigmoid_prime1_int64(int64_t a)
{
    return a*(1-a);
}

static int64_t softplus_int64(int64_t a)
{
    return log(1.0+exp(a));
}

static int64_t softplus_prime_int64(int64_t a)
{
    return op_sigmoid(a);
}

static int64_t relu_int64(int64_t a)
{
    return op_max(0,a);
}

static int64_t relu_prime_int64(int64_t a)
{
    return (a>0)? 1 : 0;
}

static int64_t leaky_relu_int64(int64_t a)
{
    return (a>0)? a : 0.1*a;
}

static int64_t leaky_relu_prime_int64(int64_t a)
{
    return (a>0)?1:0;
}

static int64_t exp_int64(int64_t a)
{
    return exp(a);
}

static int64_t tanh_int64(int64_t a)
{
    return tanh(a);
}

static int64_t tanh_prime_int64(int64_t a)
{
    double th = tanh(a);
    return 1.0-th*th;
}

static int64_t tanh_prime1_int64(int64_t a)
{
    return 1.0-a*a;
}

static int64_t negate_int64(int64_t a)
{
    return -a;
}

static int64_t copy_int64(int64_t a)
{
    return a;
}

static int64_t uniform_int64(int64_t a)
{
    UNUSED(a);
    return rand_64(MATRIX_RAND_ALG);
}

static int64_t normal_int64(int64_t a)
{
    UNUSED(a);
    return 0;
}

static int64_t zero_int64(int64_t a)
{
    UNUSED(a);
    return 0;
}

static int64_t one_int64(int64_t a)
{
    UNUSED(a);
    return 1;
}

static int64_t (*unaryop_int64[NUM_UNARYOP])(int64_t) = {
    [ZERO] = zero_int64,
    [ONE]  = one_int64,
    [COPY] = copy_int64,
    [NEGATE] = negate_int64,
    [SIGMOID] = sigmoid_int64,
    [SIGMOID_PRIME] = sigmoid_prime_int64,
    [SIGMOID_PRIME1] = sigmoid_prime1_int64,
    [SOFTPLUS] = softplus_int64,
    [SOFTPLUS_PRIME] = softplus_prime_int64,
    [RELU] = relu_int64,
    [RELU_PRIME] = relu_prime_int64,
    [LEAKY_RELU] = leaky_relu_int64,
    [LEAKY_RELU_PRIME] = leaky_relu_prime_int64,
    [TANH] = tanh_int64,
    [TANH_PRIME] = tanh_prime_int64,
    [TANH_PRIME1] = tanh_prime1_int64,
    [EXP] = exp_int64,
    [UNIFORM] = uniform_int64,
    [NORMAL] = normal_int64,
};

static void apply1(unary_operation_t func,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    size_t i, j;

    au = size_of_array(at,au);
    av = size_of_array(at,av);
    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);

    if (is_float(ct)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	float64_t (*opf)(float64_t) = unaryop_float64[func];
	for (i=0; i<n; i++) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		float64_t a = read_af(ap1);
		float64_t c = opf(a);
		ap1 += av;
		write_cf(cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
    else if (is_complex(ct)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	complex128_t (*opf)(complex128_t) = unaryop_complex128[func];
	for (i=0; i<n; i++) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		complex128_t a = read_af(ap1);
		complex128_t c = opf(a);
		ap1 += av;
		write_cf(cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t (*opf)(int64_t) = unaryop_int64[func];
	for (i=0; i<n; i++) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		int64_t a = read_af(ap1);
		int64_t c = opf(a);
		ap1 += av;
		write_cf(cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
}

// int64 operation add/sub/mul
static int64_t add_int64(int64_t a, int64_t b)
{
    return op_add(a,b);
}

static int64_t sub_int64(int64_t a, int64_t b)
{
    return op_sub(a,b);
}

static int64_t mul_int64(int64_t a, int64_t b)
{
    return op_mul(a,b);
}

static int64_t (*binop_int64[NUM_BINOP])(int64_t, int64_t) = {
    [ADD] = add_int64,
    [SUB] = sub_int64,
    [MUL] = mul_int64,
};

// float64 operation add/sub/mul
static float64_t add_float64(float64_t a, float64_t b)
{
    return op_add(a,b);
}

static float64_t sub_float64(float64_t a, float64_t b)
{
    return op_sub(a,b);
}

static float64_t mul_float64(float64_t a, float64_t b)
{
    return op_mul(a,b);
}

static float64_t (*binop_float64[NUM_BINOP])(float64_t, float64_t) = {
    [ADD] = add_float64,
    [SUB] = sub_float64,
    [MUL] = mul_float64,
};

// complex128 operation add/sub/mul
static complex128_t add_complex128(complex128_t a, complex128_t b)
{
    return op_add(a,b);
}

static complex128_t sub_complex128(complex128_t a, complex128_t b)
{
    return op_sub(a,b);
}

static complex128_t mul_complex128(complex128_t a, complex128_t b)
{
    return op_mul(a,b);
}

static complex128_t (*binop_complex128[NUM_BINOP])(complex128_t,complex128_t)={
    [ADD] = add_complex128,
    [SUB] = sub_complex128,
    [MUL] = mul_complex128,
};

// a more general function for binary operations but a lot slower
static void apply2(binary_operation_t func,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    matrix_type_t t = combine_type(at, bt);
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    bu = size_of_array(bt,bu);
    bv = size_of_array(bt,bv);
    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);
    if (is_float(t)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	float64_t (*opf)(float64_t, float64_t) = binop_float64[func];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_af(ap1);
		float64_t b = read_bf(bp1);
		float64_t c = opf(a,b);
		ap1 += av;
		bp1 += bv;
		write_cf(cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    bp += bu;
	    cp += cu;
	}
    }
    else if (is_complex(t)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	complex128_t (*read_bf)(byte_t*) = read_complex128_func[bt];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	complex128_t (*opf)(complex128_t,complex128_t)=binop_complex128[func];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		complex128_t a = read_af(ap1);
		complex128_t b = read_bf(bp1);
		complex128_t c = opf(a, b);
		ap1 += av;
		bp1 += bv;
		write_cf(cp1, c);
		cp1 += cv;
	    }
	    ap += au;
	    bp += bu;
	    cp += cu;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t (*opf)(int64_t, int64_t) = binop_int64[func];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_af(ap1);
		int64_t b = read_bf(bp1);
		int64_t c = opf(a, b);
		ap1 += av;
		bp1 += bv;
		write_cf(cp1, c);
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
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_add(at, ap, au, bp, bu, cp, cu, n, m);
	}
	else
#endif
	{
	    mt_add(at, ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	apply2(ADD, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
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
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_subtract(at, ap, au, bp, bu, cp, cu, n, m);
	}
	else
#endif
	{
	    mt_subtract(at, ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	apply2(SUB, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
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
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_times(at, ap, au, bp, bu, cp, cu, n, m);
	}
	else
#endif
	{
	    mt_times(at, ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	apply2(MUL, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
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
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp)) {
	    mtv_negate(at, ap, au, cp, cu, n, m);
	}
	else
#endif
	{
	    mt_negate(at, ap, au, av, cp, cu, cv, n, m);
	}
    }
    else {
	apply1(NEGATE, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// argmax
///////////////////////////////////////////////////////////////////////////////

static void argmax(matrix_type_t at,byte_t* ap, int au, int av,
		   int32_t* cp, int cv,
		   size_t n, size_t m)
{
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    // cv is step in size of int32 (index type)

    if (is_float(at)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t max_i = 1;
	    float64_t max_v = read_af(ap1);

	    ap1 += au;
	    while(n1--) {
		float64_t v = read_af(ap1);
		ap1 += au;
		i++;
		if (v > max_v) { max_v = v; max_i = i; }
	    }
	    *cp = max_i;
	    cp += cv;
	    ap += av;
	}
    }
    else if (is_complex(at)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t max_i = 1;
	    complex128_t max_v = read_af(ap1);

	    ap1 += au;
	    while(n1--) {
		complex128_t v = read_af(ap1);
		ap1 += au;
		i++;
		if (cabs(v) > cabs(max_v)) { max_v = v; max_i = i; }
	    }
	    *cp = max_i;
	    cp += cv;
	    ap += av;

	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t max_i = 1;
	    int64_t max_v = read_af(ap1);

	    ap1 += au;
	    while(n1--) {
		int64_t v = read_af(ap1);
		ap1 += au;
		i++;
		if (v > max_v) { max_v = v; max_i = i; }
	    }
	    *cp = max_i;
	    cp += cv;
	    ap += av;
	}
    }
}


///////////////////////////////////////////////////////////////////////////////
// max along axis or if cv=0 then total max
///////////////////////////////////////////////////////////////////////////////

static void t_max(matrix_type_t at,byte_t* ap, int au, int av,
		  matrix_type_t ct,byte_t* cp, int cv,
		  size_t n, size_t m)
{
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    cv = size_of_array(ct,cv);

    if (is_float(ct)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	float64_t max_v = read_af(ap);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		max_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		float64_t v = read_af(ap1);
		ap1 += au;
		max_v = op_max(v, max_v);
	    }
	    write_cf(cp, max_v);
	    cp += cv;
	    ap += av;
	}
    }
    else if (is_complex(ct)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	complex128_t max_v = read_af(ap);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		max_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		complex128_t v = read_af(ap1);
		ap1 += au;
		max_v = complex128_max(v, max_v);
	    }
	    write_cf(cp, max_v);
	    cp += cv;
	    ap += av;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t max_v = read_af(ap);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		max_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		int64_t v = read_af(ap1);
		ap1 += au;
		max_v = op_max(v, max_v);
	    }
	    write_cf(cp, max_v);
	    cp += cv;
	    ap += av;
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// min
///////////////////////////////////////////////////////////////////////////////

static void t_min(matrix_type_t at,byte_t* ap, int au, int av,
		  matrix_type_t ct,byte_t* cp, int cv,
		  size_t n, size_t m)
{
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    cv = size_of_array(ct,cv);

    if (is_float(ct)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	float64_t min_v = read_af(ap);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		min_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		float64_t v = read_af(ap1);
		ap1 += au;
		min_v = op_min(v, min_v);
	    }
	    write_cf(cp, min_v);
	    cp += cv;
	    ap += av;
	}
    }
    else if (is_complex(ct)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	complex128_t min_v = read_af(ap);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		min_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		complex128_t v = read_af(ap1);
		ap1 += au;
		min_v = complex128_max(v, min_v);
	    }
	    write_cf(cp, min_v);
	    cp += cv;
	    ap += av;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t min_v = read_af(ap);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		min_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		int64_t v = read_af(ap1);
		ap1 += au;
		min_v = op_min(v, min_v);
	    }
	    write_cf(cp, min_v);
	    cp += cv;
	    ap += av;
	}
    }
}


///////////////////////////////////////////////////////////////////////////////
// sum
///////////////////////////////////////////////////////////////////////////////

static void t_sum(matrix_type_t at,byte_t* ap, int au, int av,
		  matrix_type_t ct,byte_t* cp, int cv,
		  size_t n, size_t m)
{
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    cv = size_of_array(ct,cv);

    if (is_float(ct)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	float64_t sum_v = 0.0;
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		sum_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		float64_t v = read_af(ap1);
		ap1 += au;
		sum_v = op_add(v, sum_v);
	    }
	    write_cf(cp, sum_v);
	    cp += cv;
	    ap += av;
	}
    }
    else if (is_complex(ct)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	complex128_t sum_v = CMPLX(0.0,0.0);
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		sum_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		complex128_t v = read_af(ap1);
		ap1 += au;
		sum_v = op_add(v, sum_v);
	    }
	    write_cf(cp, sum_v);
	    cp += cv;
	    ap += av;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t sum_v = 0;
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n;
	    if (cv) {
		sum_v = read_af(ap1);
		ap1 += au;
		n1--;
	    }
	    while(n1--) {
		int64_t v = read_af(ap1);
		ap1 += au;
		sum_v = op_add(v, sum_v);
	    }
	    write_cf(cp, sum_v);
	    cp += cv;
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

#if 0
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
#endif

static void sigmoid_prime1(matrix_type_t at, byte_t* ap, int au, int av,
			   matrix_type_t ct, byte_t* cp, int cu, int cv,
			   size_t n, size_t m)
{
    if (at == ct) {
	mt_sigmoid_prime1(at, ap, au, av, cp, cu, cv, n, m);
    }
    else {
	apply1(SIGMOID_PRIME1, at, ap, au, av, ct, cp, cu, av, n, m);
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
	apply1(RELU, at, ap, au, av, ct, cp, cu, cv, n, m);
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
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_multiply(at,ap,au,an,am,bp,bu,bn,bm,cp,cu,cv);
	}
	else
#endif
	{
	    mt_multiply(at,ap,au,av,an,am,bp,bu,bv,bn,bm,cp,cu,cv);
	}
    }
    else {
	byte_t* bp0 = bp;
	au = size_of_array(at,au);
	av = size_of_array(at,av);
	bu = size_of_array(bt,bu);
	bv = size_of_array(bt,bv);
	cu = size_of_array(ct,cu);
	cv = size_of_array(ct,cv);

	if (is_float(ct)) {
	    float64_t (*read_af)(byte_t*) = read_float64_func[at];
	    float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	    void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	    while(an--) {
		byte_t* cp1 = cp;
		size_t m = bm;
		bp = bp0;
		while(m--) {
		    float64_t sum = 0.0;
		    byte_t* bp1 = bp;
		    byte_t* ap1 = ap;
		    size_t  k = am;
		    while(k--) {
			float64_t a = read_af(ap1);
			float64_t b = read_bf(bp1);
			float64_t c = op_mul(a,b);
			sum = op_add(sum, c);
			ap1 += av;
			bp1 += bu;
		    }
		    write_cf(cp1, sum);
		    cp1 += cv;
		    bp += bv;
		}
		
		ap += au;
		cp += cu;
	    }
	}
	else if (is_complex(ct)) {
	    complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	    complex128_t (*read_bf)(byte_t*) = read_complex128_func[bt];
	    void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	    while(an--) {
		byte_t* cp1 = cp;
		size_t m = bm;
		bp = bp0;
		while(m--) {
		    complex128_t sum =  CMPLX(0.0,0.0);
		    byte_t* bp1 = bp;
		    byte_t* ap1 = ap;
		    size_t  k = am;
		    while(k--) {
			complex128_t a = read_af(ap1);
			complex128_t b = read_bf(bp1);
			complex128_t c = op_mul(a,b);
			sum = op_add(sum, c);
			ap1 += av;
			bp1 += bu;
		    }
		    write_cf(cp1, sum);
		    cp1 += cv;
		    bp += bv;
		}
		ap += au;
		cp += cu;
	    }
	}
	else {
	    int64_t (*read_af)(byte_t*) = read_int64_func[at];
	    int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	    void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	    while(an--) {
		byte_t* cp1 = cp;
		size_t m = bm;
		bp = bp0;
		while(m--) {
		    int64_t sum = 0;
		    byte_t* bp1 = bp;
		    byte_t* ap1 = ap;
		    size_t  k = am;
		    while(k--) {
			int64_t a = read_af(ap1);
			int64_t b = read_bf(bp1);
			int64_t c = op_mul(a,b);
			sum = op_add(sum, c);
			ap1 += av;
			bp1 += bu;
		    }
		    write_cf(cp1, sum);
		    cp1 += cv;
		    bp += bv;
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
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_multiply_transposed(at,ap,au,an,am,bp,bu,bn,bm,
				    cp,cu,cv);
	else
#endif
	    mt_multiply_transposed(at,ap,au,av,an,am,bp,bu,bv,bn,bm,
				   cp,cu,cv);
    }
    else {
	byte_t* bp0 = bp;
	au = size_of_array(at,au);
	av = size_of_array(at,av);
	bu = size_of_array(bt,bu);
	bv = size_of_array(bt,bv);
	cu = size_of_array(ct,cu);
	cv = size_of_array(ct,cv);

	if (is_float(ct)) {
	    float64_t (*read_af)(byte_t*) = read_float64_func[at];
	    float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	    void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	    while(an--) {
		byte_t* cp1 = cp;
		size_t n = bn;
		bp = bp0;
		while(n--) {
		    float64_t sum = 0.0;
		    byte_t* ap1 = ap;
		    byte_t* bp1 = bp;
		    size_t k = bm;
		    while(k--) {
			float64_t a = read_af(ap1);
			float64_t b = read_bf(bp1);
			float64_t c = op_mul(a,b);
			sum = op_add(sum, c);
			ap1 += av;
			bp1 += bv;
		    }
		    write_cf(cp1, sum);
		    cp1 += cv;
		    bp  += bu;
		}
		ap += au;
		cp += cu;
	    }
	}
	else if (is_complex(ct)) {
	    complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	    complex128_t (*read_bf)(byte_t*) = read_complex128_func[bt];
	    void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
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
			complex128_t a = read_af(ap1);
			complex128_t b = read_bf(bp1);
			complex128_t c = op_mul(a,b);
			sum = op_add(sum, c);
			ap1 += av;
			bp1 += bv;
		    }
		    write_cf(cp1, sum);
		    cp1 += cv;
		    bp += bu;
		}
		ap += au;
		cp += cu;
	    }
	}
	else {
	    int64_t (*read_af)(byte_t*) = read_int64_func[at];
	    int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	    void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
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
			int64_t a = read_af(ap1);
			int64_t b = read_bf(bp1);
			int64_t c = op_mul(a,b);
			sum = op_add(sum, c);
			ap1 += av;
			bp1 += bv;
		    }
		    write_cf(cp1, sum);
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
// kmultiply
///////////////////////////////////////////////////////////////////////////////

static void kmultiply(
    bool_t use_vector,
    matrix_type_t at,byte_t* ap,int au,int av,size_t an,size_t am,
    matrix_type_t bt,byte_t* bp,int bu,int bv,size_t bn,size_t bm,
    int32_t* kp,int kv,size_t km,
    matrix_type_t ct,byte_t* cp,int cu,int cv)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_kmultiply(at,ap,au,an,am,bp,bu,bn,bm,kp,kv,km,cp,cu,cv);
	}
	else
#endif
	{
	    mt_kmultiply(at,ap,au,av,an,am,bp,bu,bv,bn,bm,kp,kv,km,cp,cu,cv);
	}
    }
    else {
	au = size_of_array(at,au);
	av = size_of_array(at,av);
	bu = size_of_array(bt,bu);
	bv = size_of_array(bt,bv);
	cu = size_of_array(ct,cu);
	cv = size_of_array(ct,cv);

	if (is_float(ct)) {
	    float64_t (*read_af)(byte_t*) = read_float64_func[at];
	    float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	    float64_t (*read_cf)(byte_t*) = read_float64_func[ct];
	    void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	    while(km--) {
		int32_t i = *kp - 1;
		if ((i >= 0) && (i < (int)an)) {
		    size_t m = bm;
		    byte_t* cp1 = cp + cu*i;
		    byte_t* ap1 = ap + au*i;
		    byte_t* bp1 = bp;
		    while(m--) {
			float64_t sum = read_cf(cp1);
			byte_t* bp2 = bp1;
			byte_t* ap2 = ap1;
			size_t  k = am;
			while(k--) {
			    float64_t a = read_af(ap2);
			    float64_t b = read_bf(bp2);
			    float64_t c = op_mul(a,b);
			    sum = op_add(sum, c);
			    ap2 += av;
			    bp2 += bu;
			}
			write_cf(cp1, sum);
			cp1 += cv;
			bp1 += bv;
		    }
		}
		kp += kv;
	    }
	}
	else if (is_complex(ct)) {
	    complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	    complex128_t (*read_bf)(byte_t*) = read_complex128_func[bt];
	    complex128_t (*read_cf)(byte_t*) = read_complex128_func[ct];
	    void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	    while(km--) {
		int32_t i = *kp - 1;
		if ((i >= 0) && (i < (int)an)) {
		    size_t m = bm;
		    byte_t* cp1 = cp + cu*i;
		    byte_t* ap1 = ap + au*i;
		    byte_t* bp1 = bp;		    
		    while(m--) {
			complex128_t sum = read_cf(cp1);
			byte_t* bp2 = bp1;
			byte_t* ap2 = ap1;
			size_t  k = am;
			while(k--) {
			    complex128_t a = read_af(ap2);
			    complex128_t b = read_bf(bp2);
			    complex128_t c = op_mul(a,b);
			    sum = op_add(sum, c);
			    ap2 += av;
			    bp2 += bu;
			}
			write_cf(cp1, sum);
			cp1 += cv;
			bp += bv;
		    }
		}
		kp += kv;
	    }
	}
	else {
	    int64_t (*read_af)(byte_t*) = read_int64_func[at];
	    int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	    int64_t (*read_cf)(byte_t*) = read_int64_func[ct];
	    void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	    while(km--) {
		int32_t i = *kp - 1;
		if ((i >= 0) && (i < (int)an)) {
		    size_t m = bm;
		    byte_t* cp1 = cp + cu*i;
		    byte_t* ap1 = ap + au*i;
		    byte_t* bp1 = bp;
		    while(m--) {
			int64_t sum = read_cf(cp1);
			byte_t* bp2 = bp1;
			byte_t* ap2 = ap1;
			size_t  k = am;
			while(k--) {
			    int64_t a = read_af(ap2);
			    int64_t b = read_bf(bp2);
			    int64_t c = op_mul(a,b);
			    sum = op_add(sum, c);
			    ap1 += av;
			    bp1 += bu;
			}
			write_cf(cp1, sum);
			cp1 += cv;
			bp += bv;
		    }
		}
		kp += kv;
	    }
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// kmultiply_transposed A*Bt = C
///////////////////////////////////////////////////////////////////////////////

static void kmultiply_t(
    bool_t use_vector,
    matrix_type_t at,byte_t* ap,int au,int av,size_t an,size_t am,
    matrix_type_t bt,byte_t* bp,int bu,int bv,size_t bn,size_t bm,
    int32_t* kp,int kv,size_t km,
    matrix_type_t ct,byte_t* cp,int cu,int cv)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_kmultiply_transposed(at,ap,au,an,am,bp,bu,bn,bm,
				     kp,kv,km,cp,cu,cv);
	else
#endif
	    mt_kmultiply_transposed(at,ap,au,av,an,am,bp,bu,bv,bn,bm,
				    kp,kv,km,cp,cu,cv);
    }
    else {
	au = size_of_array(at,au);
	av = size_of_array(at,av);
	bu = size_of_array(bt,bu);
	bv = size_of_array(bt,bv);
	cu = size_of_array(ct,cu);
	cv = size_of_array(ct,cv);

	if (is_float(ct)) {
	    float64_t (*read_af)(byte_t*) = read_float64_func[at];
	    float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	    float64_t (*read_cf)(byte_t*) = read_float64_func[ct];
	    void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	    while(km--) {
		int32_t i = *kp - 1;
		if ((i >= 0) && (i < (int)an)) {
		    byte_t* cp1 = cp + cu*i;
		    byte_t* ap1 = ap + au*i;
		    byte_t* bp1 = bp;
		    size_t n = bn;
		    while(n--) {
			float64_t sum = read_cf(cp1);
			byte_t* ap2 = ap1;
			byte_t* bp2 = bp1;
			size_t k = bm;
			while(k--) {
			    float64_t a = read_af(ap2);
			    float64_t b = read_bf(bp2);
			    float64_t c = op_mul(a,b);
			    sum = op_add(sum, c);
			    ap2 += av;
			    bp2 += bv;
			}
			write_cf(cp1, sum);
			cp1 += cv;
			bp  += bu;
		    }
		}
		kp += kv;
	    }
	}
	else if (is_complex(ct)) {
	    complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	    complex128_t (*read_bf)(byte_t*) = read_complex128_func[bt];
	    complex128_t (*read_cf)(byte_t*) = read_complex128_func[ct];
	    void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	    while(km--) {
		int32_t i = *kp - 1;
		if ((i >= 0) && (i < (int)an)) {
		    byte_t* cp1 = cp + cu*i;
		    byte_t* ap1 = ap + au*i;
		    byte_t* bp1 = bp;
		    size_t n = bn;
		    while(n--) {
			complex128_t sum = read_cf(cp1);
			byte_t* ap2 = ap1;
			byte_t* bp2 = bp1;
			size_t k = bm;
			while(k--) {
			    complex128_t a = read_af(ap2);
			    complex128_t b = read_bf(bp2);
			    complex128_t c = op_mul(a,b);
			    sum = op_add(sum, c);
			    ap2 += av;
			    bp2 += bv;
			}
			write_cf(cp1, sum);
			cp1 += cv;
			bp += bu;
		    }
		}
		kp += kv;
	    }
	}
	else {
	    int64_t (*read_af)(byte_t*) = read_int64_func[at];
	    int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	    int64_t (*read_cf)(byte_t*) = read_int64_func[ct];
	    void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	    while(km--) {
		int32_t i = *kp - 1;
		if ((i >= 0) && (i < (int)an)) {
		    byte_t* cp1 = cp + cu*i;
		    byte_t* ap1 = ap + au*i;
		    byte_t* bp1 = bp;
		    size_t n = bn;		    		    
		    while(n--) {
			int64_t sum = read_cf(cp1);
			byte_t* ap2 = ap1;
			byte_t* bp2 = bp1;
			size_t k = bm;
			while(k--) {
			    int64_t a = read_af(ap2);
			    int64_t b = read_bf(bp2);
			    int64_t c = op_mul(a,b);
			    sum = op_add(sum, c);
			    ap2 += av;
			    bp2 += bv;
			}
			write_cf(cp1, sum);
			cp1 += cv;
			bp += bu;
		    }
		}
		kp += kv;
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
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);

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

#ifdef USE_VECTOR
static void mtv_copy(matrix_type_t at, byte_t* ap, int au,
		     matrix_type_t ct, byte_t* cp, int cu,
		     size_t n, size_t m)
{
    size_t sz = element_size(at);
    au = size_of_array(at,au);
    cu = size_of_array(ct,cu);

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
#endif

///////////////////////////////////////////////////////////////////////////////
//  simple copy
///////////////////////////////////////////////////////////////////////////////

static void copy1(bool_t use_vector,
		  matrix_type_t at, byte_t* ap, int au, int av,
		  matrix_type_t ct, byte_t* cp, int cu, int cv,
		  size_t n, size_t m)
{
    if (at == ct) {
#ifdef USE_VECTOR
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

    av = size_of_array(at,av);
    au = size_of_array(at,au);
    cv = size_of_array(ct,cv);
    cu = size_of_array(ct,cu);

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
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
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
		value = read_af(ap1);
		write_cf(cp1, value);
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
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
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
		value = read_af(ap1);
		write_cf(cp1, value);
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
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
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
		value = read_af(ap1);
		write_cf(cp1,value);
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

    av = size_of_array(at,av);
    au = size_of_array(at,au);
    cv = size_of_array(ct,cv);
    cu = size_of_array(ct,cu);

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
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		float64_t value;
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_af(ap1);
		write_cf(cp1, value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }
    else if (is_complex(ct)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	void (*write_cf)(byte_t*, complex128_t) = write_complex128_func[ct];
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		complex128_t value;
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_af(ap1);
		write_cf(cp1, value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		int64_t value;
		if (aj >= am) { aj = 0; ai++; ap += au; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_af(ap1);
		write_cf(cp1,value);
		cp1 += cv;
		ap1 += av;
		aj++;
	    }
	    cp += cu;
	}
    }
}

// initialize a static matrix, used for constants etc
static matrix_t* init_matrix(matrix_t* mp,
			     bool_t rowmajor,
			     unsigned int n, unsigned int m,
			     int nstep, int mstep,
			     matrix_type_t type, byte_t* data)
{
    mp->type    = type;
    mp->n       = n;
    mp->m       = m;
    mp->nstep  = nstep;
    mp->mstep  = mstep;
    mp->size    = n*m*element_size(type);
    mp->offset  = 0;
    mp->rowmajor = rowmajor;
    mp->base    = NULL;
    mp->data    = data;
    mp->first   = data;
    mp->rw_lock = NULL;
    mp->ptr     = 0;
    return mp;
}

matrix_t* alloc_matrix_resource(size_t n, size_t m,
				bool_t rowmajor,matrix_type_t type,size_t align)
{
    matrix_t* mp = enif_alloc_resource(matrix_r, sizeof(matrix_t));

    if (mp != NULL) {
	size_t byte_stride = (size_of_array(type,m)+align-1) & ~(align-1);
	size_t size   = n*byte_stride;

	mp->n       = n;
	mp->m       = m;
	mp->type    = type;
	mp->size    = 0;
	mp->offset  = 0;
	mp->rowmajor = rowmajor;
	mp->data    = NULL;
	mp->first   = NULL;
	mp->rw_lock = NULL;
	mp->ptr     = (uintptr_t)mp;

	if ((mp->base = enif_alloc(size+align-1)) != NULL) {
	    mp->size  = size;
	    mp->nstep = byte_stride / element_size(type);
	    mp->mstep = 1;
	    mp->rw_lock = enif_rwlock_create("matrix");
	    mp->data = align_ptr(mp->base,align);
	    mp->first = mp->data;
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
	// NOTE! compiler generate code that crash on *mp = *np!
	// probalbly bad opcodes generated because of vector elements!
	memcpy(mp, np, sizeof(matrix_t));
	enif_release_resource(np);
	return 1;
    }
    return 0;
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

static int get_scalar(ErlNifEnv* env, ERL_NIF_TERM arg, bool_t coerce,
		      matrix_type_t type, matrix_type_t* otype, scalar_t* sp)
{
    double dv;
    ErlNifSInt64 iv;
    complex128_t cv;

    if (enif_get_double(env, arg, &dv)) {
	if (coerce && !is_float(type))
	    type = combine_type(type, FLOAT64);
	write_float64(type, sp->data, dv);
    }
    else if (enif_get_int64(env, arg, &iv)) {
	if (coerce && !is_integer(type))
	    type = combine_type(type, INT64);
	write_int64(type, sp->data, iv);
    }
    else if (get_complex(env, arg, &cv)) {
	if (coerce && !is_complex(type))
	    type = combine_type(type, COMPLEX128);
	write_complex128(type, sp->data, cv);
    }
    else
	return 0;
    if (otype != NULL) *otype = type;
    return 1;
}


// Get matrix argument
// { 'matrix', n, m, type, ptr, offset, nstep, mstep, row-major, binary-data }
//   Must match #matrix record in the code!
//

static int get_matrix(ErlNifEnv* env, ERL_NIF_TERM arg, matrix_t* mp)
{
    int arity;
    unsigned int type;
    ErlNifUInt64 ptr;
    size_t n_stride;  // in bytes
    size_t m_stride;  // in bytes
    size_t vsize;
    size_t byte_offset;
    const ERL_NIF_TERM* elems;
    matrix_t* rmp;

    if (!enif_get_tuple(env, arg, &arity, &elems)) return 0;
    if (arity != 10) return 0;
    if (elems[0] != ATOM(matrix)) return 0;
    if (!enif_get_uint(env, elems[1], &type)) return 0;
    if (type >= NUM_TYPES) return 0;
    if (!enif_get_uint(env, elems[2], &mp->n)) return 0;
    if (!enif_get_uint(env, elems[3], &mp->m)) return 0;
    if (!enif_get_int(env, elems[4], &mp->nstep)) return 0;
    if (!enif_get_int(env, elems[5], &mp->mstep)) return 0;
    if (!enif_get_uint64(env, elems[6], &ptr)) return 0;
    if (!enif_get_uint(env, elems[7], &mp->offset)) return 0;
    if (!get_bool(env, elems[8], &mp->rowmajor)) return 0;

    mp->type = type;
    byte_offset = mp->offset*element_size(type);
    n_stride = mp->nstep*element_size(type);
    m_stride = mp->mstep*element_size(type);
    if (mp->nstep == 0 && mp->mstep == 0)
	vsize = element_size(type);  // at least one element
    else
	vsize = (mp->n-1)*n_stride + mp->m*m_stride;

    if (ptr && enif_get_resource(env, elems[9], matrix_r, (void**)&rmp) &&
	(rmp->ptr == ptr)) {
	if ((byte_offset + vsize) > rmp->size)
	    return 0;
	mp->size = rmp->size;
	mp->base = rmp->base;
	mp->data = rmp->data;
	mp->first = mp->data + byte_offset;
	mp->rw_lock = rmp->rw_lock;
	mp->ptr   = rmp->ptr;
    }
    else {
	ErlNifBinary bin;
	if (!enif_inspect_binary(env, elems[9], &bin))
	    return 0;
	// check bounds
	if ((byte_offset + vsize) > bin.size)
	    return 0;
	mp->size = bin.size;
	mp->base = NULL;
	mp->data = bin.data;
	mp->first = mp->data + byte_offset;
	mp->rw_lock = NULL;
	mp->ptr  = 0;
    }
    return 1;
}

// Get a writable resource matrix!
static int get_w_matrix(ErlNifEnv* env, ERL_NIF_TERM arg, matrix_t* mp)
{
    return get_matrix(env, arg, mp) && (mp->ptr != 0);
}

// get_scalar_matrix
// Parse one scalar argument int,float or complex and
// generate a matrix with that argument as data, setting
//
static int get_scalar_matrix(ErlNifEnv* env, ERL_NIF_TERM arg, matrix_t* mp,
			     bool_t rowmajor, matrix_type_t type,
			     unsigned int n, unsigned int m)
{
    if (!get_scalar(env, arg, TRUE, type, &type, &mp->sdata))
	return 0;
    // init with nstep=0 and mstep=0 that is a single element is
    // repeated for n and m
    init_matrix(mp, rowmajor, n, m, 0, 0, type, mp->sdata.data);
    return 1;
}

static ERL_NIF_TERM make_matrix(ErlNifEnv* env,
				unsigned int n, unsigned int m,
				bool_t rowmajor,
				matrix_type_t type,
				matrix_t* mp,
				ERL_NIF_TERM bin)
{
    return enif_make_tuple(env, 10,
			   ATOM(matrix),
			   enif_make_uint(env, type),
			   enif_make_uint(env, n),
			   enif_make_uint(env, m),
			   enif_make_uint(env, mp->nstep),
			   enif_make_uint(env, mp->mstep),
			   enif_make_uint64(env, (uintptr_t)mp->ptr),
			   enif_make_uint(env, mp->offset),
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
    if (type >= NUM_TYPES)
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
	if (c.nstep == (int)c.m)
	    memcpy(c.data, binary.data, c.size);
	else {
	    byte_t* ap = c.data;
	    byte_t* bp = binary.data;
	    size_t  bu = c.m*element_size(type);
	    size_t  au = c.nstep*element_size(type);
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

// element(I, J, A) -> A[I][J]   I and J are 1-based
// addme:
// element(I,A) -> A[I], row(I) or column(I) depening on rowmajor
//
// element([[r1,r2,..,rm]],A) -> [[A[r1][1], A[r2][2], ... A[rm][m]]]
//
// element([[c1],[c2],..,[cn]],A) -> [[A[1][c1]], [A[2][c2]], ... [A[n][cn]]]
//
// pickout max element with (example)
// Index = argmax(A,0),
// Max = element(Index, A),
// or diagonal with:
// Index = matrix:from_list([lists:seq(1,N)],int32),
// Diag = element(Index, A),
//

///////////////////////////////////////////////////////////////////////////////
//  copy element by element via index, assume at == ct
///////////////////////////////////////////////////////////////////////////////

static void index_copy(matrix_type_t at, byte_t* ap, int au, int av,
		       int32_t* ip, int iu, int iv,
		       matrix_type_t ct, byte_t* cp, int cu, int cv,
		       size_t n, size_t m)
{
    size_t sz = element_size(at);
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);

    while(n--) {
	byte_t* ap1  = ap;
	int32_t* ip1 = ip;
	byte_t* cp1  = cp;
	size_t m1 = m;
	while(m1--) {
	    int32_t i = (*ip1-1)*au;  // get offset
	    memcpy(cp1, ap1+i, sz);
	    ip1 += iv;
	    cp1 += cv;
	    ap1 += av;  // next column
	}
	ip += iu;
	cp += cu;
    }
}

// check that all indices in matrix ip are with in range 0 < i < m
int index_check(int32_t* ip, int iu, int iv, size_t in, size_t im, size_t k)
{
    while(in--) {
	size_t imm = im;
	int32_t* ip1 = ip;
	while(imm--) {
	    int32_t ix = *ip1;
	    if (ix < 1) return 0;
	    if (ix > (int32_t)k) return 0;
	    ip1 += iv;
	}
	if (iu == 0) return 1; // repeating data
	ip += iu;
    }
    return 1;
}

ERL_NIF_TERM matrix_element(ErlNifEnv* env, int argc,
			    const ERL_NIF_TERM argv[])
{
    if (argc == 2) {
	matrix_t index;
	matrix_t a;
	matrix_t c;
	ERL_NIF_TERM bin;

	if (!get_matrix(env, argv[1], &a))
	    return BADARG(env);
	if (!get_matrix(env, argv[0], &index)) {
	    if (!get_scalar_matrix(env,argv[0],&index,a.rowmajor,INT32,a.n,1))
		return BADARG(env);
	}
	if (index.type != INT32)
	    return BADARG(env);
	if (a.rowmajor == index.rowmajor) {
	    if ((index.n > a.n) || (index.m > a.m))
		return BADARG(env);
	    if (!index_check((int32_t*)index.first, index.nstep, index.mstep,
			     index.n, index.m, a.n))
		return BADARG(env);

	    if (!create_matrix(env,index.n,index.m,index.rowmajor,
			       a.type,&c,&bin))
		return BADARG(env);

	    matrix_rr_lock(&a,&index);
	    index_copy(a.type, a.first, a.nstep, a.mstep,
		       (int32_t*)index.first, index.nstep, index.mstep,
		       c.type, c.first, c.nstep, c.mstep,
		       c.n, c.m);
	    matrix_rr_unlock(&a,&index);
	    return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
	}
	else { // a.rowmajor / index.rowmajor
	    if ((index.m > a.n) || (index.n > a.m))
		return BADARG(env);
	    if (!index_check((int32_t*)index.first, index.nstep, index.mstep,
			     index.n, index.m, a.m))
		return BADARG(env);

	    if (!create_matrix(env,index.n,index.m,index.rowmajor,
			       a.type,&c,&bin))
		return BADARG(env);

	    matrix_rr_lock(&a,&index);
	    index_copy(a.type, a.first, a.mstep, a.nstep,
		       (int32_t*)index.first, index.nstep, index.mstep,
		       c.type, c.first, c.nstep, c.mstep,
		       c.n, c.m);
	    matrix_rr_unlock(&a,&index);
	    return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
	}
    }
    else {
	unsigned int i, j;
	matrix_t a;
	byte_t* ptr;
	ERL_NIF_TERM r;

	if (!enif_get_uint(env, argv[0], &i))
	    return enif_make_badarg(env);
	if (!enif_get_uint(env, argv[1], &j))
	    return enif_make_badarg(env);
	if (!get_matrix(env, argv[2], &a))
	    return enif_make_badarg(env);
	if (a.rowmajor) {
	    if ((i < 1) || (i > a.n))
		return enif_make_badarg(env);
	    if ((j < 1) || (j > a.m))
		return enif_make_badarg(env);
	    if ((a.nstep == 0) && (a.mstep == 0))  // special case
		ptr = a.first;
	    else // fixme use mstep?
		ptr = a.first + (i-1)*size_of_array(a.type, a.nstep) +
		    size_of_array(a.type, j-1);
	}
	else {
	    if ((j < 1) || (j > a.n)) // row when column major
		return enif_make_badarg(env);
	    if ((i < 1) || (i > a.m))
		return enif_make_badarg(env);
	    if ((a.nstep == 0) && (a.mstep == 0))
		ptr = a.first;
	    else // fixme use mstep?
		ptr = a.first + (j-1)*size_of_array(a.type, a.nstep) +
		    size_of_array(a.type, i-1);
	}
	matrix_r_lock(&a);
	r = read_term(env, a.type, ptr);
	matrix_r_unlock(&a);
	return r;
    }
}

// DESTRUCTIVE
// setelement(I, J, A, V) -> A[I][J] = V  I and J are 1-based
ERL_NIF_TERM matrix_setelement(ErlNifEnv* env, int argc,
			       const ERL_NIF_TERM argv[])
{
    unsigned int i, j;
    matrix_t a;
    byte_t* ptr;
    scalar_t value;
    UNUSED(argc);

    if (!enif_get_uint(env, argv[0], &i))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &j))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[2], &a))
	return enif_make_badarg(env);
    if (!get_scalar(env, argv[3], FALSE, a.type, NULL, &value))
	return enif_make_badarg(env);

    if (a.rowmajor) {
	if ((i < 1) || (i > a.n))
	    return enif_make_badarg(env);
	if ((j < 1) || (j > a.m))
	    return enif_make_badarg(env);
	if (a.ptr == 0) // only resource matrix may be update!
	    return enif_make_badarg(env);
	if ((a.nstep == 0) && (a.mstep == 0))
	    ptr = a.first;
	else // fixme use mstep?
	    ptr = a.first + (i-1)*size_of_array(a.type, a.nstep) +
		size_of_array(a.type, j-1);
    }
    else {
	if ((j < 1) || (j > a.n)) // row when column major
	    return enif_make_badarg(env);
	if ((i < 1) || (i > a.m))
	    return enif_make_badarg(env);
	if (a.ptr == 0) // only resource matrix may be update!
	    return enif_make_badarg(env);
	if ((a.nstep == 0) && (a.mstep == 0))
	    ptr = a.first;
	else // fixme use mstep?
	    ptr = a.first + (j-1)*size_of_array(a.type, a.nstep) +
		size_of_array(a.type, i-1);
    }
    matrix_w_lock(&a);
    memcpy(ptr, value.data, element_size(a.type));
    matrix_w_unlock(&a);
    return argv[2];
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
    bool_t use_vector = (ap->mstep==1);

    if (cp->rowmajor == ap->rowmajor)
	func(use_vector,
	     ap->type, ap->first, ap->nstep, ap->mstep,
	     cp->type, cp->first, cp->nstep, 1, cp->n, cp->m);
    else
	func(FALSE,
	     ap->type, ap->first, ap->mstep, ap->nstep,
	     cp->type, cp->first, cp->nstep, 1, cp->n, cp->m);
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
    bool_t use_vector = (ap->mstep==1) && (bp->mstep==1);

    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor)
	    func(use_vector,
		ap->type, ap->first, ap->nstep, ap->mstep,
		bp->type, bp->first, bp->nstep, bp->mstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
	else if (!ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->mstep, ap->nstep,
		 bp->type, bp->first, bp->nstep, bp->mstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
	else if (ap->rowmajor && !bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->nstep, ap->mstep,
		bp->type, bp->first, bp->mstep, bp->nstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
	else
	    func(FALSE,
		ap->type, ap->first, ap->mstep, ap->nstep,
		bp->type, bp->first, bp->mstep, bp->nstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
    }
    else {
	if (ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->mstep, ap->nstep,
		bp->type, bp->first, bp->mstep, bp->nstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
	else if (!ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->nstep, ap->mstep,
		bp->type, bp->first, bp->mstep, bp->nstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
	else if (ap->rowmajor && !bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->mstep, ap->nstep,
		bp->type, bp->first, bp->nstep, bp->mstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
	else
	    func(use_vector,
		ap->type, ap->first, ap->nstep, ap->mstep,
		bp->type, bp->first, bp->nstep, bp->mstep,
		cp->type, cp->first, cp->nstep, 1,
		cp->n, cp->m);
    }
}

// matrix apply2(func,A,B) -> C, check all rowmajor variants
// with and without accelerations return a scalar values as erlang term

static void s_apply2(
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t bt, byte_t* bp, int bu, int bv,
		 matrix_type_t ct, byte_t* cp,
		 size_t n, size_t m),
    matrix_t* ap, matrix_t* bp, matrix_type_t ct, byte_t* cp)
{
    bool_t use_vector = (ap->mstep==1) && (bp->mstep==1);

    if (ap->rowmajor && bp->rowmajor)
	func(use_vector,
	     ap->type, ap->first, ap->nstep, ap->mstep,
	     bp->type, bp->first, bp->nstep, bp->mstep,
	     ct, cp, ap->n, ap->m);
    else if (!ap->rowmajor && bp->rowmajor)
	func(FALSE,
	     ap->type, ap->first, ap->mstep, ap->nstep,
	     bp->type, bp->first, bp->nstep, bp->mstep,
	     ct, cp, bp->n, bp->m);
    else if (ap->rowmajor && !bp->rowmajor)
	func(FALSE,
	     ap->type, ap->first, ap->nstep, ap->mstep,
	     bp->type, bp->first, bp->mstep, bp->nstep,
	     ct, cp, ap->n, ap->m);
    else
	func(use_vector,
	     ap->type, ap->first, ap->mstep, ap->nstep,
	     bp->type, bp->first, bp->mstep, bp->nstep,
	     ct, cp, bp->m, bp->n);
}

//
// add(matrix(), matrix() [,matrix()]) -> matrix();
//    (matrix(), scalar() [,matrix()]) -> matrix();
//    (scalar(), matrix() [,matrix()]) -> matrix();
//
ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;

    if (get_matrix(env, argv[0], &a)) {
	if (!get_matrix(env, argv[1], &b)) {
	    if (!get_scalar_matrix(env,argv[1],&b,a.rowmajor,a.type,a.n,a.m))
		return enif_make_badarg(env);
	}
    }
    else {
	if (!get_matrix(env, argv[1], &b))
	    return enif_make_badarg(env);
	if (!get_scalar_matrix(env,argv[0],&a,b.rowmajor,b.type,b.n,b.m))
	    return enif_make_badarg(env);
    }

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
	matrix_rr_lock(&a, &b);
	m_apply2(add, &a, &b, &c);
	matrix_rr_unlock(&a, &b);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 3
	if (!get_w_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rrw_lock(&a, &b, &c);
	m_apply2(add, &a, &b, &c);
	matrix_rrw_unlock(&a, &b, &c);
	return argv[2];
    }
}
//
// subtract(matrix(), matrix() [,matrix()]) -> matrix();
//         (matrix(), scalar() [,matrix()]) -> matrix();
//         (scalar(), matrix() [,matrix()]) -> matrix();
//
ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;

    if (get_matrix(env, argv[0], &a)) {
	if (!get_matrix(env, argv[1], &b)) {
	    if (!get_scalar_matrix(env,argv[1],&b,a.rowmajor,a.type,a.n,a.m))
		return enif_make_badarg(env);
	}
    }
    else {
	if (!get_matrix(env, argv[1], &b))
	    return enif_make_badarg(env);
	if (!get_scalar_matrix(env,argv[0],&a,b.rowmajor,b.type,b.n,b.m))
	    return enif_make_badarg(env);
    }

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

	matrix_rr_lock(&a,&b);
	m_apply2(subtract,&a, &b, &c);
	matrix_rr_unlock(&a,&b);

	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 3
	if (!get_w_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rrw_lock(&a, &b, &c);
	m_apply2(subtract, &a, &b, &c);
	matrix_rrw_unlock(&a, &b, &c);
	return argv[2];
    }
}


// multiply two matrices element wise
ERL_NIF_TERM matrix_times(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;

    if (get_matrix(env, argv[0], &a)) {
	if (!get_matrix(env, argv[1], &b)) {
	    if (!get_scalar_matrix(env,argv[1],&b,a.rowmajor,a.type,a.n,a.m))
		return enif_make_badarg(env);
	}
    }
    else {
	if (!get_matrix(env, argv[1], &b))
	    return enif_make_badarg(env);
	if (!get_scalar_matrix(env,argv[0],&a,b.rowmajor,b.type,b.n,b.m))
	    return enif_make_badarg(env);
    }

    if ((a.rowmajor == b.rowmajor) && ((a.n != b.n) || (a.m != b.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != b.rowmajor) && ((a.n != b.m) || (a.m != b.n)))
	return enif_make_badarg(env);

    if (argc == 2) {  // FIXME: maybe coerce type argument for result?
	ERL_NIF_TERM bin;
	matrix_type_t ct;

	ct = combine_type(a.type, b.type);

	if (!create_matrix(env,a.n,a.m,a.rowmajor,ct,&c,&bin))
	    return enif_make_badarg(env);

	matrix_rr_lock(&a,&b);
	m_apply2(times,&a, &b, &c);
	matrix_rr_unlock(&a,&b);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 3
	if (!get_w_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rrw_lock(&a, &b, &c);
	m_apply2(times, &a, &b, &c);
	matrix_rrw_unlock(&a, &b, &c);
	return argv[2];
    }
}

// multiply A and B element wise but only the rows as
// controlled by matrix K
//
ERL_NIF_TERM matrix_ktimes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c, k;

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[2], &k))
	return enif_make_badarg(env);

    if ((a.rowmajor == b.rowmajor) && ((a.n != b.n) || (a.m != b.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != b.rowmajor) && ((a.n != b.m) || (a.m != b.n)))
	return enif_make_badarg(env);

    if (k.type != INT32)
	return enif_make_badarg(env);
    if (!k.rowmajor || (k.n != 1))
	return enif_make_badarg(env);

    if (argc == 3) {
	ERL_NIF_TERM bin;
	matrix_type_t ct;

	ct = combine_type(a.type, b.type);

	if (!create_matrix(env,a.n,a.m,a.rowmajor,ct,&c,&bin))
	    return enif_make_badarg(env);

	matrix_rr_lock(&a,&b);
	m_apply2(times,&a, &b, &c);
	matrix_rr_unlock(&a,&b);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {  // argc == 4
	if (!get_w_matrix(env, argv[3], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rrw_lock(&a, &b, &c);
	m_apply2(times, &a, &b, &c);
	matrix_rrw_unlock(&a, &b, &c);
	return argv[2];
    }
}

static void m_multiply(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor) {
	    multiply(TRUE,
		     ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		     bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		     cp->type,cp->first,cp->nstep,cp->mstep);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    multiply_t(TRUE,
		       ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		       bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		       cp->type,cp->first,cp->nstep,cp->mstep);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    multiply(FALSE,
		     ap->type,ap->first,ap->mstep,ap->nstep,ap->m,ap->n,
		     bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		     cp->type,cp->first,cp->nstep,cp->mstep);
	}
	else { // !ap->rowmajor && !bp->rowmajor (NOTE A/B swap!)
	    multiply(TRUE,
		     bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		     ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		     cp->type,cp->first,1,cp->nstep);
	}
    }
    else { // !cp->rowmajor
	if (ap->rowmajor && bp->rowmajor) {
	    multiply(TRUE,
		     ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		     bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		     cp->type,cp->first,1,cp->nstep);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    multiply_t(TRUE,
		       ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		       bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		       cp->type,cp->first,1,cp->nstep);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    multiply(FALSE,
		     ap->type,ap->first,ap->mstep,ap->nstep,ap->m,ap->n,
		     bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		     cp->type,cp->first,1,cp->nstep);
	}
	else { // !ap->rowmajor && !bp->rowmajor
	    multiply(TRUE,
		     bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		     ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		     cp->type,cp->first,cp->nstep,cp->mstep);
	}
    }
}

// multiply A*B = C matrices
// NOTE!  C should not be any of A or B, but keep it for a while!
// FIXME!

ERL_NIF_TERM matrix_multiply(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    matrix_t a, b, c;
    size_t n, m;
    bool_t rowmajor = TRUE;
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

    if ((argc == 2) ||
	((argc == 3) && get_bool(env, argv[2], &rowmajor))) {
	matrix_type_t ct = combine_type(a.type, b.type);
	ERL_NIF_TERM bin = 0;

	if (rowmajor) {
	    if (!create_matrix(env,n,m,TRUE,ct,&c,&bin))
		return enif_make_badarg(env);
	}
	else {
	    if (!create_matrix(env,m,n,FALSE,ct,&c,&bin))
		return enif_make_badarg(env);
	}
	matrix_rr_lock(&a,&b);
	m_multiply(&a, &b, &c);
	matrix_rr_unlock(&a,&b);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else { // argc == 3
	if (!get_w_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);

	if (c.rowmajor && ((c.n != n) || (c.m != m)))
	    return enif_make_badarg(env);
	if (!c.rowmajor && ((c.n != m) || (c.m != n)))
	    return enif_make_badarg(env);

	matrix_rrw_lock(&a, &b, &c);
	// FIXME if a, b or both is equal to c then C
	// must be computed in a copy then copied back!
	m_multiply(&a, &b, &c);
	matrix_rrw_unlock(&a, &b, &c);
	return argv[2];
    }
}


static void k_multiply(matrix_t* ap, matrix_t* bp, matrix_t* kp,
		       matrix_t* cp)
{
    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor) {
	    kmultiply(TRUE,
		      ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		      bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		      (int32_t*)kp->first,kp->mstep,kp->m,
		      cp->type,cp->first,cp->nstep,cp->mstep);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    kmultiply_t(TRUE,
			ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
			bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
			(int32_t*)kp->first,kp->mstep,kp->m,
			cp->type,cp->first,cp->nstep,cp->mstep);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    kmultiply(FALSE,
		      ap->type,ap->first,ap->mstep,ap->nstep,ap->m,ap->n,
		      bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		      (int32_t*)kp->first,kp->mstep,kp->m,
		      cp->type,cp->first,cp->nstep,cp->mstep);
	}
	else { // !ap->rowmajor && !bp->rowmajor (NOTE A/B swap!)
	    // FIXME: check how to check k in this case
	    kmultiply(TRUE,
		      bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		      ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		      (int32_t*)kp->first,kp->mstep,kp->m,
		      cp->type,cp->first,1,cp->nstep);
	}
    }
    else { // !cp->rowmajor
	if (ap->rowmajor && bp->rowmajor) {
	    kmultiply(TRUE,
		      ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		      bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		      (int32_t*)kp->first,kp->mstep,kp->m,
		      cp->type,cp->first,1,cp->nstep);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    kmultiply_t(TRUE,
			ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
			bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
			(int32_t*)kp->first,kp->mstep,kp->m,
			cp->type,cp->first,1,cp->nstep);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    kmultiply(FALSE,
		      ap->type,ap->first,ap->mstep,ap->nstep,ap->m,ap->n,
		      bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		      (int32_t*)kp->first,kp->mstep,kp->m,
		      cp->type,cp->first,1,cp->nstep);
	}
	else { // !ap->rowmajor && !bp->rowmajor
	    // FIXME: check how to check k in this case
	    kmultiply(TRUE,
		      bp->type,bp->first,bp->nstep,bp->mstep,bp->n,bp->m,
		      ap->type,ap->first,ap->nstep,ap->mstep,ap->n,ap->m,
		      (int32_t*)kp->first,kp->mstep,kp->m,
		      cp->type,cp->first,cp->nstep,cp->mstep);
	}
    }
}

// muladd C += (A*B) o K
// K is used to select rows in A/C to operate on
//
// kmultiply(A,B,K)       -> C  (row major)
// kmultiply(A,B,K,true)  -> C (row major)
// kmultiply(A,B,K,false) -> C (coumn major)
// kmultiply(A,B,K,C)     -> C
//

ERL_NIF_TERM matrix_kmultiply(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    matrix_t a, b, k, c;
    size_t n, m;
    bool_t rowmajor = TRUE;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[2], &k))
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

    // K is a row of boolean data
    if (k.type != INT32)
	return enif_make_badarg(env);
    if (!k.rowmajor || (k.n != 1))
	return enif_make_badarg(env);

    if ((argc == 3) ||
	((argc == 4) && get_bool(env, argv[3], &rowmajor))) {
	matrix_type_t ct = combine_type(a.type, b.type);
	ERL_NIF_TERM bin = 0;

	if (rowmajor) {
	    if (!create_matrix(env,n,m,TRUE,ct,&c,&bin))
		return enif_make_badarg(env);
	}
	else {
	    if (!create_matrix(env,m,n,FALSE,ct,&c,&bin))
		return enif_make_badarg(env);
	}
	matrix_rrr_lock(&a, &b, &k);
	k_multiply(&a, &b, &k, &c);
	matrix_rrr_unlock(&a, &b, &k);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else { // argc == 4
	if (!get_w_matrix(env, argv[3], &c))
	    return enif_make_badarg(env);

	if (c.rowmajor && ((c.n != n) || (c.m != m)))
	    return enif_make_badarg(env);
	if (!c.rowmajor && ((c.n != m) || (c.m != n)))
	    return enif_make_badarg(env);

	matrix_rrrw_lock(&a, &b, &k, &c);
	// FIXME if a, b or both is equal to c then C
	// must be computed in a copy then copied back!
	k_multiply(&a, &b, &k, &c);
	matrix_rrrw_unlock(&a, &b, &k, &c);
	return argv[3];
    }
}

#define swap(a,i,j) do { \
    typeof((a)[0]) _tmp = (a)[i];		\
    (a)[i] = a[j];				\
    (a)[j] = _tmp;				\
    } while(0)

// partition around a pivot element high -> low !!!
static int topk_partition_f(float64_t* ap, int* ip, int l, int h)
{
    float64_t p = ap[l];
    int i = l-1;
    int j = h+1;

    while(1) {
	do { i++; } while(ap[i] > p);
	do { j--; } while(ap[j] < p);
	if (i >= j) return j;
	swap(ap, i, j);
	swap(ip, i, j);
    }
    return -1;
}

// partition around a pivot element high -> low !!!
static int topk_partition_c(complex128_t* ap, int* ip, int l, int h)
{
    complex128_t p = ap[l];
    int i = l-1;
    int j = h+1;

    while(1) {
	do { i++; } while(complex128_gt(ap[i], p));
	do { j--; } while(complex128_lt(ap[j], p));
	if (i >= j) return j;
	swap(ap, i, j);
	swap(ip, i, j);
    }
    return -1;
}

static int topk_partition_i(int64_t* ap, int* ip, int l, int h)
{
    int64_t p = ap[l];
    int i = l-1;
    int j = h+1;

    while(1) {
	do { i++; } while(ap[i] > p);
	do { j--; } while(ap[j] < p);
	if (i >= j) return j;
	swap(ap, i, j);
	swap(ip, i, j);
    }
    return -1;
}

// simple topk select algorithm
// get data as integer, float, complex and label data with index
static void topk(int k, matrix_type_t at,byte_t* ap,int au,
		 int32_t* cp,size_t m)
{
    int32_t index[m];
    int i, p, l, h;
    int ki = k;

    au = size_of_array(at,au);

    if (is_float(at)) {
	float64_t value[m];
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	for (i = 0; i < (int)m; i++) {
	    index[i] = i;
	    value[i] = read_af(ap);
	    ap += au;
	}
	l = 0;
	h = m-1;
	while((ki > 0) && (ki < (int)m)) {
	    int nl;
	    p = topk_partition_f(value,index,l,h);
	    nl = (p-l)+1;
	    if (nl == ki) break;
	    else if (nl > ki) h = p;
	    else { // p < k
		l = p+1;
		ki -= nl;
	    }
	}
    }
    else if (is_complex(at)) {
	complex128_t value[m];
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	for (i = 0; i < (int)m; i++) {
	    index[i] = i;
	    value[i] = read_af(ap);
	    ap += au;
	}
	l = 0;
	h = m-1;
	while((ki > 0) && (ki < (int)m)) {
	    int nl;
	    p = topk_partition_c(value,index,l,h);
	    nl = (p-l)+1;
	    if (nl == ki) break;
	    else if (nl > ki) h = p;
	    else { // p < k
		l = p+1;
		ki -= nl;
	    }
	}
    }
    else {
	int64_t value[m];
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	for (i = 0; i < (int)m; i++) {
	    index[i] = i;
	    value[i] = read_af(ap);
	    ap += au;
	}
	l = 0;
	h = m-1;
	while((ki > 0) && (ki < (int)m)) {
	    int nl;
	    p = topk_partition_i(value,index,l,h);
	    nl = (p-l)+1;
	    if (nl == ki) break;
	    else if (nl > ki) h = p;
	    else { // p < k
		l = p+1;
		ki -= nl;
	    }
	}
    }
    for (i = 0; (i < k) && (i < (int)m); i++)
	cp[i] = index[i]+1;
}


ERL_NIF_TERM matrix_topk(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, c;
    ERL_NIF_TERM bin;
    int k;
    size_t m;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (a.rowmajor) { // column vector
	if (a.m != 1) return enif_make_badarg(env);
	m = a.n;
    }
    else { // column vector
	if (!a.rowmajor && (a.n != 1))
	    return enif_make_badarg(env);
	m = a.m;
    }
    if (!enif_get_int(env, argv[1], &k) || (k < 0))
	return enif_make_badarg(env);
    if (k == 0)
	return ATOM(undefined);
    // c matrix is a 1xM matrix of INT32
    if (!create_matrix(env,1,k,TRUE,INT32,&c,&bin))
	return enif_make_badarg(env);
    // the elements in A are split and top K elements indices are stored in C
    if (a.rowmajor)
	topk(k, a.type, a.first, a.nstep, (int32_t*)c.first, m);
    else
	topk(k, a.type, a.first, 1, (int32_t*)c.first, m);

    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// negate a matrix
ERL_NIF_TERM matrix_negate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, c;

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);

    if (argc == 1) {
	ERL_NIF_TERM bin;

	if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);

	matrix_r_lock(&a);
	m_apply1(negate, &a, &c);
	matrix_r_unlock(&a);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else { // args == 2
	if (!get_w_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rw_lock(&a, &c);
	m_apply1(negate, &a, &c);
	matrix_rw_unlock(&a, &c);
	return argv[1];
    }
}

static void mulsum(bool_t use_vector, bool_t square_root,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp,
		   size_t n, size_t m)
{
    UNUSED(use_vector);

    // FIXME! acceleration at=bt
    au = size_of_array(at,au);
    av = size_of_array(at,av);
    bu = size_of_array(bt,bu);
    bv = size_of_array(bt,bv);

    if (is_float(ct)) {
	float64_t sum = 0.0;
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_af(ap1);
		float64_t b = read_bf(bp1);
		float64_t c;
		ap1 += av;
		bp1 += bv;
		c = op_mul(a,b);
		sum = op_add(sum, c);
	    }
	    ap += au;
	    bp += bu;
	}
	if (square_root) sum = sqrt(sum);
	write_float64(ct, cp, sum);
    }
    else if (is_complex(ct)) {
	complex128_t sum = CMPLX(0.0,0.0);
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	complex128_t (*read_bf)(byte_t*) = read_complex128_func[bt];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    size_t m1 = m;
	    while(m1--) {
		complex128_t a = read_af(ap1);
		complex128_t b = read_bf(bp1);
		complex128_t c;
		ap1 += av;
		bp1 += bv;
		c = op_mul(a,b);
		sum = op_add(sum, c);
	    }
	    ap += au;
	    bp += bu;
	}
	if (square_root) sum = csqrt(sum);
	write_complex128(ct, cp, sum);
    }
    else {
	int64_t sum = 0;
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_af(ap1);
		int64_t b = read_bf(bp1);
		int64_t c;
		ap1 += av;
		bp1 += bv;
		c = op_mul(a,b);
		sum = op_add(sum, c);
	    }
	    ap += au;
	    bp += bu;
	}
	if (square_root) sum = (int64_t) sqrt((double)sum);
	write_int64(ct, cp, sum);
    }
}

// wrapper to mulsum that avoid the square root calculation
static void mulsum1(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t bt, byte_t* bp, int bu, int bv,
		     matrix_type_t ct, byte_t* cp,
		     size_t n, size_t m)
{
    mulsum(use_vector, FALSE, at, ap, au, av, bt, bp, bu, bv, ct, cp, n, m);
}

// Multiply A and B element-wise and sum the result
ERL_NIF_TERM matrix_mulsum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, b;
    matrix_type_t ct;
    scalar_t c;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &b))
	return enif_make_badarg(env);
    if ((a.rowmajor == b.rowmajor) && ((a.n != b.n) || (a.m != b.m)))
	return enif_make_badarg(env);
    else if ((a.rowmajor != b.rowmajor) && ((a.n != b.m) || (a.m != b.n)))
	return enif_make_badarg(env);

    ct = combine_type(a.type, b.type);

    matrix_rr_lock(&a,&b);
    s_apply2(mulsum1, &a, &b, ct, c.data);
    matrix_rr_unlock(&a,&b);

    return read_term(env, ct, c.data);
}

// square root of the sum of squares of the activation of
// region rnxrm stepping ru in row and rv in column -direction.
//
static void l2pool(matrix_type_t at, byte_t* ap, int au, int av,
		   size_t an, size_t am,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t cn, size_t cm,
		   int ru, int rv, size_t rn, size_t rm)
{
    UNUSED(an);
    UNUSED(am);

    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);
    ru = size_of_array(at,au)*ru;
    rv = size_of_array(at,av)*rv;

    while(cn--) {
	byte_t* cp1 = cp;
	byte_t* ap1 = ap;
	size_t m = cm;
	while(m--) {
	    mulsum(FALSE, TRUE,
		   at, ap1, au, av,
		   at, ap1, au, av,
		   ct, cp1, rn, rm);
	    cp1 += cv;
	    ap1 += rv;
	}
	cp += cu;
	ap += ru;
    }
}


// l2pool
// calculate sqrt(sum(Aij*Aij)) over
// region Rn, Rm, with stride step Ru, Rv
//
ERL_NIF_TERM matrix_l2pool(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, c;
    unsigned int rn, rm, ru, rv;
    size_t cn, cm;

    if (!enif_get_uint(env, argv[0], &rn))  // number of rows in region
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &rm))  // number of columns in region
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &ru) || (ru == 0))  // region row step
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[3], &rv) || (rv == 0))  // column step
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[4], &a))
	return enif_make_badarg(env);

    if (!a.rowmajor) {
	unsigned int t;
	t = ru; ru = rv; rv = t;  // swap ru and rv
	t = rn; rn = rm; rm = t;  // swap rn and rm
    }

    if ((a.n < rn) || (a.m < rm))
	return enif_make_badarg(env);
    cn = ((a.n-rn) / ru)+1;
    cm = ((a.m-rm) / rv)+1;

    if (argc == 5) {
	ERL_NIF_TERM bin;

	if (!create_matrix(env,cn,cm,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);
	matrix_r_lock(&a);
	l2pool(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
	       c.type, c.first, c.nstep, c.mstep, c.n, c.m,
	       ru, rv, rn, rm);
	matrix_r_unlock(&a);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else { // argc == 6 with destination matrix
	if (!get_w_matrix(env, argv[5], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((c.n != cn) || (c.m != cm)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((c.n != cm) || (c.m != cn)))
	    return enif_make_badarg(env);

	if (a.rowmajor == c.rowmajor) {
	    if ((c.n != cn) || (c.m != cm))
		return enif_make_badarg(env);

	    matrix_rw_lock(&a,&c);
	    l2pool(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		   c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		   ru, rv, rn, rm);
	    matrix_rw_unlock(&a,&c);
	}
	else {
	    if ((c.n != cm) || (c.m != cn))
		return enif_make_badarg(env);

	    matrix_rw_lock(&a,&c);
	    l2pool(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		   c.type, c.first, c.mstep, c.nstep, c.m, c.n,
		   ru, rv, rn, rm);
	    matrix_rw_unlock(&a,&c);
	}
	return argv[5];
    }
}

static void mmax(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t ct, byte_t* cp,
		 size_t n, size_t m)
{
    UNUSED(use_vector);

    // FIXME! acceleration at=bt
    au = size_of_array(at,au);
    av = size_of_array(at,av);

    if (is_float(ct)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t mx = read_af(ap);
	while(n--) {
	    byte_t* ap1 = ap;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_af(ap1);
		ap1 += av;
		mx = op_max(mx,a);
	    }
	    ap += au;
	}
	write_float64(ct, cp, mx);
    }
    else if (is_complex(ct)) {
	complex128_t (*read_af)(byte_t*) = read_complex128_func[at];
	complex128_t mx = read_af(ap);
	while(n--) {
	    byte_t* ap1 = ap;
	    size_t m1 = m;
	    while(m1--) {
		complex128_t a = read_af(ap1);
		ap1 += av;
		mx = complex128_max(mx, a);
	    }
	    ap += au;
	}
	write_complex128(ct, cp, mx);
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t mx = read_af(ap);
	while(n--) {
	    byte_t* ap1 = ap;
	    size_t m1 = m;
	    while(m1--) {
		int64_t a = read_af(ap1);
		ap1 += av;
		mx = op_max(mx, a);
	    }
	    ap += au;
	}
	write_int64(ct, cp, mx);
    }
}


static void maxpool(matrix_type_t at, byte_t* ap, int au, int av,
		    size_t an, size_t am,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t cn, size_t cm,
		    int ru, int rv, size_t rn, size_t rm)
{
    UNUSED(an);
    UNUSED(am);

    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);
    ru = size_of_array(at,au)*ru;
    rv = size_of_array(at,av)*rv;

    while(cn--) {
	byte_t* cp1 = cp;
	byte_t* ap1 = ap;
	size_t m = cm;
	while(m--) {
	    mmax(FALSE,
		 at, ap1, au, av,
		 ct, cp1, rn, rm);
	    cp1 += cv;
	    ap1 += rv;
	}
	cp += cu;
	ap += ru;
    }
}

// maxpool
// calculate max over the region Rn, Rm, with stride step Ru, Rv
//
ERL_NIF_TERM matrix_maxpool(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a, c;
    unsigned int rn, rm, ru, rv;
    size_t cn, cm;

    if (!enif_get_uint(env, argv[0], &rn))  // number of rows in region
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &rm))  // number of columns in region
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &ru) || (ru == 0))  // region row step
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[3], &rv) || (rv == 0))  // column step
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[4], &a))
	return enif_make_badarg(env);

    if (!a.rowmajor) {
	unsigned int t;
	t = ru; ru = rv; rv = t;  // swap ru and rv
	t = rn; rn = rm; rm = t;  // swap rn and rm
    }

    if ((a.n < rn) || (a.m < rm))
	return enif_make_badarg(env);
    cn = ((a.n-rn) / ru)+1;
    cm = ((a.m-rm) / rv)+1;

    if (argc == 5) {
	ERL_NIF_TERM bin;

	if (!create_matrix(env,cn,cm,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);
	matrix_r_lock(&a);
	maxpool(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		ru, rv, rn, rm);
	matrix_r_unlock(&a);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else { // argc == 6 with destination matrix
	if (!get_w_matrix(env, argv[5], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((c.n != cn) || (c.m != cm)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((c.n != cm) || (c.m != cn)))
	    return enif_make_badarg(env);

	if (a.rowmajor == c.rowmajor) {
	    if ((c.n != cn) || (c.m != cm))
		return enif_make_badarg(env);

	    matrix_rw_lock(&a,&c);
	    maxpool(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		    c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		    ru, rv, rn, rm);
	    matrix_rw_unlock(&a,&c);
	}
	else {
	    if ((c.n != cm) || (c.m != cn))
		return enif_make_badarg(env);

	    matrix_rw_lock(&a,&c);
	    maxpool(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		    c.type, c.first, c.mstep, c.nstep, c.m, c.n,
		    ru, rv, rn, rm);
	    matrix_rw_unlock(&a,&c);
	}
	return argv[5];
    }
}


// sum of squares of the activation of

static void filter(matrix_type_t at, byte_t* ap, int au, int av,
		   size_t an, size_t am,
		   matrix_type_t wt, byte_t* wp, int wu, int wv,
		   size_t wn, size_t wm,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t cn, size_t cm,
		   int ru, int rv)
{
    UNUSED(an);
    UNUSED(am);

    cu = size_of_array(ct,cu);
    cv = size_of_array(ct,cv);
    ru = size_of_array(at,au)*ru;
    rv = size_of_array(at,av)*rv;

    while(cn--) {
	byte_t* cp1 = cp;
	byte_t* ap1 = ap;
	size_t m = cm;
	while(m--) {
	    mulsum(FALSE, FALSE,
		   wt, wp,  wu, wv,
		   at, ap1, au, av,
		   ct, cp1, wn, wm);
	    cp1 += cv;
	    ap1 += rv;
	}
	cp += cu;
	ap += ru;
    }
}

// maxpool
// filter a region with weight W, with stride step Ru, Rv
//
ERL_NIF_TERM matrix_filter(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t w, a, c;
    unsigned int wn, wm;
    unsigned int ru, rv;
    int wu, wv;
    size_t cn, cm;

    if (!get_matrix(env, argv[0], &w))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &ru) || (ru == 0))  // region row step
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &rv) || (rv == 0))  // column step
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[3], &a))
	return enif_make_badarg(env);

    if (a.rowmajor == w.rowmajor) {
	wn = w.n; wm = w.m; wu = w.nstep; wv = 1;
    }
    else {
	wn = w.m; wm = w.n; wu = 1, wv = w.nstep;
    }

    if ((a.n < wn) || (a.m < wm))
	return enif_make_badarg(env);
    cn = ((a.n-wn) / ru)+1;
    cm = ((a.m-wm) / rv)+1;

    if (argc == 4) {
	ERL_NIF_TERM bin;

	if (!create_matrix(env,cn,cm,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);
	matrix_r_lock(&a);
	filter(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
	       w.type, w.first, wu, wv, wn, wm,
	       c.type, c.first, c.nstep, c.mstep, c.n, c.m,
	       ru, rv);
	matrix_r_unlock(&a);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else { // argc == 5 with destination matrix
	if (!get_w_matrix(env, argv[4], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((c.n != cn) || (c.m != cm)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((c.n != cm) || (c.m != cn)))
	    return enif_make_badarg(env);

	if (a.rowmajor == c.rowmajor) {
	    if ((c.n != cn) || (c.m != cm))
		return enif_make_badarg(env);

	    matrix_rw_lock(&a,&c);
	    filter(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		   w.type, w.first, wu, wv, wn, wm,
		   c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		   ru, rv);
	    matrix_rw_unlock(&a,&c);
	}
	else {
	    if ((c.n != cm) || (c.m != cn))
		return enif_make_badarg(env);

	    matrix_rw_lock(&a,&c);
	    filter(a.type, a.first, a.mstep, a.nstep, a.m, a.n,
		   w.type, w.first, wv, wu, wm, wn,
		   c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		   rv, ru);
	    matrix_rw_unlock(&a,&c);
	}
	return argv[4];
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
	matrix_r_lock(&a);
	m_apply1(copy1, &a, &c);
	matrix_r_unlock(&a);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else if (argc == 2) {
	if (!get_w_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);

	// copy into C
	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rw_lock(&a, &c);
	m_apply1(copy1, &a, &c);
	matrix_rw_unlock(&a, &c);
	return argv[1];
    }
    else {
	unsigned int repeat_m;
	unsigned int repeat_n;

	if (!get_w_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);

	if (!enif_get_uint(env, argv[2], &repeat_m))
	    return enif_make_badarg(env);
	if (!enif_get_uint(env, argv[3], &repeat_n))
	    return enif_make_badarg(env);

	matrix_rw_lock(&a, &c);
	if (a.rowmajor == c.rowmajor)
	    tile(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
		 c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		 repeat_m, repeat_n);
	else
	    tile(a.type, a.first, a.mstep, a.nstep, a.n, a.m,
		 c.type, c.first, c.nstep, c.mstep, c.n, c.m,
		 repeat_m, repeat_n);

	matrix_rw_unlock(&a, &c);
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
    if (c.ptr == 0) // must be resource matrix!
	return enif_make_badarg(env);

    matrix_rw_lock(&a, &c);
    if (a.rowmajor == c.rowmajor)
	fill(a.type, a.first, a.nstep, a.mstep, a.n, a.m,
	     c.type, c.first, c.nstep, c.mstep, c.n, c.m);
    else
	fill(a.type, a.first, a.mstep, a.nstep, a.n, a.m,
	     c.type, c.first, c.nstep, c.mstep, c.n, c.m);
    matrix_rw_unlock(&a, &c);

    return argv[1];
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

    matrix_r_lock(&a);
    sigmoid(a.type, a.first, a.nstep, a.mstep,
	    c.type, c.first, c.nstep, c.mstep,
	    c.n, c.m);
    matrix_r_unlock(&a);
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// sigmoid_prime a matrix use the output version d(y) = y*(1-y)
ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[])
{
    matrix_t a,y;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &y))
	return enif_make_badarg(env);

    if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	return enif_make_badarg(env);

    matrix_r_lock(&y);
    sigmoid_prime1(y.type, y.first, y.nstep, y.mstep,
		  c.type, c.first, c.nstep, c.mstep,
		  c.n, c.m);
    matrix_r_unlock(&y);
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// rectifier a matrix
ERL_NIF_TERM matrix_relu(ErlNifEnv* env, int argc,
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

    matrix_r_lock(&a);
    rectifier(a.type, a.first, a.nstep, a.mstep,
	      c.type, c.first, c.nstep, c.mstep,
	      c.n, c.m);
    matrix_r_unlock(&a);
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// n m and type
ERL_NIF_TERM matrix_identity(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    unsigned int n, m, k, type;
    unsigned int i;
    matrix_t c;
    ERL_NIF_TERM bin;
    size_t elem_size;
    size_t row_step;
    size_t col_step;
    byte_t* cp;
    scalar_t one;
    UNUSED(argc);

    if (!enif_get_uint(env, argv[0], &n))  // rows
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &m))   // columns
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &type)) // type
	return enif_make_badarg(env);
    if (type >= NUM_TYPES)
	return enif_make_badarg(env);
    if (!create_matrix(env,n,m,TRUE,type,&c,&bin))
	return enif_make_badarg(env);

    memset(c.base, 0, c.size);  // set to zero

    elem_size     = element_size(type);
    col_step = elem_size;
    row_step = size_of_array(type, c.nstep);
    // format the 1 element
    if (is_float(type))
	write_float64(type, one.data, 1.0);
    else if (is_complex(type))
	write_complex128(type, one.data, CMPLX(1.0,0.0));
    else
	write_int64(type, one.data, 1);
    k = (n<m) ? n : m;
    cp = c.first;
    for (i = 0; i < k; i++) {
	memcpy(cp, one.data, elem_size);
	cp += row_step;
	cp += col_step;
    }
    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
}

// matrix_apply1 (func, Src [,Dst])
ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t a, c;
    unary_operation_t op;
    UNUSED(argc);

    if (!enif_is_atom(env, argv[0]))
	return enif_make_badarg(env);
    if (argv[0] == ATOM(zero))               op = ZERO;
    else if (argv[0] == ATOM(one))           op = ONE;
    else if (argv[0] == ATOM(copy))          op = COPY;
    else if (argv[0] == ATOM(negate))        op = NEGATE;
    else if (argv[0] == ATOM(sigmoid))       op = SIGMOID;
    else if (argv[0] == ATOM(sigmoid_prime)) op = SIGMOID_PRIME;
    else if (argv[0] == ATOM(sigmoid_prime1)) op = SIGMOID_PRIME1;
    else if (argv[0] == ATOM(relu))          op = RELU;
    else if (argv[0] == ATOM(relu_prime))    op = RELU_PRIME;
    else if (argv[0] == ATOM(leaky_relu))    op = LEAKY_RELU;
    else if (argv[0] == ATOM(leaky_relu_prime)) op = LEAKY_RELU_PRIME;
    else if (argv[0] == ATOM(tanh))          op = TANH;
    else if (argv[0] == ATOM(tanh_prime))    op = TANH_PRIME;
    else if (argv[0] == ATOM(tanh_prime1))   op = TANH_PRIME1;
    else if (argv[0] == ATOM(softplus))      op = SOFTPLUS;
    else if (argv[0] == ATOM(softplus_prime)) op = SOFTPLUS_PRIME;
    else if (argv[0] == ATOM(uniform))       op = UNIFORM;
    else if (argv[0] == ATOM(normal))        op = NORMAL;
    else if (argv[0] == ATOM(exp))           op = EXP;
    else return enif_make_badarg(env);

    if (!get_matrix(env, argv[1], &a))
	return enif_make_badarg(env);

    if (argc == 2) {
	ERL_NIF_TERM bin;
	if (!create_matrix(env,a.n,a.m,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);
	matrix_r_lock(&a);
	apply1(op,
	       a.type, a.first, a.nstep, a.mstep,
	       c.type, c.first, c.nstep, c.mstep,
	       c.n, c.m);
	matrix_r_unlock(&a);
	return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
    }
    else {
	if (!get_w_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rw_lock(&a, &c);
	if (c.rowmajor == a.rowmajor)
	    apply1(op,
		   a.type, a.first, a.nstep, a.mstep,
		   c.type, c.first, c.nstep, c.mstep,
		   c.n, c.m);
	else
	    apply1(op,
		   a.type, a.first, a.mstep, a.nstep,
		   c.type, c.first, c.nstep, c.mstep,
		   c.n, c.m);
	matrix_rw_unlock(&a, &c);
	return argv[1];
    }
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
	    matrix_r_lock(&a);
	    argmax(a.type, a.first,a.nstep,a.mstep,
		   (int32_t*)c.first, 1,
		   a.n,a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // argmax for each row is returned (as a column)
	    if (!create_matrix(env,1,a.n,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    argmax(a.type, a.first,1,a.nstep,
		   (int32_t*)c.first, 1,
		   a.m,a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    // argmax for each column is returned (as a row)
	    if (!create_matrix(env,1,a.n,TRUE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    argmax(a.type, a.first,1,a.nstep,
		   (int32_t*)c.first,1,
		   a.m,a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // argmax for each row is returned (as a column)
	    if (!create_matrix(env,1,a.m,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    argmax(a.type, a.first,a.nstep,a.mstep,
		   (int32_t*)c.first,1,
		   a.n,a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
}

// find max in matrix
ERL_NIF_TERM matrix_max(ErlNifEnv* env, int argc,
			const ERL_NIF_TERM argv[])
{
    matrix_t a;
    int axis;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (argc == 1)
	axis = -1;
    else {
	if (!enif_get_int(env, argv[1], &axis))
	    return enif_make_badarg(env);
	if ((axis < -1) || (axis > 1))
	    return enif_make_badarg(env);
    }

    if (a.rowmajor) {
	if (axis == -1) {
	    scalar_t max_v;
	    matrix_r_lock(&a);
	    t_max(a.type, a.first, a.nstep, a.mstep,
		  a.type, max_v.data, 0,
		  a.n,a.m);
	    matrix_r_unlock(&a);
	    return read_term(env, a.type, max_v.data);
	}
	else if (axis == 0) {
	    // max for each column is returned (as a row)
	    if (!create_matrix(env,1,a.m,TRUE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_max(a.type, a.first, a.nstep, a.mstep,
		  c.type, c.first, 1,
		  a.n,a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else { // axis == 1
	    // max for each row is returned (as a column)
	    if (!create_matrix(env,1,a.n,FALSE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_max(a.type, a.first, a.mstep, a.nstep,
		  c.type, c.first, 1,
		  a.m,a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
    else { // !a.rowmajor
	if (axis == -1) {
	    scalar_t max_v;
	    matrix_r_lock(&a);
	    t_max(a.type, a.first, a.mstep, a.nstep,
		  a.type, max_v.data, 0,
		  a.m,a.n);
	    matrix_r_unlock(&a);
	    return read_term(env, a.type, max_v.data);
	}
	else if (axis == 0) {
	    // max for each column is returned (as a row)
	    if (!create_matrix(env,1,a.n,TRUE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_max(a.type, a.first, a.mstep, a.nstep,
		  c.type, c.first, 1,
		  a.m, a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // max for each row is returned (as a column)
	    if (!create_matrix(env,1,a.m,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_max(a.type, a.first, a.nstep, a.mstep,
		  c.type, c.first, 1,
		  a.n, a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
}

// find min in matrix
ERL_NIF_TERM matrix_min(ErlNifEnv* env, int argc,
			const ERL_NIF_TERM argv[])
{
    matrix_t a;
    int axis;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (argc == 1)
	axis = -1;
    else {
	if (!enif_get_int(env, argv[1], &axis))
	    return enif_make_badarg(env);
	if ((axis < -1) || (axis > 1))
	    return enif_make_badarg(env);
    }

    if (a.rowmajor) {
	if (axis == -1) {
	    scalar_t min_v;
	    matrix_r_lock(&a);
	    t_min(a.type, a.first, a.nstep, a.mstep,
		  a.type, min_v.data, 0,
		  a.n,a.m);
	    matrix_r_unlock(&a);
	    return read_term(env, a.type, min_v.data);
	}
	else if (axis == 0) {
	    // min for each column is returned (as a row)
	    if (!create_matrix(env,1,a.m,TRUE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_min(a.type, a.first, a.nstep, a.mstep,
		  c.type, c.first, 1,
		  a.n,a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // min for each row is returned (as a column)
	    if (!create_matrix(env,1,a.n,FALSE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_min(a.type, a.first, a.mstep, a.nstep,
		  c.type, c.first, 1,
		  a.m,a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
    else { // !a.rowmajor
	if (axis == -1) {
	    scalar_t min_v;
	    matrix_r_lock(&a);
	    t_min(a.type, a.first, a.mstep, a.nstep,
		  a.type, min_v.data, 0,
		  a.m,a.n);
	    matrix_r_unlock(&a);
	    return read_term(env, a.type, min_v.data);
	}
	else if (axis == 0) {
	    // min for each column is returned (as a row)
	    if (!create_matrix(env,1,a.n,TRUE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_min(a.type, a.first, a.mstep, a.nstep,
		  c.type, c.first, 1,
		  a.m, a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // min for each row is returned (as a column)
	    if (!create_matrix(env,1,a.m,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_min(a.type, a.first, a.nstep, a.mstep,
		  c.type, c.first, 1,
		  a.n, a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
}


// find sum in matrix
ERL_NIF_TERM matrix_sum(ErlNifEnv* env, int argc,
			const ERL_NIF_TERM argv[])
{
    matrix_t a;
    int axis;
    matrix_t c;
    ERL_NIF_TERM bin;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (argc == 1)
	axis = -1;
    else {
	if (!enif_get_int(env, argv[1], &axis))
	    return enif_make_badarg(env);
	if ((axis < -1) || (axis > 1))
	    return enif_make_badarg(env);
    }

    if (a.rowmajor) {
	if (axis == -1) {
	    scalar_t sum_v;
	    matrix_r_lock(&a);
	    t_sum(a.type, a.first, a.nstep, a.mstep,
		  a.type, sum_v.data, 0,
		  a.n,a.m);
	    matrix_r_unlock(&a);
	    return read_term(env, a.type, sum_v.data);
	}
	else if (axis == 0) {
	    // sum for each column is returned (as a row)
	    if (!create_matrix(env,1,a.m,TRUE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_sum(a.type, a.first, a.nstep, a.mstep,
		  c.type, c.first, 1,
		  a.n,a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // sum for each row is returned (as a column)
	    if (!create_matrix(env,1,a.n,FALSE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_sum(a.type, a.first, a.mstep, a.nstep,
		  c.type, c.first, 1,
		  a.m,a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
    else { // !a.rowmajor
	if (axis == -1) {
	    scalar_t sum_v;
	    matrix_r_lock(&a);
	    t_sum(a.type, a.first, a.mstep, a.nstep,
		  a.type, sum_v.data, 0,
		  a.m,a.n);
	    matrix_r_unlock(&a);
	    return read_term(env, a.type, sum_v.data);
	}
	else if (axis == 0) {
	    // sum for each column is returned (as a row)
	    if (!create_matrix(env,1,a.n,TRUE,a.type,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_sum(a.type, a.first, a.mstep, a.nstep,
		  c.type, c.first, 1,
		  a.m, a.n);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
	else {
	    // min for each row is returned (as a column)
	    if (!create_matrix(env,1,a.m,FALSE,INT32,&c,&bin))
		return enif_make_badarg(env);
	    matrix_r_lock(&a);
	    t_sum(a.type, a.first, a.nstep, a.mstep,
		  c.type, c.first, 1,
		  a.n, a.m);
	    matrix_r_unlock(&a);
	    return make_matrix(env,c.n,c.m,c.rowmajor,c.type,&c,bin);
	}
    }
}



// transpose data rather then toggle rowmajor
ERL_NIF_TERM matrix_transpose_data(ErlNifEnv* env, int argc,
				   const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    ERL_NIF_TERM bin;

    if (!get_matrix(env,argv[0],&a))
	return enif_make_badarg(env);

    if (argc == 1) {
	if (!create_matrix(env,a.m,a.n,a.rowmajor,a.type,&c,&bin))
	    return enif_make_badarg(env);

	matrix_r_lock(&a);
	m_apply1(transpose1, &a, &c);
	matrix_r_unlock(&a);
	return make_matrix(env, c.n, c.m, c.rowmajor, c.type, &c, bin);
    }
    else {
	if (!get_w_matrix(env,argv[1],&c))
	    return enif_make_badarg(env);

	if ((a.rowmajor == c.rowmajor) && ((a.n != c.n) || (a.m != c.m)))
	    return enif_make_badarg(env);
	else if ((a.rowmajor != c.rowmajor) && ((a.n != c.m) || (a.m != c.n)))
	    return enif_make_badarg(env);

	matrix_rw_lock(&a,&c);
	m_apply1(transpose1, &a, &c);
	matrix_rw_unlock(&a,&c);
	return argv[1];
    }
}

static void matrix_dtor(ErlNifEnv* env, matrix_t* mat)
{
    UNUSED(env);
    if (mat->rw_lock)
	enif_rwlock_destroy(mat->rw_lock);
    if (mat->base)
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
    LOAD_ATOM(sigmoid_prime1);
    LOAD_ATOM(relu);
    LOAD_ATOM(relu_prime);
    LOAD_ATOM(leaky_relu);
    LOAD_ATOM(leaky_relu_prime);
    LOAD_ATOM(tanh);
    LOAD_ATOM(tanh_prime);
    LOAD_ATOM(tanh_prime1);
    LOAD_ATOM(softplus);
    LOAD_ATOM(softplus_prime);
    LOAD_ATOM(negate);
    LOAD_ATOM(uniform);
    LOAD_ATOM(normal);
    LOAD_ATOM(zero);
    LOAD_ATOM(one);
    LOAD_ATOM(copy);
    LOAD_ATOM(true);
    LOAD_ATOM(false);
    LOAD_ATOM(undefined);
    LOAD_ATOM(exp);

    matrix_r = enif_open_resource_type(env, 0, "matrix",
				       (ErlNifResourceDtor*) matrix_dtor,
				       ERL_NIF_RT_CREATE, &tried);

    enif_tsd_key_create("rand", &matrix_k);

    *priv_data = 0;
    return 0;
}

static int matrix_upgrade(ErlNifEnv* env, void** priv_data,
			  void** old_priv_data,
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
