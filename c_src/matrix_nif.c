//
// Matrix operations
//
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

#include "erl_nif.h"

//#define NIF_TRACE
//#define DEBUG

#ifdef DEBUG
#include <stdio.h>
#define DBG(...) printf(__VA_ARGS__)
#define BADARG(env) printf("matrix_nif.c: badarg line=%d\r\n", __LINE__), enif_make_badarg((env))
#else
#define DBG(...)
#define BADARG(env) enif_make_badarg((env))
#endif

#define EXCP_ERROR_N(env, arg_num, str)  raise_exception((env), ATOM(error),  (arg_num), (str), __FILE__, __LINE__)
#define EXCP_NOTSUP_N(env, arg_num, str) raise_exception((env), ATOM(notsup), (arg_num), (str), __FILE__, __LINE__)
#define EXCP_BADARG_N(env, arg_num, str) raise_exception((env), ATOM(badarg), (arg_num), (str), __FILE__, __LINE__)

#include "matrix_types.h"

typedef enum {
    ZERO           = 0,
    ONE            = 1,
    COPY           = 2,
    NEGATE         = 3,
    SIGMOID        = 4,
    SIGMOID_PRIME  = 5,
    SIGMOID_PRIME1 = 6,
    SOFTPLUS       = 7,
    SOFTPLUS_PRIME = 8,
    RELU           = 9,
    RELU_PRIME     = 10,
    LEAKY_RELU     = 11,
    LEAKY_RELU_PRIME = 12,
    TANH           = 13,
    TANH_PRIME     = 14,
    TANH_PRIME1    = 15,
    EXP            = 16,
    UNIFORM        = 17,
    NORMAL         = 18,
    RECIPROCAL     = 19,
    SQRT           = 20,
    LAST_UNARY_OP  = SQRT,
} unary_operation_t;

#define NUM_UNARYOP (LAST_UNARY_OP+1)

typedef enum {
    BNOT = 21,
    LAST_IUNARY_OP = BNOT,
} iunary_operation_t;

#define NUM_IUNARYOP (LAST_IUNARY_OP+1)

#define OPT_ABS      0x0001   // max/min/argmax/argmin/sort
#define OPT_DESCEND  0x0002   // sort
#define OPT_REAL     0x0004   // sort real part only
#define OPT_IMAG     0x0008   // sort imag part only

#define CMP_FGT(a, b) \
    (((opts)&OPT_ABS) ? (fabs((a))>fabs((b))) : ((a)>(b)))
#define CMP_IGT(a, b) \
    (((opts)&OPT_ABS) ? (labs((a))>labs((b))) : ((a)>(b)))

#define CMP_FLT(a, b) \
    (((opts)&OPT_ABS) ? (fabs((a))<fabs((b))) : ((a)<(b)))
#define CMP_ILT(a, b) \
    (((opts)&OPT_ABS) ? (labs((a))<labs((b))) : ((a)<(b)))

#define FCMP(a,b) (((opts)&OPT_DESCEND) ? CMP_FGT((a),(b)) : CMP_FLT((a),(b)))
#define ICMP(a,b) (((opts)&OPT_DESCEND) ? CMP_IGT((a),(b)) : CMP_ILT((a),(b)))

typedef enum {
    ADD = 0,
    SUB = 1,
    MUL = 2,
    MIN = 3,
    MAX = 4,
    DIV = 5,
    REM = 6,
    LAST_BINARY_OP = REM
} binary_operation_t;

#define NUM_BINOP (LAST_BINARY_OP+1)

typedef enum {
    BAND = 0,
    BOR  = 1,
    BXOR = 2,
    LAST_IBINARY_OP = BXOR,
} ibinary_operation_t;

#define NUM_IBINOP (LAST_IBINARY_OP+1)

typedef enum {
    EQ  = 0,
    LT  = 1,
    LTE = 2,
    LAST_COMPARE_OP = LTE
} compare_operation_t;

#define NUM_CMPOP (LAST_COMPARE_OP+1)

#define ATOM(name) atm_##name

#define DECL_ATOM(name) \
    ERL_NIF_TERM atm_##name = 0

#define LOAD_ATOM(env,name)			\
    atm_##name = enif_make_atom((env),#name)

#define LOAD_ATOM_STRING(env,name,string)	\
    atm_##name = enif_make_atom((env),string)

#define CAT_HELPER3(p,x,y) p ## x ## y
#define CAT3(p,x,y) CAT_HELPER3(p,x,y)

#define CAT_HELPER2(x,y) x ## y
#define CAT2(x,y) CAT_HELPER2(x,y)

#define MT_NAME(p,x) CAT3(p,NAME,x)
#define VTYPE        CAT2(v,TYPE)
#define VTYPE_R      CAT2(v,TYPE_R)
#define VTYPE_ZERO   CAT3(v,TYPE,_zero)
#define VTYPE_CONST(name) CAT3(v,TYPE,_const)(name)

#define TYPE_ZERO   CAT2(TYPE,_zero)

#define swap_array_elem(a,i,j) do {			\
	typeof((a)[0]) _tmp = (a)[i];			\
	(a)[i] = a[j];					\
	(a)[j] = _tmp;					\
    } while(0)

#define swap(a,b) do {				\
	typeof((a)) _tmp = (a);			\
	(a) = (b);				\
	(b) = _tmp;				\
    } while(0)

typedef struct _matrix_t {
    size_t    n;              // #elements in n direction
    size_t    m;              // #elements in m direction
    ssize_t   n_stride;       // #bytes in n direction (row stride)
    ssize_t   m_stride;       // #bytes per column, element size (column stride)
    ssize_t   k_stride;       // #bytes per component (vector element size)
    uintptr_t offset;         // byte offset to first element
    matrix_type_t type;
    bool_t       rowmajor;    // stored row-by-row
    ErlNifRWLock* rw_lock;    // make sure we can read/write "safe"
    size_t size;              // allocated memory size
    byte_t* base;             // allocated memory
    byte_t* data;             // aligned data
    byte_t* first;            // pointer to first element within data
    uintptr_t ptr;            // resource pointer (may be self)
    ERL_NIF_TERM parent;      // resource or real binary
    byte_t  smem[sizeof(float64_t)*ALLOC_COMPONENTS+VSIZE-1];
} matrix_t;
// FIXME  ((packed)) do not work with the above declaration!!!???
// __attribute__ ((packed)) matrix_t;

// used for add/sub/times ...
typedef void (*mt_binary_func_t)(byte_t* ap, int au, int av,
				 byte_t* bp, int bu, int bv,
				 byte_t* cp, int cu, int cv,
				 size_t n, size_t m);

typedef void (*mt_unary_func_t)(byte_t* ap, int au, int av,
				byte_t* cp, int cu, int cv,
				size_t n, size_t m);

typedef void (*mtv_binary_func_t)(void* ap, int au,
				  void* bp, int bu,
				  void* cp, int cu,
				  size_t n, size_t m);

typedef void (*mtv_unary_func_t)(void* ap, int au,
				 void* cp, int cu,
				 size_t n, size_t m);

typedef void (*mt_mulop_func_t)(byte_t* ap, int au, int av,size_t an, size_t am,
				byte_t* bp, int bu, int bv,size_t bn, size_t bm,
				byte_t* cp, int cu, int cv);

typedef void (*mtv_mulop_func_t)(byte_t* ap,int au,size_t an, size_t am,
				 byte_t* bp,int bu,size_t bn, size_t bm,
				 byte_t* cp,int cu,int cv);

typedef void (*mt_kmulop_func_t)(byte_t* ap,int au,int av,size_t an,size_t am,
				 byte_t* bp,int bu,int bv,size_t bn,size_t bm,
				 int32_t* kp,int kv,size_t km,
				 byte_t* cp,int cu,int cv);

typedef void (*mtv_kmulop_func_t)(byte_t* ap,int au,size_t an, size_t am,
				  byte_t* bp,int bu,size_t bn, size_t bm,
				  int32_t* kp, int kv, size_t km,
				  byte_t* cp,int cu,int cv);

// typedef vector_t (*vector_unary_func_t)(vector_t a);
// typedef vector_t (*vector_binary_func_t)(vector_t a, vector_t b);

// Global data (store in env?)
static ErlNifResourceType* matrix_res;
static ErlNifTSDKey matrix_k;


ERL_NIF_TERM raise_exception(ErlNifEnv* env, ERL_NIF_TERM id, int arg_num, char* explanation, char* file, int line)
{
    ERL_NIF_TERM file_info, exception;
    char *error_msg = explanation;
    UNUSED(file);
    UNUSED(line);

//  enif_fprintf(stderr, "exeception %s:%d: arg_num=%d, msg=%s\n",
//	    file, line, arg_num, error_msg);

    /* Make the data for exception */
    file_info = enif_make_new_map(env);
//    enif_make_map_put(env, file_info,
//                      enif_make_atom(env,"c_file_name"),
//                      enif_make_string(env, file, (ERL_NIF_LATIN1)),
//                      &file_info);
//    enif_make_map_put(env, file_info,
//                      enif_make_atom(env,"c_file_line_num"),
//                      enif_make_int(env, line),
//                      &file_info);
//    enif_make_map_put(env, file_info,
//                      enif_make_atom(env,"c_function_arg_num"),
//                      enif_make_int(env, arg_num+1),  // +1 for erlang
//                      &file_info);
    enif_make_map_put(env, file_info,
                      enif_make_atom(env,"argument"),
                      enif_make_int(env, arg_num+1),  // +1 for erlang
                      &file_info);
    exception =
        enif_make_tuple3(env,
                         id,
                         file_info,
                         enif_make_string(env, error_msg, (ERL_NIF_LATIN1))
                         );
    return enif_raise_exception(env, exception);
}


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

// Dirty optional since 2.7 and mandatory since 2.12
#if (ERL_NIF_MAJOR_VERSION > 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION >= 7))
#ifdef USE_DIRTY_SCHEDULER
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(ERL_NIF_DIRTY_JOB_CPU_BOUND)}
#define NIF_DIRTY_FUNC(name,arity,fptr) {(name),(arity),(fptr),(ERL_NIF_DIRTY_JOB_CPU_BOUND)}
#else
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr),(0)}
#define NIF_DIRTY_FUNC(name,arity,fptr) {(name),(arity),(fptr),(ERL_NIF_DIRTY_JOB_CPU_BOUND)}
#endif
#else
#define NIF_FUNC(name,arity,fptr) {(name),(arity),(fptr)}
#define NIF_DIRTY_FUNC(name,arity,fptr) {(name),(arity),(fptr)}
#endif

#define NIF_LIST \
    NIF("create_",      5, matrix_create) \
    NIF("native_vector_width", 1, matrix_native_vector_width) \
    NIF("preferred_vector_width", 1, matrix_preferred_vector_width) \
    NIF("identity_",    3, matrix_identity) \
    NIF("size",         1, matrix_size) \
    NIF("element",      2, matrix_element) \
    NIF("element",      3, matrix_element) \
    NIF("setelement",   4, matrix_setelement) \
    NIF("add",          2, matrix_add) \
    NIF("add",          3, matrix_add) \
    NIF("subtract",     2, matrix_subtract) \
    NIF("subtract",     3, matrix_subtract) \
    NIF("times",        2, matrix_times) \
    NIF("times",        3, matrix_times) \
    NIF("divide",       2, matrix_divide) \
    NIF("divide",       3, matrix_divide) \
    NIF("remainder",    2, matrix_remainder) \
    NIF("remainder",    3, matrix_remainder) \
    NIF("ktimes_",       3, matrix_ktimes) \
    NIF("ktimes_",       4, matrix_ktimes) \
    NIF("multiply",     2, matrix_multiply) \
    NIF("multiply",     3, matrix_multiply) \
    NIF("kmultiply_",    3, matrix_kmultiply) \
    NIF("kmultiply_",    4, matrix_kmultiply) \
    NIF("topk",         2, matrix_topk) \
    NIF("negate",       1, matrix_negate) \
    NIF("negate",       2, matrix_negate) \
    NIF("reciprocal",   1, matrix_reciprocal) \
    NIF("reciprocal",   2, matrix_reciprocal) \
    NIF("sqrt",         1, matrix_sqrt) \
    NIF("sqrt",         2, matrix_sqrt) \
    NIF("mulsum",       2, matrix_mulsum) \
    NIF("l2pool",       5, matrix_l2pool) \
    NIF("l2pool",       6, matrix_l2pool) \
    NIF("maxpool",      5, matrix_maxpool) \
    NIF("maxpool",      6, matrix_maxpool) \
    NIF("filter",       4, matrix_filter) \
    NIF("filter",       5, matrix_filter) \
    NIF("submatrix",    5, matrix_submatrix) \
    NIF("transpose",    1, matrix_transpose) \
    NIF("transpose",    2, matrix_transpose) \
    NIF("transpose_data",1, matrix_transpose_data) \
    NIF("transpose_data",2, matrix_transpose_data) \
    NIF("sigmoid",      1, matrix_sigmoid) \
    NIF("sigmoid_prime",2, matrix_sigmoid_prime) \
    NIF("relu",          1, matrix_relu) \
    NIF("copy",          1, matrix_copy) \
    NIF("copy",          2, matrix_copy) \
    NIF("copy",          4, matrix_copy) \
    NIF("fill",          2, matrix_fill) \
    NIF("apply1",        2, matrix_apply1) \
    NIF("apply1",        3, matrix_apply1) \
    NIF("argmax",        3, matrix_argmax) \
    NIF("argmin",        3, matrix_argmin) \
    NIF("max",           3, matrix_max) \
    NIF("min",           3, matrix_min) \
    NIF("maximum",       2, matrix_maximum) \
    NIF("maximum",       3, matrix_maximum) \
    NIF("minimum",       2, matrix_minimum) \
    NIF("minimum",       3, matrix_minimum) \
    NIF("sum",           1, matrix_sum) \
    NIF("sum",           2, matrix_sum) \
    NIF("sort",          4, matrix_sort) \
    NIF("swap",          4, matrix_swap) \
    NIF("swap",          6, matrix_swap) \
    NIF("eq",            2, matrix_eq) \
    NIF("eq",            3, matrix_eq) \
    NIF("lt",            2, matrix_lt) \
    NIF("lt",            3, matrix_lt) \
    NIF("lte",           2, matrix_lte) \
    NIF("lte",           3, matrix_lte) \
    NIF("bitwise_and",   2, matrix_band) \
    NIF("bitwise_and",   3, matrix_band) \
    NIF("bitwise_or",    2, matrix_bor) \
    NIF("bitwise_or",    3, matrix_bor) \
    NIF("bitwise_xor",   2, matrix_bxor) \
    NIF("bitwise_xor",   3, matrix_bxor) \
    NIF("bitwise_not",   1, matrix_bnot) \
    NIF("bitwise_not",   2, matrix_bnot) \
    NIF("eval1",         2, matrix_eval1) \
    NIF("eval1",         3, matrix_eval1) \
    NIF("eval2",         3, matrix_eval2) \
    NIF("eval2",         4, matrix_eval2) \
    NIF("info",          2, matrix_info)


// Declare all nif functions
#undef NIF
#ifdef NIF_TRACE
#define NIF(name, arity, func)						\
    static ERL_NIF_TERM func(ErlNifEnv* env, int argc,const ERL_NIF_TERM argv[]); \
    static ERL_NIF_TERM trace##_##func##_##arity(ErlNifEnv* env, int argc,const ERL_NIF_TERM argv[]);
#else
#define NIF(name, arity, func)						\
    static ERL_NIF_TERM func(ErlNifEnv* env, int argc,const ERL_NIF_TERM argv[]);
#endif

NIF_LIST

#undef NIF
#ifdef NIF_TRACE
#define NIF(name,arity,func) NIF_FUNC(name, arity, trace##_##func##_##arity),
#else
#define NIF(name,arity,func) NIF_FUNC(name, arity, func),
#endif

ErlNifFunc matrix_funcs[] =
{
    NIF_LIST
};

/*
matrix_type_t integer_type_[NUM_TYPES] = {
    [INT8] = INT8,
    [INT16] = INT16,
    [INT32] = INT32,
    [INT64] = INT64,
    [INT128] = INT128,
    [FLOAT32] = INT32,
    [FLOAT64] = INT64,
};
*/

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
DECL_ATOM(matrix_t);
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
DECL_ATOM(abs);
DECL_ATOM(ascend);
DECL_ATOM(descend);
// info
DECL_ATOM(rowmajor);
DECL_ATOM(n);
DECL_ATOM(n_stride);
DECL_ATOM(m);
DECL_ATOM(m_stride);
DECL_ATOM(k_stride);
DECL_ATOM(rows);
DECL_ATOM(columns);
DECL_ATOM(type);
DECL_ATOM(size);
DECL_ATOM(parent);
// exception
DECL_ATOM(error);
DECL_ATOM(notsup);
DECL_ATOM(badarg);
// instructions
DECL_ATOM(ret);
DECL_ATOM(mov);
DECL_ATOM(r);
DECL_ATOM(a);
DECL_ATOM(c);
DECL_ATOM(add);
DECL_ATOM(sub);
DECL_ATOM(mul);
DECL_ATOM(neg);
DECL_ATOM(inv);
DECL_ATOM(lt);
DECL_ATOM(lte);
DECL_ATOM(eq);
DECL_ATOM(band);
DECL_ATOM(bor);
DECL_ATOM(bxor);
DECL_ATOM(bnot);
// types
DECL_ATOM(uint8);
DECL_ATOM(uint16);
DECL_ATOM(uint32);
DECL_ATOM(uint64);
DECL_ATOM(uint128);
DECL_ATOM(int8);
DECL_ATOM(int16);
DECL_ATOM(int32);
DECL_ATOM(int64);
DECL_ATOM(int128);
DECL_ATOM(float16);
DECL_ATOM(float32);
DECL_ATOM(float64);

static size_t element_size(matrix_type_t type)
{
    return (get_vector_size(type) << get_scalar_exp_size(type));
}

#ifdef USE_VECTOR
static size_t components_per_vector_t(matrix_type_t type)
{
    return (sizeof(vector_t) >> get_scalar_exp_size(type));
}
#endif

static size_t component_size(matrix_type_t type)
{
    return (1 << get_scalar_exp_size(type));
}

/*
static int size_of_components(matrix_type_t type, size_t ncomponents)
{
    return (ncomponents << get_scalar_exp_size(type));
}
*/

static int size_of_array(matrix_type_t type, size_t nelems)
{
    return (nelems*element_size(type));
}

// element poistion from pointer diff
#if 0
static intptr_t element_pos(matrix_type_t type, intptr_t offs)
{
    return (offs >> get_elem_comp_size(type));
}
#endif

static int is_integer(matrix_type_t type)
{
    return (type >= INT8) && (type <= INT128);
}

static int is_float(matrix_type_t type)
{
    return (type >= FLOAT32) && (type <= FLOAT64);
}

static matrix_type_t combine_type(matrix_type_t at, matrix_type_t bt)
{
    return (at > bt) ? at : bt;
}

static matrix_type_t copy_type(matrix_type_t at)
{
    return at;
}

// convert float type to integer type with the same size
static matrix_type_t integer_type(matrix_type_t at)
{
    return ((at & ~BASE_TYPE_MASK) | INT);
}

static matrix_type_t compare_type(matrix_type_t at, matrix_type_t bt)
{
    matrix_type_t ct = combine_type(at, bt);
    return integer_type(ct);
}

////////////////////////////////////////////////////////////////////////////
//
//  rw_op.i  is generated from ../priv/rw_op.term (with texgen.erl)
//  it generates all functions needed to load value into a general class
//
////////////////////////////////////////////////////////////////////////////

#include "rw_op.i"

static void write_int64(matrix_type_t type, byte_t* ptr, int64_t v)
{
    (write_int64_func[type])(ptr, v);
}

static void write_float64(matrix_type_t type, byte_t* ptr, float64_t v)
{
    (write_float64_func[type])(ptr, v);
}

static int64_t read_int64(matrix_type_t type, byte_t* ptr)
{
    return (*read_int64_func[type])(ptr);
}

static uint64_t read_uint64(matrix_type_t type, byte_t* ptr)
{
    return (*read_uint64_func[type])(ptr);
}

static float64_t read_float64(matrix_type_t type, byte_t* ptr)
{
    return (*read_float64_func[type])(ptr);
}

// convert to erlang term vector_size=1 (no singletons allowed)
// or tuple size 2,3,4,8,16
static ERL_NIF_TERM make_element(ErlNifEnv* env, matrix_type_t type, byte_t* ptr)
{
    size_t n = get_vector_size(type);
    matrix_type_t t = get_base_type(type);
    matrix_type_t et = get_scalar_type(type);
    
    if (n == 1) {
	switch(t) {
	case INT:  return enif_make_int64(env, read_int64(et, ptr));
	case UINT: return enif_make_uint64(env, read_uint64(et, ptr));
	case FLOAT: return enif_make_double(env, read_float64(et, ptr));
	default:
	    return EXCP_BADARG_N(env, 0, "internal error, type unknow");
	}
    }
    else {
	ERL_NIF_TERM arr[16];
	size_t step = get_scalar_size(et);
	int i;
	switch(t) {
	case INT:
	    for (i = 0; i < (int)n; i++) {
		arr[i]=enif_make_int64(env, read_int64(et, ptr));
		ptr += step;
	    }
	    break;
	case UINT:
	    for (i = 0; i < (int)n; i++) {	    
		arr[i]=enif_make_uint64(env, read_uint64(et, ptr));
		ptr += step;
	    }
	    break;
	case FLOAT:
	    for (i = 0; i < (int)n; i++) {	    	    
		arr[i]=enif_make_double(env, read_float64(et, ptr));
		ptr += step;
	    }
	    break;
	default:
	    return EXCP_BADARG_N(env, 0, "internal error, type unknow");
	}
	return enif_make_tuple_from_array(env, arr, n);
    }
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

// FIXME: init from erlang!?
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

// functional macros of arithmetic operators
// supported vector operators: +, -, *, /, unary minus, ^, |, &, ~, %.
// shift operators: << and >> for integer vectors
// comparison operators: ==, !=, <, <=, >, >=
// nullary
#define op_zero() (0)
#define op_one()  (1)

// unary
#define op_neg(x)        (-(x))
#define op_bnot(x)       (~(x))
#define op_reciprocal(x) (1/(x))

// binary
#define op_add(x,y)     ((x)+(y))
#define op_sub(x,y)     ((x)-(y))
#define op_mul(x,y)     ((x)*(y))
#define op_div(x,y)   ((x)/(y))
#define op_rem(x,y)   ((x)%(y))
#define op_bxor(x,y)  ((x)^(y))
#define op_bor(x,y)   ((x)|(y))
#define op_band(x,y)  ((x)&(y))
#define op_bsl(x,y)   ((x)<<(y))
#define op_bsr(x,y)   ((x)>>(y))
#define op_eq(x,y)    ((x)==(y))
#define op_lt(x,y)    ((x)<(y))
#define op_lte(x,y)   ((x)<=(y))
// scalar version
#define op_ieq(x,y)    (-((x)==(y)))
#define op_ilt(x,y)    (-((x)<(y)))
#define op_ilte(x,y)   (-((x)<=(y)))

// <= avoids warning!
#define op_min(x,y)     ((x)<=(y)?(x):(y))
#define op_max(x,y)     (((y)<=(x))?(x):(y))

#define op_sigmoid(x)         (1.0/(1.0 + exp(-(x))))
#define op_rectifier(x)       op_max(0,(x))
#define op_sigmoid_prime1(x)  ((x)*(1-(x)))

static inline float64_t op_sigmoid_prime(float64_t x)
{
    float64_t z = op_sigmoid(x);
    return z*(1-z);
}


int128_t op128_band(int128_t a, int128_t b)
{
    int128_t r = {a.hi & b.hi, a.lo & b.lo};
    return r;
}

int128_t op128_bor(int128_t a, int128_t b)
{
    int128_t r = {a.hi | b.hi, a.lo | b.lo};
    return r;
}

int128_t op128_bxor(int128_t a, int128_t b)
{
    int128_t r = {a.hi ^ b.hi, a.lo ^ b.lo};
    return r;
}

int128_t op128_bnot(int128_t a)
{
    int128_t r = {~a.hi, ~a.lo};
    return r;
}

///////////////////////////////////////////////////////////////////////////////
//  operations
///////////////////////////////////////////////////////////////////////////////

#include "matrix_arith.i"
#include "matrix_bitwise.i"
#include "matrix_cmp.i"
//#include "matrix_minmax.i"

#include "matrix_dot.i"
#include "matrix_multiply.i"
#include "matrix_multiply_t.i"
#include "matrix_kmultiply.i"
#include "matrix_kmultiply_t.i"
#include "matrix_sigmoid.i"
#include "matrix_sigmoid_prime1.i"
#include "matrix_rectifier.i"

// SIMD versions
#ifdef USE_VECTOR
#include "matrix_vdot.i"
#include "matrix_vmultiply.i"
#include "matrix_vmultiply_t.i"
#include "matrix_vkmultiply.i"
#include "matrix_vkmultiply_t.i"
#endif

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

static float64_t reciprocal_float64(float64_t a)
{
    return 1/a;
}

static float64_t sqrt_float64(float64_t a)
{
    return sqrt(a);
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
    [RECIPROCAL] = reciprocal_float64,
    [SQRT] = sqrt_float64,    
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

static int64_t reciprocal_int64(int64_t a)
{
    UNUSED(a);
    return 0;
}

static int64_t bnot_int64(int64_t a)
{
    return ~a;
}

static int64_t sqrt_int64(int64_t a)
{
    int64_t xk, ak;
    if (a <= 0) return 0;
    if (a == 1) return 1;
    xk = a / 2;
    ak = a / xk;
    while(ak < xk) {
	int64_t xk1 = (xk+ak) >> 1;
	ak = a / xk1;
	xk = xk1;
    }
    return xk;
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
    [RECIPROCAL] = reciprocal_int64,
    [SQRT] = sqrt_int64,
};

static int64_t (*iunaryop_int64[NUM_IUNARYOP])(int64_t) = {
    [BNOT] = bnot_int64,
};

#define MAX_PROG_SIZE 64   // enough?

#define OP_CND   0x80  // use condition register
#define OP_BIN   0x40  // binary operation / else unary

// FIXME: make register machine 8 vector regs?
typedef enum {
    // unary
    OP_RET  = 0,     // return top of stack
    OP_MOVR = 1,     // move register to register    
    OP_MOVA = 2,     // move argument to register
    OP_MOVC = 3,     // move constant to register    
    OP_NEG  = 4,     // negate    
    OP_BNOT = 5,     // bitwise negate
    OP_INV  = 6,     // recipocal    

    OP_ADD  = OP_BIN+1,   // add
    OP_SUB  = OP_BIN+2,   // subtract
    OP_MUL  = OP_BIN+3,   // mul
    OP_BAND = OP_BIN+4,   // bitwise and
    OP_BOR  = OP_BIN+5,   // bitwise or
    OP_BXOR = OP_BIN+6,   // bitwise xor
    OP_LT   = OP_BIN+7,   // less
    OP_LTE  = OP_BIN+8,   // less or equal
    OP_EQ   = OP_BIN+9,   // equal
} opcode_t;

// 32bit
typedef struct {
    unsigned op:8;   // CND|BIN|<op>
    unsigned type:8; // element type
    unsigned ri:4;   // src1
    unsigned rj:4;   // src2
    unsigned rd:4;   // dst
    unsigned rc:4;   // condition mask
} instr_t;

// run op over all element in array
static void ev_unary(unary_op_t* opv,matrix_type_t type,
		     void* src, void* dst)
		    
{
    matrix_type_t t = get_scalar_type(type);
    unary_op_t op = opv[t];    
    size_t len = get_vector_size(type);

    if (len == 1)
	(*op)(src, dst);
    else {
        size_t m = get_scalar_size(t);
	while(len--) {
	    (*op)(src, dst);
	    src = ((byte_t*)src) + m;
	    dst = ((byte_t*)dst) + m;
	}
    }
}

// run op over all element in array
static void ev_binary(binary_op_t* opv,matrix_type_t type,
		      void* src1, void* src2, void* dst)
		    
{
    matrix_type_t t = get_scalar_type(type);
    binary_op_t op = opv[t];
    size_t len = get_vector_size(type);

    if (len == 1)
	(*op)(src1, src2, dst);
    else {
	size_t m = get_scalar_size(t);
	while(len--) {
	    (*op)(src1, src2, dst);
	    src1 = ((byte_t*)src1) + m;
	    src2 = ((byte_t*)src2) + m;
	    dst = ((byte_t*)dst) + m;
	}
    }
}

#define EVAL_STACK_SIZE 8

// splat data into dst vector
static void set_const(matrix_type_t t, uint8_t* data, scalar_t* dst)
{
    switch(t) {
    case INT8:  memcpy(&dst->i8, data, sizeof(dst->i8)); break;
    case INT16: memcpy(&dst->i16, data, sizeof(dst->i16)); break;
    case INT32: memcpy(&dst->i32, data, sizeof(dst->i32)); break;
    case INT64: memcpy(&dst->i64, data, sizeof(dst->i64)); break;
    case INT128: memcpy(&dst->i128, data, sizeof(dst->i128)); break;		
    case UINT8: memcpy(&dst->u8, data, sizeof(dst->u8)); break;
    case UINT16: memcpy(&dst->u16, data, sizeof(dst->u16)); break;
    case UINT32: memcpy(&dst->u32, data, sizeof(dst->u32)); break;
    case UINT64: memcpy(&dst->u64, data, sizeof(dst->u64)); break;
    case UINT128: memcpy(&dst->u128, data, sizeof(dst->u128)); break;	
	
    case FLOAT16: memcpy(&dst->f16, data, sizeof(dst->f16)); break;
    case FLOAT32: memcpy(&dst->f32, data, sizeof(dst->f32)); break;
    case FLOAT64: memcpy(&dst->f64, data, sizeof(dst->f64)); break;
    default: break;
    }
}

static void eval_prog(instr_t* prog, int argc, scalar_t argv[], scalar_t* dst)
{
    scalar_t r[16];
    scalar_t ack;
    matrix_type_t t;
    unsigned i;
    int ri, rj, rd, rc;
    int pc = 0;
next:
    i = prog[pc].op;
    t = prog[pc].type;
    ri = prog[pc].ri;
    rj = prog[pc].rj;
    rd = prog[pc].rd;
    rc = prog[pc].rc; 

    switch(i & 0x7f) {
    case OP_MOVR: ack = r[ri]; break;
    case OP_MOVA:
	if (ri >= argc) return;  // ERROR
	ack = argv[ri];
	break;
    case OP_MOVC: {
	int len = get_scalar_size(t);  // number of constant bytes
	set_const(t, (uint8_t*) &prog[pc+1], &ack);
	pc += ((len+3)/4);
	break;
    }
    case OP_NEG:  (*fun_neg_ops[t])(&r[ri],&ack); break;
    case OP_BNOT: (*fun_bnot_ops[t])(&r[ri],&ack); break;
    case OP_INV:  (*fun_reciprocal_ops[t])(&r[ri],&ack); break;
    case OP_RET:  *dst = r[ri]; return;
	
    case OP_ADD: (*fun_add_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_SUB: (*fun_sub_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_MUL: (*fun_times_ops[t])(&r[ri],&r[rj],&ack); break;

    case OP_LT:  (*fun_lt_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_LTE: (*fun_lte_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_EQ:  (*fun_eq_ops[t])(&r[ri],&r[rj],&ack); break;

    case OP_BAND: (*fun_band_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_BOR:  (*fun_bor_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_BXOR: (*fun_bxor_ops[t])(&r[ri],&r[rj],&ack); break;
    default: break;
    }
    if (i & OP_CND) {  // conditional update preserve elements not masked
	scalar_t cond = r[rc];
	// rd = (rc & a) | (~rc & rd)
	(*fun_band_ops[t])(&cond,&ack,&ack);
	(*fun_bnot_ops[t])(&cond,&cond);
	(*fun_band_ops[t])(&cond,&r[rd],&r[rd]);
	(*fun_bor_ops[t])(&ack,&r[rd],&r[rd]);
    }
    else {
	r[rd] = ack;
    }
    pc++;
    goto next;
}


// splat data into dst vector
static void set_vconst(matrix_type_t t, uint8_t* data, vscalar_t* dst)
{
    switch(t) {
    case INT8: {
	int8_t iv;
	memcpy(&iv, data, sizeof(iv));
	vint8_t v = vint8_t_const(iv);
	dst->vi8 = v;
	break;
    }
    case INT16: {
	int16_t iv;
	memcpy(&iv, data, sizeof(iv));
	vint16_t v = vint16_t_const(iv);
	dst->vi16 = v;
	break;
    }
    case INT32: {
	int32_t iv;	
	memcpy(&iv, data, sizeof(iv));
	vint32_t v = vint32_t_const(iv);
	dst->vi32 = v;
	break;
    }
    case INT64: {
	int64_t iv;	
	memcpy(&iv, data, sizeof(iv));
	vint64_t v = vint64_t_const(iv);
	dst->vi64 = v;
	break;
    }
    case UINT8: {
	uint8_t uv;	
	memcpy(&uv, data, sizeof(uv));
	vuint8_t v = vuint8_t_const(uv);
	dst->vu8 = v;
	break;
    }
    case UINT16: {
	uint16_t uv;
	memcpy(&uv, data, sizeof(uv));	
	vuint16_t v = vuint16_t_const(uv);
	dst->vu16 = v;
	break;
    }
    case UINT32: {
	uint32_t uv;
	memcpy(&uv, data, sizeof(uv));	
	vuint32_t v = vuint32_t_const(uv);
	dst->vu32 = v;
	break;
    }
    case UINT64: {
	uint64_t uv;
	memcpy(&uv, data, sizeof(uv));	
	vuint64_t v = vuint64_t_const(uv);
	dst->vu64 = v;
	break;
    }
    case FLOAT16: {
	float16_t fv;	
	memcpy(&fv, data, sizeof(fv));
	vfloat16_t v = vfloat16_t_const(fv);
	dst->vf16 = v;
	break;
    }	
    case FLOAT32: {
	float32_t fv;
	memcpy(&fv, data, sizeof(fv));
	vfloat32_t v = vfloat32_t_const(fv);
	dst->vf32 = v;
	break;
    }
    case FLOAT64: {
	float64_t fv;
	memcpy(&fv, data, sizeof(fv));
	vfloat64_t v = vfloat64_t_const(fv);
	dst->vf64 = v; 
	break;
    }
    default:
	break;
    }
}

static void eval_vprog(instr_t* prog, int argc, vector_t* argv[], vector_t* dst)
{
    vscalar_t r[16];
    vscalar_t ack;
    matrix_type_t t;
    unsigned i;
    int ri, rj, rd, rc;
    int pc = 0;
next:
    i = prog[pc].op;
    t = prog[pc].type;
    ri = prog[pc].ri;
    rj = prog[pc].rj;
    rd = prog[pc].rd;
    rc = prog[pc].rc; 

    switch(i & 0x7f) {
    case OP_MOVR: ack = r[ri]; break;
    case OP_MOVA:
	if (ri >= argc) return;  // ERROR
	ack.vi8 = *argv[ri];
	break;
    case OP_MOVC: {
	int len = get_scalar_size(t);  // number of constant bytes
	set_vconst(t, (uint8_t*) &prog[pc+1], &ack);
	pc += ((len+3)/4);
	break;
    }
    case OP_NEG:  (*vfun_neg_ops[t])(&r[ri],&ack); break;
    case OP_BNOT: (*vfun_bnot_ops[t])(&r[ri],&ack); break;
    case OP_INV:  (*vfun_reciprocal_ops[t])(&r[ri],&ack); break;
    case OP_RET:  *dst = r[ri].vi8; return;
	
    case OP_ADD: (*vfun_add_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_SUB: (*vfun_sub_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_MUL: (*vfun_times_ops[t])(&r[ri],&r[rj],&ack); break;

    case OP_LT:  (*vfun_lt_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_LTE: (*vfun_lte_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_EQ:  (*vfun_eq_ops[t])(&r[ri],&r[rj],&ack); break;

    case OP_BAND: (*vfun_band_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_BOR:  (*vfun_bor_ops[t])(&r[ri],&r[rj],&ack); break;
    case OP_BXOR: (*vfun_bxor_ops[t])(&r[ri],&r[rj],&ack); break;
    default: break;
    }
    if (i & OP_CND) {  // conditional update preserve elements not masked
	vscalar_t cond = r[rc];
	// rd = (rc & a) | (~rc & rd)
	(*vfun_band_ops[t])(&cond,&ack,&ack);
	(*vfun_bnot_ops[t])(&cond,&cond);
	(*vfun_band_ops[t])(&cond,&r[rd],&r[rd]);
	(*vfun_bor_ops[t])(&ack,&r[rd],&r[rd]);
    }
    else {
	r[rd] = ack;
    }
    pc++;
    goto next;
}

static void fetch_vector_element(matrix_type_t type, scalar_t* ep, int i)
{
    switch(component_size(type)) {
    case 1: ep->u8  = ep->vu8[i]; break;
    case 2: ep->u16 = ep->vu16[i]; break;
    case 4: ep->u32 = ep->vu32[i]; break;
    case 8: ep->u64 = ep->vu64[i]; break;
    default: break;
    }
}

static void apply1(unary_operation_t func,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    size_t i, j;
    size_t k = get_vector_size(at);    

    if ((get_vector_size(ct)) != k)
	return; // FIXME: error
    if (k > 1) {
	at = get_scalar_type(at);
	ct = get_scalar_type(ct);
	m *= k;
	av /= k;
	cv /= k;
    }

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

static void eval1(instr_t* prog,
		  matrix_type_t at, byte_t* ap, int au, int av,
		  matrix_type_t ct, byte_t* cp, int cu, int cv,
		  size_t n, size_t m)
{
    /*
    enif_fprintf(stderr, "eval1:n=%ld,m=%ld,at=%x,ct=%x\r\n",
		 n,m,at,ct);
    enif_fprintf(stderr, "  au=%d, av=%d\n", au, av);
    enif_fprintf(stderr, "  cu=%d, cv=%d\n", cu, cv);
    */
    
    if (is_float(ct)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;    
	    size_t m1 = m;
	    while(m1--) {
		scalar_t argv[1];
		scalar_t res;
		argv[0].f64 = read_af(ap1);
		eval_prog(prog,1,argv,&res);
		write_cf(cp1, res.i64);
		ap1 += av;
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];	
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;    
	    size_t m1 = m;
	    while(m1--) {
		scalar_t argv[1];
		scalar_t res;
		argv[0].i64 = read_af(ap1);
		eval_prog(prog,1,argv,&res);
		write_cf(cp1, res.i64);
		ap1 += av;
		cp1 += cv;
	    }
	    ap += au;
	    cp += cu;
	}
    }	
}

static void iapply1(iunary_operation_t func,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m)
{
    size_t i, j;
    size_t k = get_vector_size(at);

    if (get_vector_size(ct) != k)
	return; // FIXME: error!
    if (k > 1) {
	at = get_scalar_type(at);
	m *= k;
	av /= k;
	cv /= k;
    }
    
    if (is_integer(ct)) {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t (*opf)(int64_t) = iunaryop_int64[func];
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

static int64_t min_int64(int64_t a, int64_t b)
{
    return op_min(a,b);
}

static int64_t max_int64(int64_t a, int64_t b)
{
    return op_max(a,b);
}

static int64_t div_int64(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : a / b;
}

static int64_t rem_int64(int64_t a, int64_t b)
{
    return (b == 0) ? 0 : a % b;
}

static int64_t (*binop_int64[NUM_BINOP])(int64_t, int64_t) = {
    [ADD] = add_int64,
    [SUB] = sub_int64,
    [MUL] = mul_int64,
    [MIN] = min_int64,
    [MAX] = max_int64,
    [DIV] = div_int64,
    [REM] = rem_int64,
};

static int64_t band_int64(int64_t a, int64_t b)
{
    return op_band(a,b);
}

static int64_t bor_int64(int64_t a, int64_t b)
{
    return op_bor(a,b);
}

static int64_t bxor_int64(int64_t a, int64_t b)
{
    return op_bxor(a,b);
}

static int64_t (*ibinop_int64[NUM_IBINOP])(int64_t, int64_t) = {
    [BAND] = band_int64,
    [BOR]  = bor_int64,
    [BXOR] = bxor_int64,
};

static int64_t eq_int64(int64_t a, int64_t b)
{
    return op_ieq(a,b);
}

static int64_t lt_int64(int64_t a, int64_t b)
{
    return op_ilt(a,b);
}

static int64_t lte_int64(int64_t a, int64_t b)
{
    return op_ilte(a,b);
}

static int64_t (*cmpop_int64[NUM_CMPOP])(int64_t, int64_t) = {
    [EQ]  = eq_int64,
    [LT]  = lt_int64,
    [LTE] = lte_int64,
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

static float64_t min_float64(float64_t a, float64_t b)
{
    return op_min(a,b);
}

static float64_t max_float64(float64_t a, float64_t b)
{
    return op_max(a,b);
}

static float64_t div_float64(float64_t a, float64_t b)
{
    return (b == 0.0) ? 0.0 : a / b;
}

static float64_t rem_float64(float64_t a, float64_t b)
{
    return (b == 0.0) ? 0.0 : fmod(a, b);
}

static float64_t (*binop_float64[NUM_BINOP])(float64_t, float64_t) = {
    [ADD]  = add_float64,
    [SUB]  = sub_float64,
    [MUL]  = mul_float64,
    [MIN]  = min_float64,
    [MAX]  = max_float64,
    [DIV]  = div_float64,
    [REM]  = rem_float64,
};

static int64_t eq_float64(float64_t a, float64_t b)
{
    return op_eq(a,b);
}

static int64_t lt_float64(float64_t a, float64_t b)
{
    return op_ilt(a,b);
}

static int64_t lte_float64(float64_t a, float64_t b)
{
    return op_ilte(a,b);
}

static int64_t (*cmpop_float64[NUM_CMPOP])(float64_t, float64_t) = {
    [EQ]  = eq_float64,
    [LT]  = lt_float64,
    [LTE] = lte_float64,
};


static void eval_prog_unary(instr_t* prog, matrix_type_t type,
			    int use_vector,
			    byte_t* ap, int au, int av,
			    byte_t* cp, int cu, int cv,
			    size_t n, size_t m)
{
    // matrix_type_t t = get_scalar_type(type);
    size_t k = get_vector_size(type);
#ifdef USE_VECTOR
    size_t kk = components_per_vector_t(type);
#endif
    
    if (k > 1) {
	m *= k;
	av /= k;
	cv /= k;
    }
    
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* cp1 = cp;
	size_t m1 = m;
#ifdef USE_VECTOR
	if (use_vector) {
	    while(m1 >= kk) {
		vector_t* argv[2];
		argv[0] = (vector_t*) ap1;
		eval_vprog(prog, 1, argv, (vector_t*)cp1);
		ap1 += sizeof(vector_t);
		cp1 += sizeof(vector_t);
		m1  -= kk;
	    }
	}
#endif	
	while(m1--) {	    
	    scalar_t argv[1];
	    scalar_t res;

	    memcpy(argv[0].data, ap1, av);
	    eval_prog(prog, 1, argv, &res);
	    memcpy(cp1, res.data, cv);
	    
	    ap1 += av;
	    cp1 += cv;
	}
        ap += au;
        cp += cu;
    }
}

static void eval_prog_binary(instr_t* prog, matrix_type_t type,
			     int use_vector,
			     byte_t* ap, int au, int av,
			     byte_t* bp, int bu, int bv,
			     byte_t* cp, int cu, int cv,
			     size_t n, size_t m)
{
    // matrix_type_t t = get_scalar_type(type);
    size_t k = get_vector_size(type);
#ifdef USE_VECTOR        
    size_t kk = components_per_vector_t(type);
#endif
    
    if (k > 1) {
	m *= k;
	av /= k;
	bv /= k;
	cv /= k;
    }
    
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* bp1 = bp;
	byte_t* cp1 = cp;
	size_t m1 = m;
#ifdef USE_VECTOR
	if (use_vector) {
	    while(m1 >= kk) {
		vector_t* argv[2];
		argv[0] = (vector_t*) ap1;
		argv[1] = (vector_t*) bp1;
		eval_vprog(prog, 2, argv, (vector_t*)cp1);
		ap1 += sizeof(vector_t);
		bp1 += sizeof(vector_t);
		cp1 += sizeof(vector_t);
		m1  -= kk;
	    }
	}
#endif	
	while(m1--) {
	    scalar_t argv[2];
	    scalar_t res;
	    memcpy(argv[0].data, ap1, av);
	    memcpy(argv[1].data, bp1, bv);
	    eval_prog(prog, 2, argv, &res);
	    memcpy(cp1, res.data, cv);
	    ap1 += av;
	    bp1 += bv;
	    cp1 += cv;
	}
        ap += au;
        bp += bu;
        cp += cu;
    }
}
    
static void mt_unary_eval(unary_op_t* opv, matrix_type_t type,
			  byte_t* ap, int au, int av,
			  byte_t* cp, int cu, int cv,
			  size_t n, size_t m)
{
//    matrix_type_t t = get_scalar_type(type);
    //    unary_op_t op = opv[t];    
/*    size_t k = get_vector_size(type);
    if (k > 1) {
	m *= k;
	av /= k;
	cv /= k;
    }
*/
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* cp1 = cp;
	size_t m1 = m;
	while(m1--) {
	    ev_unary(opv, type, (void*)ap1,(void*)cp1);
	    // (*op)((void*)ap1, (void*)cp1);
	    ap1 += av;
	    cp1 += cv;
	}
	ap += au;
	cp += cu;
    }
}

static void mtv_unary_eval(unary_vop_t* vopv, unary_op_t* opv,
			   matrix_type_t type,
			   byte_t* ap, int au, int av,
			   byte_t* cp, int cu, int cv,
			   size_t n, size_t m)
{
    matrix_type_t t = get_scalar_type(type);
    size_t k = get_vector_size(type);    
    unary_vop_t vop = vopv[t];
    unary_op_t op = opv[t];
#ifdef USE_VECTOR
    size_t kk = components_per_vector_t(type);
#endif

    if (k > 1) {
	m *= k;
	av /= k;
	cv /= k;
    }
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* cp1 = cp;
	size_t m1 = m;
	while(m1 >= kk) {
	    (*vop)((vector_t*) ap1, (vector_t*) cp1);
	    ap1 += sizeof(vector_t);
	    cp1 += sizeof(vector_t);
	    m1  -= kk;
	}
	while(m1--) {
	    (*op)((void*)ap1, (void*) cp1);
	    ap1 += av;
	    cp1 += cv;
	}
        ap += au;
        cp += cu;
    }
}

// av, au ... are byte steps
static void mt_binary_eval(binary_op_t* opv, matrix_type_t type,
			   byte_t* ap, int au, int av,
			   byte_t* bp, int bu, int bv,
			   byte_t* cp, int cu, int cv,
			   size_t n, size_t m)
{
//    matrix_type_t t = get_scalar_type(type);
//    binary_op_t op = opv[t];
//    size_t k = get_vector_size(type);
/*
    enif_fprintf(stderr, "mt_binary_eval:n=%ld,m=%ld,type=%x\r\n",
		 n,m,type);
    enif_fprintf(stderr, "  au=%d, av=%d\n", au, av);
    enif_fprintf(stderr, "  bu=%d, bv=%d\n", bu, bv);
    enif_fprintf(stderr, "  cu=%d, cv=%d\n", cu, cv);
*/
/*
    if (k > 1) {
	m *= k;
	av /= k;
	bv /= k;
	cv /= k;
    }
*/
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* bp1 = bp;
	byte_t* cp1 = cp;
	size_t m1 = m;
	while(m1--) {
	    ev_binary(opv, type, (void*)ap1,(void*)bp1,(void*)cp1);
	    // (*op)((void*)ap1,(void*)bp1,(void*)cp1);
	    ap1 += av;
	    bp1 += bv;
	    cp1 += cv;
	}
	ap += au;
	bp += bu;
	cp += cu;
    }
}

// au,bu,cu ... are byte steps
static void mtv_binary_eval(binary_vop_t* vopv, binary_op_t* opv,
			    matrix_type_t type,
			    byte_t* ap, int au, int av,
			    byte_t* bp, int bu, int bv,
			    byte_t* cp, int cu, int cv,
			    size_t n, size_t m)
{
    matrix_type_t t = get_scalar_type(type);
    size_t k = get_vector_size(type);    
    binary_vop_t vop = vopv[t];
    binary_op_t op = opv[t];
#ifdef USE_VECTOR    
    size_t kk = components_per_vector_t(type);
#endif
    /*
    enif_fprintf(stderr, "mtv_binary_eval:n=%ld,m=%ld,k=%ld,kk=%ld,type=%x\r\n",
		 n,m,k,kk,type);
    enif_fprintf(stderr, "  au=%d, av=%d\n", au, av);
    enif_fprintf(stderr, "  bu=%d, bv=%d\n", bu, bv);
    enif_fprintf(stderr, "  cu=%d, cv=%d\n", cu, cv);
    */
    
    if (k > 1) {
	m *= k;
	av /= k;
	bv /= k;
	cv /= k;
    }
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* bp1 = bp;
	byte_t* cp1 = cp;
	size_t m1 = m;

	while(m1 >= kk) {
	    (*vop)((vector_t*)ap1,(vector_t*)bp1,(vector_t*)cp1);
	    ap1 += sizeof(vector_t);
	    bp1 += sizeof(vector_t);
	    cp1 += sizeof(vector_t);
	    m1  -= kk;
	}
	while(m1--) {
	    (*op)((void*)ap1,(void*)bp1,(void*)cp1);
	    ap1 += av;
	    bp1 += bv;
	    cp1 += cv;
	}
        ap += au;
        bp += bu;
        cp += cu;
    }
}


// a more general function for binary operations but a lot slower
static void apply2(binary_operation_t func,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    matrix_type_t t = combine_type(at, bt);
    size_t k = get_vector_size(at);

    /*
    enif_fprintf(stderr, "apply2:n=%ld,m=%ld,k=%ld,at=%x,bt=%x,ct=%x\r\n",
		 n,m,k,at,bt,ct);
    enif_fprintf(stderr, "  au=%d, av=%d\n", au, av);
    enif_fprintf(stderr, "  bu=%d, bv=%d\n", bu, bv);
    enif_fprintf(stderr, "  cu=%d, cv=%d\n", cu, cv);
    */
    
    if (((get_vector_size(bt)) != k) ||
	((get_vector_size(ct)) != k))
	return; // FIXME: error
    if (k > 1) {
	at = get_scalar_type(at);
	bt = get_scalar_type(bt);
	ct = get_scalar_type(ct);	
	m *= k;
	av /= k;
	bv /= k;
	cv /= k;
    }
    
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

static void eval2(instr_t* prog,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m)
{
    matrix_type_t t = combine_type(at, bt);

    /*
    enif_fprintf(stderr, "eval2:n=%ld,m=%ld,at=%x,bt=%x,ct=%x\r\n",
		 n,m,at,bt,ct);
    enif_fprintf(stderr, "  au=%d, av=%d\n", au, av);
    enif_fprintf(stderr, "  bu=%d, bv=%d\n", bu, bv);
    enif_fprintf(stderr, "  cu=%d, cv=%d\n", cu, cv);
    */
    
    if (is_float(t)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	void (*write_cf)(byte_t*, float64_t) = write_float64_func[ct];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		scalar_t argv[2];
		scalar_t res;

		argv[0].f64 = read_af(ap1);
		argv[1].f64 = read_bf(bp1);
		eval_prog(prog, 2, argv, &res);
		write_cf(cp1, res.f64);
		ap1 += av;
		bp1 += bv;
		cp1 += cv;
	    }
	}
	ap += au;
	bp += bu;
	cp += cu;
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	void (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		scalar_t argv[2];
		scalar_t res;

		argv[0].i64 = read_af(ap1);
		argv[1].i64 = read_bf(bp1);
		eval_prog(prog, 2, argv, &res);
		write_cf(cp1, res.i64);
		cp1 += cv;
	    }
	}
	ap += au;
	bp += bu;
	cp += cu;	
    }
}

// a more general function for compare operations but a lot slower
static void compare2(compare_operation_t func,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t bt, byte_t* bp, int bu, int bv,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m)
{
    matrix_type_t t = combine_type(at, bt);
    size_t k;
    
    if ((k = get_vector_size(at)) > 1) {
	at = get_scalar_type(at);
	m *= k;
	av /= k;
    }
    if ((k = get_vector_size(bt)) > 1) {
	bt = get_scalar_type(bt);
	bv /= k;
    }
    if ((k = get_vector_size(ct)) > 1) {
	ct = get_scalar_type(ct);
	cv /= k;
    }
    
    if (is_float(t)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t (*read_bf)(byte_t*) = read_float64_func[bt];
	void (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t (*opf)(float64_t, float64_t) = cmpop_float64[func];
	while(n--) {
	    byte_t* ap1 = ap;
	    byte_t* bp1 = bp;
	    byte_t* cp1 = cp;
	    size_t m1 = m;
	    while(m1--) {
		float64_t a = read_af(ap1);
		float64_t b = read_bf(bp1);
		int64_t c = opf(a,b);
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
	int64_t (*opf)(int64_t, int64_t) = cmpop_int64[func];
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

// a more general function for integer binary operations but a lot slower
static void iapply2(ibinary_operation_t func,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t bt, byte_t* bp, int bu, int bv,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m)
{
    matrix_type_t t = combine_type(at, bt);
    size_t k = get_vector_size(at);

    if ((get_vector_size(bt) != k) ||
	(get_vector_size(ct) != k))
	return;  // FIXME: error
    if (k > 1) {
	at = get_scalar_type(at);
	bt = get_scalar_type(bt);
	ct = get_scalar_type(ct);	
	m *= k;
	av /= k;
	bv /= k;
	cv /= k;
	t = combine_type(at, bt);
    }

    if (is_integer(t)) {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t (*read_bf)(byte_t*) = read_int64_func[bt];
	void    (*write_cf)(byte_t*, int64_t) = write_int64_func[ct];
	int64_t (*opf)(int64_t, int64_t) = ibinop_int64[func];
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

static void mop_add(bool_t use_vector,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t bt, byte_t* bp, int bu, int bv,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if ((at == bt) && (bt == ct)) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_add_ops, fun_add_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_add_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	apply2(ADD, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// subtract
///////////////////////////////////////////////////////////////////////////////

static void mop_subtract(bool_t use_vector,
			 matrix_type_t at, byte_t* ap, int au, int av,
			 matrix_type_t bt, byte_t* bp, int bu, int bv,
			 matrix_type_t ct, byte_t* cp, int cu, int cv,
			 size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if ((at == bt) && (bt == ct)) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_sub_ops, fun_sub_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_sub_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	apply2(SUB, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// minimum
///////////////////////////////////////////////////////////////////////////////

static void mop_minimum(bool_t use_vector,
			matrix_type_t at, byte_t* ap, int au, int av,
			matrix_type_t bt, byte_t* bp, int bu, int bv,
			matrix_type_t ct, byte_t* cp, int cu, int cv,
			size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    UNUSED(use_vector);
    apply2(MIN, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
}

///////////////////////////////////////////////////////////////////////////////
// maximum
///////////////////////////////////////////////////////////////////////////////

static void mop_maximum(bool_t use_vector,
			matrix_type_t at, byte_t* ap, int au, int av,
			matrix_type_t bt, byte_t* bp, int bu, int bv,
			matrix_type_t ct, byte_t* cp, int cu, int cv,
			size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    UNUSED(use_vector);
    apply2(MAX, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
}

///////////////////////////////////////////////////////////////////////////////
// eq
///////////////////////////////////////////////////////////////////////////////

static void mop_eq(bool_t use_vector,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    if ((at == bt) && (ct == integer_type(bt))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_eq_ops, fun_eq_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_eq_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	compare2(EQ, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}


///////////////////////////////////////////////////////////////////////////////
// lt
///////////////////////////////////////////////////////////////////////////////

static void mop_lt(bool_t use_vector,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp, int cu, int cv,
		   size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    if ((at == bt) && (ct == integer_type(bt))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_lt_ops, fun_lt_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_eq_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);	    
	}
    }
    else {
	compare2(LT, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// lte
///////////////////////////////////////////////////////////////////////////////

static void mop_lte(bool_t use_vector,
		    matrix_type_t at, byte_t* ap, int au, int av,
		    matrix_type_t bt, byte_t* bp, int bu, int bv,
		    matrix_type_t ct, byte_t* cp, int cu, int cv,
		    size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    if ((at == bt) && (ct == integer_type(bt))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_lte_ops, fun_lte_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_lt_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	compare2(LTE, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// band
///////////////////////////////////////////////////////////////////////////////

static void mop_band(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t bt, byte_t* bp, int bu, int bv,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if ((at == bt) && (ct == integer_type(bt))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_band_ops, fun_band_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_band_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);	    
	}
    }
    else {
	iapply2(BAND, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}


static void mop_bor(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t bt, byte_t* bp, int bu, int bv,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if ((at == bt) && (ct == integer_type(bt))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_bor_ops, fun_bor_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_bor_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	iapply2(BOR, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

static void mop_bxor(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t bt, byte_t* bp, int bu, int bv,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if ((at == bt) && (ct == integer_type(bt))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_bxor_ops, fun_bxor_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_binary_eval(fun_bxor_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
    }
    else {
	iapply2(BXOR, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// times
///////////////////////////////////////////////////////////////////////////////

static void mop_times(bool_t use_vector,
		      matrix_type_t at, byte_t* ap, int au, int av,
		      matrix_type_t bt, byte_t* bp, int bu, int bv,
		      matrix_type_t ct, byte_t* cp, int cu, int cv,
		      size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if ((at == bt) && (bt == ct)) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    mtv_binary_eval(vfun_times_ops, fun_times_ops, at,
			    ap, au, av, bp, bu, bv, cp, cu, cv, n, m); 
	}
	else
#endif
	{
	    mt_binary_eval(fun_times_ops, at,
			   ap, au, av, bp, bu, bv, cp, cu, cv, n, m);	    
	}
    }
    else {
	apply2(MUL, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
    }
}


///////////////////////////////////////////////////////////////////////////////
// divide
///////////////////////////////////////////////////////////////////////////////

static void mop_divide(bool_t use_vector,
		       matrix_type_t at, byte_t* ap, int au, int av,
		       matrix_type_t bt, byte_t* bp, int bu, int bv,
		       matrix_type_t ct, byte_t* cp, int cu, int cv,
		       size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    UNUSED(use_vector);
    apply2(DIV, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
}

///////////////////////////////////////////////////////////////////////////////
// remainder
///////////////////////////////////////////////////////////////////////////////

static void mop_remainder(bool_t use_vector,
			  matrix_type_t at, byte_t* ap, int au, int av,
			  matrix_type_t bt, byte_t* bp, int bu, int bv,
			  matrix_type_t ct, byte_t* cp, int cu, int cv,
			  size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    UNUSED(use_vector);
    apply2(REM, at, ap, au, av, bt, bp, bu, bv, ct, cp, cu, cv, n, m);
}

///////////////////////////////////////////////////////////////////////////////
// negate
///////////////////////////////////////////////////////////////////////////////

static void mop_negate(bool_t use_vector,
		       matrix_type_t at, byte_t* ap, int au, int av,
		       matrix_type_t ct, byte_t* cp, int cu, int cv,
		       size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    if (at == ct) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp)) {
	    mtv_unary_eval(vfun_neg_ops, fun_neg_ops, at,
			   ap, au, av, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_unary_eval(fun_neg_ops, at,
			  ap, au, av, cp, cu, cv, n, m);
	}
    }
    else {
	apply1(NEGATE, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// reciprocal
///////////////////////////////////////////////////////////////////////////////

static void mop_reciprocal(bool_t use_vector,
			   matrix_type_t at, byte_t* ap, int au, int av,
			   matrix_type_t ct, byte_t* cp, int cu, int cv,
			   size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if (at == ct) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp)) {
	    mtv_unary_eval(vfun_reciprocal_ops, fun_reciprocal_ops, at,
			   ap, au, av, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_unary_eval(fun_reciprocal_ops, at,
			  ap, au, av, cp, cu, cv, n, m);
	}
    }
    else {
	apply1(RECIPROCAL, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// sqrt
///////////////////////////////////////////////////////////////////////////////

static void mop_sqrt(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m, void* extra)
{
    UNUSED(extra);    
    UNUSED(use_vector);
    apply1(SQRT, at, ap, au, av, ct, cp, cu, cv, n, m);
}


///////////////////////////////////////////////////////////////////////////////
// bnot
///////////////////////////////////////////////////////////////////////////////

static void mop_bnot(bool_t use_vector,
		     matrix_type_t at, byte_t* ap, int au, int av,
		     matrix_type_t ct, byte_t* cp, int cu, int cv,
		     size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if (at == ct) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp)) {
	    mtv_unary_eval(vfun_bnot_ops, fun_bnot_ops, at,
			   ap, au, av, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    mt_unary_eval(fun_bnot_ops, at,
			  ap, au, av, cp, cu, cv, n, m);
	}
    }
    else {
	iapply1(BNOT, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
// eval1
///////////////////////////////////////////////////////////////////////////////

static void mop_eval1(bool_t use_vector,
		      matrix_type_t at, byte_t* ap, int au, int av,
		      matrix_type_t ct, byte_t* cp, int cu, int cv,
		      size_t n, size_t m, void* prog)
{
    if (element_size(at) == element_size(ct)) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp)) {
	    eval_prog_unary((instr_t*)prog, at, TRUE,			    
			    ap, au, av, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    eval_prog_unary((instr_t*)prog, at, FALSE,
			    ap, au, av, cp, cu, cv, n, m);
	}
    }
    else {
	eval1((instr_t*)prog, at, ap, au, av,
	      ct, cp, cu, cv, n, m);	
    }
}


///////////////////////////////////////////////////////////////////////////////
// eval2
///////////////////////////////////////////////////////////////////////////////

static void mop_eval2(bool_t use_vector,
		      matrix_type_t at, byte_t* ap, int au, int av,
		      matrix_type_t bt, byte_t* bp, int bu, int bv,
		      matrix_type_t ct, byte_t* cp, int cu, int cv,
		      size_t n, size_t m, void* prog)
{
    if ((at == bt) && (element_size(bt) == element_size(ct))) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    eval_prog_binary((instr_t*)prog, at, TRUE,
			     ap, au, av, bp, bu, bv, cp, cu, cv, n, m);
	}
	else
#endif
	{
	    eval_prog_binary((instr_t*)prog, at, FALSE,
			     ap, au, av, bp, bu, bv, cp, cu, cv, n, m);	    
	}
    }
    else {
	eval2((instr_t*)prog, at, ap, au, av, bt, bp, bu, bv,
	      ct, cp, cu, cv, n, m);
    }
}


///////////////////////////////////////////////////////////////////////////////
// argmax 
///////////////////////////////////////////////////////////////////////////////


static void argmax(matrix_type_t at,byte_t* ap, int au, int av,
		   int32_t* cp, int cv,
		   size_t n, size_t m, int opts)
{
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
		if (CMP_FGT(v,max_v)) { max_v = v; max_i = i; }
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
		if (CMP_IGT(v,max_v)) { max_v = v; max_i = i; }
	    }
	    *cp = max_i;
	    cp += cv;
	    ap += av;
	}
    }
}

static void argmax_0(matrix_type_t at,byte_t* ap, int au, int av,
		     int32_t* ci, int32_t* cj,
		     size_t n, size_t m, int opts)
{
    int32_t   max_i = 1;
    int32_t   max_j = 1;
    
    if (is_float(at)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t max_v = read_af(ap);  // initialize
	int32_t i, j;

	for (i = 1; i <= (int32_t)n; i++) {
	    byte_t* ap1 = ap;
	    for (j = 1; j <= (int32_t)m; j++) {
		float64_t v = read_af(ap1);
		ap1 += av;
		if (CMP_FGT(v,max_v)) { max_v = v; max_i = i; max_j = j; }
	    }
	    ap += au;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t max_v = read_af(ap);  // initialize
	int32_t i,j;

	for (i = 1; i <= (int32_t)n; i++) {
	    byte_t* ap1 = ap;
	    for (j = 1; j <= (int32_t)m; j++) {
		int64_t v = read_af(ap1);
		ap1 += av;
		if (CMP_IGT(v,max_v)) { max_v = v; max_i = i; max_j = j; }
	    }
	    ap += au;
	}
    }
    *ci = max_i;
    *cj = max_j;
}

///////////////////////////////////////////////////////////////////////////////
// argmin
///////////////////////////////////////////////////////////////////////////////

static void argmin(matrix_type_t at,byte_t* ap, int au, int av,
		   int32_t* cp, int cv,
		   size_t n, size_t m, int opts)
{
    if (is_float(at)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	while(m--) {
	    byte_t* ap1 = ap;
	    size_t  n1 = n-1;
	    int32_t i = 1;
	    int32_t min_i = 1;
	    float64_t min_v = read_af(ap1);

	    ap1 += au;
	    while(n1--) {
		float64_t v = read_af(ap1);
		ap1 += au;
		i++;
		if (CMP_FLT(v,min_v)) { min_v = v; min_i = i; }
	    }
	    *cp = min_i;
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
	    int32_t min_i = 1;
	    int64_t min_v = read_af(ap1);

	    ap1 += au;
	    while(n1--) {
		int64_t v = read_af(ap1);
		ap1 += au;
		i++;
		if (CMP_ILT(v,min_v)) { min_v = v; min_i = i; }
	    }
	    *cp = min_i;
	    cp += cv;
	    ap += av;
	}
    }
}


static void argmin_0(matrix_type_t at,byte_t* ap, int au, int av,
		     int32_t* ci, int32_t* cj,
		     size_t n, size_t m, int opts)
{
    int32_t   min_i = 1;
    int32_t   min_j = 1;
    
    if (is_float(at)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[at];
	float64_t min_v = read_af(ap);  // initialize
	int32_t i, j;

	for (i = 1; i <= (int32_t)n; i++) {
	    byte_t* ap1 = ap;
	    for (j = 1; j <= (int32_t)m; j++) {
		float64_t v = read_af(ap1);
		ap1 += av;
		if (CMP_FLT(v,min_v)) { min_v = v; min_i = i; min_j = j; }
	    }
	    ap += au;
	}
    }
    else {
	int64_t (*read_af)(byte_t*) = read_int64_func[at];
	int64_t min_v = read_af(ap);  // initialize
	int32_t i,j;

	for (i = 1; i <= (int32_t)n; i++) {
	    byte_t* ap1 = ap;
	    for (j = 1; j <= (int32_t)m; j++) {
		int64_t v = read_af(ap1);
		ap1 += av;
		if (CMP_ILT(v,min_v)) { min_v = v; min_i = i; min_j = j; }
	    }
	    ap += au;
	}
    }
    *ci = min_i;
    *cj = min_j;
}


///////////////////////////////////////////////////////////////////////////////
// max along axis or if cv=0 then total max
///////////////////////////////////////////////////////////////////////////////

static void t_max(matrix_type_t at,byte_t* ap, int au, int av,
		  matrix_type_t ct,byte_t* cp, int cv,
		  size_t n, size_t m, int opts)
{
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
		if (CMP_FGT(v, max_v)) { max_v = v; }
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
		if (CMP_IGT(v, max_v)) { max_v = v; }
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
		  size_t n, size_t m, int opts)
{
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
		if (CMP_FLT(v,min_v)) { min_v = v; }
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
		if (CMP_ILT(v,min_v)) { min_v = v; }
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
	// (*mt_sigmoid_funcs[at])(ap, au, av, cp, cu, cv, n, m);
	mt_unary_eval(fun_sigmoid_ops, at,
		      ap, au, av, cp, cu, cv, n, m);	    	
    }
    else {
	apply1(SIGMOID, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

static void sigmoid_prime1(matrix_type_t at, byte_t* ap, int au, int av,
			   matrix_type_t ct, byte_t* cp, int cu, int cv,
			   size_t n, size_t m)
{
    if (at == ct) {
	// (*mt_sigmoid_prime1_funcs[at])(ap, au, av, cp, cu, cv, n, m);
	mt_unary_eval(fun_sigmoid_prime1_ops, at,
		      ap, au, av, cp, cu, cv, n, m);

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
	//mt_rectifier(at, ap, au, av, cp, cu, cv, n, m);
	mt_unary_eval(fun_rectifier_ops, at,
		      ap, au, av, cp, cu, cv, n, m);	
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
    // all types must be component types! 
    if ((at == bt) && (bt == ct)) {
#ifdef DEBUG
    enif_fprintf(stderr, "multiply: use_vector=%d\r\n", use_vector);
    enif_fprintf(stderr, "at=%d,au=%d,av=%d,an=%ld,am=%ld\r\n", at,au,av,an,am);
    enif_fprintf(stderr, "bt=%d,bu=%d,bv=%d,bn=%ld,bm=%ld\r\n", bt,bu,bv,bn,bm);
    enif_fprintf(stderr, "ct=%d,cu=%d,cv=%d\r\n", ct,cu,cv);

#endif
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(bp) && is_aligned(cp)) {
	    (*vproc_multiply_funcs[at])(ap,au,an,am,bp,bu,bn,bm,cp,cu,cv);
	}
	else
#endif
	{
	    (*proc_multiply_funcs[at])(ap,au,av,an,am,bp,bu,bv,bn,bm,cp,cu,cv);
	}
    }
    else {
	byte_t* bp0 = bp;

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
	    (*vproc_multiply_transposed_funcs[at])(ap,au,an,am,bp,bu,bn,bm,
						 cp,cu,cv);
	else
#endif
	    (*proc_multiply_transposed_funcs[at])(ap,au,av,an,am,bp,bu,bv,bn,bm,
						  cp,cu,cv);
    }
    else {
	byte_t* bp0 = bp;

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
	    (*vproc_kmultiply_funcs[at])(ap,au,an,am,bp,bu,bn,bm,kp,kv,km,
					 cp,cu,cv);
	}
	else
#endif
	{
	    (*proc_kmultiply_funcs[at])(ap,au,av,an,am,bp,bu,bv,bn,bm,kp,kv,km,
					cp,cu,cv);	    
	}
    }
    else {
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
	    (*vproc_kmultiply_transposed_funcs[at])(ap,au,an,am,bp,bu,bn,bm,
						    kp,kv,km,cp,cu,cv);
	else
#endif
	    (*proc_kmultiply_transposed_funcs[at])(ap,au,av,an,am,bp,bu,bv,bn,bm,kp,kv,km,cp,cu,cv);
    }
    else {
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
		    byte_t* cp, int cu, int cv,
		    size_t n, size_t m)
{
    size_t sz = element_size(at);

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
		     byte_t* cp, int cu,
		     size_t n, size_t m)
{
    m *= element_size(at);
    while(n--) {
	byte_t* ap1 = (byte_t*) ap;
	byte_t* cp1 = (byte_t*) cp;
	size_t m1 = m;
	while(m1 >= sizeof(vector_t)) {
	    vint8_t v = *(vint8_t*)ap1;
	    ap1 += sizeof(vector_t);
	    *(vint8_t*)cp1 = v;
	    cp1 += sizeof(vector_t);
	    m1  -= sizeof(vector_t);
	}
	if (m1) {
	    memcpy(cp1, ap1, m1);
	}
	ap += au;
	cp += cu;
    }
}
#endif

///////////////////////////////////////////////////////////////////////////////
//  simple copy
///////////////////////////////////////////////////////////////////////////////

static void mop_copy1(bool_t use_vector,
		      matrix_type_t at, byte_t* ap, int au, int av,
		      matrix_type_t ct, byte_t* cp, int cu, int cv,
		      size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    if (at == ct) {
#ifdef USE_VECTOR
	if (use_vector && is_aligned(ap) && is_aligned(cp))
	    mtv_copy(at, ap, au, cp, cu, n, m);
	else
#endif
	    mt_copy(at, ap, au, av, cp, cu, cv, n, m);
    }
    else {
	apply1(COPY, at, ap, au, av, ct, cp, cu, cv, n, m);
    }
}

///////////////////////////////////////////////////////////////////////////////
//  simple tranpose
///////////////////////////////////////////////////////////////////////////////

static void mop_transpose(bool_t use_vector,
			  matrix_type_t at, byte_t* ap, int au, int av,
			  matrix_type_t ct, byte_t* cp, int cu, int cv,
			  size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    UNUSED(use_vector);
    if (at == ct) {
	mt_copy(at, ap, au, av, cp, cv, cu, m, n);
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
			     size_t n_stride, size_t m_stride, size_t k_stride,
			     matrix_type_t type, byte_t* data)
{
    mp->n       = n;
    mp->m       = m;
    mp->n_stride = n_stride;
    mp->m_stride = m_stride;
    mp->k_stride = k_stride;
    mp->offset  = 0;
    mp->type    = type;
    mp->rowmajor = rowmajor;
    mp->rw_lock = NULL;
    mp->size    = n*m*element_size(type);
    mp->base    = NULL;
    mp->data    = data;
    mp->first   = data;
    mp->ptr     = 0;
    mp->parent  = ATOM(undefined);
    return mp;
}

//
// Allocate matrix memory space
// n = rows and m = columns
// each row is padded so each row is aligned to internal vector size
// so vector size is 64 then each row will be padded to 64 byte alignment.
// Extra memory is created to allow for memory to be transposed
// inline or by copy.
// example: align=8
//   create | 1 2 3 |
//   then memory that will be create is | 1 2 3 0 0 0 0 0 |
// but when transposed the data used must be
//   | 1 0 0 0 0 0 0 0 |
//   | 2 0 0 0 0 0 0 0 |
//   | 3 0 0 0 0 0 0 0 |
// so allocation is the max of a nxm matrix and mxn matrix
//
matrix_t* alloc_matrix_resource(size_t n, size_t m,
				bool_t rowmajor,matrix_type_t type,size_t align)
{
    matrix_t* mp = enif_alloc_resource(matrix_res, sizeof(matrix_t));

    if (mp != NULL) {
	size_t r_stride = (size_of_array(type,m)+align-1) & ~(align-1);
	size_t r_size   = n*r_stride;
	size_t c_stride = (size_of_array(type,n)+align-1) & ~(align-1);
	size_t c_size   = m*c_stride;
	size_t size     = op_max(r_size, c_size);
#ifdef DEBUG
	enif_fprintf(stderr, "n=%ld\n",  n);
	enif_fprintf(stderr, "m=%ld\n",  m);
	enif_fprintf(stderr, "align=%ld\n",  align);
	enif_fprintf(stderr, "r_stride=%ld\n",  r_stride);
	enif_fprintf(stderr, "r_size=%ld\n",    r_size);
	enif_fprintf(stderr, "c_stride=%ld\n",  c_stride);
	enif_fprintf(stderr, "c_size=%ld\n",    c_size);
	enif_fprintf(stderr, "size=%ld\n",      size);
	enif_fprintf(stderr, "align=%ld\n",     align);
	enif_fprintf(stderr, "component_size=%ld\n", component_size(type));
	enif_fprintf(stderr, "element_size=%ld\n", element_size(type));
	enif_fprintf(stderr, "before alloc mp=%p\n", mp);
#endif
	mp->n        = n;
	mp->m        = m;
	
	mp->n_stride = r_stride;
	mp->m_stride = element_size(type);
	mp->k_stride = component_size(type);
	mp->type    = type;
	mp->size    = 0;
	mp->offset  = 0;
	mp->rowmajor = rowmajor;
	mp->data    = NULL;
	mp->first   = NULL;
	mp->rw_lock = NULL;
	mp->ptr     = (uintptr_t)mp;
	mp->parent  = ATOM(undefined);

	if ((mp->base = enif_alloc(size+align-1)) != NULL) {
	    // enif_fprintf(stderr, "base=%p\n", mp->base);	    
	    mp->size  = size;
	    mp->rw_lock = enif_rwlock_create("matrix");
	    mp->data = align_ptr(mp->base,align);
	    mp->first = mp->data;
#ifdef DEBUG	    
	    enif_fprintf(stderr, "size=%ld\n", size);
	    enif_fprintf(stderr, "n_stride=%ld\n",     mp->n_stride);
	    enif_fprintf(stderr, "m_stride=%ld\n",     mp->m_stride);
	    enif_fprintf(stderr, "k_stride=%ld\n",     mp->k_stride);
#endif
	}
	else {
	    enif_release_resource(mp);
	    return NULL;
	}
    }
    return mp;
}

matrix_t* create_matrix(ErlNifEnv* env, unsigned int n, unsigned int m,
			bool_t rowmajor, matrix_type_t type,
			ERL_NIF_TERM* resp)
{
    matrix_t* np;

    if ((np = alloc_matrix_resource(n, m, rowmajor, type, ALIGN)) != NULL) {
	*resp = enif_make_resource(env,np);
	enif_release_resource(np);
	return np;
    }
    return NULL;
}

static int get_opts(ErlNifEnv* env, ERL_NIF_TERM arg, int* opts_ptr)
{
    ERL_NIF_TERM list = arg;
    ERL_NIF_TERM head, tail;    
    int opts = 0;

    while(enif_get_list_cell(env, list, &head, &tail)) {
	if (head == ATOM(abs)) opts |= OPT_ABS;
	else if (head == ATOM(descend)) opts |= OPT_DESCEND;
	else if (head == ATOM(ascend)) opts &= ~OPT_DESCEND;
	else return 0;
	list = tail;
    }
    if (!enif_is_empty_list(env, list))
	return 0;
    *opts_ptr = opts;
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

#ifdef DEBUG
static void print_hex(FILE* f, byte_t* ptr, size_t len)
{
    while(len--)
	enif_fprintf(f, "%02x", *ptr++);
}
#endif

//
// Read a element value assume sp points to an area that is
// possibly SIMD aligned and contain at least
//   ALLOC_COMPONENTS*sizeof(float64_t) bytes
// elements are then copied to fill at least SIMD vector size
// 
static int get_scalar(ErlNifEnv* env, ERL_NIF_TERM arg,
		      matrix_type_t type, byte_t* sp)
{
    double dv;
    ErlNifSInt64 iv;
    int i;
    size_t n = get_vector_size(type);
    size_t step = get_scalar_size(type);
    size_t m = VSIZE / step;  // components per vector
    size_t rep = 0;           // number of copies of components
    byte_t* sp0 = sp;
    
    if (n == 1) { // scalar 
	// example: value = 1000.0
	// int16, VSIZE=16, 
	// fill rep=8:
	//   | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
	rep = m;
	if (enif_get_double(env, arg, &dv))
	    write_float64(type, sp0, dv);
	else if (enif_get_int64(env, arg, &iv))
	    write_int64(type, sp0, iv);
	else
	    return 0;
	sp += step;
    }
    else {  // vector n=2,3,4,8,16 (valid)
	int arity;
	const ERL_NIF_TERM* elems;	
	type = get_scalar_type(type);

	// example: value = 1000.0, n = 2
	// int16x2, VSIZE=16
	//   |{1000,1000}|{1000,1000}|{1000,1000}|{1000,1000}|
	// fill rep=8 | 
	if (enif_get_double(env, arg, &dv)) {	    
	    rep = op_max(VSIZE/step, n);
	    write_float64(type, sp0, dv);
	    sp += step;	    
	}
	else if (enif_get_int64(env, arg, &iv)) {
	    rep = op_max(VSIZE / step, n);
	    write_int64(type, sp0, iv);
	    sp += step;
	}
	else if (enif_get_tuple(env, arg, &arity, &elems) && (arity==(int)n)) {
	    // example: value = {1000.0,1234.0,5678.0}, n = 3
	    // int16x3, VSIZE=16  
	    //   |{1000,1234,5678}|{1000,1234,5678}|{1000,1234,5678}|
	    //    {1000,1234,5678}|{1000,1234,5678}|{1000,1234,5678}|
	    //    {1000,1234,5678}|{1000,1234,5678}|
	    // fill rep=8 |
	    // rep = (2*3 + ((16-2*3) % 16)) / 2  #elements!
	    for (i = 0; i < (int)n; i++) {
		if (enif_get_double(env, elems[i], &dv))
		    write_float64(type, sp, dv);
		else if (enif_get_int64(env, elems[i], &iv))
		    write_int64(type, sp, iv);
		else
		    return 0;
		sp += step;
	    }
	    step *= n; // element size
	    rep = (step + ((VSIZE-step) % VSIZE)) / get_scalar_size(type);
	}
	else
	    return 0;
    }
    // fill (may be improved for power of two)
    for (i = 1; i < (int)rep; i++) {
	memcpy(sp, sp0, step);
	sp += step;
    }
#ifdef DEUBG
    enif_fprintf(stderr, "scalar:n=%ld, step=%ld, rep=%ld\r\n", n, step, rep);
    sp = sp0;
    enif_fprintf(stderr, "data=|");
    for (i = 0; i < (int)rep; i++) {
	print_hex(stderr, sp, step);
	enif_fprintf(stderr, "|");	
	sp += step;
    }
    enif_fprintf(stderr, "\r\n");
#endif
    return 1;
}

int is_resource(ErlNifEnv* env, ERL_NIF_TERM res)
{
    void* ptr;
    return enif_get_resource(env, res, matrix_res, &ptr);
}

int get_resource(ErlNifEnv* env, ERL_NIF_TERM term, void** mpp)
{
    int arity;
    const ERL_NIF_TERM* elems;
    
    if (enif_get_resource(env, term, matrix_res, mpp))
	return 1;
    if (!enif_get_tuple(env, term, &arity, &elems))
	return 0;
    if (arity == 2) { // resource as empty (magic) binaries
	ErlNifUInt64 ptr;
	if (!enif_get_uint64(env,elems[1],&ptr)) return 0;
	if (!enif_get_resource(env,elems[2],matrix_res,mpp)) return 0;
	if (ptr != (ErlNifUInt64) *mpp) return 0;
	return 1;
    }
    return 0;
}

// get size of element within matrix taking n_strind into account
static size_t mat_size(matrix_t* a, size_t n, size_t m)
{
    if ((a->n_stride == 0) && (a->m_stride == 0))
	return element_size(a->type);
    else
	return (n-1)*labs(a->n_stride) + m*labs(a->m_stride);
}

// check if (byte) offset is within matrix
static int is_valid_offset(matrix_t* a, uintptr_t offset, size_t n, size_t m)
{
    size_t vsize = mat_size(a, n, m);
    return ((offset+vsize) <= a->size);
}

static int get_matrix(ErlNifEnv* env, ERL_NIF_TERM arg,
			  matrix_t* mp, matrix_t** mpp)
{
    int arity;
    unsigned int type;
    size_t vsize;
    size_t byte_offset;
    const ERL_NIF_TERM* elems;
    matrix_t* rmp;

    if (!enif_get_tuple(env, arg, &arity, &elems))
	return 0;
    if (arity == 3) {  // matrix short format
	if (elems[0] != ATOM(matrix_t)) return 0;
	if (!enif_get_uint(env, elems[1], &type)) return 0;
	if (!get_resource(env, elems[2], (void**)mpp)) return 0;
	if (type != (*mpp)->type) return 0;
	return 1;
    }
    if (arity != 10) return 0;
    if (elems[0] != ATOM(matrix)) return 0;
    if (!enif_get_uint(env, elems[1], &type)) return 0;
    if (type > TYPE_MASK) return 0;
    if (!enif_get_uint64(env, elems[3], &mp->n)) return 0;
    if (!enif_get_uint64(env, elems[4], &mp->m)) return 0;
    if (!enif_get_int64(env, elems[5], &mp->n_stride)) return 0;
    if (!enif_get_int64(env, elems[6], &mp->m_stride)) return 0;
    if (!enif_get_int64(env, elems[7], &mp->k_stride)) return 0;
    if (!enif_get_uint64(env, elems[8], &mp->offset)) return 0;
    if (!get_bool(env, elems[9], &mp->rowmajor)) return 0;

    mp->type = type;
    byte_offset = mp->offset;

    vsize = mat_size(mp, mp->n, mp->m);
    
    if (get_resource(env, elems[2], (void**)&rmp)) {
	if ((byte_offset + vsize) > rmp->size)
	    return 0;
	mp->size = rmp->size;
	mp->base = rmp->base;
	mp->data = rmp->data;
	mp->first = rmp->data + byte_offset;
	mp->rw_lock = rmp->rw_lock;
	mp->parent = elems[2];
	mp->ptr   = rmp->ptr;
    }
    else {
	ErlNifBinary bin;
	if (!enif_inspect_binary(env, elems[2], &bin))
	    return 0;
	// check bounds
	if ((byte_offset + vsize) > bin.size)
	    return 0;
	mp->size = bin.size;
	mp->base = NULL;
	mp->data = bin.data;  // temporary!!!
	mp->first = mp->data + byte_offset;
	mp->rw_lock = NULL;
	mp->parent = elems[2];
	mp->ptr  = 0; // signals that we are not allowed to write
    }
    *mpp = mp;
    return 1;
}

// check if matrix a is submatrix of b or wiseversa
static bool_t is_overlapping(matrix_t* a, matrix_t* b)
{
    intptr_t a_offs, b_offs;
    intptr_t a0, a1, b0, b1;
    intptr_t c0, c1;
    size_t esize;

    if (a->base != b->base) return FALSE;  // check me!
    if (a->data != b->data) return FALSE;  // check me!
    if ((a->n_stride == 0) || (b->n_stride == 0)) return FALSE;

    esize = element_size(a->type);
    
    a_offs = a->first - a->data;
    b_offs = b->first - b->data;

    // left column
    a0 = (a_offs % a->n_stride) / esize;
    b0 = (b_offs % b->n_stride) / esize;
    c0 = op_max(a0, b0);

    // right column
    a1 = a0 + a->m - 1;
    b1 = b0 + b->m - 1;
    c1 = op_min(a1, b1);

    if (c0 > c1) return FALSE;

    // top row
    a0 = a_offs / a->n_stride;
    b0 = b_offs / b->n_stride;
    c0 = op_max(a0, b0);

    // bottom row
    a1 = a0 + a->n - 1;
    b1 = b0 + b->n - 1;
    c1 = op_min(a1, b1);
    
    if (c0 > c1) return FALSE;
    
    return TRUE;
}


// Get a writable resource matrix!
static int get_w_matrix(ErlNifEnv* env, ERL_NIF_TERM arg, matrix_t* mp,
			matrix_t** mpp)
{
    return get_matrix(env, arg, mp, mpp) && ((*mpp)->ptr != 0);
}

// get_scalar_matrix
// Parse one element argument int,float or vector and
// generate a matrix with that argument as data, setting
//
static int get_scalar_matrix(ErlNifEnv* env, ERL_NIF_TERM arg, matrix_t* mp,
			     matrix_t** mpp,
			     bool_t rowmajor, matrix_type_t type,
			     unsigned int n, unsigned int m)
{
    byte_t* sptr = align_ptr(mp->smem,ALIGN);
    if (!get_scalar(env, arg, type, sptr))
	return 0;
    init_matrix(mp, rowmajor, n, m, 0, 0, 0, type, sptr);
    *mpp = mp;
    return 1;
}

static ERL_NIF_TERM make_matrix_ptr(ErlNifEnv* env,
				    matrix_t* mp,
				    ERL_NIF_TERM res)
{
#if (ERL_NIF_MAJOR_VERSION < 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION < 12))
    if (is_resource(env, res))
	res = make_tuple(env,2,res,enif_make_uint64((uint64_t)mp->ptr));
#endif
    return enif_make_tuple(env, 10,
			   ATOM(matrix),
			   enif_make_uint(env,mp->type),
			   res,
			   enif_make_uint(env,mp->n),
			   enif_make_uint(env,mp->m),
			   enif_make_int64(env,mp->n_stride),
			   enif_make_int64(env,mp->m_stride),
			   enif_make_int64(env,mp->k_stride),
			   enif_make_uint64(env,mp->offset),
			   make_bool(env,mp->rowmajor));
}

static ERL_NIF_TERM make_matrix_t(ErlNifEnv* env,matrix_t* mp,
				  ERL_NIF_TERM res)
{
#if (ERL_NIF_MAJOR_VERSION < 2) || ((ERL_NIF_MAJOR_VERSION == 2) && (ERL_NIF_MINOR_VERSION < 12))
    if (is_resource(env, res))
	res = make_tuple(env,2,res,enif_make_uint64((uint64_t)mp->ptr));
#endif
    return enif_make_tuple(env, 3,
			   ATOM(matrix_t),
			   enif_make_uint(env, mp->type),
			   res);
}

// new_(N, M, Type, RowMajor, Data)
static ERL_NIF_TERM matrix_create(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    unsigned int n, m, type;
    ErlNifBinary binary;
    matrix_t* c;
    ERL_NIF_TERM res;
    bool_t rowmajor;
    UNUSED(argc);

    if (!enif_get_uint(env, argv[0], &n))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &m))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &type))
	return enif_make_badarg(env);
    if (type >= TYPE_MASK)
	return enif_make_badarg(env);
    if (!get_bool(env, argv[3], &rowmajor))
	return enif_make_badarg(env);
    if (!enif_inspect_iolist_as_binary(env, argv[4], &binary))
	return enif_make_badarg(env);
    if ((binary.size != 0) && (n*m*element_size(type) != binary.size))
	return enif_make_badarg(env);
    if ((c=create_matrix(env,n,m,rowmajor,type,&res)) == NULL)
	return enif_make_badarg(env);
    if (binary.size == n*m*element_size(type)) {
#ifdef DEBUG	
	enif_fprintf(stderr, "size=%ld, n=%ld, m=%ld, element_size=%ld\r\n",
		     binary.size, n, m, element_size(type));
	enif_fprintf(stderr, "c->n_stride=%ld, c->m_stride=%ld, c->k_stride=%ld, c->size=%ld\r\n",
		     c->n_stride, c->m_stride, c->k_stride, c->size);
	enif_fprintf(stderr, "element_size(%d)=%d\r\n",
	             type, element_size(type));
#endif
	if (c->n_stride == (ssize_t)(c->m*element_size(type))) {
	    memcpy(c->data, binary.data, c->size);
	}
	else {
	    byte_t* ap = c->data;
	    byte_t* bp = binary.data;
	    size_t  bu = c->m*element_size(type);
	    size_t  au = c->n_stride;
	    size_t i;
	    for (i=0; i<n; i++) {
		memcpy(ap, bp, bu);
		ap += au;
		bp += bu;
	    }
	}
    }
    return make_matrix_t(env,c,res);
}

static ERL_NIF_TERM matrix_size(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (a->rowmajor)
	return enif_make_tuple(env,2,
			       enif_make_uint(env, a->n),
			       enif_make_uint(env, a->m));
    else
	return enif_make_tuple(env,2,
			       enif_make_uint(env, a->m),
			       enif_make_uint(env, a->n));
}

static ERL_NIF_TERM matrix_native_vector_width(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
    
{
    UNUSED(argc);
    matrix_type_t type;
    
    if (argv[0] == ATOM(uint8)) type = UINT8;
    else if (argv[0] == ATOM(uint16)) type = UINT16;
    else if (argv[0] == ATOM(uint32)) type = UINT32;
    else if (argv[0] == ATOM(uint64)) type = UINT64;
    else if (argv[0] == ATOM(uint128)) type = UINT128;
    else if (argv[0] == ATOM(int8)) type = INT8;
    else if (argv[0] == ATOM(int16)) type = INT16;
    else if (argv[0] == ATOM(int32)) type = INT32;
    else if (argv[0] == ATOM(int64)) type = INT64;
    else if (argv[0] == ATOM(int128)) type = INT128;
    else if (argv[0] == ATOM(float16)) type = FLOAT16;
    else if (argv[0] == ATOM(float32)) type = FLOAT32;
    else if (argv[0] == ATOM(float64)) type = FLOAT64;
    else return enif_make_badarg(env);

    return enif_make_int(env, VSIZE / element_size(type));
}

static ERL_NIF_TERM matrix_preferred_vector_width(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    UNUSED(argc);
    matrix_type_t type;
    
    if (argv[0] == ATOM(uint8)) type = UINT8;
    else if (argv[0] == ATOM(uint16)) type = UINT16;
    else if (argv[0] == ATOM(uint32)) type = UINT32;
    else if (argv[0] == ATOM(uint64)) type = UINT64;
    else if (argv[0] == ATOM(uint128)) type = UINT128;
    else if (argv[0] == ATOM(int8)) type = INT8;
    else if (argv[0] == ATOM(int16)) type = INT16;
    else if (argv[0] == ATOM(int32)) type = INT32;
    else if (argv[0] == ATOM(int64)) type = INT64;
    else if (argv[0] == ATOM(int128)) type = INT128;
    else if (argv[0] == ATOM(float16)) type = FLOAT16;
    else if (argv[0] == ATOM(float32)) type = FLOAT32;
    else if (argv[0] == ATOM(float64)) type = FLOAT64;
    else return enif_make_badarg(env);

    return enif_make_int(env, VSIZE / element_size(type));
}


///////////////////////////////////////////////////////////////////////////////
//  copy element by element via index, assume at == ct
///////////////////////////////////////////////////////////////////////////////

static void index_copy(matrix_type_t at, byte_t* ap, int au, int av,
		       byte_t* ip, int iu, int iv,
		       byte_t* cp, int cu, int cv,
		       size_t n, size_t m)
{
    size_t sz = element_size(at);

    while(n--) {
	byte_t* ap1 = ap;
	byte_t* ip1 = ip;
	byte_t* cp1  = cp;
	size_t m1 = m;
	while(m1--) {
	    int32_t i = ((*(int32_t*)ip1)-1)*au;  // get offset
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
int index_check(byte_t* ip, int iu, int iv, size_t in, size_t im, size_t k)
{
    while(in--) {
	size_t imm = im;
	byte_t* ip1 = ip;
	while(imm--) {
	    int32_t ix = *(int32_t*)ip1;
	    if (ix < 1) return 0;
	    if (ix > (int32_t)k) return 0;
	    ip1 += iv;
	}
	if (iu == 0) return 1; // repeating data
	ip += iu;
    }
    return 1;
}

static ERL_NIF_TERM matrix_elem(ErlNifEnv* env, matrix_t* a,
				unsigned int i,  unsigned int j)
{
    byte_t* ptr;
    ERL_NIF_TERM r;
    
    if (a->rowmajor) {
	if ((i < 1) || (i > a->n))
	    return EXCP_BADARG_N(env, 0, "index out of bounds");
	if ((j < 1) || (j > a->m))
	    return EXCP_BADARG_N(env, 1, "index out of bounds");
	if ((a->n_stride == 0) && (a->m_stride == 0))  // special case
	    ptr = a->first;
	else
	    ptr = a->first + (i-1)*a->n_stride + (j-1)*a->m_stride;
    }
    else { // row when column major
	if ((j < 1) || (j > a->n))
	    return EXCP_BADARG_N(env, 1, "index out of bounds");
	if ((i < 1) || (i > a->m))
	    return EXCP_BADARG_N(env, 0, "index out of bounds");
	if ((a->n_stride == 0) && (a->m_stride == 0))  // special case	
	    ptr = a->first;
	else
	    ptr = a->first + (i-1)*a->m_stride + (j-1)*a->n_stride;
    }
    matrix_r_lock(a);
    r = make_element(env, a->type, ptr);
    matrix_r_unlock(a);
    return r;
}


ERL_NIF_TERM matrix_element(ErlNifEnv* env, int argc,
			    const ERL_NIF_TERM argv[])
{
    if (argc == 2) {
	const ERL_NIF_TERM* elems;
	matrix_t mat[2];
	matrix_t *a, *index, *c;
	ERL_NIF_TERM res;
	unsigned int i, j;	
	int arity;
	
	if (!get_matrix(env, argv[1], &mat[1], &a))
	    return EXCP_BADARG_N(env, 1, "matrix expected");	

	if (enif_get_tuple(env, argv[0], &arity, &elems) && (arity == 2)) {
	    if (!enif_get_uint(env, elems[0], &i))
		return EXCP_BADARG_N(env, 0, "fst not positive integer");
	    if (!enif_get_uint(env, elems[1], &j))
		return EXCP_BADARG_N(env, 0, "snd not positive integer");
	    return matrix_elem(env, a, i, j);
	}
	    
	if (!get_matrix(env, argv[0], &mat[0], &index)) {
	    if (!get_scalar_matrix(env,argv[0],&mat[0],&index,
				   a->rowmajor,INT32,a->n,1))
		return EXCP_ERROR_N(env, 0, "index matrix expected");
	}
	if (index->type != INT32)
	    return EXCP_BADARG_N(env, 0, "index matrix must be int32");
	if (a->rowmajor == index->rowmajor) {
	    if ((index->n > a->n) || (index->m > a->m))
		return EXCP_BADARG_N(env, 0, "index matrix too large");
	    if (!index_check(index->first, index->n_stride, index->m_stride,
			     index->n, index->m, a->n))
		return EXCP_BADARG_N(env, 0, "index matrix out of bounds");
	    if ((c=create_matrix(env,index->n,index->m,index->rowmajor,
				 a->type,&res)) == NULL)
		return EXCP_ERROR_N(env, 0, "allocation failure");
	    matrix_rr_lock(a,index);
	    index_copy(a->type, a->first, a->n_stride, a->m_stride,
		       index->first, index->n_stride, index->m_stride,
		       c->first, c->n_stride, c->m_stride,
		       c->n, c->m);
	    matrix_rr_unlock(a,index);
	    return make_matrix_t(env,c,res);
	}
	else { // a.rowmajor / index.rowmajor
	    if ((index->m > a->n) || (index->n > a->m))
		return EXCP_BADARG_N(env, 0, "index matrix too large");		
	    if (!index_check(index->first, index->n_stride, index->m_stride,
			     index->n, index->m, a->m))
		return EXCP_BADARG_N(env, 0, "index matrix out of bounds");
	    if ((c=create_matrix(env,index->n,index->m,index->rowmajor,
				 a->type,&res)) == NULL)
		return EXCP_ERROR_N(env, 0, "allocation failure");
	    matrix_rr_lock(a,index);
	    index_copy(a->type, a->first, a->m_stride, a->n_stride,
		       index->first, index->n_stride, index->m_stride,
		       c->first, c->n_stride, c->m_stride,
		       c->n, c->m);
	    matrix_rr_unlock(a,index);
	    return make_matrix_t(env,c,res);
	}
    }
    else {
	unsigned int i, j;
	matrix_t mat[1];
	matrix_t* a;

	if (!enif_get_uint(env, argv[0], &i))
	    return EXCP_BADARG_N(env, 0, "index not positive integer");
	if (!enif_get_uint(env, argv[1], &j))
	    return EXCP_BADARG_N(env, 1, "index not positive integer");
	if (!get_matrix(env, argv[2], &mat[0], &a))
	    return EXCP_BADARG_N(env, 2, "matrix expected");
	return matrix_elem(env, a, i, j);
    }
}

// DESTRUCTIVE
// setelement(I, J, A, V) -> A[I][J] = V  I and J are 1-based
ERL_NIF_TERM matrix_setelement(ErlNifEnv* env, int argc,
			       const ERL_NIF_TERM argv[])
{
    unsigned int i, j;
    matrix_t mat[1];
    matrix_t *a;
    byte_t* ptr;
    scalar_t value;
    UNUSED(argc);

    if (!enif_get_uint(env, argv[0], &i))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &j))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[2], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_scalar(env, argv[3], a->type, value.data))
	return enif_make_badarg(env);
    if (a->rowmajor) {
	if ((i < 1) || (i > a->n))
	    return enif_make_badarg(env);
	if ((j < 1) || (j > a->m))
	    return enif_make_badarg(env);
	if (a->ptr == 0) // only resource matrix may be update!
	    return enif_make_badarg(env);
	if ((a->n_stride == 0) && (a->m_stride == 0))
	    ptr = a->first;
	else
	    ptr = a->first + (i-1)*a->n_stride + (j-1)*a->m_stride;
    }
    else {
	if ((j < 1) || (j > a->n)) // row when column major
	    return enif_make_badarg(env);
	if ((i < 1) || (i > a->m))
	    return enif_make_badarg(env);
	if (a->ptr == 0) // only resource matrix may be update!
	    return enif_make_badarg(env);
	if ((a->n_stride == 0) && (a->m_stride == 0))
	    ptr = a->first;
	else
	    ptr = a->first + (i-1)*a->m_stride + (j-1)*a->n_stride;
    }
    matrix_w_lock(a);
    memcpy(ptr, value.data, element_size(a->type));
    matrix_w_unlock(a);
    return argv[2];
}

// matrix apply1(func,A,B) -> C, check all rowmajor variants
// with and without accelerations
static void m_apply1(
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t ct, byte_t* cp, int cu, int cv,
		 size_t n, size_t m, void* extra),
    void* extra,
    matrix_t* ap, matrix_t* cp)
{
    bool_t use_vector = (labs(ap->m_stride)==element_size(ap->type));

    if (cp->rowmajor == ap->rowmajor)
	func(use_vector,
	     ap->type, ap->first, ap->n_stride, ap->m_stride,
	     cp->type, cp->first, cp->n_stride, cp->m_stride,
	     cp->n, cp->m, extra);
    else
	func(FALSE,
	     ap->type, ap->first, ap->m_stride, ap->n_stride,
	     cp->type, cp->first, cp->n_stride, cp->m_stride,
	     cp->n, cp->m, extra);
}


// matrix apply2(func,A,B) -> C, check all rowmajor variants
// with and without accelerations

static void m_apply2(
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t bt, byte_t* bp, int bu, int bv,
		 matrix_type_t ct, byte_t* cp, int cu, int cv,
		 size_t n, size_t m, void* extra),
    void* extra,
    matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    bool_t use_vector =
	(labs(ap->m_stride)==element_size(ap->type)) &&
	(labs(bp->m_stride)==element_size(bp->type));

    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor)
	    func(use_vector,
		 ap->type, ap->first, ap->n_stride, ap->m_stride,
		 bp->type, bp->first, bp->n_stride, bp->m_stride,
		 cp->type, cp->first, cp->n_stride, cp->m_stride,
		 cp->n, cp->m, extra);
	else if (!ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		 ap->type, ap->first, ap->m_stride, ap->n_stride,
		 bp->type, bp->first, bp->n_stride, bp->m_stride,
		 cp->type, cp->first, cp->n_stride, cp->m_stride,
		 cp->n, cp->m, extra);
	else if (ap->rowmajor && !bp->rowmajor)
	    func(FALSE,
		 ap->type, ap->first, ap->n_stride, ap->m_stride,
		 bp->type, bp->first, bp->m_stride, bp->n_stride,
		 cp->type, cp->first, cp->n_stride, cp->m_stride,
		 cp->n, cp->m, extra);
	else
	    func(FALSE,
		 ap->type, ap->first, ap->m_stride, ap->n_stride,
		 bp->type, bp->first, bp->m_stride, bp->n_stride,
		 cp->type, cp->first, cp->n_stride, cp->m_stride,
		 cp->n, cp->m, extra);
    }
    else {
	if (ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->n_stride, ap->m_stride,
		bp->type, bp->first, bp->n_stride, bp->m_stride,
		cp->type, cp->first, cp->m_stride, cp->n_stride,
		cp->n, cp->m, extra);
	else if (!ap->rowmajor && bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->m_stride, ap->n_stride,
		bp->type, bp->first, bp->n_stride, bp->m_stride,
		cp->type, cp->first, cp->m_stride, cp->n_stride,
		cp->n, cp->m, extra);
	else if (ap->rowmajor && !bp->rowmajor)
	    func(FALSE,
		ap->type, ap->first, ap->n_stride, ap->m_stride,
		bp->type, bp->first, bp->m_stride, bp->n_stride,
		cp->type, cp->first, cp->m_stride, cp->n_stride,
		cp->n, cp->m, extra);
	else
	    func(use_vector,
		ap->type, ap->first, ap->m_stride, ap->n_stride,
		bp->type, bp->first, bp->m_stride, bp->n_stride,
		cp->type, cp->first, cp->m_stride, cp->n_stride,
		cp->n, cp->m, extra);
    }
}

// matrix apply2(func,A,B) -> C, check all rowmajor variants
// with and without accelerations return a scalar values as erlang term

static void s_apply2(
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t bt, byte_t* bp, int bu, int bv,
		 matrix_type_t ct, byte_t* cp,
		 size_t n, size_t m, void* extra),
    void* extra,
    matrix_t* ap, matrix_t* bp, matrix_type_t ct, byte_t* cp)
{
    bool_t use_vector =
	(labs(ap->m_stride)==element_size(ap->type)) &&
	(labs(bp->m_stride)==element_size(bp->type));

    if (ap->rowmajor && bp->rowmajor)
	func(use_vector,
	     ap->type, ap->first, ap->n_stride, ap->m_stride,
	     bp->type, bp->first, bp->n_stride, bp->m_stride,
	     ct, cp, ap->n, ap->m, extra);
    else if (!ap->rowmajor && bp->rowmajor)
	func(FALSE,
	     ap->type, ap->first, ap->m_stride, ap->n_stride,
	     bp->type, bp->first, bp->n_stride, bp->m_stride,
	     ct, cp, bp->n, bp->m, extra);
    else if (ap->rowmajor && !bp->rowmajor)
	func(FALSE,
	     ap->type, ap->first, ap->n_stride, ap->m_stride,
	     bp->type, bp->first, bp->m_stride, bp->n_stride,
	     ct, cp, ap->n, ap->m, extra);
    else
	func(use_vector,
	     ap->type, ap->first, ap->m_stride, ap->n_stride,
	     bp->type, bp->first, bp->m_stride, bp->n_stride,
	     ct, cp, bp->m, bp->n, extra);
}

//
// binary_op(env,arc,argv,func)
//       
ERL_NIF_TERM binary_op(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[],
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t bt, byte_t* bp, int bu, int bv,
		 matrix_type_t ct, byte_t* cp, int cu, int cv,
		 size_t n, size_t m, void* extra),
    void* extra,
    matrix_type_t (*coerce_type)(matrix_type_t at,
				 matrix_type_t bt),
    matrix_type_flags_t arg1_types,
    matrix_type_flags_t arg2_types,
    matrix_type_flags_t return_types)
{
    matrix_t mat[3];
    matrix_t *a, *b, *c;
    
    if (get_matrix(env, argv[0], &mat[0], &a)) {
	if (!get_matrix(env, argv[1], &mat[1], &b)) {
	    if (!get_scalar_matrix(env,argv[1],&mat[1],&b,
				   a->rowmajor,a->type,a->n,a->m))
		return enif_make_badarg(env);
	}
    }
    else {
	if (!get_matrix(env, argv[1], &mat[1], &b))
	    return enif_make_badarg(env);
	if (!get_scalar_matrix(env,argv[0],&mat[0],&a,
			       b->rowmajor,b->type,b->n,b->m))
	    return enif_make_badarg(env);
    }
    
    if ((a->rowmajor == b->rowmajor) && ((a->n != b->n) || (a->m != b->m)))
	return enif_make_badarg(env);
    else if ((a->rowmajor != b->rowmajor) && ((a->n != b->m) || (a->m != b->n)))
	return enif_make_badarg(env);

    if (!IS_TYPE(a->type, arg1_types))
	return enif_make_badarg(env);
    if (!IS_TYPE(b->type, arg2_types))
	return enif_make_badarg(env);

    if (argc == 2) {
	ERL_NIF_TERM res;
	matrix_type_t ct = (*coerce_type)(a->type, b->type);

	if (!IS_TYPE(ct, return_types))
	    return enif_make_badarg(env);
	
	if ((c=create_matrix(env,a->n,a->m,a->rowmajor,ct,&res)) == NULL)
	    return enif_make_badarg(env);
	
	matrix_rr_lock(a, b);
	m_apply2(func, extra, a, b, c);
	matrix_rr_unlock(a, b);
	return make_matrix_t(env, c, res);
    }
    else { // argc == 3
	if (!get_w_matrix(env, argv[2], &mat[2], &c))
	    return enif_make_badarg(env);

	if (!IS_TYPE(c->type, return_types))
	    return enif_make_badarg(env);

	if ((a->rowmajor == c->rowmajor) && ((a->n != c->n) || (a->m != c->m)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) &&
		 ((a->n != c->m) || (a->m != c->n)))
	    return enif_make_badarg(env);
	matrix_rrw_lock(a, b, c);
	m_apply2(func, extra, a, b, c);
	matrix_rrw_unlock(a, b, c);
	return argv[2];
    }
}

ERL_NIF_TERM unary_op(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[],
    void (*func)(bool_t use_vector,
		 matrix_type_t at, byte_t* ap, int au, int av,
		 matrix_type_t ct, byte_t* cp, int cu, int cv,
		 size_t n, size_t m, void* extra),
    void* extra,
    matrix_type_t (*coerce_type)(matrix_type_t at),
    matrix_type_flags_t arg_types,
    matrix_type_flags_t return_types)    
{
    matrix_t mat[2];
    matrix_t *a, *c;

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);

    if (!IS_TYPE(a->type, arg_types))
	return enif_make_badarg(env);    
    
    if (argc == 1) {
	ERL_NIF_TERM res;
	matrix_type_t ct = (*coerce_type)(a->type);

	if (!IS_TYPE(ct, return_types))
	    return enif_make_badarg(env);	
	
	if ((c=create_matrix(env,a->n,a->m,a->rowmajor,ct,&res)) == NULL)
	    return enif_make_badarg(env);
	
	matrix_r_lock(a);
	m_apply1(func,extra,a,c);
	matrix_r_unlock(a);
	return make_matrix_t(env,c,res);
    }
    else { // args == 2
	if (!get_w_matrix(env, argv[1], &mat[1], &c))
	    return enif_make_badarg(env);
	
	if (!IS_TYPE(c->type, return_types))
	    return enif_make_badarg(env);

	if ((a->rowmajor == c->rowmajor) && ((a->n != c->n) || (a->m != c->m)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) &&
		 ((a->n != c->m) || (a->m != c->n)))
	    return enif_make_badarg(env);

	matrix_rw_lock(a, c);
	m_apply1(func, extra, a, c);
	matrix_rw_unlock(a, c);
	return argv[1];
    }
}

// add element wise
ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_add, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}


// subtract element wise
ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_subtract, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}


// multiply element wise
ERL_NIF_TERM matrix_times(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_times, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}

// divide element wise
ERL_NIF_TERM matrix_divide(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_divide, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}

// remainder element wise
ERL_NIF_TERM matrix_remainder(ErlNifEnv* env,int argc,const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_remainder, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}

// min element wise
ERL_NIF_TERM matrix_minimum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_minimum, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}

// max element wise
ERL_NIF_TERM matrix_maximum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_maximum, NULL,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);
}

// multiply A and B element wise but only the rows as
// controlled by matrix K
//
ERL_NIF_TERM matrix_ktimes(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t mat[4];
    matrix_t *a, *b, *k, *c;

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &mat[1], &b))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[2], &mat[2], &k))
	return enif_make_badarg(env);

    if ((a->rowmajor == b->rowmajor) && ((a->n != b->n) || (a->m != b->m)))
	return enif_make_badarg(env);
    else if ((a->rowmajor != b->rowmajor) && ((a->n != b->m) || (a->m != b->n)))
	return enif_make_badarg(env);

    if (k->type != INT32)
	return enif_make_badarg(env);
    if (!k->rowmajor || (k->n != 1))
	return enif_make_badarg(env);

    if (argc == 3) {
	ERL_NIF_TERM res;
	matrix_type_t ct = combine_type(a->type, b->type);

	if ((c=create_matrix(env,a->n,a->m,a->rowmajor,ct,&res)) == NULL)
	    return enif_make_badarg(env);

	matrix_rr_lock(a,b);
	m_apply2(mop_times,NULL,a,b,c);
	matrix_rr_unlock(a,b);
	return make_matrix_t(env,c,res);
    }
    else {  // argc == 4
	if (!get_w_matrix(env, argv[3], &mat[3], &c))
	    return enif_make_badarg(env);
	if ((a->rowmajor == c->rowmajor) && ((a->n != c->n) || (a->m != c->m)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) &&
		 ((a->n != c->m) || (a->m != c->n)))
	    return enif_make_badarg(env);

	matrix_rrw_lock(a, b, c);
	m_apply2(mop_times, NULL, a, b, c);
	matrix_rrw_unlock(a, b, c);
	return argv[2];
    }
}

static void m_multiply(matrix_t* ap, matrix_t* bp, matrix_t* cp)
{
    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor) {
	    multiply(TRUE,
		     ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		     bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		     cp->type,cp->first,cp->n_stride,cp->m_stride);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    multiply_t(TRUE,
		       ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		       bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		       cp->type,cp->first,cp->n_stride,cp->m_stride);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    multiply(FALSE,
		     ap->type,ap->first,ap->m_stride,ap->n_stride,ap->m,ap->n,
		     bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		     cp->type,cp->first,cp->n_stride,cp->m_stride);
	}
	else { // !ap->rowmajor && !bp->rowmajor (NOTE A/B swap!)
	    multiply(TRUE,
		     bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		     ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		     cp->type,cp->first,cp->m_stride,cp->n_stride);
	}
    }
    else { // !cp->rowmajor
	if (ap->rowmajor && bp->rowmajor) {
	    multiply(TRUE,
		     ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		     bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		     cp->type,cp->first,cp->m_stride,cp->n_stride);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    multiply_t(TRUE,
		       ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		       bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		       cp->type,cp->first,cp->m_stride,cp->n_stride);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    multiply(FALSE,
		     ap->type,ap->first,ap->m_stride,ap->n_stride,ap->m,ap->n,
		     bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		     cp->type,cp->first,cp->m_stride,cp->n_stride);
	}
	else { // !ap->rowmajor && !bp->rowmajor
	    multiply(TRUE,
		     bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		     ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		     cp->type,cp->first,cp->n_stride,cp->m_stride);
	}
    }
}

// multiply A*B = C matrices
// NOTE!  C should not be any of A or B, but keep it for a while!
// FIXME!

ERL_NIF_TERM matrix_multiply(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    matrix_t mat[3];
    matrix_t *a, *b, *c;
    size_t n, m;
    bool_t rowmajor = TRUE;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &mat[1], &b))
	return enif_make_badarg(env);

    if (a->rowmajor) {
	if (b->rowmajor) {
	    if (a->m != b->n) return enif_make_badarg(env);
	    n = a->n; m = b->m;
	} else {
	    if (a->m != b->m) return enif_make_badarg(env);
	    n = a->n; m = b->n;
	}
    }
    else {
	if (b->rowmajor) {
	    if (a->n != b->n) return enif_make_badarg(env);
	    n = a->m; m = b->m;
	} else {
	    if (a->n != b->m) return enif_make_badarg(env);
	    n = a->m; m = b->n;
	}
    }
    // check that elements have no vector type (yet)
    if ((get_vector_size(a->type) > 1) ||
	(get_vector_size(b->type) > 1))
	return enif_make_badarg(env);

    if ((argc == 2) ||
	((argc == 3) && get_bool(env, argv[2], &rowmajor))) {
	matrix_type_t ct = combine_type(a->type, b->type);
	ERL_NIF_TERM res = 0;

	if (rowmajor) {
	    if ((c=create_matrix(env,n,m,TRUE,ct,&res)) == NULL)
		return enif_make_badarg(env);
	}
	else {
	    if ((c=create_matrix(env,m,n,FALSE,ct,&res)) == NULL)
		return enif_make_badarg(env);
	}
	matrix_rr_lock(a,b);
	m_multiply(a,b,c);
	matrix_rr_unlock(a,b);
	return make_matrix_t(env,c,res);
    }
    else { // argc == 3
	if (!get_w_matrix(env, argv[2], &mat[2], &c))
	    return enif_make_badarg(env);

	if (c->rowmajor && ((c->n != n) || (c->m != m)))
	    return enif_make_badarg(env);
	if (!c->rowmajor && ((c->n != m) || (c->m != n)))
	    return enif_make_badarg(env);

	if (get_vector_size(c->type) > 1)
	    return enif_make_badarg(env);

	matrix_rrw_lock(a, b, c);
	// FIXME if a, b or both are equal to c then C
	// must be computed in a copy then copied back!
	m_multiply(a, b, c);
	matrix_rrw_unlock(a, b, c);
	return argv[2];
    }
}


static void k_multiply(matrix_t* ap, matrix_t* bp, matrix_t* kp,
		       matrix_t* cp)
{
    if (cp->rowmajor) {
	if (ap->rowmajor && bp->rowmajor) {
	    kmultiply(TRUE,
		      ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		      bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		      (int32_t*)kp->first,kp->m_stride,kp->m,
		      cp->type,cp->first,cp->n_stride,cp->m_stride);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    kmultiply_t(TRUE,
			ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
			bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
			(int32_t*)kp->first,kp->m_stride,kp->m,
			cp->type,cp->first,cp->n_stride,cp->m_stride);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    kmultiply(FALSE,
		      ap->type,ap->first,ap->m_stride,ap->n_stride,ap->m,ap->n,
		      bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		      (int32_t*)kp->first,kp->m_stride,kp->m,
		      cp->type,cp->first,cp->n_stride,cp->m_stride);
	}
	else { // !ap->rowmajor && !bp->rowmajor (NOTE A/B swap!)
	    // FIXME: check how to check k in this case
	    kmultiply(TRUE,
		      bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		      ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		      (int32_t*)kp->first,kp->m_stride,kp->m,
		      cp->type,cp->first,1,cp->n_stride);
	}
    }
    else { // !cp->rowmajor
	if (ap->rowmajor && bp->rowmajor) {
	    kmultiply(TRUE,
		      ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		      bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		      (int32_t*)kp->first,kp->m_stride,kp->m,
		      cp->type,cp->first,1,cp->n_stride);
	} else if (ap->rowmajor && !bp->rowmajor) {
	    kmultiply_t(TRUE,
			ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
			bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
			(int32_t*)kp->first,kp->m_stride,kp->m,
			cp->type,cp->first,1,cp->n_stride);
	} else if (!ap->rowmajor && bp->rowmajor) {
	    kmultiply(FALSE,
		      ap->type,ap->first,ap->m_stride,ap->n_stride,ap->m,ap->n,
		      bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		      (int32_t*)kp->first,kp->m_stride,kp->m,
		      cp->type,cp->first,1,cp->n_stride);
	}
	else { // !ap->rowmajor && !bp->rowmajor
	    // FIXME: check how to check k in this case
	    kmultiply(TRUE,
		      bp->type,bp->first,bp->n_stride,bp->m_stride,bp->n,bp->m,
		      ap->type,ap->first,ap->n_stride,ap->m_stride,ap->n,ap->m,
		      (int32_t*)kp->first,kp->m_stride,kp->m,
		      cp->type,cp->first,cp->n_stride,cp->m_stride);
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
    matrix_t mat[4];
    matrix_t *a, *b, *k, *c;
    size_t n, m;
    bool_t rowmajor = TRUE;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &mat[1], &b))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[2], &mat[2], &k))
	return enif_make_badarg(env);

    if (a->rowmajor) {
	if (b->rowmajor) {
	    if (a->m != b->n) return enif_make_badarg(env);
	    n = a->n; m = b->m;
	} else {
	    if (a->m != b->m) return enif_make_badarg(env);
	    n = a->n; m = b->n;
	}
    }
    else {
	if (b->rowmajor) {
	    if (a->n != b->n) return enif_make_badarg(env);
	    n = a->m; m = b->m;
	} else {
	    if (a->n != b->m) return enif_make_badarg(env);
	    n = a->m; m = b->n;
	}
    }

    if (k->type != INT32)
	return enif_make_badarg(env);
    if (!k->rowmajor || (k->n != 1))
	return enif_make_badarg(env);

    if ((argc == 3) ||
	((argc == 4) && get_bool(env, argv[3], &rowmajor))) {
	matrix_type_t ct = combine_type(a->type, b->type);
	ERL_NIF_TERM res = 0;

	if (rowmajor) {
	    if ((c=create_matrix(env,n,m,TRUE,ct,&res))==NULL)
		return enif_make_badarg(env);
	}
	else {
	    if ((c=create_matrix(env,m,n,FALSE,ct,&res)) == NULL)
		return enif_make_badarg(env);
	}
	matrix_rrr_lock(a,b,k);
	k_multiply(a,b,k,c);
	matrix_rrr_unlock(a,b,k);
	return make_matrix_t(env,c,res);
    }
    else { // argc == 4
	if (!get_w_matrix(env, argv[3], &mat[3], &c))
	    return enif_make_badarg(env);

	if (c->rowmajor && ((c->n != n) || (c->m != m)))
	    return enif_make_badarg(env);
	if (!c->rowmajor && ((c->n != m) || (c->m != n)))
	    return enif_make_badarg(env);

	matrix_rrrw_lock(a, b, k, c);
	// FIXME if a, b or both is equal to c then C
	// must be computed in a copy then copied back!
	k_multiply(a, b, k, c);
	matrix_rrrw_unlock(a, b, k, c);
	return argv[3];
    }
}

// partition around a pivot element high -> low !!! abs sort?
static int topk_partition_f(float64_t* ap, int* ip, int l, int h)
{
    float64_t p = ap[l];
    int i = l-1;
    int j = h+1;

    while(1) {
	do { i++; } while(ap[i] > p);
	do { j--; } while(ap[j] < p);
	if (i >= j) return j;
	swap_array_elem(ap, i, j);
	swap_array_elem(ip, i, j);
    }
    return -1;
}


// abs sort?
static int topk_partition_i(int64_t* ap, int* ip, int l, int h)
{
    int64_t p = ap[l];
    int i = l-1;
    int j = h+1;

    while(1) {
	do { i++; } while(ap[i] > p);
	do { j--; } while(ap[j] < p);
	if (i >= j) return j;
	swap_array_elem(ap, i, j);
	swap_array_elem(ip, i, j);
    }
    return -1;
}

// simple topk select algorithm
// get data as integer, float  and label data with index
static void topk(int k, matrix_type_t at,byte_t* ap,int au,
		 int32_t* cp,size_t m)
{
    int32_t index[m];
    int i, p, l, h;
    int ki = k;

//    au = size_of_array(at,au);

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
    matrix_t mat[2];
    matrix_t *a, *c;
    ERL_NIF_TERM res;
    int k;
    size_t m;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (a->rowmajor) { // column vector
	if (a->m != 1) return enif_make_badarg(env);
	m = a->n;
    }
    else { // column vector
	if (!a->rowmajor && (a->n != 1))
	    return enif_make_badarg(env);
	m = a->m;
    }
    if (!enif_get_int(env, argv[1], &k) || (k < 0))
	return enif_make_badarg(env);
    if (k == 0)
	return ATOM(undefined);
    // c matrix is a 1xM matrix of INT32
    if ((c=create_matrix(env,1,k,TRUE,INT32,&res)) == NULL)
	return enif_make_badarg(env);
    // the elements in A are split and top K elements indices are stored in C
    if (a->rowmajor)
	topk(k, a->type, a->first, a->n_stride, (int32_t*)c->first, m);
    else
	topk(k, a->type, a->first, 1, (int32_t*)c->first, m);

    return make_matrix_t(env,c,res);
}

// negate a matrix
ERL_NIF_TERM matrix_negate(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return unary_op(env, argc, argv, mop_negate, NULL,
		    copy_type,
		    ALL_TYPES, ALL_TYPES);
}

// Invert float arguments element wise
ERL_NIF_TERM matrix_reciprocal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return unary_op(env, argc, argv, mop_reciprocal, NULL,
		    copy_type,
		    NONE_INTEGER_TYPES, NONE_INTEGER_TYPES);
}

// sqare root integer/float arguments element wise
ERL_NIF_TERM matrix_sqrt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return unary_op(env, argc, argv, mop_sqrt, NULL,
		    copy_type,
		    ALL_TYPES, ALL_TYPES);
}


static void mulsum(bool_t use_vector, bool_t square_root,
		   matrix_type_t at, byte_t* ap, int au, int av,
		   matrix_type_t bt, byte_t* bp, int bu, int bv,
		   matrix_type_t ct, byte_t* cp,
		   size_t n, size_t m)
{
    UNUSED(use_vector);

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
		    size_t n, size_t m, void* extra)
{
    UNUSED(extra);
    mulsum(use_vector, FALSE, at, ap, au, av, bt, bp, bu, bv, ct, cp, n, m);
}

// Multiply A and B element-wise and sum the result
static ERL_NIF_TERM matrix_mulsum(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a, *b;
    matrix_type_t ct;
    scalar_t c;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &mat[1], &b))
	return enif_make_badarg(env);
    if ((a->rowmajor == b->rowmajor) && ((a->n != b->n) || (a->m != b->m)))
	return enif_make_badarg(env);
    else if ((a->rowmajor != b->rowmajor) && ((a->n != b->m) || (a->m != b->n)))
	return enif_make_badarg(env);

    ct = combine_type(a->type, b->type);

    matrix_rr_lock(a,b);
    s_apply2(mulsum1, NULL, a, b, ct, c.data);
    matrix_rr_unlock(a,b);

    return make_element(env, ct, c.data);
}

// compare element-wise
static ERL_NIF_TERM matrix_eq(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_eq, NULL,
		     compare_type, ALL_TYPES, ALL_TYPES, INT_TYPES);
}

// compare less element-wise
static ERL_NIF_TERM matrix_lt(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_lt, NULL,
		     compare_type, ALL_TYPES, ALL_TYPES, INT_TYPES);
}

// compare less or equal element-wise
static ERL_NIF_TERM matrix_lte(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_lte, NULL,
		     compare_type, ALL_TYPES, ALL_TYPES, INT_TYPES);		     
}

// bitwise and element wise
static ERL_NIF_TERM matrix_band(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_band, NULL,
		     combine_type, INT_TYPES, INT_TYPES, INT_TYPES);
}

// bitwise and element wise
static ERL_NIF_TERM matrix_bor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_bor, NULL,
		     combine_type, INT_TYPES, INT_TYPES, INT_TYPES);
}

// bitwise and element wise
static ERL_NIF_TERM matrix_bxor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return binary_op(env, argc, argv, mop_bxor, NULL,
		     combine_type, INT_TYPES, INT_TYPES, INT_TYPES);
}

// bitwise complement element wise
static ERL_NIF_TERM matrix_bnot(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    return unary_op(env, argc, argv, mop_bnot, NULL,
		    copy_type, INT_TYPES, INT_TYPES);
}

//
// {{ret,false,type},Ri}
// {{instr,false,type},Ri,Rd}
// {{instr,false,type},Ri,Rj,Rd}
// {{instr,true,type},Ri,Rd,Rc}
// {{instr,true,type},Ri,Rj,Rd,Rc}
//

static int load_program(ErlNifEnv* env, ERL_NIF_TERM arg, int nargs,
			instr_t* prog, size_t maxlen)
{
    ERL_NIF_TERM list = arg;
    ERL_NIF_TERM head, tail;    
    int pc = 0;
    int len = maxlen-2;  // int|float arg followed by (auto) ret
    
    while((len>0) && enif_get_list_cell(env, list, &head, &tail)) {
	const ERL_NIF_TERM* elems;
	int arity;
	const ERL_NIF_TERM* ie;
	int ia;
	int pos = 0;
	opcode_t op;
	matrix_type_t type;
	int ilen = 0;  // extra instruction len MOVC!

	// {inst, ri[, rj], rd[, rc]}
	if (!enif_get_tuple(env, head, &arity, &elems) || (arity < 2))
	    return -1;

	// {instruction,cond,type}
	if (!enif_get_tuple(env, elems[pos], &ia, &ie) || (ia != 3))
	    return -1;

	// first decode the instruction name!
	if      (ie[0] == ATOM(ret))  op = OP_RET;
	else if (ie[0] == ATOM(mov))  op = OP_MOVR; // updated!
	else if (ie[0] == ATOM(neg))  op = OP_NEG;
	else if (ie[0] == ATOM(bnot)) op = OP_BNOT;
	else if (ie[0] == ATOM(inv))  op = OP_INV;
	else if (ie[0] == ATOM(band)) op = OP_BAND;
	else if (ie[0] == ATOM(bor))  op = OP_BOR;
	else if (ie[0] == ATOM(bxor)) op = OP_BXOR;	
	
	else if (ie[0] == ATOM(add))  op = OP_ADD;
	else if (ie[0] == ATOM(sub))  op = OP_SUB;
	else if (ie[0] == ATOM(mul))  op = OP_MUL;
	else if (ie[0] == ATOM(lt))   op = OP_LT;
	else if (ie[0] == ATOM(lte))  op = OP_LTE;
	else if (ie[0] == ATOM(eq))   op = OP_EQ;
	else return -1;

	if (ie[1] == ATOM(true)) op |= OP_CND;
	prog[pc].op = op;

	if (ie[2] == ATOM(uint8))       type = UINT8;
	else if (ie[2] == ATOM(uint16)) type = UINT16;
	else if (ie[2] == ATOM(uint32)) type = UINT32;
	else if (ie[2] == ATOM(uint64)) type = UINT64;
	else if (ie[2] == ATOM(uint128)) type = UINT128;
	else if (ie[2] == ATOM(int8))   type = INT8;
	else if (ie[2] == ATOM(int16))  type = INT16;
	else if (ie[2] == ATOM(int32))  type = INT32;
	else if (ie[2] == ATOM(int64))  type = INT64;
	else if (ie[2] == ATOM(int128)) type = INT128;
	else if (ie[2] == ATOM(float16)) type = FLOAT16;
	else if (ie[2] == ATOM(float32)) type = FLOAT32;
	else if (ie[2] == ATOM(float64)) type = FLOAT64;
	else return -1;
	prog[pc].type = type;

	pos++;
	// Ri  pos=1
	{
	    const ERL_NIF_TERM* rs;
	    int ri;
	    // {r,i} | {a,i} | {c,<int>|<float>>}
	    if (!enif_get_tuple(env, elems[pos], &ri, &rs) || (ri != 2))
		return -1;
	    if (rs[0] == ATOM(r)) {
		int i;
		if (!enif_get_int(env, rs[1], &i) || (i < 0) || (i > 15))
		    return -1;
		prog[pc].ri = i;
	    }
	    else if (rs[0] == ATOM(a)) {
		int i;
		if (!enif_get_int(env, rs[1], &i) || (i < 0) || (i > 15))
		    return -1;
		if (i >= nargs) // FIXME?
		    return -1;
		if ((op & ~OP_CND) != OP_MOVR)
		    return -1;  // {a,i} only supported for MOV!
		prog[pc].op = OP_MOVA | (op & OP_CND);
		prog[pc].ri = i;  // interpreted as argument i
	    }
	    else if (rs[0] == ATOM(c)) {
		int64_t   ival;
		float64_t fval;
		uint8_t* ptr = (uint8_t*) &prog[pc+1];
		
		if ((op & ~OP_CND) != OP_MOVR)
		    return -1;  // {c,int|float} only supported for MOV!
		prog[pc].op = OP_MOVC | (op & OP_CND);
		// fixme use type to store constants

		if (enif_get_int64(env, rs[1], &ival)) {
		    switch(type) {
		    case UINT8: {
			uint8_t uv = ival;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 1;
			break;
		    }
		    case UINT16: {
			uint16_t uv = ival;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 1;
			break;
		    }
		    case UINT32: {
			uint32_t uv = ival;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 1;
			break;
		    }
		    case UINT64: {
			uint64_t uv = ival;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 2;
			break;
		    }
		    case UINT128: {
			uint128_t uv;
			uv.lo = ival;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 4;
			break;			
		    }
		    case INT8: {
			int8_t iv = ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case INT16: {
			int16_t iv = ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case INT32: {
			int32_t iv = ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case INT64: {
			int64_t iv = ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 2;
			break;
		    }
		    case INT128: {
			int128_t iv;
			iv.lo = ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 4;
			break;
		    }
		    case FLOAT16:{
			float16_t iv = (float16_t) ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case FLOAT32: {
			float32_t iv = (float32_t) ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case FLOAT64: {
			float64_t iv = (float64_t) ival;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 2;
			break;
		    }
		    default:
			return -1;
		    }
		}
		else if (enif_get_double(env, rs[1], &fval)) {
		    switch(type) {
		    case UINT8: {
			uint8_t uv = fval;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 1;
			break;
		    }
		    case UINT16: {
			uint16_t uv = fval;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 1;
			break;
		    }
		    case UINT32: {
			uint32_t uv = fval;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 1;
			break;
		    }
		    case UINT64: {
			uint64_t uv = fval;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 2;
			break;
		    }
		    case UINT128: {
			uint128_t uv;
			uv.lo = fval;
			memcpy(ptr, &uv, sizeof(uv));
			ilen = 4;
			break;			
		    }
		    case INT8: {
			int8_t iv = fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case INT16: {
			int16_t iv = fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case INT32: {
			int32_t iv = fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case INT64: {
			int64_t iv = fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 2;
			break;
		    }
		    case INT128: {
			int128_t iv;
			iv.lo = fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 4;
			break;
		    }
		    case FLOAT16:{
			float16_t iv = (float16_t) fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;			
			break;
		    }
		    case FLOAT32: {
			float32_t iv = (float32_t) fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 1;
			break;
		    }
		    case FLOAT64: {
			float64_t iv = (float64_t) fval;
			memcpy(ptr, &iv, sizeof(iv));
			ilen = 2;
			break;
		    }
		    default:
			return -1;
		    }
		}
		else
		    return -1;
	    }
	    else
		return -1;
	    pos++;
	}

	// Rj (second source) is present if binary op OP_BIN
	if ((op & OP_BIN) && (op != OP_RET)) {
	    const ERL_NIF_TERM* rs;
	    int ri;
	    if (pos >= arity) return -1;  // missing rj register
	    if (!enif_get_tuple(env, elems[pos], &ri, &rs) || (ri != 2))
		return -1;
	    if (rs[0] == ATOM(r)) {
		int j;
		if (!enif_get_int(env, rs[1], &j) || (j < 0) || (j > 15))
		    return -1;
		prog[pc].rj = j;
	    }
	    pos++;
	}

	// pos = 2 | 3
	// Rd is present except for RET
	if (op != OP_RET) {
	    const ERL_NIF_TERM* rs;
	    int ri;
	    if (pos >= arity) return -1;  // missing rd register
	    if (!enif_get_tuple(env, elems[pos], &ri, &rs) || (ri != 2))
		return -1;
	    if (rs[0] == ATOM(r)) {
		int d;
		if (!enif_get_int(env,rs[1], &d) || (d < 0) || (d > 15))
		return -1;
		prog[pc].rd = d;
	    }
	    pos++;
	}

	// pos = 3|4
	// Rc is if condition code is set
	if (op & OP_CND) {
	    const ERL_NIF_TERM* rs;
	    int ri;
	    if (pos >= arity) return -1;  // missing rc register
	    if (!enif_get_tuple(env, elems[pos], &ri, &rs) || (ri != 2))
		return -1;
	    if (rs[0] == ATOM(r)) {
		int c;
		if (!enif_get_int(env,rs[1], &c) || (c < 0) || (c > 15))
		return -1;
		prog[pc].rc = c;
	    }
	    pos++;
	}
	pc++;         // next instruction
	pc += ilen;   // skip instruction data MOVC
	len--;
	list = tail;
    }
    if (enif_is_empty_list(env, list)) {
	prog[pc].op = OP_RET;
	return 0;
    }
    return -1;
}

static ERL_NIF_TERM matrix_eval1(ErlNifEnv* env, int argc,
				 const ERL_NIF_TERM argv[])
{
    instr_t prog[MAX_PROG_SIZE];

    if (load_program(env, argv[0], 1, prog, MAX_PROG_SIZE) < 0) {
	return EXCP_BADARG_N(env, 1, "invalid program");
    }
    return unary_op(env, argc-1, argv+1, mop_eval1, prog,
		    copy_type, ALL_TYPES, ALL_TYPES);
}

static ERL_NIF_TERM matrix_eval2(ErlNifEnv* env, int argc,
				 const ERL_NIF_TERM argv[])
{
    instr_t prog[MAX_PROG_SIZE];

    if (load_program(env, argv[0], 2, prog, MAX_PROG_SIZE) < 0) {
	return EXCP_BADARG_N(env, 1, "invalid program");
    }    
    return binary_op(env, argc-1, argv+1, mop_eval2, prog,
		     combine_type, ALL_TYPES, ALL_TYPES, ALL_TYPES);    
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

//    cu = size_of_array(ct,cu);
//    cv = size_of_array(ct,cv);
//    ru = size_of_array(at,au)*ru;
//    rv = size_of_array(at,av)*rv;
    ru = au*ru;
    rv = av*rv;    

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
    matrix_t mat[2];
    matrix_t *a, *c;
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
    if (!get_matrix(env, argv[4], &mat[0], &a))
	return enif_make_badarg(env);

    if (!a->rowmajor) {
	unsigned int t;
	t = ru; ru = rv; rv = t;  // swap ru and rv
	t = rn; rn = rm; rm = t;  // swap rn and rm
    }

    if ((a->n < rn) || (a->m < rm))
	return enif_make_badarg(env);
    cn = ((a->n-rn) / ru)+1;
    cm = ((a->m-rm) / rv)+1;

    if (argc == 5) {
	ERL_NIF_TERM res;

	if ((c=create_matrix(env,cn,cm,a->rowmajor,a->type,&res))==NULL)
	    return enif_make_badarg(env);
	matrix_r_lock(a);
	l2pool(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
	       c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
	       ru, rv, rn, rm);
	matrix_r_unlock(a);
	return make_matrix_t(env,c,res);
    }
    else { // argc == 6 with destination matrix
	if (!get_w_matrix(env, argv[5], &mat[1], &c))
	    return enif_make_badarg(env);

	if ((a->rowmajor == c->rowmajor) && ((c->n != cn) || (c->m != cm)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) && ((c->n != cm) || (c->m != cn)))
	    return enif_make_badarg(env);

	if (a->rowmajor == c->rowmajor) {
	    if ((c->n != cn) || (c->m != cm))
		return enif_make_badarg(env);

	    matrix_rw_lock(a,c);
	    l2pool(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		   c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		   ru, rv, rn, rm);
	    matrix_rw_unlock(a,c);
	}
	else {
	    if ((c->n != cm) || (c->m != cn))
		return enif_make_badarg(env);

	    matrix_rw_lock(a,c);
	    l2pool(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		   c->type, c->first, c->m_stride, c->n_stride, c->m, c->n,
		   ru, rv, rn, rm);
	    matrix_rw_unlock(a,c);
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
//    au = size_of_array(at,au);
//    av = size_of_array(at,av);

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

    ru = au*ru;
    rv = av*rv;

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
    matrix_t mat[2];
    matrix_t *a, *c;
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
    if (!get_matrix(env, argv[4], &mat[0], &a))
	return enif_make_badarg(env);

    if (!a->rowmajor) {
	unsigned int t;
	t = ru; ru = rv; rv = t;  // swap ru and rv
	t = rn; rn = rm; rm = t;  // swap rn and rm
    }

    if ((a->n < rn) || (a->m < rm))
	return enif_make_badarg(env);
    cn = ((a->n-rn) / ru)+1;
    cm = ((a->m-rm) / rv)+1;

    if (argc == 5) {
	ERL_NIF_TERM res;
	
	if ((c=create_matrix(env,cn,cm,a->rowmajor,a->type,&res))==NULL)
	    return enif_make_badarg(env);
	matrix_r_lock(a);
	maxpool(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		ru, rv, rn, rm);
	matrix_r_unlock(a);
	return make_matrix_t(env,c,res);
    }
    else { // argc == 6 with destination matrix
	if (!get_w_matrix(env, argv[5], &mat[1], &c))
	    return enif_make_badarg(env);

	if ((a->rowmajor == c->rowmajor) && ((c->n != cn) || (c->m != cm)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) && ((c->n != cm) || (c->m != cn)))
	    return enif_make_badarg(env);

	if (a->rowmajor == c->rowmajor) {
	    if ((c->n != cn) || (c->m != cm))
		return enif_make_badarg(env);

	    matrix_rw_lock(a,c);
	    maxpool(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		    c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		    ru, rv, rn, rm);
	    matrix_rw_unlock(a,c);
	}
	else {
	    if ((c->n != cm) || (c->m != cn))
		return enif_make_badarg(env);

	    matrix_rw_lock(a,c);
	    maxpool(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		    c->type, c->first, c->m_stride, c->n_stride, c->m, c->n,
		    ru, rv, rn, rm);
	    matrix_rw_unlock(a,c);
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

    ru = au*ru;
    rv = av*rv;    

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
    matrix_t mat[3];
    matrix_t *w, *a, *c;
    unsigned int wn, wm;
    unsigned int ru, rv;
    int wu, wv;
    size_t cn, cm;

    if (!get_matrix(env, argv[0], &mat[0], &w))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &ru) || (ru == 0))  // region row step
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &rv) || (rv == 0))  // column step
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[3], &mat[1], &a))
	return enif_make_badarg(env);

    if (a->rowmajor == w->rowmajor) {
	wn = w->n; wm = w->m; wu = w->n_stride; wv = 1;
    }
    else {
	wn = w->m; wm = w->n; wu = 1, wv = w->n_stride;
    }

    if ((a->n < wn) || (a->m < wm))
	return enif_make_badarg(env);
    cn = ((a->n-wn) / ru)+1;
    cm = ((a->m-wm) / rv)+1;

    if (argc == 4) {
	ERL_NIF_TERM res;

	if ((c=create_matrix(env,cn,cm,a->rowmajor,a->type,&res))==NULL)
	    return enif_make_badarg(env);
	matrix_r_lock(a);
	filter(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
	       w->type, w->first, wu, wv, wn, wm,
	       c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
	       ru, rv);
	matrix_r_unlock(a);
	return make_matrix_t(env,c,res);
    }
    else { // argc == 5 with destination matrix
	
	if (!get_w_matrix(env, argv[4], &mat[2], &c))
	    return enif_make_badarg(env);

	if ((a->rowmajor == c->rowmajor) && ((c->n != cn) || (c->m != cm)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) && ((c->n != cm) || (c->m != cn)))
	    return enif_make_badarg(env);

	if (a->rowmajor == c->rowmajor) {
	    if ((c->n != cn) || (c->m != cm))
		return enif_make_badarg(env);

	    matrix_rw_lock(a,c);
	    filter(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		   w->type, w->first, wu, wv, wn, wm,
		   c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		   ru, rv);
	    matrix_rw_unlock(a,c);
	}
	else {
	    if ((c->n != cm) || (c->m != cn))
		return enif_make_badarg(env);

	    matrix_rw_lock(a,c);
	    filter(a->type, a->first, a->m_stride, a->n_stride, a->m, a->n,
		   w->type, w->first, wv, wu, wm, wn,
		   c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		   rv, ru);
	    matrix_rw_unlock(a,c);
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
    if (argc < 3)
	return unary_op(env, argc, argv, mop_copy1, NULL,
			copy_type,
			ALL_TYPES, ALL_TYPES);
    else {
	matrix_t mat[2];
	matrix_t *a, *c;
	unsigned int repeat_m;
	unsigned int repeat_n;
	
	if (!get_matrix(env, argv[0], &mat[0], &a))
	    return enif_make_badarg(env);
	
	if (!get_w_matrix(env, argv[1], &mat[1], &c))
	    return enif_make_badarg(env);

	if (!enif_get_uint(env, argv[2], &repeat_m))
	    return enif_make_badarg(env);
	if (!enif_get_uint(env, argv[3], &repeat_n))
	    return enif_make_badarg(env);

	matrix_rw_lock(a, c);
	if (a->rowmajor == c->rowmajor)
	    tile(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
		 c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		 repeat_m, repeat_n);
	else
	    tile(a->type, a->first, a->m_stride, a->n_stride, a->n, a->m,
		 c->type, c->first, c->n_stride, c->m_stride, c->n, c->m,
		 repeat_m, repeat_n);

	matrix_rw_unlock(a, c);
	return argv[1];
    }
}

// copy data row by row from A sequentially into C
ERL_NIF_TERM matrix_fill(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a, *c;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &mat[1], &c))
	return enif_make_badarg(env);
    if (c->ptr == 0) // must be resource matrix!
	return enif_make_badarg(env);

    matrix_rw_lock(a, c);
    if (a->rowmajor == c->rowmajor)
	fill(a->type, a->first, a->n_stride, a->m_stride, a->n, a->m,
	     c->type, c->first, c->n_stride, c->m_stride, c->n, c->m);
    else
	fill(a->type, a->first, a->m_stride, a->n_stride, a->n, a->m,
	     c->type, c->first, c->n_stride, c->m_stride, c->n, c->m);
    matrix_rw_unlock(a, c);

    return argv[1];
}

// sigmoid a matrix
ERL_NIF_TERM matrix_sigmoid(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a, *c;
    ERL_NIF_TERM res;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);

    if ((c=create_matrix(env,a->n,a->m,a->rowmajor,a->type,&res))==NULL)
	return enif_make_badarg(env);

    matrix_r_lock(a);
    sigmoid(a->type, a->first, a->n_stride, a->m_stride,
	    c->type, c->first, c->n_stride, c->m_stride,
	    c->n, c->m);
    matrix_r_unlock(a);
    return make_matrix_t(env,c,res);
}

// sigmoid_prime a matrix use the output version d(y) = y*(1-y)
ERL_NIF_TERM matrix_sigmoid_prime(ErlNifEnv* env, int argc,
				  const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a,*y,*c;
    ERL_NIF_TERM res;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &mat[1], &y))
	return enif_make_badarg(env);

    if ((c=create_matrix(env,a->n,a->m,a->rowmajor,a->type,&res))==NULL)
	return enif_make_badarg(env);

    matrix_r_lock(y);
    sigmoid_prime1(y->type, y->first, y->n_stride, y->m_stride,
		  c->type, c->first, c->n_stride, c->m_stride,
		  c->n, c->m);
    matrix_r_unlock(y);
    return make_matrix_t(env,c,res);
}

// rectifier a matrix
ERL_NIF_TERM matrix_relu(ErlNifEnv* env, int argc,
			 const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a, *c;
    ERL_NIF_TERM res;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);

    if ((c=create_matrix(env,a->n,a->m,a->rowmajor,a->type,&res))==NULL)
	return enif_make_badarg(env);

    matrix_r_lock(a);
    rectifier(a->type, a->first, a->n_stride, a->m_stride,
	      c->type, c->first, c->n_stride, c->m_stride,
	      c->n, c->m);
    matrix_r_unlock(a);
    return make_matrix_t(env,c,res);
}

// n m and type
ERL_NIF_TERM matrix_identity(ErlNifEnv* env, int argc,
			     const ERL_NIF_TERM argv[])
{
    unsigned int n, m, k, type;
    unsigned int i;
    matrix_t *c;
    ERL_NIF_TERM res;
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
    if ((c=create_matrix(env,n,m,TRUE,type,&res))==NULL)
	return enif_make_badarg(env);

    memset(c->data, 0, c->size);  // set to zero

    col_step  = c->m_stride;
    row_step  = c->n_stride;
    // format the 1 element
    if (is_float(type))
	write_float64(type, one.data, 1.0);
    else
	write_int64(type, one.data, 1);
    k = (n<m) ? n : m;
    cp = c->first;
    elem_size = element_size(type);
    
    for (i = 0; i < k; i++) {
	memcpy(cp, one.data, elem_size);
	cp += row_step;
	cp += col_step;
    }
    return make_matrix_t(env,c,res);
}

// matrix_apply1 (func, Src [,Dst])
ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a, *c;
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

    if (!get_matrix(env, argv[1], &mat[0], &a))
	return enif_make_badarg(env);

    if (argc == 2) {
	ERL_NIF_TERM res;
	if ((c=create_matrix(env,a->n,a->m,a->rowmajor,a->type,&res))==NULL)
	    return enif_make_badarg(env);
	matrix_r_lock(a);
	apply1(op,
	       a->type, a->first, a->n_stride, a->m_stride,
	       c->type, c->first, c->n_stride, c->m_stride,
	       c->n, c->m);
	matrix_r_unlock(a);
	return make_matrix_t(env,c,res);
    }
    else {
	if (!get_w_matrix(env, argv[2],&mat[1],&c))
	    return enif_make_badarg(env);

	if ((a->rowmajor == c->rowmajor) && ((a->n != c->n) || (a->m != c->m)))
	    return enif_make_badarg(env);
	else if ((a->rowmajor != c->rowmajor) && ((a->n != c->m) || (a->m != c->n)))
	    return enif_make_badarg(env);

	matrix_rw_lock(a, c);
	if (c->rowmajor == a->rowmajor)
	    apply1(op,
		   a->type, a->first, a->n_stride, a->m_stride,
		   c->type, c->first, c->n_stride, c->m_stride,
		   c->n, c->m);
	else
	    apply1(op,
		   a->type, a->first, a->m_stride, a->n_stride,
		   c->type, c->first, c->n_stride, c->m_stride,
		   c->n, c->m);
	matrix_rw_unlock(a, c);
	return argv[1];
    }
}

// find argmax in matrix
ERL_NIF_TERM matrix_argmax(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a,*c;
    unsigned int axis;
    ERL_NIF_TERM res;
    int opts;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &axis))
	return enif_make_badarg(env);
    if (!get_opts(env, argv[2], &opts))
	return enif_make_badarg(env);
    if (axis > 2)
	return enif_make_badarg(env);

    if (a->rowmajor) {
	if (axis == 0) {
	    int32_t max_i, max_j;
	    argmax_0(a->type, a->first,a->n_stride,a->m_stride,
		     &max_i, &max_j,
		     a->n,a->m,opts);
	    return enif_make_tuple2(env,
				    enif_make_int(env, max_i),
				    enif_make_int(env, max_j));
	}
	else if (axis == 1) {
	    // argmax for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->m,TRUE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmax(a->type, a->first,a->n_stride,a->m_stride,
		   (int32_t*)c->first, 1,
		   a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // argmax for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->n,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmax(a->type, a->first,1,a->n_stride,
		   (int32_t*)c->first, 1,
		   a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    int32_t max_i, max_j;
	    argmax_0(a->type,a->first,a->m_stride,a->n_stride,
		     &max_j, &max_i,
		     a->m,a->n,opts);
	    return enif_make_tuple2(env,
				    enif_make_int(env, max_i),
				    enif_make_int(env, max_j));
	}
	else if (axis == 1) {
	    // argmax for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->n,TRUE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmax(a->type, a->first,1,a->n_stride,
		   (int32_t*)c->first,1,
		   a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // argmax for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->m,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmax(a->type, a->first,a->n_stride,a->m_stride,
		   (int32_t*)c->first,1,
		   a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
}

// find max in matrix
ERL_NIF_TERM matrix_max(ErlNifEnv* env, int argc,
			const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a, *c;
    unsigned int axis;
    ERL_NIF_TERM res;
    int opts;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &axis))
	return enif_make_badarg(env);
    if (!get_opts(env, argv[2], &opts))
	return enif_make_badarg(env);
    if (axis > 2)
	return enif_make_badarg(env);

    if (a->rowmajor) {
	if (axis == 0) {
	    scalar_t max_v;
	    matrix_r_lock(a);
	    t_max(a->type, a->first, a->n_stride, a->m_stride,
		  a->type, max_v.data, 0,
		  a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_element(env, a->type, max_v.data);
	}
	else if (axis == 1) {
	    // max for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->m,TRUE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_max(a->type, a->first, a->n_stride, a->m_stride,
		  c->type, c->first, 1,
		  a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else { // axis == 2
	    // max for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->n,FALSE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_max(a->type, a->first, a->m_stride, a->n_stride,
		  c->type, c->first, 1,
		  a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    scalar_t max_v;
	    matrix_r_lock(a);
	    t_max(a->type, a->first, a->m_stride, a->n_stride,
		  a->type, max_v.data, 0,
		  a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_element(env, a->type, max_v.data);
	}
	else if (axis == 1) {
	    // max for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->n,TRUE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_max(a->type, a->first, a->m_stride, a->n_stride,
		  c->type, c->first, 1,
		  a->m, a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // max for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->m,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_max(a->type, a->first, a->n_stride, a->m_stride,
		  c->type, c->first, 1,
		  a->n, a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
}

// find argmin in matrix
ERL_NIF_TERM matrix_argmin(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a,*c;
    unsigned int axis;
    ERL_NIF_TERM res;
    int opts;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &axis))
	return enif_make_badarg(env);
    if (!get_opts(env, argv[2], &opts))
	return enif_make_badarg(env);
    if (axis > 2)
	return enif_make_badarg(env);

    if (a->rowmajor) {
	if (axis == 0) {
	    int32_t min_i, min_j;
	    argmin_0(a->type, a->first,a->n_stride,a->m_stride,
		     &min_i, &min_j,
		     a->n,a->m,opts);
	    return enif_make_tuple2(env,
				    enif_make_int(env, min_i),
				    enif_make_int(env, min_j));
	}
	else if (axis == 1) {
	    // argmin for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->m,TRUE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmin(a->type, a->first,a->n_stride,a->m_stride,
		   (int32_t*)c->first, 1,
		   a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // argmin for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->n,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmin(a->type, a->first,1,a->n_stride,
		   (int32_t*)c->first, 1,
		   a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    int32_t min_i, min_j;
	    argmin_0(a->type,a->first,a->m_stride,a->n_stride,
		     &min_j, &min_i,
		     a->m,a->n,opts);
	    return enif_make_tuple2(env,
				    enif_make_int(env, min_i),
				    enif_make_int(env, min_j));
	}
	else if (axis == 1) {
	    // argmin for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->n,TRUE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmin(a->type, a->first,1,a->n_stride,
		   (int32_t*)c->first,1,
		   a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // argmin for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->m,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    argmin(a->type, a->first,a->n_stride,a->m_stride,
		   (int32_t*)c->first,1,
		   a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
}

// find min in matrix
ERL_NIF_TERM matrix_min(ErlNifEnv* env, int argc,
			const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a,*c;
    unsigned int axis;
    ERL_NIF_TERM res;
    int opts;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &axis))
	return enif_make_badarg(env);
    if (!get_opts(env, argv[2], &opts))
	return enif_make_badarg(env);
    if (axis > 2)
	return enif_make_badarg(env);
    if (a->rowmajor) {
	if (axis == 0) {
	    scalar_t min_v;
	    matrix_r_lock(a);
	    t_min(a->type, a->first, a->n_stride, a->m_stride,
		  a->type, min_v.data, 0,
		  a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_element(env, a->type, min_v.data);
	}
	else if (axis == 1) {
	    // min for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->m,TRUE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_min(a->type, a->first, a->n_stride, a->m_stride,
		  c->type, c->first, 1,
		  a->n,a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // min for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->n,FALSE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_min(a->type, a->first, a->m_stride, a->n_stride,
		  c->type, c->first, 1,
		  a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    scalar_t min_v;
	    matrix_r_lock(a);
	    t_min(a->type, a->first, a->m_stride, a->n_stride,
		  a->type, min_v.data, 0,
		  a->m,a->n,opts);
	    matrix_r_unlock(a);
	    return make_element(env, a->type, min_v.data);
	}
	else if (axis == 1) {
	    // min for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->n,TRUE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_min(a->type, a->first, a->m_stride, a->n_stride,
		  c->type, c->first, 1,
		  a->m, a->n,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // min for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->m,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_min(a->type, a->first, a->n_stride, a->m_stride,
		  c->type, c->first, 1,
		  a->n, a->m,opts);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
}


// find sum in matrix
ERL_NIF_TERM matrix_sum(ErlNifEnv* env, int argc,
			const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a, *c;
    unsigned int axis;
    ERL_NIF_TERM res;
    UNUSED(argc);

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (argc == 1)
	axis = 0;
    else {
	if (!enif_get_uint(env, argv[1], &axis))
	    return enif_make_badarg(env);
	if (axis > 2)
	    return enif_make_badarg(env);
    }

    if (a->rowmajor) {
	if (axis == 0) {
	    scalar_t sum_v;
	    matrix_r_lock(a);
	    t_sum(a->type, a->first, a->n_stride, a->m_stride,
		  a->type, sum_v.data, 0,
		  a->n,a->m);
	    matrix_r_unlock(a);
	    return make_element(env, a->type, sum_v.data);
	}
	else if (axis == 1) {
	    // sum for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->m,TRUE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_sum(a->type, a->first, a->n_stride, a->m_stride,
		  c->type, c->first, 1,
		  a->n,a->m);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // sum for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->n,FALSE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_sum(a->type, a->first, a->m_stride, a->n_stride,
		  c->type, c->first, 1,
		  a->m,a->n);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
    else { // !a.rowmajor
	if (axis == 0) {
	    scalar_t sum_v;
	    matrix_r_lock(a);
	    t_sum(a->type, a->first, a->m_stride, a->n_stride,
		  a->type, sum_v.data, 0,
		  a->m,a->n);
	    matrix_r_unlock(a);
	    return make_element(env, a->type, sum_v.data);
	}
	else if (axis == 1) {
	    // sum for each column is returned (as a row)
	    if ((c=create_matrix(env,1,a->n,TRUE,a->type,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_sum(a->type, a->first, a->m_stride, a->n_stride,
		  c->type, c->first, 1,
		  a->m, a->n);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
	else {
	    // min for each row is returned (as a column)
	    if ((c=create_matrix(env,1,a->m,FALSE,INT32,&res))==NULL)
		return enif_make_badarg(env);
	    matrix_r_lock(a);
	    t_sum(a->type, a->first, a->n_stride, a->m_stride,
		  c->type, c->first, 1,
		  a->n, a->m);
	    matrix_r_unlock(a);
	    return make_matrix_t(env,c,res);
	}
    }
}

static void swap_elem(matrix_type_t at, byte_t* ptr1, byte_t* ptr2)
{
    if (ptr1 != ptr2) {
	size_t esize = element_size(at);
	byte_t temp[sizeof(scalar_t)];
	memcpy(temp, ptr1, esize);
	memcpy(ptr1, ptr2, esize);
	memcpy(ptr2, temp, esize);
    }
}

// swap rows or columns 
static void swap_data(matrix_type_t at,byte_t* ptr1, byte_t* ptr2,
		      int u, size_t n)
{
    if (ptr1 == ptr2)  // nothing to swap
	return;
    else if (u == (int)element_size(at)) { // elements are consecutive
	n = size_of_array(at,n);
	while(n > 0) {
	    byte_t temp[1024];
	    size_t k = (n > 1024) ? 1024 : n;
	    memcpy(temp, ptr1, k);
	    memcpy(ptr1, ptr2, k);
	    memcpy(ptr2, temp, k);
	    ptr1 += k;
	    ptr2 += k;
	    n -= k;
	}
    }
    else {
	size_t esize = element_size(at);
	// u = size_of_array(at,u);
	while(n--) {
	    byte_t temp[16*sizeof(scalar_t)];
	    memcpy(temp, ptr1, esize);
	    memcpy(ptr1, ptr2, esize);
	    memcpy(ptr2, temp, esize);
	    ptr1 += u;
	    ptr2 += u;
	}
    }
}

static ERL_NIF_TERM matrix_swap(ErlNifEnv* env, int argc,
				const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a;
    unsigned int k, l;
    unsigned int j, i;
    unsigned int len;
    unsigned int js;
    int axis;
    byte_t* k_ptr;
    byte_t* i_ptr;
    ERL_NIF_TERM res;
    UNUSED(argc);
    
    if (!enif_get_uint(env, argv[0], &k))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &i))
	return enif_make_badarg(env);
    if (argc == 6) {
	unsigned int ln;
	if (!enif_get_uint(env, argv[2], &j))
	    return enif_make_badarg(env);
	if (!enif_get_uint(env, argv[3], &l))
	    return enif_make_badarg(env);
	if (!get_w_matrix(env, argv[4], &mat[0], &a))
	    return enif_make_badarg(env);
	if (!enif_get_int(env, argv[5], &axis))
	    return enif_make_badarg(env);
	if (j < 1)
	    return enif_make_badarg(env);
	res = argv[4];
	if ((j == 1) && (l == 0)) // special case
	    return res;
	else if (l < j)
	    return enif_make_badarg(env);
	if (axis == 1)
	    ln = (a->rowmajor) ? a->m : a->n;
	else if (axis == 2)
	    ln = (a->rowmajor) ? a->n : a->m;
	else
	    return enif_make_badarg(env);
	if (l > ln)
	    return enif_make_badarg(env);
    }
    else {
	if (!get_w_matrix(env, argv[2], &mat[0], &a))
	    return enif_make_badarg(env);
	if (!enif_get_int(env, argv[3], &axis))
	    return enif_make_badarg(env);
	res = argv[2];
	j = 1;
	if (axis == 1)
	    l = (a->rowmajor) ? a->m : a->n;
	else if (axis == 2)
	    l = (a->rowmajor) ? a->n : a->m;
	else 
	    return enif_make_badarg(env);
    }
    if ((k < 1) || (i < 1))
	return enif_make_badarg(env);

    len = l-j+1;  // number of elements to swap
    if (axis == 1) {  // swap row k and row i
	matrix_w_lock(a);
	if (a->rowmajor) {
	    if ((k > a->n) || (i > a->n)) return enif_make_badarg(env);
	    js = (j-1)*a->m_stride;
	    k_ptr = a->first + (k-1)*a->n_stride + js;
	    i_ptr = a->first + (i-1)*a->n_stride + js;
	    swap_data(a->type, k_ptr, i_ptr, a->m_stride, len);
	}
	else {  // column major
	    if ((k > a->m) || (i > a->m)) return enif_make_badarg(env);
	    js = (j-1)*size_of_array(a->type,a->n_stride);
	    k_ptr = a->first + (k-1)*a->m_stride + js;
	    i_ptr = a->first + (i-1)*a->m_stride + js;
	    swap_data(a->type, k_ptr, i_ptr, a->n_stride, len);
	}
	matrix_w_unlock(a);
    }
    else if (axis == 2) { // swap column k and column i
	matrix_w_lock(a);	
	if (a->rowmajor) {
	    if ((k > a->m) || (i > a->m)) return enif_make_badarg(env);
	    js = (j-1)*a->n_stride;
	    k_ptr = a->first + (k-1)*a->m_stride + js;
	    i_ptr = a->first + (i-1)*a->m_stride + js;
	    swap_data(a->type, k_ptr, i_ptr, a->n_stride, len);
	}
	else {  // column major
	    if ((k > a->n) || (i > a->n)) return enif_make_badarg(env);
	    js = (j-1)*a->m_stride;
	    k_ptr = a->first + (k-1)*a->n_stride + js;
	    i_ptr = a->first + (i-1)*a->n_stride + js;
	    swap_data(a->type, k_ptr, i_ptr, a->m_stride, len);
	}
	matrix_w_unlock(a);
    }
    else {
	return enif_make_badarg(env);
    }
    return res;
}

// simple sort (try to avoid many swaps)
// et, ep, eu, n is the key row/column
// at, ap, av, m is the matrix to be sorted
// overlap is true of e "vector" overlap with matrix a
// FIXME: what if e overlap a bit?
static void sort_data(matrix_type_t et, byte_t* ep, int eu, size_t n,
		      matrix_type_t at, byte_t* ap, int au, int av, size_t m,
		      bool_t overlap, int opts)
{
    //int eub = size_of_array(et,eu);
    //int aub = size_of_array(at,au);
    int en = (int)n;
    
    if (is_float(et)) {
	float64_t (*read_af)(byte_t*) = read_float64_func[et];
	// sort row/column from ptr ... ptr+n
	while(en-- > 1) {
	    float64_t v0 = read_af(ep);
	    byte_t* ep1 = ep + eu;
	    byte_t* ap1;
	    byte_t* ep0 = ep;
	    int k = 0;
	    int k0 = 0;
	    for (k=0; k<en;k++) {
		float64_t v = read_af(ep1);
		if (FCMP(v,v0)) { v0=v; ep0=ep1; k0=k; }
		ep1 += eu;
	    }
	    ap1 = ap + ((au == 1) ? size_of_array(at,k0) : au*k0);
	    swap_data(at, ap, ap1, av, m);
	    if (!overlap) swap_elem(et, ep, ep0);
	    ep += eu;
	    ap += au;
	}
    }
    else { // int64_t
	int64_t (*read_af)(byte_t*) = read_int64_func[et];
	// sort row/column from ptr ... ptr+n
	while(en-- > 1) {
	    int64_t v0 = read_af(ep);
	    byte_t* ep1 = ep + eu;
	    byte_t* ap1;
	    byte_t* ep0 = ep;
	    int k = 0;
	    int k0 = 0;
	    for (k=0; k<en-1;k++) {
		int64_t v = read_af(ep1);
		if (ICMP(v,v0)) { v0=v; ep0=ep1; k0=k; }
		ep1 += eu;
	    }
	    ap1 = ap + ((au == 1) ? size_of_array(at,k0) : au*k0);
	    swap_data(at, ap, ap1, av, m);
	    if (!overlap) swap_elem(et, ep, ep0);	    
	    ep += eu;
	    ap += au;
	}
    }    
}

static ERL_NIF_TERM matrix_sort(ErlNifEnv* env, int argc,
				const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a;
    unsigned int k;
    int axis;
    int opts;
    byte_t* k_ptr;
    UNUSED(argc);
    
    if (!get_w_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &axis))
	return enif_make_badarg(env);
    if (!get_opts(env, argv[3], &opts))
	return enif_make_badarg(env);
    if ((axis < 1) || (axis > 2))
	return enif_make_badarg(env);
    
    if (enif_get_uint(env, argv[1], &k)) {
	if (k < 1) return enif_make_badarg(env);
	if (axis == 1) { // sort columns according to row k
	    if (a->rowmajor) {
		if (k>a->n) return enif_make_badarg(env);
		k_ptr = a->first + (k-1)*a->n_stride;
		matrix_w_lock(a);
		sort_data(a->type, k_ptr, a->m_stride, a->m,
			  a->type, a->first, a->m_stride, a->n_stride, a->n,
			  TRUE, opts);
		matrix_w_unlock(a);		
	    }
	    else {
		if (k>a->m) return enif_make_badarg(env);
		k_ptr = a->first + (k-1)*a->m_stride;
		matrix_w_lock(a);
		sort_data(a->type, k_ptr, a->n_stride, a->n,
			  a->type, a->first, a->n_stride, a->m_stride, a->m,
			  TRUE, opts);
		matrix_w_unlock(a);
	    }
	}
	else if (axis == 2) { // sort rows according to column k
	    if (a->rowmajor) {	    
		if (k>a->m) return enif_make_badarg(env);
		k_ptr = a->first + (k-1)*a->m_stride;
		matrix_w_lock(a);
		sort_data(a->type, k_ptr, a->n_stride, a->n,
			  a->type, a->first, a->n_stride, a->m_stride, a->m,
			  TRUE, opts);
		matrix_w_unlock(a);
	    }	    
	    else {
		if (k>a->n) return enif_make_badarg(env);
		k_ptr = a->first + (k-1)*a->n_stride;
		matrix_w_lock(a);
		sort_data(a->type, k_ptr, a->m_stride, a->m,
			  a->type, a->first, a->m_stride, a->n_stride, a->n,
			  TRUE, opts);
		matrix_w_unlock(a);
	    }
	}
    }
    else {  // sort according to matrix e (1,n) or (m,1)
	matrix_t* e;
	bool_t overlap;
	if (!get_matrix(env, argv[1], &mat[1], &e))
	    return enif_make_badarg(env);
	overlap = is_overlapping(a, e);
	if (axis == 1) {   // sort columns according to row vector e
	    if (a->rowmajor) {  // matrix to be sorted is in rowmajor order
		if (e->rowmajor) { // key vector (to be sorted) is rowmajor
		    if ((e->n > 1) || (e->m > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->m_stride,e->m,
			      a->type,a->first,a->m_stride,a->n_stride,a->n,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
		else {
		    if ((e->m > 1) || (e->n > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->n_stride,e->n,
			      a->type,a->first,a->m_stride,a->n_stride,a->n,
			      overlap,opts);
		    matrix_rw_unlock(e,a);		    
		}
	    }
	    else {
		if (e->rowmajor) {
		    if ((e->n > 1) || (e->m > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->m_stride,e->m,
			      a->type,a->first,a->n_stride,a->m_stride,a->m,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
		else {
		    if ((e->m > 1) || (e->n > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->n_stride,e->n,
			      a->type,a->first,a->n_stride,a->m_stride,a->m,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
	    }
	}
	else if (axis == 2) { // sort rows according to column vector e
	    if (a->rowmajor) {  // matrix to be sorted is in rowmajor order
		if (e->rowmajor) { // key vector (to be sorted) is rowmajor
		    if ((e->m > 1) || (e->n > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->n_stride,e->n,
			      a->type,a->first,a->n_stride,a->m_stride,a->m,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
		else {
		    if ((e->n > 1) || (e->m > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->m_stride,e->m,
			      a->type,a->first,a->m_stride,a->n_stride,a->n,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
	    }
	    else {
		if (e->rowmajor) {
		    if ((e->m > 1) || (e->n > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->n_stride,e->n,
			      a->type,a->first,a->m_stride,a->n_stride,a->n,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
		else {
		    if ((e->n > 1) || (e->m > a->m))
			return enif_make_badarg(env);
		    matrix_rw_lock(e,a);
		    sort_data(e->type,e->first,e->m_stride,e->m,
			      a->type,a->first,a->m_stride,a->n_stride,a->n,
			      overlap,opts);
		    matrix_rw_unlock(e,a);
		}
	    }   
	}
    }
    return argv[0];
}


// transpose a matrix
ERL_NIF_TERM matrix_transpose(ErlNifEnv* env, int argc,
			      const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a;

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (argc == 1) {
	ERL_NIF_TERM res;
	
	if (a == &mat[0]) { // constant or submatrix, update matrix and return
	    res = a->parent;
	}
	else {
	    res = enif_make_resource(env,a);
	    memcpy(&mat[0], a, sizeof(matrix_t));
	    a = &mat[0];
	}
	a->rowmajor = !a->rowmajor;
	return make_matrix_ptr(env, a, res);
    }
    else {  // transpose into c  argc==2
	matrix_t *c;
	if (!get_w_matrix(env, argv[1], &mat[1], &c))
	    return enif_make_badarg(env);
	if ((a->rowmajor != c->rowmajor) && ((a->n != c->n) || (a->m != c->m)))
	    return enif_make_badarg(env);
	if ((a->rowmajor == c->rowmajor) && ((a->n != c->m) || (a->m != c->n)))
	    return enif_make_badarg(env);
	matrix_rw_lock(a,c);
	m_apply1(mop_transpose, NULL, a, c);
	matrix_rw_unlock(a,c);	
	return argv[1];
    }
    return enif_make_badarg(env);
}

// transpose data rather then toggle rowmajor
ERL_NIF_TERM matrix_transpose_data(ErlNifEnv* env, int argc,
				   const ERL_NIF_TERM argv[])
{
    matrix_t mat[2];
    matrix_t *a, *c;

    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);

    if (argc == 1) {
	ERL_NIF_TERM res;
	matrix_type_t ct = a->type;

	if ((c = create_matrix(env,a->m,a->n,a->rowmajor,ct,&res)) == NULL)
	    return enif_make_badarg(env);
	matrix_r_lock(a);
	m_apply1(mop_transpose,NULL,a,c);
	matrix_r_unlock(a);
	return make_matrix_t(env,c,res);
    }
    else { // args == 2
	if (!get_w_matrix(env, argv[1], &mat[1], &c))
	    return enif_make_badarg(env);
	// FIXME: allow destructive update on if A == C, then
	// rearrange data and update toggle rowmajor!
	if ((a->rowmajor != c->rowmajor) && ((a->n != c->n) || (a->m != c->m)))
	    return enif_make_badarg(env);
	if ((a->rowmajor == c->rowmajor) && ((a->n != c->m) || (a->m != c->n)))
	    return enif_make_badarg(env);
	matrix_rw_lock(a, c);
	m_apply1(mop_transpose,NULL, a, c);
	matrix_rw_unlock(a, c);
	return argv[1];
    }
}

ERL_NIF_TERM matrix_submatrix(ErlNifEnv* env, int argc,
			      const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a;
    unsigned int i, j, n, m;
    unsigned int offset;
    ERL_NIF_TERM res;
    UNUSED(argc);
    
    if (!enif_get_uint(env, argv[0], &i)) return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[1], &j)) return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &n)) return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[3], &m)) return enif_make_badarg(env);
    if (!get_matrix(env, argv[4], &mat[0], &a))
	return enif_make_badarg(env);
    
    if ((i < 1) || (i > a->n)) return enif_make_badarg(env);
    if ((j < 1) || (j > a->m)) return enif_make_badarg(env);
    if (n > a->n) return enif_make_badarg(env);
    if (m > a->m) return enif_make_badarg(env);
    
    if (a->rowmajor) {	
	if ((a->n_stride == 0) && (a->m_stride == 0))
	    offset = a->offset;
	else
	    offset = a->offset + (i-1)*a->n_stride + (j-1)*a->m_stride;
	if (!is_valid_offset(a, offset, n, m))
	    return enif_make_badarg(env);
	if (a == &mat[0])
	    res = a->parent;
	else {
	    res = enif_make_resource(env,a);
	    memcpy(&mat[0], a, sizeof(matrix_t));
	    a = &mat[0];
	}
	a->n = n;
	a->m = m;
	a->offset = offset;
	return make_matrix_ptr(env, a, res);
    }
    else {
	if ((a->n_stride == 0) && (a->m_stride == 0))
	    offset = a->offset;
	else
	    offset = a->offset + (j-1)*a->n_stride + (i-1);
	if (!is_valid_offset(a, offset, m, n))
	    return enif_make_badarg(env);
	if (a == &mat[0])
	    res = a->parent;
	else {
	    res = enif_make_resource(env,a);
	    memcpy(&mat[0], a, sizeof(matrix_t));
	    a = &mat[0];
	}
	a->n = m;  // yes, swapped!
	a->m = n;
	a->offset = offset;
	return make_matrix_ptr(env, a, res);
    }
}

ERL_NIF_TERM matrix_info(ErlNifEnv* env, int argc,
			      const ERL_NIF_TERM argv[])
{
    matrix_t mat[1];
    matrix_t *a;
    UNUSED(argc);
    
    if (!get_matrix(env, argv[0], &mat[0], &a))
	return enif_make_badarg(env);
    if (argv[1] == ATOM(rowmajor))
	return make_bool(env, a->rowmajor);
    
    if (argv[1] == ATOM(n))
	return enif_make_uint(env, a->n);
    if (argv[1] == ATOM(n_stride))
	return enif_make_uint(env, a->n_stride);
    
    if (argv[1] == ATOM(m))
	return enif_make_uint(env, a->m);
    if (argv[1] == ATOM(m_stride))
	return enif_make_uint(env, a->m_stride);
    if (argv[1] == ATOM(k_stride))
	return enif_make_uint(env, a->k_stride);    
    if (argv[1] == ATOM(rows))
	return a->rowmajor ? enif_make_uint(env, a->n) :
	    enif_make_uint(env, a->m);
    if (argv[1] == ATOM(columns))
	return !a->rowmajor ? enif_make_uint(env, a->n) :
	    enif_make_uint(env, a->m);
    if (argv[1] == ATOM(type))
	return enif_make_uint(env, a->type);
    if (argv[1] == ATOM(size))
	return enif_make_uint(env, a->size);
    if (argv[1] == ATOM(parent))
	return a->parent;
    return enif_make_badarg(env);    
}



// create all tracing NIFs
#ifdef NIF_TRACE

#undef NIF

static void trace_print_arg_list(ErlNifEnv* env,int argc,const ERL_NIF_TERM argv[])
{
    enif_fprintf(stdout, "(");
    if (argc > 0) {
	int i;
	if (enif_is_ref(env, argv[0])) {
	    // FIXME print object type if available
	    enif_fprintf(stdout, "%T", argv[0]);
	}
	else
	    enif_fprintf(stdout, "%T", argv[0]);
	for (i = 1; i < argc; i++)
	    enif_fprintf(stdout, ",%T", argv[i]);
    }
    enif_fprintf(stdout, ")");
}

#define NIF(name, arity, func)					\
static ERL_NIF_TERM trace##_##func##_##arity(ErlNifEnv* env, int argc,const ERL_NIF_TERM argv[]) \
{ \
    ERL_NIF_TERM result;					\
    enif_fprintf(stdout, "ENTER %s", (name));			\
    trace_print_arg_list(env, argc, argv);			\
    enif_fprintf(stdout, "\r\n");				\
    result = func(env, argc, argv);				\
    enif_fprintf(stdout, "  RESULT=%T\r\n", (result));		\
    enif_fprintf(stdout, "LEAVE %s\r\n", (name));		\
    return result;						\
}

NIF_LIST

#endif


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
    UNUSED(load_info);
    ErlNifResourceFlags tried;

    DBG("matrix_load\r\n");
    LOAD_ATOM(env,matrix);
    LOAD_ATOM(env,matrix_t);
    LOAD_ATOM(env,sigmoid);
    LOAD_ATOM(env,sigmoid_prime);
    LOAD_ATOM(env,sigmoid_prime1);
    LOAD_ATOM(env,relu);
    LOAD_ATOM(env,relu_prime);
    LOAD_ATOM(env,leaky_relu);
    LOAD_ATOM(env,leaky_relu_prime);
    LOAD_ATOM(env,tanh);
    LOAD_ATOM(env,tanh_prime);
    LOAD_ATOM(env,tanh_prime1);
    LOAD_ATOM(env,softplus);
    LOAD_ATOM(env,softplus_prime);
    LOAD_ATOM(env,negate);
    LOAD_ATOM(env,uniform);
    LOAD_ATOM(env,normal);
    LOAD_ATOM(env,zero);
    LOAD_ATOM(env,one);
    LOAD_ATOM(env,copy);
    LOAD_ATOM(env,true);
    LOAD_ATOM(env,false);
    LOAD_ATOM(env,undefined);
    LOAD_ATOM(env,exp);
    LOAD_ATOM(env,abs);
    LOAD_ATOM(env,ascend);
    LOAD_ATOM(env,descend);
    // info
    LOAD_ATOM(env,rowmajor);
    LOAD_ATOM(env,n);
    LOAD_ATOM(env,n_stride);
    LOAD_ATOM(env,m);
    LOAD_ATOM(env,m_stride);
    LOAD_ATOM(env,k_stride);    
    LOAD_ATOM(env,rows);
    LOAD_ATOM(env,columns);
    LOAD_ATOM(env,type);
    LOAD_ATOM(env,size);
    LOAD_ATOM(env,parent);
    LOAD_ATOM(env,error);
    LOAD_ATOM(env,notsup);
    LOAD_ATOM(env,badarg);
    // instructions
    LOAD_ATOM(env,r);
    LOAD_ATOM(env,a);
    LOAD_ATOM(env,c);
    LOAD_ATOM(env,ret);
    LOAD_ATOM(env,mov);    
    LOAD_ATOM(env,neg);
    LOAD_ATOM(env,inv);
    LOAD_ATOM(env,add);
    LOAD_ATOM(env,sub);
    LOAD_ATOM(env,mul);
    LOAD_ATOM(env,lt);
    LOAD_ATOM(env,lte);
    LOAD_ATOM(env,eq);
    LOAD_ATOM(env,band);
    LOAD_ATOM(env,bor);
    LOAD_ATOM(env,bxor);
    LOAD_ATOM(env,bnot);

    LOAD_ATOM(env,uint8);
    LOAD_ATOM(env,uint16);
    LOAD_ATOM(env,uint32);
    LOAD_ATOM(env,uint64);
    LOAD_ATOM(env,uint128);
    LOAD_ATOM(env,int8);
    LOAD_ATOM(env,int16);
    LOAD_ATOM(env,int32);
    LOAD_ATOM(env,int64);
    LOAD_ATOM(env,int128);
    LOAD_ATOM(env,float16);
    LOAD_ATOM(env,float32);
    LOAD_ATOM(env,float64);
        
    matrix_res = enif_open_resource_type(env, 0, "matrix",
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
