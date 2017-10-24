//
// Matrix operations
//
#include <stdio.h>
#include <stdint.h>
#include <memory.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>

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
//    INT128  = 4,
    FLOAT32 = 5,
    FLOAT64 = 6,
//    FLOAT128 = 7,
//    COMPLEX64 = 8,
//    COMPLEX128 = 9
} matrix_type_t;

typedef enum {
    SIGMOID,
    SIGMOID_PRIME,
    RECTIFIER,
    TANH,
    NEGATE,
    UNIFORM,
    NORMAL,
    ONE,
    ZERO,
    IDENTITY,
} unary_operation_t;

typedef enum {
    PLUS,
    MINUS,
    TIMES,
} binary_operation_t;

typedef unsigned char byte_t;
typedef float  float32_t;   // fixme: configure
typedef double float64_t;   // fixme: configure

#define VOIDPTR(x) ((void*)&(x))

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
static ErlNifTSDKey matrix_k;

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
static ERL_NIF_TERM matrix_multiply_transposed(ErlNifEnv* env, int argc, 
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
static ERL_NIF_TERM matrix_copy(ErlNifEnv* env, int argc, 
				const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_copy_data(ErlNifEnv* env, int argc, 
				     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc, 
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
    NIF_FUNC("new_",          4, matrix_new),
    NIF_FUNC("add_",          2, matrix_add),
    NIF_FUNC("add_",          3, matrix_add),
    NIF_FUNC("subtract",      2, matrix_subtract),
    NIF_FUNC("times",         2, matrix_times),
    NIF_FUNC("multiply_",     2, matrix_multiply),
    NIF_FUNC("multiply_",     3, matrix_multiply),
    NIF_FUNC("multiply_transposed_", 2, matrix_multiply_transposed),
    NIF_FUNC("multiply_transposed_", 3, matrix_multiply_transposed),    
    NIF_FUNC("negate",        1, matrix_negate),
    NIF_FUNC("negate",        2, matrix_negate),
    NIF_FUNC("scale",         2, matrix_scale),
    NIF_FUNC("transpose",     1, matrix_transpose),
    NIF_FUNC("sigmoid",       1, matrix_sigmoid),
    NIF_FUNC("sigmoid_prime", 1, matrix_sigmoid_prime),
    NIF_FUNC("copy",          4, matrix_copy),
    NIF_FUNC("copy_data",     2, matrix_copy_data),
    NIF_FUNC("apply1",        3, matrix_apply1),
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

void copy_circular(uint8_t* dst, size_t n, uint8_t* src, size_t m)
{
    if (src == 0)
	return;
    else if (m == 0)
	memset(dst, 0x55, n);
    else {
	size_t i, j=0;
	for (i = 0; i < n; i++) {
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

float32_t normal_32_(rand_state_t* sp, float m, float s)
{
    float x1 = uniform_32_(sp);
    float x2 = uniform_32_(sp);
    return m + s*sqrtf(-2*logf(x1))*cosf(2*M_PI*x2);
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

float64_t normal_64_(rand_state_t* state, float64_t m, float64_t s)
{
    float64_t x1 = uniform_64_(state);
    float64_t x2 = uniform_64_(state);
    return m + s*sqrt(-2*log(x1))*cosf(2*M_PI*x2);
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

float32_t normal_32(rand_alg_t a, float m, float s)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return normal_32_(sp, m, s);
}

float64_t normal_64(rand_alg_t a, float64_t m, float64_t s)
{
    rand_state_t* sp = get_tsd_rand_state(a);
    return normal_64_(sp, m, s);
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
		   matrix_type_t at, byte_t* ap, size_t as,
		   matrix_type_t ct, byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    size_t elem_size_a = element_size(at);
    size_t elem_size_c = element_size(ct);
    size_t i, j;
    if (element_is_float(at) || element_is_float(ct)) {
	for (i = 0; i < n; i++) {
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		float64_t a = read_float(at, ap1);
		float64_t c;
		ap1 += elem_size_a;
		switch(func) {
		case SIGMOID:       c = op_sigmoid(a); break;
		case SIGMOID_PRIME: c = op_sigmoid_prime(a); break;
		case RECTIFIER:     c = op_max(0,a); break;
		case TANH:          c = tanh(a); break;		
		case NEGATE:        c = -a; break;
		case UNIFORM: c = uniform_64(MATRIX_RAND_ALG); break;
		case NORMAL:  c = normal_64(MATRIX_RAND_ALG,0.0,1.0); break;
		case ONE:     c = 1.0; break;
		case ZERO:     c= 0.0; break;
		case IDENTITY: c = (i==j)?1.0:0.0; break;
		default:      c = 0.0; break;
		}
		write_float(ct, cp1, c);
		cp1 += elem_size_c;
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}
    }
    else {
	for (i = 0; i < n; i++) {	
	    byte_t* ap1 = ap;
	    byte_t* cp1 = cp;
	    for (j = 0; j < m; j++) {
		int64_t a = read_int(at, ap1);
		int64_t c;
		ap1 += elem_size_a;
		switch(func) {
		case SIGMOID:   c = op_sigmoid(a); break;
		case SIGMOID_PRIME: c = op_sigmoid_prime(a); break;
		case RECTIFIER: c = op_max(0,a); break;
		case TANH:      c = tanh(a); break;		
		case NEGATE:    c = -a; break;
		case UNIFORM:   c = rand_64(MATRIX_RAND_ALG); break;
		case ONE:       c = 1; break;
		case ZERO:      c= 0; break;
		case IDENTITY:  c = (i==j); break;
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

///////////////////////////////////////////////////////////////////////////////
// add
///////////////////////////////////////////////////////////////////////////////


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

///////////////////////////////////////////////////////////////////////////////
// subtract
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// times
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// negate
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// scale_i with integer factor
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// scale_f with floating point factor
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// sigmoid
///////////////////////////////////////////////////////////////////////////////


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

///////////////////////////////////////////////////////////////////////////////
// sigmoid_prime
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// multiply
///////////////////////////////////////////////////////////////////////////////

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

///////////////////////////////////////////////////////////////////////////////
// multiply_tranposed
///////////////////////////////////////////////////////////////////////////////

static void multiply_transposed(
    matrix_type_t at,byte_t* ap,size_t as,size_t an,size_t am,
    matrix_type_t bt,byte_t* bp,size_t bs,size_t bn,size_t bm,
    matrix_type_t ct,byte_t* cp,size_t cs)
{
    if ((at == bt) && (bt == ct)) {
#ifdef USE_GCC_VECTOR
	if (is_aligned(ap) && is_aligned(bp) && is_aligned(cp))
	    mtv_multiply_transposed(at,ap,as,an,am,bp,bs,bn,bm,cp,cs);
	else
#endif	    
	    mt_multiply_transposed(at,ap,as,an,am,bp,bs,bn,bm,cp,cs);
    }
    else if (element_is_float(at) || element_is_float(bt) ||
	     element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_b = element_size(bt);
	size_t elem_size_c = element_size(ct);
	unsigned int i, j, k;
	
	for (i=0; i<an; i++) {
	    byte_t* cp1 = cp;
	    for (j=0; j<bn; j++) {
		float64_t sum = 0;
		byte_t* ap1 = ap;     // row pointer		
		byte_t* bp1 = bp;     // "column" pointer
		for (k = 0; k < bm; k++) {
		    float64_t a = read_float(at, ap1);
		    float64_t b = read_float(bt, bp1);
		    sum += a*b;
		    ap1 += elem_size_a;
		    bp1 += elem_size_b;
		}
		write_float(ct, cp1, sum);
		cp1 += elem_size_c;
		bp += bs*elem_size_b;
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
		byte_t* ap1 = ap; // row pointer
		byte_t* bp1 = bp; // "column" pointer

		for (k = 0; k < am; k++) {
		    int64_t a = read_int(at, ap1);
		    int64_t b = read_int(bt, bp1);
		    sum += a*b;
		    ap1 += elem_size_a;
		    bp1 += elem_size_b;
		}
		write_int(ct, cp1, sum);
		cp1 += elem_size_c;
		bp += bs*elem_size_b;		
	    }
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	}
    }
}

///////////////////////////////////////////////////////////////////////////////
// copy
// copy matrix data in A with repeatition into matrix C
///////////////////////////////////////////////////////////////////////////////

static void copy(matrix_type_t at,byte_t* ap,size_t as,size_t an,size_t am,
		 matrix_type_t ct,byte_t* cp,size_t cs,size_t cn, size_t cm,
		 unsigned int repeat_h, unsigned int repeat_v)
{
    // each row in A copy with repeat into row in C until C is filled
    size_t  ai = 0;
    byte_t* ap0 = ap;
    size_t  n = cn;

    if (at == ct) { // simple copy row with wrap
	size_t elem_size = element_size(at);
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
		memcpy(cp1, ap1, elem_size);
		cp1 += elem_size;
		ap1 += elem_size;
		aj++;
	    }

	    // next line
	    ap += as*elem_size;
	    cp += cs*elem_size;
	    ai++;
	}
    }
    else if (element_is_float(at) || element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

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
		cp1 += elem_size_c;
		ap1 += elem_size_a;
		aj++;
	    }
	    // next line
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;
	    ai++;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);


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
		cp1 += elem_size_c;
		ap1 += elem_size_a;
		aj++;
	    }
	    // next line
	    ap += as*elem_size_a;
	    cp += cs*elem_size_c;	    
	    ai++;
	}
    }
}



///////////////////////////////////////////////////////////////////////////////
// copy_data
// copy sequential data with repeat from matrix A into matrix C
// until matrix C is filled.
///////////////////////////////////////////////////////////////////////////////

static void copy_data(matrix_type_t at,byte_t* ap,size_t as,size_t an,size_t am,
		      matrix_type_t ct,byte_t* cp,size_t cs,size_t cn,size_t cm)
{
    byte_t* ap0 = ap;
    byte_t* ap1 = ap;
    size_t  ai = 0;
    size_t  aj = 0;
    size_t  n = cn;

    if (at == ct) {
	size_t elem_size = element_size(at);
	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		if (aj >= am) { aj = 0; ai++; ap += as*elem_size; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		memcpy(cp1, ap1, elem_size);
		cp1 += elem_size;
		ap1 += elem_size;
		aj++;
	    }
	    // next line
	    cp += cs*elem_size;
	}
    }
    else if (element_is_float(at) || element_is_float(ct)) {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;
	    
	    while(m--) {
		float64_t value;
		if (aj >= am) { aj = 0; ai++; ap += as*elem_size_a; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_float(at, ap1);
		write_float(ct, cp1, value);
		cp1 += elem_size_c;
		ap1 += elem_size_a;
		aj++;
	    }
	    cp += cs*elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(at);
	size_t elem_size_c = element_size(ct);

	while(n--) {
	    byte_t* cp1 = cp;
	    size_t m = cm;

	    while(m--) {
		int64_t value;
		if (aj >= am) { aj = 0; ai++; ap += as*elem_size_a; ap1 = ap; }
		if (ai >= an) { ai = 0; ap = ap0; ap1 = ap; }
		value = read_int(at, ap1);
		write_int(ct,cp1,value);
		cp1 += elem_size_c;
		ap1 += elem_size_a;
		aj++;
	    }
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
    byte_t* ap;
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
    if ((bin.size != 0) && (n*m*element_size(type) < bin.size))
	return enif_make_badarg(env);
    if (!make_matrix_resource(env, n, m, type, &a_bin_term, &ap, &mp))
	return enif_make_badarg(env);
    if (bin.size == n*m*element_size(type)) {
	if (mp->stride == mp->m)
	    memcpy(ap, bin.data, mp->size);
	else {
	    byte_t* bp = bin.data;
	    size_t  bs = mp->m*element_size(type);
	    size_t  as = mp->stride*element_size(type);
	    size_t i;
	    for (i = 0; i < n; i++) {
		memcpy(ap, bp, bs);
		ap += as;
		bp += bs;
	    }
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

    if (argc == 2) {
	c_t = combine_type(a.type, b.type);
	if (!make_matrix_resource(env,a.n,b.m,c_t,&c_bin_term,&c_data,&cp))
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
    else { // argc == 3
	matrix_t c;
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if ((a.n != c.n) || (b.m != c.m))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);
	multiply(a.type, a.data+a.byte_offset, a.stride, a.n, a.m, 
		 b.type, b.data+b.byte_offset, b.stride, b.n, b.m,
		 c.type, c.data+c.byte_offset, c.stride);
	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_runlock(b.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);
	return argv[2];
    }
}


// multiply matrix with a tranposed matrix
ERL_NIF_TERM matrix_multiply_transposed(
    ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
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
    if (a.m != b.m)
	return enif_make_badarg(env);

    if (argc == 2) {
	c_t = combine_type(a.type, b.type);
	if (!make_matrix_resource(env,a.n,b.n,c_t,&c_bin_term,&c_data,&cp))
	    return enif_make_badarg(env);
	enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rlock(b.rw_lock);
	multiply_transposed(
	    a.type, a.data+a.byte_offset, a.stride, a.n, a.m,
	    b.type, b.data+b.byte_offset, b.stride, b.n, b.m,
	    c_t, c_data+cp->byte_offset, cp->stride);
	enif_rwlock_runlock(b.rw_lock);    
	enif_rwlock_runlock(a.rw_lock);
	c_matrix = make_matrix(env, a.n, b.m, c_t, cp, c_bin_term);
	return c_matrix;
    }
    else { // argc == 3
	matrix_t c;
	if (!get_matrix(env, argv[2], &c))
	    return enif_make_badarg(env);
	if ((a.n != c.n) || (b.n != c.m))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	if (c.rw_lock != b.rw_lock) enif_rwlock_rlock(b.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);
	multiply_transposed(	
	    a.type, a.data+a.byte_offset, a.stride, a.n, a.m, 
	    b.type, b.data+b.byte_offset, b.stride, b.n, b.m,
	    c.type, c.data+c.byte_offset, c.stride);
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
	ERL_NIF_TERM c_bin_term; 
	matrix_type_t c_t = a.type;
	byte_t* c_data;
	matrix_t* cp;
	
	if (!make_matrix_resource(env,a.n,a.m,c_t,&c_bin_term,&c_data,&cp))
	    return enif_make_badarg(env);
	enif_rwlock_rlock(a.rw_lock);
	negate(a.type, a.data+a.byte_offset, a.stride,
	       c_t, c_data+cp->byte_offset, cp->stride, a.n, a.m);
	enif_rwlock_runlock(a.rw_lock);
	return make_matrix(env, a.n, a.m, c_t, cp, c_bin_term);
    }
    else { // args == 2
	matrix_t c;
	if (!get_matrix(env, argv[1], &c))
	    return enif_make_badarg(env);

	if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
	enif_rwlock_rwlock(c.rw_lock);

	negate(a.type, a.data+a.byte_offset, a.stride,
	       c.type, c.data+c.byte_offset, c.stride, a.n, a.m);

	enif_rwlock_rwunlock(c.rw_lock);
	if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

	return argv[1];
    }
}

// copy a matrix
//   matrix a is tiled into matrix c
//   it is tiled horizontal repeat_h times
//   and vertical repeat_v times, if 0 is specified it repeasts for ever
// 
ERL_NIF_TERM matrix_copy(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    unsigned int repeat_h;
    unsigned int repeat_v;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &c))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[2], &repeat_h))
	return enif_make_badarg(env);
    if (!enif_get_uint(env, argv[3], &repeat_v))
	return enif_make_badarg(env);
    
    if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rwlock(c.rw_lock);

    copy(a.type, a.data+a.byte_offset, a.stride, a.n, a.m,
	 c.type, c.data+c.byte_offset, c.stride, c.n, c.m,
	 repeat_h, repeat_v);
    
    enif_rwlock_rwunlock(c.rw_lock);
    if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

    return argv[1];
}

// copy data row by row from A sequentially into C
ERL_NIF_TERM matrix_copy_data(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    matrix_t a;
    matrix_t c;
    (void) argc;
    
    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &c))
	return enif_make_badarg(env);

    if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rwlock(c.rw_lock);

    copy_data(a.type, a.data+a.byte_offset, a.stride, a.n, a.m,
	      c.type, c.data+c.byte_offset, c.stride, c.n, c.m);

    enif_rwlock_rwunlock(c.rw_lock);
    if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

    return argv[1];
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
    float64_t    f_scale;
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

// matrix_apply1
ERL_NIF_TERM matrix_apply1(ErlNifEnv* env, int argc,
			   const ERL_NIF_TERM argv[])
{
    matrix_t a;
    unary_operation_t op;
    matrix_t c;
    (void) argc;
    
    if (!enif_is_atom(env, argv[2]))
	return enif_make_badarg(env);
    if (argv[2] == ATOM(sigmoid)) op = SIGMOID;
    else if (argv[2] == ATOM(sigmoid_prime)) op = SIGMOID_PRIME;
    else if (argv[2] == ATOM(rectifier)) op = RECTIFIER;
    else if (argv[2] == ATOM(tanh))    op = TANH;
    else if (argv[2] == ATOM(negate))  op = NEGATE;
    else if (argv[2] == ATOM(uniform)) op = UNIFORM;
    else if (argv[2] == ATOM(normal))  op = NORMAL;
    else if (argv[2] == ATOM(zero))    op = ZERO;
    else if (argv[2] == ATOM(one))     op = ONE;
    else if (argv[2] == ATOM(identity)) op = IDENTITY;
    else return enif_make_badarg(env);

    if (!get_matrix(env, argv[0], &a))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[1], &c))
	return enif_make_badarg(env);

    if (c.rw_lock != a.rw_lock) enif_rwlock_rlock(a.rw_lock);
    enif_rwlock_rwlock(c.rw_lock);

    apply1(op,
	   a.type, a.data+a.byte_offset, a.stride,
	   c.type, c.data+c.byte_offset, c.stride,
	   a.n, a.m);

    enif_rwlock_rwunlock(c.rw_lock);
    if (c.rw_lock != a.rw_lock) enif_rwlock_runlock(a.rw_lock);

    return argv[1];	
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

    // how to find all thread data?
    enif_tsd_key_destroy(matrix_k);
    
    DBG("matrix_unload\r\n");
}

ERL_NIF_INIT(matrix, matrix_funcs,
	     matrix_load, NULL,
	     matrix_upgrade, matrix_unload)
