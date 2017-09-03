//
// Matrix operations
//
// defined for int8, int16, int32, int64, float32, float64
// both mixed and non mixed operations
// FIXME use vector ops if possible
//
#include <stdio.h>
#include <stdint.h>
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

typedef unsigned char byte_t;
typedef float  float32_t;   // fixme: configure
typedef double float64_t;   // fixme: configure

#define ATOM(name) atm_##name

#define DECL_ATOM(name) \
    ERL_NIF_TERM atm_##name = 0

// require env in context (ugly)
#define LOAD_ATOM(name)			\
    atm_##name = enif_make_atom(env,#name)

#define LOAD_ATOM_STRING(name,string)			\
    atm_##name = enif_make_atom(env,string)


static int matrix_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info);
static int matrix_upgrade(ErlNifEnv* env, void** priv_data, void** old_priv_data,
		       ERL_NIF_TERM load_info);
static void matrix_unload(ErlNifEnv* env, void* priv_data);

static ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, 
			     const ERL_NIF_TERM argv[]);
static ERL_NIF_TERM matrix_subtract(ErlNifEnv* env, int argc, 
			     const ERL_NIF_TERM argv[]);

ErlNifFunc matrix_funcs[] =
{
    { "add",        2, matrix_add },
    { "subtract",   2, matrix_subtract },
};


size_t element_size_[6] = { 1, 2, 4, 8, 4, 8 };

DECL_ATOM(matrix);


static size_t element_size(matrix_type_t type)
{
    return element_size_[type];
}

static int element_is_float(matrix_type_t type)
{
    return (type >= FLOAT32_T);
}

static matrix_type_t combine_type(matrix_type_t type_a, matrix_type_t type_b)
{
    // max(type_a, type_b)
    if (type_a > type_b) return type_a; else return type_b;
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

#define MATRIX_BINOP_FUNC(name,op,type)			     \
static void matrix_##name##_(type* ap, type* bp, type* cp, size_t n) \
{ \
    while(n--) { \
	*cp++ = *ap++ op *bp++; \
    } \
}

#define MATRIX_UNOP_FUNC(name,op,type)		      \
static void matrix_##name##_(type* ap, type* cp, size_t n) \
{ \
    while(n--) { \
	*cp++ = op *ap++; \
    } \
}

#define MATRIX_BINOP_SELECT(name) \
static void matrix_##name##_(matrix_type_t type, byte_t* ap, byte_t* bp, byte_t* cp, size_t n) \
{ \
  switch(type) { \
  case INT8: matrix_##name##_int8_((int8_t*)ap,(int8_t*)bp,(int8_t*)cp, n); break; \
  case INT16: matrix_##name##_int16_((int16_t*)ap,(int16_t*)bp,(int16_t*)cp,n); break; \
  case INT32: matrix_##name##_int32_((int32_t*)ap,(int32_t*)bp,(int32_t*)cp,n); break; \
  case INT64: matrix_##name##_int64_((int64_t*)ap,(int64_t*)bp,(int64_t*)cp,n); break; \
  case FLOAT32: matrix_##name##_float32_((float32_t*)ap,(float32_t*)bp,(float32_t*)cp,n); break; \
  case FLOAT64: matrix_##name##_float64_((float64_t*)ap,(float64_t*)bp,(float64_t*)cp,n); break; \
  default: break;  \
  }  \
}

#define MATRIX_UNOP_SELECT(name) \
static void matrix_##name##_(matrix_type_t type, byte_t* ap, byte_t* cp, size_t n) \
{ \
  switch(type) { \
  case INT8: matrix_##name##_int8_((int8_t*)ap,(int8_t*)cp, n); break; \
  case INT16: matrix_##name##_int16_((int16_t*)ap,(int16_t*)cp,n); break; \
  case INT32: matrix_##name##_int32_((int32_t*)ap,(int32_t*)cp,n); break; \
  case INT64: matrix_##name##_int64_((int64_t*)ap,(int64_t*)cp,n); break; \
  case FLOAT32: matrix_##name##_float32_((float32_t*)ap,(float32_t*)cp,n); break; \
  case FLOAT64: matrix_##name##_float64_((float64_t*)ap,(float64_t*)cp,n); break; \
  default: break;  \
  }  \
}


MATRIX_BINOP_FUNC(add_int8,+,int8_t)
MATRIX_BINOP_FUNC(add_int16,+,int16_t)
MATRIX_BINOP_FUNC(add_int32,+,int32_t)
MATRIX_BINOP_FUNC(add_int64,+,int64_t)
MATRIX_BINOP_FUNC(add_float32,+,float)
MATRIX_BINOP_FUNC(add_float64,+,double)
MATRIX_BINOP_SELECT(add)

MATRIX_BINOP_FUNC(subtract_int8,-,int8_t)
MATRIX_BINOP_FUNC(subtract_int16,-,int16_t)
MATRIX_BINOP_FUNC(subtract_int32,-,int32_t)
MATRIX_BINOP_FUNC(subtract_int64,-,int64_t)
MATRIX_BINOP_FUNC(subtract_float32,-,float)
MATRIX_BINOP_FUNC(subtract_float64,-,double)
MATRIX_BINOP_SELECT(subtract)

MATRIX_UNOP_FUNC(negate_int8,-,int8_t)
MATRIX_UNOP_FUNC(negate_int16,-,int16_t)
MATRIX_UNOP_FUNC(negate_int32,-,int32_t)
MATRIX_UNOP_FUNC(negate_int64,-,int64_t)
MATRIX_UNOP_FUNC(negate_float32,-,float)
MATRIX_UNOP_FUNC(negate_float64,-,double)
MATRIX_UNOP_SELECT(negate)



static void add(matrix_type_t type_a, byte_t* ptr_a,
		matrix_type_t type_b, byte_t* ptr_b,
		matrix_type_t type_c, byte_t* ptr_c,
		size_t nm)
{
    if (type_a == type_b) {
	matrix_add_(type_a, ptr_a, ptr_b, ptr_c, nm);
    }
    else if (element_is_float(type_a) || element_is_float(type_b)) {
	size_t elem_size_a = element_size(type_a);
	size_t elem_size_b = element_size(type_b);
	size_t elem_size_c = element_size(type_c);

	while(nm--) {
	    float64_t a = read_float(type_a, ptr_a);
	    float64_t b = read_float(type_b, ptr_b);

	    ptr_a += elem_size_a;
	    ptr_b += elem_size_b;
	    write_float(type_c, ptr_c, a+b);
	    ptr_c += elem_size_c;
	}
    }
    else {
	size_t elem_size_a = element_size(type_a);
	size_t elem_size_b = element_size(type_b);
	size_t elem_size_c = element_size(type_c);

	while(nm--) {
	    int64_t a = read_int(type_a, ptr_a);
	    int64_t b = read_int(type_b, ptr_b);

	    ptr_a += elem_size_a;
	    ptr_b += elem_size_b;
	    write_int(type_c, ptr_c, a+b);
	    ptr_c += elem_size_c;
	}
    }
}

// Get matrix argument
// { 'matrix', n, m, type, ptr, binary-data }

static int get_matrix(ErlNifEnv* env, ERL_BIF_TERM arg,
		      unsigned int* np, unsigned int* mp,
		      matrix_type_t* tp, ErlBinBinary* bp)
{
    int arity;
    unsigned int type;
    unsigned long ptr;
    const ERL_NIF_TERM* elems;
    
    
    if (!enif_get_tuple(env, arg, &arity, &elems)) return 0;
    if (arity != 6) return 0;
    if (elems[0] != ATOM(matrix)) return 0;
    if (!enif_get_uint(env, elems[1], np)) return 0;
    if (!enif_get_uint(env, elems[2], mp)) return 0;
    if (!enif_get_uint(env, elems[3], &type)) return 0;
    if (type > FLOAT64) return 0;
    *tp = type;
    if (!enif_get_ulong(env, elems[4], &ptr)) return 0;
    if (!enif_inspect_binary(env, a_elems[5], bp)) return 0;
    if (bp->size < (*np)*(*mp)*element_size(type)) return 0;
    return 1;
}
		      

// add two matrices on form
//
ERL_NIF_TERM matrix_add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    int a_arity;
    unsigned int a_n, a_m;
    matrix_type_t a_t;
    const ERL_NIF_TERM* a_elems;
    ErlNifBinary a_bin;
    
    int b_arity;
    unsigned int b_n, b_m;
    matrix_type_t b_t;
    const ERL_NIF_TERM* b_elems;
    ErlNifBinary b_bin;

    ErlNifBinary c_bin;
    matrix_type_t c_t;
    ERL_NIF_TERM c_matrix;

    if (!get_matrix(env, argv[0], &a_n, &a_m, &a_t, &a_bin))
	return enif_make_badarg(env);
    if (!get_matrix(env, argv[0], &b_n, &b_m, &b_t, &b_bin))
	return enif_make_badarg(env);
    if ((a_n != b_n) || (a_m != b_m))
	return enif_make_badarg(env);

    c_t = combine_types(a_t, b_t);
    if (!enif_alloc_binary(a_n*a_m*element_size(c_t), &c_bin))
	return enif_make_badarg(env);

    add(a_t, a_bin.data, b_t, b_bin.data, c_t, c_bin.data);

    c_matrix = enif_make_binary(&c_bin);
    return c_matrix;
}




static int matrix_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info)
{
    (void) env;
    (void) load_info;
    DBG("matrix_load\r\n");
    LOAD_ATOM(matrix);    
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
    DBG("matrix_unload\r\n");
}

ERL_NIF_INIT(matrix, matrix_funcs,
	     matrix_load, NULL,
	     matrix_upgrade, matrix_unload)
