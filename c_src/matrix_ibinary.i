/////////////////////////////////////////////////////////////////////////////
// INTEGER BINARY OP
/////////////////////////////////////////////////////////////////////////////

// REQUIRE MACRO NAME,OP to be defined

//#define PROC      CAT3(proc_,NAME,_int8)
#define FUN      CAT3(fun_,NAME,_int8)
#define TYPE           int8_t
#define TYPE_R         int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) CAT2(op_,OP)((a),(b))
#include "mt_binary_op.i"

//#define PROC      CAT3(proc_,NAME,_int16)
#define FUN      CAT3(fun_,NAME,_int16)
#define TYPE           int16_t
#define TYPE_R         int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) CAT2(op_,OP)((a),(b))
#include "mt_binary_op.i"

//#define PROC      CAT3(proc_,NAME,_int32)
#define FUN      CAT3(fun_,NAME,_int32)
#define TYPE           int32_t
#define TYPE_R         int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) CAT2(op_,OP)((a),(b))
#include "mt_binary_op.i"

// #define PROC      CAT3(proc_,NAME,_int64)
#define FUN      CAT3(fun_,NAME,_int64)
#define TYPE           int64_t
#define TYPE_R         int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) CAT2(op_,OP)((a),(b))
#include "mt_binary_op.i"

// #define PROC      CAT3(proc_,NAME,_int128)
#define FUN      CAT3(fun_,NAME,_int128)
#define TYPE           int128_t
#define TYPE_R         int128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) CAT2(op128_,OP)((a),(b))
#include "mt_binary_op.i"

/*
static mt_binary_func_t CAT3(proc_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mt_binary_func_t) CAT3(proc_,NAME,_int8),
    [INT16] = (mt_binary_func_t) CAT3(proc_,NAME,_int16),
    [INT32] = (mt_binary_func_t) CAT3(proc_,NAME,_int32),
    [INT64] = (mt_binary_func_t) CAT3(proc_,NAME,_int64),
    [INT128] = (mt_binary_func_t) CAT3(proc_,NAME,_int128)
};
*/

static binary_op_t CAT3(fun_,NAME,_ops)[NUM_TYPES] = {
    [INT8]  = (binary_op_t) CAT3(fun_,NAME,_int8),
    [INT16] = (binary_op_t) CAT3(fun_,NAME,_int16),
    [INT32] = (binary_op_t) CAT3(fun_,NAME,_int32),
    [INT64] = (binary_op_t) CAT3(fun_,NAME,_int64),
    [INT128] = (binary_op_t) CAT3(fun_,NAME,_int128),
    [FLOAT32] = (binary_op_t) CAT3(fun_,NAME,_int32),  // mask
    [FLOAT64] = (binary_op_t) CAT3(fun_,NAME,_int64),  // mask
};
