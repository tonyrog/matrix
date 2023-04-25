//
// INTEGER UNARY OPERATIONS
//
//#define PROC      CAT3(proc_,NAME,_int8)
#define FUN      CAT3(fun_,NAME,_int8)
#define TYPE           int8_t
#define TYPE_R         int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mt_unary_op.i"

//#define PROC      CAT3(proc_,NAME,_int16)
#define FUN      CAT3(fun_,NAME,_int16)
#define TYPE           int16_t
#define TYPE_R         int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mt_unary_op.i"

//#define PROC      CAT3(proc_,NAME,_int32)
#define FUN      CAT3(fun_,NAME,_int32)
#define TYPE           int32_t
#define TYPE_R         int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mt_unary_op.i"

//#define PROC      CAT3(proc_,NAME,_int64)
#define FUN      CAT3(fun_,NAME,_int64)
#define TYPE           int64_t
#define TYPE_R         int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mt_unary_op.i"

//#define PROC      CAT3(proc_,NAME,_int128)
#define FUN      CAT3(fun_,NAME,_int128)
#define TYPE           int128_t
#define TYPE_R         int128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CAT2(op128_,OP)((a))
#include "mt_unary_op.i"

/*
static mt_unary_func_t CAT3(proc_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mt_unary_func_t) CAT3(proc_,NAME,_int8),
    [INT16] = (mt_unary_func_t) CAT3(proc_,NAME,_int16),
    [INT32] = (mt_unary_func_t) CAT3(proc_,NAME,_int32),
    [INT64] = (mt_unary_func_t) CAT3(proc_,NAME,_int64),
    [INT128] = (mt_unary_func_t) CAT3(proc_,NAME,_int128),
};
*/


static unary_op_t CAT3(fun_,NAME,_ops)[NUM_TYPES] = {
    [INT8] = (unary_op_t) CAT3(fun_,NAME,_int8),
    [INT16] = (unary_op_t) CAT3(fun_,NAME,_int16),
    [INT32] = (unary_op_t) CAT3(fun_,NAME,_int32),
    [INT64] = (unary_op_t) CAT3(fun_,NAME,_int64),
    [INT128] = (unary_op_t) CAT3(fun_,NAME,_int128),
    [FLOAT32] = (unary_op_t) CAT3(fun_,NAME,_int32),  // mask
    [FLOAT64] = (unary_op_t) CAT3(fun_,NAME,_int64),  // mask
};
