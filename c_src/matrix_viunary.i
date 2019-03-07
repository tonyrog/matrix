#define PROCEDURE      CAT3(mtv_,NAME,_int8)
#define TYPE           int8_t
#define TYPE_R         int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_int16)
#define TYPE           int16_t
#define TYPE_R         int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_int32)
#define TYPE           int32_t
#define TYPE_R         int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_int64)
#define TYPE           int64_t
#define TYPE_R         int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_int128)
#define TYPE           int128_t
#define TYPE_R         int128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op128_,OP)((a))
#include "mtv_unary_op.i"

static mtv_unary_func_t CAT3(mtv_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mtv_unary_func_t) CAT3(mtv_,NAME,_int8),
    [INT16] = (mtv_unary_func_t) CAT3(mtv_,NAME,_int16),
    [INT32] = (mtv_unary_func_t) CAT3(mtv_,NAME,_int32),
    [INT64] = (mtv_unary_func_t) CAT3(mtv_,NAME,_int64),
    [INT128] = (mtv_unary_func_t) CAT3(mtv_,NAME,_int128),
};
