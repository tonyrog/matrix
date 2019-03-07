#define PROCEDURE      CAT3(mtv_,NAME,_int8)
#define TYPE           int8_t
#define TYPE_R         int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_int16)
#define TYPE           int16_t
#define TYPE_R         int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE       CAT3(mtv_,NAME,_int32)
#define TYPE            int32_t
#define TYPE_R          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE       CAT3(mtv_,NAME,_int64)
#define TYPE            int64_t
#define TYPE_R          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_float32)
#define TYPE           float32_t
#define TYPE_R         int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_float64)
#define TYPE           float64_t
#define TYPE_R         int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_complex64)
#define TYPE           complex64_t
#define TYPE_R         int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(vcop64_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(cop64_,OP)((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      CAT3(mtv_,NAME,_complex128)
#define TYPE           complex128_t
#define TYPE_R         int128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(vcop128_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(cop128_,OP)((a),(b))
#include "mtv_binary_op.i"

static mtv_binary_func_t CAT3(mtv_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mtv_binary_func_t) CAT3(mtv_,NAME,_int8),
    [INT16] = (mtv_binary_func_t) CAT3(mtv_,NAME,_int16),
    [INT32] = (mtv_binary_func_t) CAT3(mtv_,NAME,_int32),
    [INT64] = (mtv_binary_func_t) CAT3(mtv_,NAME,_int64),
    [FLOAT32] = (mtv_binary_func_t) CAT3(mtv_,NAME,_float32),
    [FLOAT64] = (mtv_binary_func_t) CAT3(mtv_,NAME,_float64),
    [COMPLEX64] = (mtv_binary_func_t) CAT3(mtv_,NAME,_complex64),
    [COMPLEX128] = (mtv_binary_func_t) CAT3(mtv_,NAME,_complex128)
};
