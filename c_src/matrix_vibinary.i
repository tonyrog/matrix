//#define VPROC      CAT3(vproc_,NAME,_int8)
#define VFUN      CAT3(vfun_,NAME,_int8)
#define TYPE           int8_t
#define TYPE_R         int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

//#define VPROC      CAT3(vproc_,NAME,_int16)
#define VFUN      CAT3(vfun_,NAME,_int16)
#define TYPE           int16_t
#define TYPE_R         int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

//#define VPROC       CAT3(vproc_,NAME,_int32)
#define VFUN       CAT3(vfun_,NAME,_int32)
#define TYPE            int32_t
#define TYPE_R          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

//#define VPROC       CAT3(vproc_,NAME,_int64)
#define VFUN       CAT3(vfun_,NAME,_int64)
#define TYPE            int64_t
#define TYPE_R          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#include "mtv_binary_op.i"

//#define VPROC       CAT3(vproc_,NAME,_int128)
#define VFUN       CAT3(vfun_,NAME,_int128)
#define TYPE            int128_t
#define TYPE_R          int128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) CAT2(op_,OP)((a),(b))
#define OPERATION(a,b)  CAT2(op128_,OP)((a),(b))
#include "mtv_binary_op.i"

/*
static mtv_binary_func_t CAT3(vproc_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mtv_binary_func_t) CAT3(vproc_,NAME,_int8),
    [INT16] = (mtv_binary_func_t) CAT3(vproc_,NAME,_int16),
    [INT32] = (mtv_binary_func_t) CAT3(vproc_,NAME,_int32),
    [INT64] = (mtv_binary_func_t) CAT3(vproc_,NAME,_int64),
    [INT128] = (mtv_binary_func_t) CAT3(vproc_,NAME,_int128),	
};
*/

static binary_vop_t CAT3(vfun_,NAME,_ops)[NUM_TYPES] = {
    [INT8] = (binary_vop_t) CAT3(vfun_,NAME,_int8),
    [INT16] = (binary_vop_t) CAT3(vfun_,NAME,_int16),
    [INT32] = (binary_vop_t) CAT3(vfun_,NAME,_int32),
    [INT64] = (binary_vop_t) CAT3(vfun_,NAME,_int64),
    [INT128] = (binary_vop_t) CAT3(vfun_,NAME,_int128),
    [FLOAT32] = (binary_vop_t) CAT3(vfun_,NAME,_int32),  // mask
    [FLOAT64] = (binary_vop_t) CAT3(vfun_,NAME,_int64),  // mask
};
