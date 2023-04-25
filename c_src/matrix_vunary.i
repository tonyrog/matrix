#define VFUN           CAT3(vfun_,NAME,_uint8)
#define TYPE           uint8_t
#define TYPE_R         uint8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_uint16)
#define TYPE           uint16_t
#define TYPE_R         uint16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_uint32)
#define TYPE           uint32_t
#define TYPE_R         uint32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_uint64)
#define TYPE           uint64_t
#define TYPE_R         uint64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_int8)
#define TYPE           int8_t
#define TYPE_R         int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_int16)
#define TYPE           int16_t
#define TYPE_R         int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_int32)
#define TYPE           int32_t
#define TYPE_R         int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_int64)
#define TYPE           int64_t
#define TYPE_R         int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_float32)
#define TYPE           float32_t
#define TYPE_R         float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

#define VFUN           CAT3(vfun_,NAME,_float64)
#define TYPE           float64_t
#define TYPE_R         float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  CAT2(op_,OP)((a))
#define OPERATION(a)   CAT2(op_,OP)((a))
#include "mtv_unary_op.i"

/*
static mtv_unary_func_t CAT3(vproc_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mtv_unary_func_t) CAT3(vproc_,NAME,_int8),
    [INT16] = (mtv_unary_func_t) CAT3(vproc_,NAME,_int16),
    [INT32] = (mtv_unary_func_t) CAT3(vproc_,NAME,_int32),
    [INT64] = (mtv_unary_func_t) CAT3(vproc_,NAME,_int64),
    [FLOAT32] = (mtv_unary_func_t) CAT3(vproc_,NAME,_float32),
    [FLOAT64] = (mtv_unary_func_t) CAT3(vproc_,NAME,_float64),
    [COMPLEX64] = (mtv_unary_func_t) CAT3(vproc_,NAME,_complex64),
    [COMPLEX128] = (mtv_unary_func_t) CAT3(vproc_,NAME,_complex128)
};
*/

static unary_vop_t CAT3(vfun_,NAME,_ops)[NUM_TYPES] = {
    [UINT8] = (unary_vop_t) CAT3(vfun_,NAME,_uint8),
    [UINT16] = (unary_vop_t) CAT3(vfun_,NAME,_uint16),
    [UINT32] = (unary_vop_t) CAT3(vfun_,NAME,_uint32),
    [UINT64] = (unary_vop_t) CAT3(vfun_,NAME,_uint64),    
    [INT8] = (unary_vop_t) CAT3(vfun_,NAME,_int8),
    [INT16] = (unary_vop_t) CAT3(vfun_,NAME,_int16),
    [INT32] = (unary_vop_t) CAT3(vfun_,NAME,_int32),
    [INT64] = (unary_vop_t) CAT3(vfun_,NAME,_int64),
    [FLOAT32] = (unary_vop_t) CAT3(vfun_,NAME,_float32),
    [FLOAT64] = (unary_vop_t) CAT3(vfun_,NAME,_float64),
//    [COMPLEX64] = (unary_vop_t) CAT3(vfun_,NAME,_complex64),
//    [COMPLEX128] = (unary_vop_t) CAT3(vfun_,NAME,_complex128)
};
