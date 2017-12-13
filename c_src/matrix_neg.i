
/////////////////////////////////////////////////////////////////////////////
//   NEGATE
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_negate_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_negate_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_neg((a))
#include "mt_unary_op.i"

#define SELECT mt_negate
#define NAME negate
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

#ifdef USE_VECTOR
#define PROCEDURE      mtv_negate_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_neg((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_neg((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_neg((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_neg((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_neg((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  op_neg((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  complex64_negate((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"

#define PROCEDURE      mtv_negate_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a)  complex128_negate((a))
#define OPERATION(a)   op_neg((a))
#include "mtv_unary_op.i"    

#define SELECT mtv_negate
#define NAME negate
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mtv_unary_op_select.i"

#endif
