
/////////////////////////////////////////////////////////////////////////////
//   SUBTRACT
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE           mt_subtract_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE           mt_subtract_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_sub((a),(b))
#include "mt_binary_op.i"

// SELECT
#define SELECT mt_subtract
#define NAME subtract
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_binary_op_select.i"

#ifdef USE_VECTOR

#define PROCEDURE           mtv_subtract_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_sub((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE           mtv_subtract_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_sub((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE           mtv_subtract_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_sub((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE           mtv_subtract_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_sub((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE           mtv_subtract_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_sub((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE           mtv_subtract_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_sub((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_subtract_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) complex64_subtract((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_subtract_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) complex128_subtract((a),(b))
#define OPERATION(a,b) op_sub((a),(b))
#include "mtv_binary_op.i"

#define SELECT mtv_subtract
#define NAME subtract
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mtv_binary_op_select.i"

#endif
