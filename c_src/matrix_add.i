/////////////////////////////////////////////////////////////////////////////
//   ADD
/////////////////////////////////////////////////////////////////////////////

// add: int8 x int8 -> int8
#define PROCEDURE      mt_add_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: int16 x int16 -> int16
#define PROCEDURE      mt_add_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: int32 x int32 -> int32
#define PROCEDURE      mt_add_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: int64 x int64 -> int64
#define PROCEDURE      mt_add_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: float32 x float32 -> float32
#define PROCEDURE      mt_add_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: float64 x float64 -> float64
#define PROCEDURE      mt_add_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: complex64 x complex64 -> complex64
#define PROCEDURE      mt_add_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// add: complex128 x complex128 -> complex128
#define PROCEDURE      mt_add_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_add((a),(b))
#include "mt_binary_op.i"

// SELECT
#define SELECT mt_add
#define NAME add
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_binary_op_select.i"


#ifdef USE_VECTOR

#define PROCEDURE      mtv_add_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_add((a),(b))
#define OPERATION(a,b) op_add((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_add_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_add((a),(b))
#define OPERATION(a,b) op_add((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_add_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_add((a),(b))
#define OPERATION(a,b) op_add((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_add_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_add((a),(b))
#define OPERATION(a,b) op_add((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_add_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_add((a),(b))
#define OPERATION(a,b) op_add((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_add_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_add((a),(b))
#define OPERATION(a,b) op_add((a),(b))
#include "mtv_binary_op.i"

// add float32_t components. Make sure number of columns are double
#define PROCEDURE      mtv_add_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) complex64_add((a),(b))
#define OPERATION(a,b)  op_add((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_add_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) complex128_add((a),(b))
#define OPERATION(a,b)  op_add((a),(b))
#include "mtv_binary_op.i"

#define SELECT mtv_add
#define NAME add
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mtv_binary_op_select.i"

#endif
