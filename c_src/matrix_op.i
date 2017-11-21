
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


#ifdef USE_GCC_VECTOR

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

#ifdef USE_GCC_VECTOR

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

/////////////////////////////////////////////////////////////////////////////
//   TIMES
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_times_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

#define PROCEDURE      mt_times_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#include "mt_binary_op.i"

// SELECT
#define SELECT mt_times
#define NAME times
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_binary_op_select.i"

#ifdef USE_GCC_VECTOR

#define PROCEDURE      mtv_times_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_mul((a),(b))
#define OPERATION(a,b) op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_mul((a),(b))
#define OPERATION(a,b) op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_mul((a),(b))
#define OPERATION(a,b) op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_mul((a),(b))
#define OPERATION(a,b) op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_mul((a),(b))
#define OPERATION(a,b) op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) op_mul((a),(b))
#define OPERATION(a,b) op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) complex64_multiply((a),(b))
#define OPERATION(a,b)  op_mul((a),(b))
#include "mtv_binary_op.i"

#define PROCEDURE      mtv_times_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define VOPERATION(a,b) complex128_multiply((a),(b))
#define OPERATION(a,b)  op_mul((a),(b))
#include "mtv_binary_op.i"

#define SELECT mtv_times
#define NAME times
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mtv_binary_op_select.i"

#endif

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

#ifdef USE_GCC_VECTOR
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

/////////////////////////////////////////////////////////////////////////////
//   SIGMOID
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_sigmoid_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"


#define PROCEDURE      mt_sigmoid_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   cop_sigmoid((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   cop_sigmoid((a))
#include "mt_unary_op.i"

#define SELECT mt_sigmoid
#define NAME sigmoid
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

/////////////////////////////////////////////////////////////////////////////
//   SIGMOID_PRIME
/////////////////////////////////////////////////////////////////////////////
#if 0
#define PROCEDURE      mt_sigmoid_prime_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"


#define PROCEDURE      mt_sigmoid_prime_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   cop_sigmoid_prime((a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   cop_sigmoid_prime((a))
#include "mt_unary_op.i"


#define SELECT mt_sigmoid_prime
#define NAME sigmoid_prime
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   SIGMOID_PRIME1
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_sigmoid_prime1_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime1_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"


#define PROCEDURE      mt_sigmoid_prime1_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime1_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime1_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime1_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime1_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define PROCEDURE      mt_sigmoid_prime1_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_mul((a),op_sub(1.0,(a)))
#include "mt_unary_op.i"

#define SELECT mt_sigmoid_prime1
#define NAME sigmoid_prime1
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

/////////////////////////////////////////////////////////////////////////////
//   RECTIFIER
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_rectifier_int8
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_max(0,(a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_rectifier_int16
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_max(0,(a))
#include "mt_unary_op.i"


#define PROCEDURE      mt_rectifier_int32
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_max(0,(a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_rectifier_int64
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_max(0,(a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_rectifier_float32
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_max(0.0,(a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_rectifier_float64
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   op_max(0.0,(a))
#include "mt_unary_op.i"

#define PROCEDURE      mt_rectifier_complex64
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CMPLXF(op_max(0.0,crealf((a))),0)
#include "mt_unary_op.i"

#define PROCEDURE      mt_rectifier_complex128
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a)   CMPLX(op_max(0.0,creal((a))),0)
#include "mt_unary_op.i"

#define SELECT mt_rectifier
#define NAME rectifier
#define LOCALS_DECL
#define PARAMS_DECL
#define PARAMS
#include "mt_unary_op_select.i"

/////////////////////////////////////////////////////////////////////////////
//   DOT used by MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_dot_int8_t
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_int16_t
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_int32_t
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_int64_t
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_float32_t
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_float64_t
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_complex64_t
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

#define PROCEDURE      mt_dot_complex128_t
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_dot.i"

/////////////////////////////////////////////////////////////////////////////
//   SIMD/DOT used by MULTIPLY
/////////////////////////////////////////////////////////////////////////////
#ifdef USE_GCC_VECTOR

#define PROCEDURE      mtv_dot_int8_t
#define TYPE           int8_t
#define TYPE2          int16_t
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_int16_t
#define TYPE           int16_t
#define TYPE2          int32_t
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_int32_t
#define TYPE           int32_t
#define TYPE2          int64_t
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_int64_t
#define TYPE           int64_t
#define TYPE2          int64_t
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_float32_t
#define TYPE           float32_t
#define TYPE2          float64_t
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_float64_t
#define TYPE           float64_t
#define TYPE2          float64_t
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_complex64_t
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define OPERATION(a,b)   op_mul((a),(b))
#define OPERATION2(a,b)  op_add((a),(b))
#define VOPERATION(a,b)  complex64_multiply((a),(b))
#define VOPERATION2(a,b) complex64_add((a),(b))
#define VELEMENT(a,i)    complex64_velement((a),(i))
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_complex128_t
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define OPERATION(a,b)   op_mul((a),(b))
#define OPERATION2(a,b)  op_add((a),(b))
#define VOPERATION(a,b)  complex128_multiply((a),(b))
#define VOPERATION2(a,b) complex128_add((a),(b))
#define VELEMENT(a,i)    complex128_velement((a),(i))
#include "mtv_dot.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_multiply_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_complex64
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define PROCEDURE      mt_multiply_complex128
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop.i"

#define SELECT mt_multiply
#define NAME multiply
#include "mt_mulop_select.i"

#ifdef USE_GCC_VECTOR

#define PROCEDURE      mtv_multiply_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_complex64
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)   op_mul((a),(b))
#define OPERATION2(a,b)  op_add((a),(b))
#define VOPERATION(a,b)  complex64_multiply((a),(b))
#define VOPERATION2(a,b) complex64_add((a),(b))
#define VELEMENT(a,i)    complex64_velement((a),(i))
#define VSETELEMENT(a,i,v) complex64_vsetelement(&(a),(i),(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_complex128
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)   op_mul((a),(b))
#define OPERATION2(a,b)  op_add((a),(b))
#define VOPERATION(a,b)  complex128_multiply((a),(b))
#define VOPERATION2(a,b) complex128_add((a),(b))
#define VELEMENT(a,i)    complex128_velement((a),(i))
#define VSETELEMENT(a,i,v) complex128_vsetelement(&(a),(i),(v))
#include "mtv_mulop.i"

#define SELECT mtv_multiply
#define NAME multiply
#include "mtv_mulop_select.i"

#endif

/////////////////////////////////////////////////////////////////////////////
//   MULTIPLY_TRANSPOSED
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_multiply_transposed_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_complex64
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define PROCEDURE      mt_multiply_transposed_complex128
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_mulop_t.i"

#define SELECT mt_multiply_transposed
#define NAME multiply_transposed
#include "mt_mulop_select.i"

#ifdef USE_GCC_VECTOR

#define PROCEDURE      mtv_multiply_transposed_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop_t.i"

#define PROCEDURE      mtv_multiply_transposed_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop_t.i"

#define PROCEDURE      mtv_multiply_transposed_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop_t.i"

#define PROCEDURE      mtv_multiply_transposed_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop_t.i"

#define PROCEDURE      mtv_multiply_transposed_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop_t.i"

#define PROCEDURE      mtv_multiply_transposed_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  op_mul((a),(b))
#define VOPERATION2(a,b) op_add((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_mulop_t.i"

#define PROCEDURE      mtv_multiply_transposed_complex64
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  complex64_multiply((a),(b))
#define VOPERATION2(a,b) complex64_add((a),(b))
#define VELEMENT(a,i)    complex64_velement((a),(i))
#define VSETELEMENT(a,i,v) complex64_vsetelement(&(a),(i),(v))
#include "mtv_mulop.i"

#define PROCEDURE      mtv_multiply_transposed_complex128
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#define VOPERATION(a,b)  complex128_multiply((a),(b))
#define VOPERATION2(a,b) complex128_add((a),(b))
#define VELEMENT(a,i)    complex128_velement((a),(i))
#define VSETELEMENT(a,i,v) complex128_vsetelement(&(a),(i),(v))
#include "mtv_mulop.i"

#define SELECT mtv_multiply_transposed
#define NAME multiply_transposed
#include "mtv_mulop_select.i"

#endif
