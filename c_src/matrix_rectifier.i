
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
