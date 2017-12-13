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
