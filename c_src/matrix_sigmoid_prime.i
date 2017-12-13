
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
