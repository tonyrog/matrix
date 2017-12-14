
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
