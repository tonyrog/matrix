
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

#ifdef USE_VECTOR

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
