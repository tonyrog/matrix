
/////////////////////////////////////////////////////////////////////////////
//   KMULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define PROCEDURE      mt_kmultiply_int8
#define TYPE           int8_t
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_int16
#define TYPE           int16_t
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_int32
#define TYPE           int32_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_int64
#define TYPE           int64_t
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_float32
#define TYPE           float32_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b) op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_float64
#define TYPE           float64_t
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_complex64
#define TYPE           complex64_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define PROCEDURE      mt_kmultiply_complex128
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  op_mul((a),(b))
#define OPERATION2(a,b) op_add((a),(b))
#include "mt_kmulop.i"

#define SELECT mt_kmultiply
#define NAME kmultiply
#include "mt_kmulop_select.i"

#ifdef USE_VECTOR

#define PROCEDURE      mtv_kmultiply_int8
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_int16
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_int32
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_int64
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_float32
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_float64
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_complex64
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
#include "mtv_kmulop.i"

#define PROCEDURE      mtv_kmultiply_complex128
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
#include "mtv_kmulop.i"

#define SELECT mtv_kmultiply
#define NAME kmultiply
#include "mtv_kmulop_select.i"

#endif
