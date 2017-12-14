/////////////////////////////////////////////////////////////////////////////
//   SIMD/DOT used by MULTIPLY
/////////////////////////////////////////////////////////////////////////////

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
#define VOPERATION(a,b)  complex64_mul((a),(b))
#define VOPERATION2(a,b) complex64_add((a),(b))
#define VELEMENT(a,i)    complex64_velement((a),(i))
#include "mtv_dot.i"

#define PROCEDURE      mtv_dot_complex128_t
#define TYPE           complex128_t
#define TYPE2          complex128_t
#define OPERATION(a,b)   op_mul((a),(b))
#define OPERATION2(a,b)  op_add((a),(b))
#define VOPERATION(a,b)  complex128_mul((a),(b))
#define VOPERATION2(a,b) complex128_add((a),(b))
#define VELEMENT(a,i)    complex128_velement((a),(i))
#include "mtv_dot.i"
