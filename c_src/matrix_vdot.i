/////////////////////////////////////////////////////////////////////////////
//   SIMD/DOT used by MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define NAME dot_
#define OP   mul
#define OP2  add

#define TYPE           int8_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          int8_t
#define OPERATION(a,b)   CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           int16_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          int16_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           int32_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          int32_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_dot.i"

#define TYPE           int64_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          int64_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           float32_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          float32_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           float64_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          float64_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           complex64_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          complex64_t
#define OPERATION(a,b)   CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b)  CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  complex64_mul((a),(b))
#define VOPERATION2(a,b) complex64_add((a),(b))
#define VELEMENT(a,i)    complex64_velement((a),(i))
#include "mtv_dot.i"

#define TYPE           complex128_t
#define PROCEDURE      CAT3(mtv_,NAME,TYPE)
#define TYPE2          complex128_t
#define OPERATION(a,b)   CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b)  CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  complex128_mul((a),(b))
#define VOPERATION2(a,b) complex128_add((a),(b))
#define VELEMENT(a,i)    complex128_velement((a),(i))
#include "mtv_dot.i"

#undef OP2
#undef OP
#undef NAME