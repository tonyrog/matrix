/////////////////////////////////////////////////////////////////////////////
//   SIMD/DOT used by MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define NAME dot_
#define OP   mul
#define OP2  add

#define TYPE           int8_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          int8_t
#define OPERATION(a,b)   CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           int16_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          int16_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           int32_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          int32_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#define VSETELEMENT(a,i,v) ((a)[(i)]=(v))
#include "mtv_dot.i"

#define TYPE           int64_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          int64_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           float32_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          float32_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

#define TYPE           float64_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          float64_t
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define VOPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#define VELEMENT(a,i)    ((a)[(i)])
#include "mtv_dot.i"

/*
#define TYPE           complex64_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          complex64_t
#define OPERATION(a,b)   CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b)  CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  vcop64_mul((a),(b))
#define VOPERATION2(a,b) vcop64_add((a),(b))
#define VELEMENT(a,i)    vcop64_velement((a),(i))
#include "mtv_dot.i"

#define TYPE           complex128_t
#define PROC      CAT3(vproc_,NAME,TYPE)
#define TYPE2          complex128_t
#define OPERATION(a,b)   CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b)  CAT2(op_,OP2)((a),(b))
#define VOPERATION(a,b)  vcop128_mul((a),(b))
#define VOPERATION2(a,b) vcop128_add((a),(b))
#define VELEMENT(a,i)    vcop128_velement((a),(i))
#include "mtv_dot.i"
*/

#undef OP2
#undef OP
#undef NAME
