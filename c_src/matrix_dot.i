/////////////////////////////////////////////////////////////////////////////
//   DOT used by MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define NAME dot_
#define OP   mul
#define OP2  add

#define TYPE           int8_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           int16_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           int32_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           int64_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           float32_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           float64_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           complex64_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#define TYPE           complex128_t
#define PROCEDURE      CAT3(mt_,NAME,TYPE)
#define TYPE2          complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#define OPERATION(a,b)  CAT2(op_,OP)((a),(b))
#define OPERATION2(a,b) CAT2(op_,OP2)((a),(b))
#include "mt_dot.i"

#undef OP2
#undef OP
#undef NAME
