/////////////////////////////////////////////////////////////////////////////
//   MULTIPLY
/////////////////////////////////////////////////////////////////////////////

#define NAME multiply

#define PROCEDURE      CAT3(mt_,NAME,_int8)
#define TYPE           int8_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_int16)
#define TYPE           int16_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_int32)
#define TYPE           int32_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_int64)
#define TYPE           int64_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_float32)
#define TYPE           float32_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_float64)
#define TYPE           float64_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_complex64)
#define TYPE           complex64_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

#define PROCEDURE      CAT3(mt_,NAME,_complex128)
#define TYPE           complex128_t
#define PARAMS_DECL
#define LOCALS_DECL
#include "mt_mulop.i"

static mt_mulop_func_t CAT3(mt_,NAME,_funcs)[NUM_TYPES] = {
    [INT8] = (mt_mulop_func_t) CAT3(mt_,NAME,_int8),
    [INT16] = (mt_mulop_func_t) CAT3(mt_,NAME,_int16),
    [INT32] = (mt_mulop_func_t) CAT3(mt_,NAME,_int32),
    [INT64] = (mt_mulop_func_t) CAT3(mt_,NAME,_int64),
    [FLOAT32] = (mt_mulop_func_t) CAT3(mt_,NAME,_float32),
    [FLOAT64] = (mt_mulop_func_t) CAT3(mt_,NAME,_float64),
    [COMPLEX64] = (mt_mulop_func_t) CAT3(mt_,NAME,_complex64),
    [COMPLEX128] = (mt_mulop_func_t) CAT3(mt_,NAME,_complex128)
};

#undef OP2
#undef OP
#undef NAME
