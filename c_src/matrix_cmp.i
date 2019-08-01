/////////////////////////////////////////////////////////////////////////////
// EQ
/////////////////////////////////////////////////////////////////////////////

#define NAME eq
#define OP eq
#include "matrix_compare.i"
#undef OP
#undef NAME

/////////////////////////////////////////////////////////////////////////////
// LT
/////////////////////////////////////////////////////////////////////////////

#define NAME lt
#define OP lt
#include "matrix_compare.i"
#undef OP
#undef NAME

/////////////////////////////////////////////////////////////////////////////
// LTE
/////////////////////////////////////////////////////////////////////////////

#define NAME lte
#define OP lte
#include "matrix_compare.i"
#undef OP
#undef NAME

#ifdef USE_VECTOR

/////////////////////////////////////////////////////////////////////////////
// EQ
/////////////////////////////////////////////////////////////////////////////
#define NAME eq
#define OP   eq
#include "matrix_vcompare.i"
#undef OP
#undef NAME

/////////////////////////////////////////////////////////////////////////////
// LT
/////////////////////////////////////////////////////////////////////////////
#define NAME lt
#define OP   lt
#include "matrix_vcompare.i"
#undef OP
#undef NAME

/////////////////////////////////////////////////////////////////////////////
// LTE
/////////////////////////////////////////////////////////////////////////////
#define NAME lte
#define OP   lte
#include "matrix_vcompare.i"
#undef OP
#undef NAME

#endif