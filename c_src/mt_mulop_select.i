
static void SELECT(matrix_type_t type,
		   byte_t* ap, int au, int av, size_t an, size_t am,
		   byte_t* bp, int bu, int bv, size_t bn, size_t bm,
		   byte_t* dp, int du, int dv,
		   byte_t* cp, int cu, int cv)
{
    switch(type) {
    case INT8: MT_NAME(mt_,_int8)((int8_t*)ap,au,av,an,am,(int8_t*)bp,bu,bv,bn,bm,dp,du,dv,(int8_t*)cp,cu,cv); break;
    case INT16: MT_NAME(mt_,_int16)((int16_t*)ap,au,av,an,am,(int16_t*)bp,bu,bv,bn,bm,dp,du,dv,(int16_t*)cp,cu,cv); break;
    case INT32: MT_NAME(mt_,_int32)((int32_t*)ap,au,av,an,am,(int32_t*)bp,bu,bv,bn,bm,dp,du,dv,(int32_t*)cp,cu,cv); break;
    case INT64: MT_NAME(mt_,_int64)((int64_t*)ap,au,av,an,am,(int64_t*)bp,bu,bv,bn,bm,dp,du,dv,(int64_t*)cp,cu,cv); break;
    case FLOAT32: MT_NAME(mt_,_float32)((float32_t*)ap,au,av,an,am,(float32_t*)bp,bu,bv,bn,bm,dp,du,dv,(float32_t*)cp,cu,cv); break;
    case FLOAT64: MT_NAME(mt_,_float64)((float64_t*)ap,au,av,an,am,(float64_t*)bp,bu,bv,bn,bm,dp,du,dv,(float64_t*)cp,cu,cv); break;
    case COMPLEX64: MT_NAME(mt_,_complex64)((complex64_t*)ap,au,av,an,am,(complex64_t*)bp,bu,bv,bn,bm,dp,du,dv,(complex64_t*)cp,cu,cv); break;
    case COMPLEX128: MT_NAME(mt_,_complex128)((complex128_t*)ap,au,av,an,am,(complex128_t*)bp,bu,bv,bn,bm,dp,du,dv,(complex128_t*)cp,cu,cv); break;
    default: break;
    }
}

#undef NAME
#undef SELECT
