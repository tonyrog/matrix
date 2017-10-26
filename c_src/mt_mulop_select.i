
static void SELECT(matrix_type_t type,
		   byte_t* ap,size_t as,size_t an, size_t am,
		   byte_t* bp,size_t bs,size_t bn, size_t bm,
		   byte_t* cp,size_t cs)				\
{
    switch(type) {
    case INT8: MT_NAME(mt_,_int8)((int8_t*)ap,as,an,am,(int8_t*)bp,bs,bn,bm,(int8_t*)cp,cs); break;
    case INT16: MT_NAME(mt_,_int16)((int16_t*)ap,as,an,am,(int16_t*)bp,bs,bn,bm,(int16_t*)cp,cs); break;
    case INT32: MT_NAME(mt_,_int32)((int32_t*)ap,as,an,am,(int32_t*)bp,bs,bn,bm,(int32_t*)cp,cs); break;
    case INT64: MT_NAME(mt_,_int64)((int64_t*)ap,as,an,am,(int64_t*)bp,bs,bn,bm,(int64_t*)cp,cs); break;
    case FLOAT32: MT_NAME(mt_,_float32)((float32_t*)ap,as,an,am,(float32_t*)bp,bs,bn,bm,(float32_t*)cp,cs); break;
    case FLOAT64: MT_NAME(mt_,_float64)((float64_t*)ap,as,an,am,(float64_t*)bp,bs,bn,bm,(float64_t*)cp,cs); break;
    case COMPLEX64: MT_NAME(mt_,_complex64)((complex64_t*)ap,as,an,am,(complex64_t*)bp,bs,bn,bm,(complex64_t*)cp,cs); break;
    case COMPLEX128: MT_NAME(mt_,_complex128)((complex128_t*)ap,as,an,am,(complex128_t*)bp,bs,bn,bm,(complex128_t*)cp,cs); break;
    default: break;
    }
}

#undef NAME
#undef SELECT
