

static void SELECT(matrix_type_t type,
		   byte_t* ap, size_t as,
		   byte_t* bp, size_t bs,
		   byte_t* cp, size_t cs,
		   size_t n, size_t m)
{
    switch(type) {
    case INT8: MT_NAME(mt_,_int8_)((int8_t*)ap,as,(int8_t*)bp,bs,(int8_t*)cp,cs,n,m); break;
    case INT16: MT_NAME(mt_,_int16_)((int16_t*)ap,as,(int16_t*)bp,bs,(int16_t*)cp,cs,n,m); break;
    case INT32: MT_NAME(mt_,_int32_)((int32_t*)ap,as,(int32_t*)bp,bs,(int32_t*)cp,cs,n,m); break;
    case INT64: MT_NAME(mt_,_int64_)((int64_t*)ap,as,(int64_t*)bp,bs,(int64_t*)cp,cs,n,m); break;
    case FLOAT32: MT_NAME(mt_,_float32_)((float32_t*)ap,as,(float32_t*)bp,bs,(float32_t*)cp,cs,n,m); break;
    case FLOAT64: MT_NAME(mt_,_float64_)((float64_t*)ap,as,(float64_t*)bp,bs,(float64_t*)cp,cs,n,m); break;
    default: break;
    }
}

#undef NAME
#undef SELECT
