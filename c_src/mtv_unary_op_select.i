

static void SELECT(matrix_type_t type,
		   byte_t* ap, size_t as,
		   byte_t* cp, size_t cs,
		   size_t n, size_t m
		   PARAMS_DECL)
{
    switch(type) {
    case INT8: MT_NAME(mtv_,_int8_)((int8_t*)ap,as,(int8_t*)cp,cs,n,m PARAMS); break;
    case INT16: MT_NAME(mtv_,_int16_)((int16_t*)ap,as,(int16_t*)cp,cs,n,m PARAMS); break;
    case INT32: MT_NAME(mtv_,_int32_)((int32_t*)ap,as,(int32_t*)cp,cs,n,m PARAMS); break;
    case INT64: MT_NAME(mtv_,_int64_)((int64_t*)ap,as,(int64_t*)cp,cs,n,m PARAMS); break;
    case FLOAT32: MT_NAME(mtv_,_float32_)((float32_t*)ap,as,(float32_t*)cp,cs,n,m PARAMS); break;
    case FLOAT64: MT_NAME(mtv_,_float64_)((float64_t*)ap,as,(float64_t*)cp,cs,n,m PARAMS); break;
    default: break;
    }
}

#undef SELECT
#undef NAME
#undef PARAMS_DECL
#undef PARAMS
