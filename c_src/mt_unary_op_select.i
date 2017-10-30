/***************************************************************************
 *
 * Copyright (C) 2017, Rogvall Invest AB, <tony@rogvall.se>
 *
 * This software is licensed as described in the file COPYRIGHT, which
 * you should have received as part of this distribution. The terms
 * are also available at http://www.rogvall.se/docs/copyright.txt.
 *
 * You may opt to use, copy, modify, merge, publish, distribute and/or sell
 * copies of the Software, and permit persons to whom the Software is
 * furnished to do so, under the terms of the COPYRIGHT file.
 *
 * This software is distributed on an "AS IS" basis, WITHOUT WARRANTY OF ANY
 * KIND, either express or implied.
 *
 ***************************************************************************/

static void SELECT(matrix_type_t type,
		   byte_t* ap, int au, int av,
		   byte_t* cp, int cu, int cv,
		   size_t n, size_t m
		   PARAMS_DECL)
{
    LOCALS_DECL
    switch(type) {
    case INT8: MT_NAME(mt_,_int8)((int8_t*)ap,au,av,(int8_t*)cp,cu,cv,n,m PARAMS); break;
    case INT16: MT_NAME(mt_,_int16)((int16_t*)ap,au,av,(int16_t*)cp,cu,cv,n,m PARAMS); break;
    case INT32: MT_NAME(mt_,_int32)((int32_t*)ap,au,av,(int32_t*)cp,cu,cv,n,m PARAMS); break;
    case INT64: MT_NAME(mt_,_int64)((int64_t*)ap,au,av,(int64_t*)cp,cu,cv,n,m PARAMS); break;
    case FLOAT32: MT_NAME(mt_,_float32)((float32_t*)ap,au,av,(float32_t*)cp,cu,cv,n,m PARAMS); break;
    case FLOAT64: MT_NAME(mt_,_float64)((float64_t*)ap,au,av,(float64_t*)cp,cu,cv,n,m PARAMS); break;
    case COMPLEX64: MT_NAME(mt_,_complex64)((complex64_t*)ap,au,av,(complex64_t*)cp,cu,cv,n,m PARAMS); break;
    case COMPLEX128: MT_NAME(mt_,_complex128)((complex128_t*)ap,au,av,(complex128_t*)cp,cu,cv,n,m PARAMS); break;
    default: break;
    }
}

#undef SELECT
#undef NAME
#undef LOCALS_DECL
#undef PARAMS_DECL
#undef PARAMS

