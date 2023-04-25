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

#ifdef VPROC
static void VPROC(TYPE* ap, int au,
		  TYPE_R* cp, int cu,
		  size_t n, size_t m
		  PARAMS_DECL)
{
    LOCALS_DECL
    while(n--) {
	byte_t* ap1 = ap;
	byte_t* cp1 = cp;	
	size_t m1 = m;
	while(m1 >= VELEMS(TYPE)) {
	    VTYPE a = *(VTYPE*)ap1;
	    ap1 += sizeof(vector_t);
	    *((VTYPE_R*)cp1) = VOPERATION(a,b);
	    cp1 += sizeof(vector_t);	    
	    m1  -= VELEMS(TYPE);
	}
	while(m1--) {
	    TYPE a = *((TYPE*)ap1);
	    *((TYPE_R*)cp1) = OPERATION(a);	    
	    ap1 += sizeof(TYPE);
	    cp1 += sizeof(TYPE_R);
	}
        ap += au;
        cp += cu;
    }
}
#endif

static void VFUN(vector_t* ap, vector_t* cp)
{
    VTYPE a = *(VTYPE*)ap;
    *cp = (vector_t) VOPERATION(a);
}

#undef VPROC
#undef VFUN
#undef TYPE
#undef TYPE_R
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef VOPERATION
#undef OPERATION

