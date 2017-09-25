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

static void PROCEDURE(TYPE* ap, size_t as, TYPE* bp, size_t bs,
		      TYPE* cp, size_t cs,
		      size_t n, size_t m
		      PARAMS_DECL)
{
    LOCALS_DECL
    while(n--) {
	TYPE* ap1 = ap;
	TYPE* bp1 = bp;
	TYPE* cp1 = cp;
	size_t m1 = m;
	while(m1 >= VELEMS(TYPE)) {
	    VTYPE a = *(VTYPE*)ap1;
	    VTYPE b = *(VTYPE*)bp1;
	    ap1 += VELEMS(TYPE);
	    bp1 += VELEMS(TYPE);
	    *(VTYPE*)cp1 = VOPERATION(a,b);
	    cp1 += VELEMS(TYPE);
	    m1  -= VELEMS(TYPE);
	}
	while(m1--) {
	    TYPE a = *ap1++;
	    TYPE b = *bp1++;
	    *cp1++ = OPERATION(a,b);
	}
        ap += as;
        bp += bs;
        cp += cs;
    }
}

#undef PROCEDURE
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef VOPERATION
#undef OPERATION
