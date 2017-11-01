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

static void PROCEDURE(TYPE* ap, int au, size_t an, size_t am,
		      TYPE* bp, int bu, size_t bn, size_t bm,
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL

    while (bm--) {
        VTYPE col[(bn+VELEMS(TYPE)-1)/VELEMS(TYPE)];
	TYPE* ap1 = ap;
	TYPE* bp1 = bp;
	TYPE* cp1 = cp;
	size_t n = bn;
	size_t j = 0;

	// load column from B
	while(n >= VELEMS(TYPE)) {
	    VTYPE  ce;
	    size_t i;
	    TYPE*  bp2 = bp1;
	    for (i = 0; i < VELEMS(TYPE); i++) {
		VSETELEMENT(ce,i,*bp2);
		bp2 += bu;
	    }
	    col[j++] = ce;
	    n -= VELEMS(TYPE);
	    bp1 += bu*VELEMS(TYPE);
        }
	if (n) {  // column tail
	    VTYPE  ce;
	    size_t i = 0;
	    TYPE*  bp2 = bp1;
	    while(i < n) {
		VSETELEMENT(ce,i,*bp2);
		bp2 += bu;
		i++;
	    }
	    while(i < VELEMS(TYPE)) {
		VSETELEMENT(ce,i,0);
		i++;
	    }
	    col[j++] = ce;
	}
	bp++;     // advance to next column
	n = an;   // multiply with all rows in A
	while(n--) {
	    TYPE2 sum = 0;
	    VTYPE vsum = VTYPE_ZERO;
	    size_t m = am;
	    TYPE* ap2 = ap1;
	    TYPE* tp = (TYPE*) &col[0];
	    size_t i;
	    while(m >= VELEMS(TYPE)) {
		VTYPE r = VOPERATION(*(VTYPE*)tp, *(VTYPE*)ap2);
		vsum = VOPERATION2(vsum, r);
		tp += VELEMS(TYPE);
		ap2 += VELEMS(TYPE);
		m -= VELEMS(TYPE);
	    }
	    for (i = 0; i < VELEMS(TYPE); i++)
		sum += VELEMENT(vsum,i);
	    while(m--) {
		TYPE p = OPERATION(*tp,*ap2);
		sum = OPERATION2(sum, p);
		tp++;
		ap2++;
	    }
	    *cp1 = sum;
	    cp1 += cu;
	    ap1 += au;
	}
	cp += cv;
    }
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2
#undef VOPERATION
#undef VOPERATION2
#undef VELEMENT
#undef VSETELEMENT