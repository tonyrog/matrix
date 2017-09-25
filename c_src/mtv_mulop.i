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

static void PROCEDURE(TYPE* ap, size_t as, size_t an, size_t am,
		      TYPE* bp, size_t bs, size_t bn, size_t bm,
		      TYPE* cp, size_t cs
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
		ce[i] = *bp2;
		bp2 += bs;
	    }
	    col[j++] = ce;
	    n -= VELEMS(TYPE);
	    bp1 += bs*VELEMS(TYPE);
        }
	if (n) {  // column tail
	    VTYPE  ce;
	    size_t i = 0;
	    TYPE*  bp2 = bp1;
	    while(i < n) {
		ce[i] = *bp2;
		bp2 += bs;
		i++;
	    }
	    while(i < VELEMS(TYPE)) {
		ce[i] = 0;
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
		VTYPE r = OPERATION(*(VTYPE*)tp, *(VTYPE*)ap2);
		vsum = OPERATION2(vsum, r);
		tp += VELEMS(TYPE);
		ap2 += VELEMS(TYPE);
		m -= VELEMS(TYPE);
	    }
	    for (i = 0; i < VELEMS(TYPE); i++)
		sum += vsum[i];
	    while(m--) {
		TYPE p = OPERATION(*tp,*ap2);
		sum = OPERATION2(sum, p);
		tp++;
		ap2++;
	    }
	    *cp1 = sum;
	    cp1 += cs;
	    ap1 += as;
	}
	cp++;
    }
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2

