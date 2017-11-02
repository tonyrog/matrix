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

static inline TYPE2 CAT2(PROCEDURE,_dotv_mulop)(TYPE* ap,TYPE* bp,size_t n)
{
    TYPE2 sum = 0;
    VTYPE vsum = VTYPE_ZERO;
    unsigned int i;
    
    while(n >= VELEMS(TYPE)) {
	VTYPE r = VOPERATION(*(VTYPE*)ap, *(VTYPE*)bp);
	vsum = VOPERATION2(vsum, r);
	ap += VELEMS(TYPE);
	bp += VELEMS(TYPE);		
	n -= VELEMS(TYPE);
    }
    for (i = 0; i < VELEMS(TYPE); i++)
	sum += VELEMENT(vsum,i);
    while(n--) {
	TYPE p = OPERATION(*ap,*bp);
	sum = OPERATION2(sum, p);
	ap++;
	bp++;
    }
    return sum;
}

// Load n elements from b into vector array cp
static inline void CAT2(PROCEDURE,_loadv_mulop)(VTYPE* cp,TYPE* bp,int bu,size_t n)
{
    while(n >= VELEMS(TYPE)) {
	VTYPE ce;
	size_t i;
	for (i = 0; i < VELEMS(TYPE); i++) {
	    VSETELEMENT(ce,i,*bp);
	    bp += bu;
	}
	*cp++ = ce;
	n -= VELEMS(TYPE);
    }
    if (n) {  // column tail
	VTYPE  ce = VTYPE_ZERO;
	size_t i = 0;
	while(i < n) {
	    VSETELEMENT(ce,i,*bp);
	    bp += bu;
	    i++;
	}
	*cp = ce;
    }
}

static void PROCEDURE(TYPE* ap, int au, size_t an, size_t am,
		      TYPE* bp, int bu, size_t bn, size_t bm,
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL

    while (bm--) {
        VTYPE col[(bn+VELEMS(TYPE)-1)/VELEMS(TYPE)];
	TYPE* ap1 = ap;
	TYPE* cp1 = cp;
	size_t n;

	CAT2(PROCEDURE,_loadv_mulop)(col,bp,bu,bn);
	
	bp++;     // advance to next column
	n = an;   // multiply with all rows in A
	while(n--) {
	    TYPE* tp = (TYPE*) &col[0];
	    *cp1 = CAT2(PROCEDURE,_dotv_mulop)(tp, ap1, am);
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
