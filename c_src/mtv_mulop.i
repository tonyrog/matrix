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

// Load n elements from b into vector array cp
static inline void CAT2(PROCEDURE,_loadv_mulop)(VTYPE* cp,TYPE* bp,int bu,size_t n)
{
    TYPE* cp1 = (TYPE*) cp;
    while(n--) {
	*cp1++ = *bp;
	bp += bu;
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
	    *cp1 = CAT2(mtv_dot_,TYPE)(tp, ap1, am);
	    cp1 += cu;
	    ap1 += au;
	}
	cp += cv;
    }
}

#undef PROCEDURE
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
