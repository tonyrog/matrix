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
		      int32_t* kp, int kv, size_t km,
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL
    UNUSED(an);
    
    while (bm--) {
	VTYPE col[(bn+VELEMS(TYPE)-1)/VELEMS(TYPE)];
	int32_t* kp1 = kp;
	size_t n = km;

	CAT2(PROCEDURE,_loadv_mulop)(col,bp,bu,bn);
	bp++;     // advance to next column
	
	while(n--) {
	    int32_t i = *kp1 - 1;
	    if ((i >= 0) && (i < (int)an)) {
		TYPE* ap1 = ap + i*au;
		TYPE* cp1 = cp + i*cu;
		TYPE* tp = (TYPE*) &col[0];
		*cp1 += CAT2(mtv_dot_,TYPE)(tp, ap1, am);
	    }
	    kp1 += kv;
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
