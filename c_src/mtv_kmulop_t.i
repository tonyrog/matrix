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
		      int32_t* kp, int kv, size_t km,
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL
    UNUSED(am);
    UNUSED(an);    
    
    while (km--) {
	int32_t i = *kp - 1;
	if ((i >= 0) && (i < (int)an)) {
	    TYPE* cp1 = cp + i*cu;
	    TYPE* ap1 = ap + i*au;  // select row
	    size_t n = bn;
	    TYPE* bp1 = bp;
	    while(n--) {
		*cp1 += CAT2(mtv_dot_,TYPE)(ap1, bp1, bm);
		cp1  += cv;
		bp1  += bu;
	    }
	}
	kp += kv;
    }
}

#undef PROCEDURE
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
