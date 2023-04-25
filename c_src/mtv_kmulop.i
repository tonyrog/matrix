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
static inline void CAT2(PROC,_loadv_mulop)(byte_t* cp,byte_t* bp,int bu,size_t n)
{
    byte_t* cp1 = cp;
    while(n--) {
	TYPE b = *((TYPE*)bp);
	*((TYPE*)cp1) = b;
	cp1 += sizeof(TYPE);
	bp += bu;
    }
}

static void PROC(byte_t* ap, int au, size_t an, size_t am,
		 byte_t* bp, int bu, size_t bn, size_t bm,
		 int32_t* kp, int kv, size_t km,
		 byte_t* cp, int cu, int cv
		 PARAMS_DECL)
{
    LOCALS_DECL
    UNUSED(an);
    
    while (bm--) {
	VTYPE col[(bn+VELEMS(TYPE)-1)/VELEMS(TYPE)];
	int32_t* kp1 = kp;
	size_t n = km;

	CAT2(PROC,_loadv_mulop)((byte_t*)col,bp,bu,bn);
	bp += bm;     // advance to next column
	
	while(n--) {
	    int32_t i = *kp1 - 1;
	    if ((i >= 0) && (i < (int)an)) {
		byte_t* ap1 = ap + i*au;
		byte_t* cp1 = cp + i*cu;
		byte_t* tp = (byte_t*) &col[0];
		*((TYPE*)cp1) += CAT2(vproc_dot_,TYPE)(tp, ap1, am);
	    }
	    kp1 += kv;
	}
	cp += cv;
    }
}

#undef PROC
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
