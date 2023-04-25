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
static inline void CAT2(VPROC,_loadv_mulop)(byte_t* cp,byte_t* bp,int bu,size_t n)
{
    byte_t* cp1 = cp;
    while(n--) {
	*((TYPE*)cp1) = *((TYPE*)bp);
	cp1 += sizeof(TYPE);
	bp += bu;
    }
}

static void VPROC(byte_t* ap, int au, size_t an, size_t am,
		  byte_t* bp, int bu, size_t bn, size_t bm,
		  byte_t* cp, int cu, int cv
		  PARAMS_DECL)
{
    LOCALS_DECL
    VTYPE col[(bn+VELEMS(TYPE)-1)/VELEMS(TYPE)];
	
    while (bm--) {
	byte_t* ap1 = ap;
	byte_t* cp1 = cp;
	size_t n;
	CAT2(VPROC,_loadv_mulop)((byte_t*)col,bp,bu,bn);

	bp += sizeof(TYPE);  // advance to next column
	n = an;              // multiply with all rows in A
	while(n--) {
	    TYPE d = CAT2(vproc_dot_,TYPE)((byte_t*)col, ap1, am);
	    *((TYPE*)cp1) = d;
	    cp1 += cu;
	    ap1 += au;
	}
	cp += cv;
    }
}

#undef VPROC
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
