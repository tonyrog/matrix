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

static void PROC(byte_t* ap, int au, size_t an, size_t am,
		 byte_t* bp, int bu, size_t bn, size_t bm,
		 byte_t* cp, int cu, int cv
		 PARAMS_DECL)
{
    LOCALS_DECL
    UNUSED(am);
    byte_t* bp0 = bp;

    while (an--) {
	byte_t* cp1 = cp;
	size_t n = bn;
	bp = bp0;
	while(n--) {
	    TYPE d = CAT2(vproc_dot_,TYPE)(ap, bp, bm);
	    *((TYPE*)cp1) = d;
	    bp  += bu;
	    cp1 += cv;
	}
	ap += au;
	cp += cu;
    }
}

#undef PROC
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
