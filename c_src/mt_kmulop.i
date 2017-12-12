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

static void PROCEDURE(TYPE* ap, int au, int av, size_t an, size_t am,
		      TYPE* bp, int bu, int bv, size_t bn, size_t bm,
		      byte_t* kp,int ku,int kv,
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL
    (void) am;
    TYPE* bp0 = bp;
    UNUSED(ku);

    while(an--) {
	TYPE* cp1 = cp;
	size_t m = bm;
	if (*kp) {
	    bp = bp0;
	    while(m--) {
		*cp1 = CAT2(mt_dot_,TYPE)(ap,av,bp,bu,bn);
		cp1 += cv;
		bp  += bv;
	    }
	}
	else {
	    while(m--) {
		*cp1 = TYPE_ZERO;
		cp1 += cv;
	    }
	}
	ap += au;
	cp += cu;
	kp += kv;
    }
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2
