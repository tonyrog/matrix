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
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL
    (void) bn;	
    size_t i, j, k;

    for (i=0; i<an; i++) {
        TYPE* cp1 = cp;
	for (j=0; j<bm; j++) {
	    TYPE2 sum = 0;
	    TYPE* bp1 = bp + j*bv;  // bv2!!
	    TYPE* ap1 = ap;
	    for (k = 0; k < am; k++) {
		TYPE2 p = OPERATION(*ap1,*bp1);
		sum = OPERATION2(sum, p);
		ap1 += av;
		bp1 += bu;
	    }
	    *cp1 = sum;
	    cp1 += cv;
	}
	ap += au;
	cp += cu;
    }
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2
