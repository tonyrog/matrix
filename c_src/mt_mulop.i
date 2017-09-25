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

// Common BINOP functions
// #define NAME
// #define TYPE
// #define PARAMS_DECL
// #define OPERATION

static void NAME(TYPE* ap, size_t as, size_t an, size_t am,
		 TYPE* bp, size_t bs, size_t bn, size_t bm,
		 TYPE* cp, size_t cs
		 PARAMS_DECL)
{
    size_t i, j, k;
    (void) bn;
    for (i=0; i<an; i++) {
        TYPE* cp1 = cp;
	for (j=0; j<bm; j++) {
	    TYPE2 sum = 0;
	    TYPE* bp1 = bp + j;
	    TYPE* ap1 = ap;
	    for (k = 0; k < am; k++) {
		TYPE2 p = OPERATION(*ap1,*bp1);
		sum = OPERATION2(sum, p);
		ap1 += 1;
		bp1 += bs;
	    }
	    *cp1++ = sum;
	}
	ap += as;
	cp += cs;
    }
}

#undef NAME
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef OPERATION
#undef OPERATION2
