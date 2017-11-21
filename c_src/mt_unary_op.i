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

static void PROCEDURE(TYPE* ap, int au, int av,
		      TYPE* cp, int cu, int cv,
		      size_t n, size_t m
		      PARAMS_DECL)
{
    LOCALS_DECL
    while(n--) {
	TYPE* ap1 = ap;
	TYPE* cp1 = cp;
	size_t m1 = m;
	while(m1--) {
	    TYPE a = *ap1;
	    *cp1 = OPERATION(a);
	    ap1 += av;
	    cp1 += cv;
	}
	ap += au;
	cp += cu;
    }
}

#undef PROCEDURE
#undef TYPE
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
