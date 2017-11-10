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

static TYPE2 CAT2(PROCEDURE,_dot_t_mulop)(TYPE* ap,int av,TYPE* bp,int bv,size_t n)
{
    TYPE2 sum = 0;

    while(n--) {
	TYPE2 p = OPERATION(*ap,*bp);
	sum = OPERATION2(sum, p);
	ap += av;
	bp += bv;
    }
    return sum;
}


static void PROCEDURE(TYPE* ap, int au, int av, size_t an, size_t am,
		      TYPE* bp, int bu, int bv, size_t bn, size_t bm,
		      byte_t* dp,int du,int dv,
		      TYPE* cp, int cu, int cv
		      PARAMS_DECL)
{
    LOCALS_DECL
    (void) am;
    TYPE* bp0 = bp;

    while(an--) {
        TYPE* cp1 = cp;
	byte_t* dp1 = dp;
	size_t n = bn;
	bp = bp0;
	while(n--) {
	    if (*dp1)
		*cp1 = CAT2(PROCEDURE,_dot_t_mulop)(ap,av,bp,bv,bm);
	    else
		*cp1 = TYPE_ZERO;
	    cp1 += cv;
	    bp += bu;
	    dp1 += du;
	}
	ap += au;
	cp += cu;
	dp += dv;
    }
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2
