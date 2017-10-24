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

static TYPE2 CAT2(PROCEDURE,_dotv_t_mulop)(TYPE* ap,  TYPE* bp,  size_t n)
{
    TYPE2 sum = 0;
    VTYPE vsum = VTYPE_ZERO;
    unsigned int i;
    
    while(n >= VELEMS(TYPE)) {
	VTYPE r = OPERATION(*(VTYPE*)ap, *(VTYPE*)bp);
	vsum = OPERATION2(vsum, r);
	ap += VELEMS(TYPE);
	bp += VELEMS(TYPE);		
	n -= VELEMS(TYPE);
    }
    for (i = 0; i < VELEMS(TYPE); i++)
	sum += vsum[i];
    while(n--) {
	TYPE p = OPERATION(*ap,*bp);
	sum = OPERATION2(sum, p);
	ap++;
	bp++;
    }
    return sum;
}

static void PROCEDURE(TYPE* ap, size_t as, size_t an, size_t am,
		      TYPE* bp, size_t bs, size_t bn, size_t bm,
		      TYPE* cp, size_t cs
		      PARAMS_DECL)
{
    LOCALS_DECL
    (void) am;
    TYPE* bp0 = bp;
    
    while (an--) {
	TYPE* cp1 = cp;
	size_t n = bn;

	bp = bp0;
	while(n--) {
	    *cp1++ = CAT2(PROCEDURE,_dotv_t_mulop)(ap, bp, bm);
	    bp  += bs;
	}
	ap += as;
	cp += cs;
    }
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2
