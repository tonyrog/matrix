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

static inline TYPE2 PROCEDURE(TYPE* ap,TYPE* bp,size_t n)
{
    TYPE2 sum = 0;
    VTYPE vsum = VTYPE_ZERO;
    unsigned int i;
    
    while(n >= VELEMS(TYPE)) {
	VTYPE r = VOPERATION(*(VTYPE*)ap, *(VTYPE*)bp);
	vsum = VOPERATION2(vsum, r);
	ap += VELEMS(TYPE);
	bp += VELEMS(TYPE);		
	n -= VELEMS(TYPE);
    }
    for (i = 0; i < VELEMS(TYPE); i++)
	sum += VELEMENT(vsum,i);
    while(n--) {
	TYPE p = OPERATION(*ap,*bp);
	sum = OPERATION2(sum, p);
	ap++;
	bp++;
    }
    return sum;
}

#undef PROCEDURE
#undef TYPE
#undef TYPE2
#undef OPERATION
#undef OPERATION2
#undef VOPERATION
#undef VOPERATION2
#undef VELEMENT


