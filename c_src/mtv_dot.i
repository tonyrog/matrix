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

static inline TYPE2 PROC(byte_t* ap,byte_t* bp,size_t n)
{
    TYPE2 sum = 0;
    VTYPE vsum = VTYPE_ZERO;
    unsigned int i;
    
    while(n >= VELEMS(TYPE)) {
	VTYPE a = *(VTYPE*)ap;
	VTYPE b = *(VTYPE*)bp;
	VTYPE r = VOPERATION(a, b);
	vsum = VOPERATION2(vsum, r);
	ap += sizeof(vector_t);
	bp += sizeof(vector_t);
	n -= VELEMS(TYPE);
    }
    for (i = 0; i < VELEMS(TYPE); i++)
	sum += VELEMENT(vsum,i);
    while(n--) {
	TYPE a = *(TYPE*)ap;
	TYPE b = *(TYPE*)bp;		
	TYPE p = OPERATION(a,b);
	sum = OPERATION2(sum, p);
	ap += sizeof(TYPE);
	bp += sizeof(TYPE);
    }
    return sum;
}

#undef PROC
#undef TYPE
#undef TYPE2
#undef OPERATION
#undef OPERATION2
#undef VOPERATION
#undef VOPERATION2
#undef VELEMENT


