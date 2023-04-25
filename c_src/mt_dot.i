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

static TYPE2 PROC(byte_t* ap,int av,byte_t* bp,int bu,size_t n)
{
    TYPE2 sum = 0;

    while(n--) {
	TYPE a = *((TYPE*) ap);
	TYPE b = *((TYPE*) bp);
	TYPE2 p = OPERATION(a,b);
	sum = OPERATION2(sum, p);
	ap += av;
	bp += bu;
    }
    return sum;
}

#undef PROC
#undef TYPE
#undef TYPE2
#undef PARAMS_DECL
#undef LOCALS_DECL
#undef OPERATION
#undef OPERATION2

