#include <asmjit/x86.h>
#include <iostream>

using namespace asmjit;

#include "matrix_types.h"
#include "matrix_kernel.h"

static void emit_vadd(x86::Assembler &a, uint8_t type, x86::Vec dst, x86::Vec src1, x86::Vec src2);

x86::Vec vreg(int i)
{
    switch(i) {
    case 0: return x86::regs::xmm0;
    case 1: return x86::regs::xmm1;
    case 2: return x86::regs::xmm2;
    case 3: return x86::regs::xmm3;
    case 4: return x86::regs::xmm4;
    case 5: return x86::regs::xmm5;
    case 6: return x86::regs::xmm6;
    case 7: return x86::regs::xmm7;
    case 8: return x86::regs::xmm8;
    case 9: return x86::regs::xmm9;
    case 10: return x86::regs::xmm10;
    case 11: return x86::regs::xmm11;
    case 12: return x86::regs::xmm12;
    case 13: return x86::regs::xmm13;
    case 14: return x86::regs::xmm14;
    case 15: return x86::regs::xmm15;
    default: return x86::regs::xmm0;
    }
}

x86::Gp reg(int i)
{
    switch(i) {
    case 0: return x86::regs::r8;
    case 1: return x86::regs::r9;
    case 2: return x86::regs::r10;
    case 3: return x86::regs::r11;
    case 4: return x86::regs::r12;
    case 5: return x86::regs::r13;
    case 6: return x86::regs::r14;
    case 7: return x86::regs::r15;
	
    case 8: return x86::regs::rax;
    case 9: return x86::regs::rbx;
    case 10: return x86::regs::rcx;
    case 11: return x86::regs::rdx;
    case 12: return x86::regs::rbp;
    case 13: return x86::regs::rsi;
    case 14: return x86::regs::rdi;
    case 15: return x86::regs::rsp;
    default: return x86::regs::rax;
    }
}

/*
x86::Gp reg(int i)
{
    switch(i) {
    case 0: return x86::regs::rax;
    case 1: return x86::regs::rbx;
    case 2: return x86::regs::rcx;
    case 3: return x86::regs::rdx;
    case 4: return x86::regs::rbp;
    case 5: return x86::regs::rsi;
    case 6: return x86::regs::rdi;
    case 7: return x86::regs::rsp;
    case 8: return x86::regs::r8;
    case 9: return x86::regs::r9;
    case 10: return x86::regs::r10;
    case 11: return x86::regs::r11;
    case 12: return x86::regs::r12;
    case 13: return x86::regs::r13;
    case 14: return x86::regs::r14;
    case 15: return x86::regs::r15;
    default: return x86::regs::rax;
    }
}
*/


// set dst = 0
static void emit_vzero(x86::Assembler &a, x86::Vec dst)
{
    a.pxor(dst.xmm(), dst.xmm());
}

static void emit_vone(x86::Assembler &a, x86::Vec dst)
{
    a.pcmpeqb(dst.xmm(), dst.xmm());
}

static void emit_vmov(x86::Assembler &a, uint8_t type,
		      x86::Vec dst, x86::Vec src)
{
    if (IS_FLOAT_TYPE(type))
	a.movaps(dst.xmm(), src.xmm());  // dst = src1;
    else
	a.movdqa(dst.xmm(), src.xmm());  // dst = src1;    
}

// dst = src
static void emit_vmovr(x86::Assembler &a, uint8_t type,
		       x86::Vec dst, x86::Vec src)
{
    if (src == dst) { // dst = dst;
	return;
    }
    else {
	emit_vmov(a, type, dst, src); // dst = src
    }
}


// x86::Gp 32-bit:
// EAX:32=(_:16,AH:8,AL:8)
// EBX:32=(_:16,BH:8,BL:8)
// ECX:32=(_:16,CH:8,CL:8)
// EDX:32=(_:16,DH:8,DL:8)
// EBP = (_:16,BP:16)
// ESI = (_:16,SI:16)
// EDI = (_:16,DI:16)
// ESP = (__16,SP:16)

// x86::Gp 64-bit:
// RAX:64 = (_:32, EAX:32=(_:16,AX:16=(AH:8,AL:8)))
// RBX:64 = EBX:32 =(_:16,BH:8,BL:8)
// RCX:64 = ECX:32 =(_:16,CH:8,CL:8)
// RDX:64 = EDX:32 =(_:16,DH:8,DL:8)
// EBP:64 = EBP:32 =(_:16,BP:16)
// ESI:64 = ESI:32 =(_:16,SI:16)
// EDI:64 = EDI:32 =(_:16,DI:16)
// RSP:64 = ESP:32 =(_:16,SP:16)
// R8D:64 = (_:32, R8:32 = (_:16,R8W:16=(_:8,R8B:8)))
// ..
// R15D:64=(_:32, R15:32 = (_:16,R15W:16=(_:8,R15B:8)))

// dst = -src  = 0 - src
static void emit_neg(x86::Assembler &a, uint8_t type, x86::Gp dst, x86::Gp src)
{
    if (src != dst)
	a.mov(dst, src);
    switch(type) {
    case UINT8:	
    case INT8:       a.neg(dst.r8()); break;
    case UINT16:	
    case INT16:      a.neg(dst.r16()); break;
    case UINT32:	
    case INT32:      a.neg(dst.r32()); break;
    case UINT64:	
    case INT64:      a.neg(dst.r64()); break;
    default: break;
    }
}

// dst = ~src 
static void emit_bnot(x86::Assembler &a, uint8_t type, x86::Gp dst, x86::Gp src)
{
    if (src != dst)
	a.mov(dst, src);
    switch(type) {
    case UINT8:	
    case INT8:       a.not_(dst.r8()); break;
    case UINT16:	
    case INT16:      a.not_(dst.r16()); break;
    case UINT32:	
    case INT32:      a.not_(dst.r32()); break;
    case UINT64:	
    case INT64:      a.not_(dst.r64()); break;
    default: break;
    }
}


static void emit_add(x86::Assembler &a, uint8_t type, x86::Gp dst, x86::Gp src1, x86::Gp src2)
{
    x86::Gp src;
    if ((dst == src1) && (dst == src2)) { // dst = dst + dst : dst += dst
	src = dst;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	switch(type) {
	case UINT8:
	case INT8:    a.mov(dst.r8(), src1.r8()); break;
	case UINT16:
	case INT16:   a.mov(dst.r16(), src1.r16()); break;
	case UINT32:
	case INT32:   a.mov(dst.r32(), src1.r32()); break;
	case UINT64:
	case INT64:   a.mov(dst.r64(), src1.r64()); break;
	default: break;
	}
	src = src2;
    }
    switch(type) {
    case UINT8:	
    case INT8:       a.add(dst.r8(), src.r8()); break;
    case UINT16:	
    case INT16:      a.add(dst.r16(), src.r16()); break;
    case UINT32:	
    case INT32:      a.add(dst.r32(), src.r32()); break;
    case UINT64:	
    case INT64:      a.add(dst.r64(), src.r64()); break;
    default: break;
    }
}


// dst = -src  = 0 - src
static void emit_vneg(x86::Assembler &a, uint8_t type, x86::Vec dst, x86::Vec src)
{
    if (src == dst) { // dst = -dst;
	src = x86::regs::xmm15;
	emit_vmov(a, type, src, dst); // copy dst to xmm15
    }
    emit_vzero(a, dst);    
    switch(type) {  // dst = src - dst
    case INT8:
    case UINT8:   a.psubb(dst.xmm(), src.xmm()); break;
    case INT16:	    
    case UINT16:  a.psubw(dst.xmm(), src.xmm()); break;
    case INT32:
    case UINT32:  a.psubd(dst.xmm(), src.xmm()); break;
    case INT64:
    case UINT64:  a.psubq(dst.xmm(), src.xmm()); break;
    case FLOAT32: a.subps(dst.xmm(), src.xmm()); break;
    case FLOAT64: a.subpd(dst.xmm(), src.xmm()); break;
    default: break;
    }    
}

static void emit_vbnot(x86::Assembler &a, uint8_t type, x86::Vec dst, x86::Vec src)
{
    (void) type;
    if (src == dst) {
	emit_vone(a, x86::regs::xmm15);  // pcmpeqb xmm15,xmm15
	if (IS_FLOAT_TYPE(type))
	    a.andnps(dst.xmm(), x86::regs::xmm15);
	else
	    a.pandn(dst.xmm(), x86::regs::xmm15);
    }
    else {
	emit_vmov(a, type, dst, src);  // dst = src
	emit_vone(a, src.xmm());      // src = -1 (mask)
	if (IS_FLOAT_TYPE(type))
	    a.andnps(dst.xmm(), src.xmm());  // dst = ~dst & src
	else
	    a.pandn(dst.xmm(), src.xmm());
    }
}

// PADDW dst,src  : dst = dst + src
// dst = src1 + src2  |  dst=src1; dst += src2;
// dst = dst + src2   |  dst += src2;
// dst = src1 + dst   |  dst += src1;
// dst = dst + dst    |  dst += dst;   == dst = 2*dst == dst = (dst<<1)

static void emit_vadd(x86::Assembler &a, uint8_t type,
		      x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    x86::Vec src;
    if ((dst == src1) && (dst == src2)) { // dst = dst + dst : dst += dst
	src = dst;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_vmov(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.paddb(dst.xmm(), src.xmm()); break;
    case INT16:	    
    case UINT16:  a.paddw(dst.xmm(), src.xmm()); break;
    case INT32:	    
    case UINT32:  a.paddd(dst.xmm(), src.xmm()); break;
    case INT64:	    
    case UINT64:  a.paddq(dst.xmm(), src.xmm()); break;
    case FLOAT32: a.addps(dst.xmm(), src.xmm()); break;
    case FLOAT64: a.addpd(dst.xmm(), src.xmm()); break;
    default: break;
    }
}

// SUB r0, r1, r2   (r2 = r0 - r1 )
// dst = src1 - src2 
// PADDW dst,src  : dst = src - dst ???
static void emit_vsub(x86::Assembler &a, uint8_t type,
		      x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    x86::Vec src;
    if ((dst == src1) &&
	(dst == src2)) { // dst = dst - dst : dst = 0
	emit_vzero(a, dst.xmm());
	return;
    }
    else if (src1 == dst) {   // dst = dst - src2 : dst -= src2
	src = src2;
    }
    else if (src2 == dst) { // dst = src - dst; dst = src1 + (0 - dst)
	emit_vneg(a, type, dst.xmm(), dst.xmm());
	emit_vadd(a, type, dst.xmm(), src1.xmm(), dst.xmm());
	return;
    }
    else {
	a.movaps(dst.xmm(), src1.xmm());  // dst = src1;
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.psubb(dst.xmm(), src.xmm()); break;
    case INT16:	    
    case UINT16:  a.psubw(dst.xmm(), src.xmm()); break;
    case INT32:
    case UINT32:  a.psubd(dst.xmm(), src.xmm()); break;
    case INT64:	    
    case UINT64:  a.psubq(dst.xmm(), src.xmm()); break;
    case FLOAT32: a.subps(dst.xmm(), src.xmm()); break;
    case FLOAT64: a.subpd(dst.xmm(), src.xmm()); break;
    default: break;
    }
}

// dst = src1*src2
static void emit_vmul(x86::Assembler &a, uint8_t type,
		      x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    x86::Vec src;
    
    if ((dst == src1) && (dst == src2)) // dst = dst * dst : dst *= dst
	src = dst;
    else if (src1 == dst)           // dst = dst * src2 : dst *= src2
	src = src2;
    else if (src2 == dst)            // dst = src1 * dst : dst *= src1
	src = src1;
    else {
	emit_vmov(a, type, dst, src1);
	src = src2;
    }
    
    switch(type) {
    case INT8:
    case UINT8: { // temp1 = xmm15
	a.movdqa(x86::xmm15, dst.xmm());
	a.pmullw(x86::xmm15, src.xmm());
	a.psllw(x86::xmm15, 8);
	a.psrlw(x86::xmm15, 8);           
    
	a.psrlw(dst.xmm(), 8);
	a.psrlw(src.xmm(), 8);
	a.pmullw(dst.xmm(), src.xmm());
	a.psllw(dst.xmm(), 8);
	a.por(dst.xmm(), x86::xmm15);
	break;
    }
    case INT16:
    case UINT16: a.pmullw(dst.xmm(), src.xmm()); break;
    case INT32:	    
    case UINT32: a.pmulld(dst.xmm(), src.xmm()); break;
	
    case UINT64:
    case INT64: { // temp1 = xmm14, temp2 = xmm15
	a.movdqa(x86::xmm14,  src.xmm());
	a.pmuludq(x86::xmm14, dst.xmm()); // xmm14=AC = BA*DC
	a.movdqa(x86::xmm15,  src.xmm()); // B = BA
	a.psrlq(x86::xmm15, 32);    // B = BA >> 32
	a.pmuludq(x86::xmm15, dst.xmm()); // BC = xmm15 = B * (DC & 0xFFFFFFFF)
	a.psrlq(dst.xmm(), 32);           // D = DC >> 32
	a.pmuludq(dst.xmm(), src.xmm());        // DA = (BA & 0xFFFFFFFF) * D;
	a.paddq(dst.xmm(), x86::xmm15);   // H = BC + DA
	a.psllq(dst.xmm(), 32);           // H <<= 32
	a.paddq(dst.xmm(), x86::xmm14);   // H + AC
	break;
    }
    case FLOAT32: a.mulps(dst.xmm(), src.xmm()); break;	    
    case FLOAT64: a.mulpd(dst.xmm(), src.xmm()); break;
    default: break;
    }
}

static void emit_vbor(x86::Assembler &a, uint8_t type,
		      x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    (void) type;
    x86::Vec src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_vmov(a, type, dst, src1);  // dst = src1;
	src = src2;
    }
    // FIXME: add a.orps / a.orpd
    a.por(dst.xmm(), src.xmm());
}

static void emit_vband(x86::Assembler &a, uint8_t type,
		       x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    (void) type;
    x86::Vec src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_vmov(a, type, dst, src1); // dst = src1;
	src = src2;
    }
    // FIXME: add a.andps / a.andpd    
    a.pand(dst.xmm(), src.xmm());    
}

static void emit_vbxor(x86::Assembler &a, uint8_t type,
		       x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    (void) type;    
    x86::Vec src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_vmov(a, type, dst, src1);  // dst = src1;
	src = src2;
    }
    // FIXME: add a.xorps / a.xorpd
    a.pxor(dst.xmm(), src.xmm());    
}

#define CMP_EQ 0
#define CMP_LT 1
#define CMP_LE 2
#define CMP_UNORD 3
#define CMP_NEQ 4
#define CMP_NLT 5
#define CMP_GE  5
#define CMP_NLE 6
#define CMP_GT  6
#define CMP_ORD 7

static void emit_vcmpeq(x86::Assembler &a, uint8_t type,
			x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    x86::Vec src;
    int cmp = CMP_EQ;  // EQ
    
    if ((dst == src1) && (dst == src2)) {  // dst = dst == dst
	emit_vone(a, dst.xmm());
	return;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_vmov(a, type, dst, src1); // dst = src1;
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.pcmpeqb(dst.xmm(), src.xmm()); break;
    case INT16:	    
    case UINT16:  a.pcmpeqw(dst.xmm(), src.xmm()); break;
    case INT32:	    
    case UINT32:  a.pcmpeqd(dst.xmm(), src.xmm()); break;
    case INT64:	    
    case UINT64:  a.pcmpeqq(dst.xmm(), src.xmm()); break;
    case FLOAT32: a.cmpps(dst.xmm(), src.xmm(), cmp); break;
    case FLOAT64: a.cmppd(dst.xmm(), src.xmm(), cmp); break;
    default: break;
    }
}

// emit dst = src1 > src2
static void emit_vcmpgt(x86::Assembler &a, uint8_t type,
			x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    x86::Vec src;
    int cmp = CMP_GT;

    if ((dst == src1) && (dst == src2)) { // dst = dst > dst
	emit_vzero(a, dst.xmm());
	return;
    }
    else if (src1 == dst) { // dst = dst > src2
	src = src2;
    }
    else if (src2 == dst) { // dst = src1 > dst => dst = dst < src1
	cmp = CMP_LT;
	src = src1;
    }
    else { // dst = src1 > src2 : dst=src1,  dst = dst > src2
	emit_vmov(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:
	a.pcmpgtb(dst.xmm(), src.xmm());
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);
	break;
    case INT16:
    case UINT16:
	a.pcmpgtw(dst.xmm(), src.xmm());
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);	
	break;
    case INT32:
    case UINT32:
	a.pcmpgtd(dst.xmm(), src.xmm());
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);	
	break;
    case INT64:	    
    case UINT64:
	a.pcmpgtq(dst.xmm(), src.xmm());
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);		
	break;
    case FLOAT32: a.cmpps(dst.xmm(), src.xmm(), cmp); break;
    case FLOAT64: a.cmppd(dst.xmm(), src.xmm(), cmp); break;
    default: break;
    }
}


// emit dst = src1 >= src2
static void emit_vcmpge(x86::Assembler &a, uint8_t type,
			x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    x86::Vec src;

    if (IS_FLOAT_TYPE(type)) {
	int cmp = CMP_GE;
	if ((dst == src1) && (dst == src2)) { // dst = dst >= dst (TRUE!)
	    emit_vone(a, dst);
	    return;
	}
	else if (src1 == dst)   // dst = dst >= src2
	    src = src2;
	else if (src2 == dst) { // dst = src1 >= dst => dst = dst <= src1
	    cmp = CMP_LE;
	    src = src1;
	}
	else {
	    emit_vmov(a, type, dst, src1); // dst = (src1 >= src2)
	    src = src2;
	}
	if (type == FLOAT32)
	    a.cmpps(dst.xmm(), src.xmm(), cmp);
	else 
	    a.cmppd(dst.xmm(), src.xmm(), cmp);
    }
    else {
	if ((dst == src1) && (dst == src2)) { // dst = dst >= dst (TRUE)
	    emit_vone(a, dst);
	    return;
	}
	else if (src1 == dst)   // dst = dst >= src2
	    src = src2;
	else if (src2 == dst)   // dst = src1 >= dst
	    src = src1;
	else {
	    emit_vmov(a, type, dst, src2); // dst = (src1 >= src2)
	    src = src1;
	}
	switch(type) {
	case INT8:
	case UINT8:   a.pcmpgtb(dst.xmm(), src.xmm()); break;
	case INT16:
	case UINT16:  a.pcmpgtw(dst.xmm(), src.xmm()); break;
	case INT32:	    
	case UINT32:  a.pcmpgtd(dst.xmm(), src.xmm()); break;
	case INT64:	    
	case UINT64:  a.pcmpgtq(dst.xmm(), src.xmm()); break;
	default: break;
	}
	emit_vbnot(a, type, dst, dst);
    }
}

static void emit_vcmplt(x86::Assembler &a, uint8_t type,
			x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    emit_vcmpgt(a, type, dst, src2, src1);
}

static void emit_vcmple(x86::Assembler &a, uint8_t type,
			x86::Vec dst, x86::Vec src1, x86::Vec src2)
{
    emit_vcmpge(a, type, dst, src2, src1);
}

// Helper function to generate instructions based on type and operation
void emit_instruction(x86::Assembler &a, instr_t* optr, x86::Gp ret)
{
    switch(optr->op) {
    case OP_RET:
	a.mov(x86::ptr(ret), reg(optr->ri));
	break;
    case OP_NEG:   emit_neg(a, optr->type, reg(optr->rd),reg(optr->ri)); break;
    case OP_BNOT:  emit_bnot(a, optr->type, reg(optr->rd), reg(optr->ri)); break;
    case OP_ADD:   emit_add(a, optr->type, reg(optr->rd), reg(optr->ri), reg(optr->rj)); break;
    case OP_VRET:
	a.movdqu(x86::ptr(ret), vreg(optr->ri).xmm());
	break;
    case OP_MOVR:   emit_vmovr(a, optr->type, vreg(optr->rd), vreg(optr->ri)); break;
    case OP_VNEG:   emit_vneg(a, optr->type, vreg(optr->rd), vreg(optr->ri)); break;
    case OP_VADD:   emit_vadd(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VSUB:   emit_vsub(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VMUL:   emit_vmul(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VBNOT:  emit_vbnot(a, optr->type, vreg(optr->rd), vreg(optr->ri)); break;	
    case OP_VBAND:  emit_vband(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VBOR:   emit_vbor(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VBXOR:  emit_vbxor(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VCMPEQ: emit_vcmpeq(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VCMPLT: emit_vcmplt(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;
    case OP_VCMPLE: emit_vcmple(a, optr->type, vreg(optr->rd), vreg(optr->ri), vreg(optr->rj)); break;	
    default: break;
    }
}

// add all dirty register 
void add_dirty_regs(FuncFrame &frame, instr_t* code, size_t n)
{
    while (n--) {
	if (code->op & OP_VEC) {
	    if (code->op == OP_VRET)
		frame.addDirtyRegs(vreg(code->ri));
	    else if (code->op == OP_VMOVI)
		frame.addDirtyRegs(vreg(code->rd));
	    else if (code->op & OP_BIN) { // ri,rj,rd
		frame.addDirtyRegs(vreg(code->ri));
		frame.addDirtyRegs(vreg(code->rj));
		frame.addDirtyRegs(vreg(code->rd));
	    }
	    else { // ri,rd
		frame.addDirtyRegs(vreg(code->ri));
		frame.addDirtyRegs(vreg(code->rd));
	    }
	}
	else {
	    if (code->op == OP_RET)
		frame.addDirtyRegs(reg(code->ri));
	    else if (code->op == OP_MOVI)
		frame.addDirtyRegs(reg(code->rd));
	    else if (code->op & OP_BIN) { // ri,rj,rd
		frame.addDirtyRegs(reg(code->ri));
		frame.addDirtyRegs(reg(code->rj));
		frame.addDirtyRegs(reg(code->rd));
	    }
	    else { // ri,rd
		frame.addDirtyRegs(reg(code->ri));
		frame.addDirtyRegs(reg(code->rd));
	    }
	}
	code++;
    }
}

void assemble(x86::Assembler &a, const Environment &env,
	      x86::Reg dst, x86::Reg src1, x86::Reg src2,
	      instr_t* code, size_t n)
{
    FuncDetail func;
    FuncFrame frame;

    // calling with pointer arguments *dst  *src1, *src2
    x86::Gp p_dst  = a.zax();
    x86::Gp p_src1 = a.zcx();
    x86::Gp p_src2 = a.zdx();
    
    func.init(FuncSignatureT<void, void*, const void*, const void*>(CallConvId::kHost), env);
    frame.init(func);

    add_dirty_regs(frame, code, n);

    FuncArgsAssignment args(&func);   // Create arguments assignment context.
    args.assignAll(p_dst, p_src1, p_src2);// Assign our registers to arguments.
    args.updateFuncFrame(frame);      // Reflect our args in FuncFrame.    
    frame.finalize();                 // Finalize the FuncFrame (updates it).

    //a.emitProlog(frame);              // Emit function prolog.
    a.emitArgsAssignment(frame, args);// Assign arguments to registers.
    // TESTING

    if (dst.isXmm()) {
	x86::Xmm* xdst = (x86::Xmm*) &dst;
	a.movdqu(xdst->xmm(), x86::ptr(p_dst));  // vector from [p_dst] to XMM2.
    }
    else {
	x86::Gp* gdst = (x86::Gp*) &dst;
	a.mov(*gdst, x86::ptr(p_dst));
    }

    // LOAD 2 arguments
    if (src1.isXmm()) {
	x86::Xmm* xsrc1 = (x86::Xmm*) &src1;
	a.movdqu(xsrc1->xmm(), x86::ptr(p_src1));  // vector from [p_src1] to XMM0.
    }
    else {
	x86::Gp* gsrc1 = (x86::Gp*) &src1;
	a.mov(*gsrc1, x86::ptr(p_src1));
    }
    
    if (src2.isXmm()) {
	x86::Xmm* xsrc2 = (x86::Xmm*) &src2;
	a.movdqu(xsrc2->xmm(), x86::ptr(p_src2));  // vector from [p_src2] to XMM1.
    }
    else {
	x86::Gp* gsrc2 = (x86::Gp*) &src2;
	a.mov(*gsrc2, x86::ptr(p_src2));
    }
    
    // assemble all code
    while (n--) {
	emit_instruction(a, code, p_dst);
	code++;
    }
    a.emitEpilog(frame);              // Emit function epilog and return.
}
