#include <asmjit/x86.h>
#include <iostream>
#include <assert.h>

using namespace asmjit;

#include "matrix_types.h"
#include "matrix_kernel.h"
#include "matrix_kernel_asm.h"

#define CMP_EQ    0
#define CMP_LT    1
#define CMP_LE    2
#define CMP_UNORD 3
#define CMP_NEQ   4
#define CMP_NLT   5
#define CMP_GE    5
#define CMP_NLE   6
#define CMP_GT    6
#define CMP_ORD   7

#define TMPREG   3   // currently r11 
#define TMPVREG0 14
#define TMPVREG1 15

void crash(const char* filename, int line, int code)
{
    fprintf(stderr, "%s:%d: CRASH code=%d\n", filename, line, code);
    assert(0);
}

x86::Vec xreg(int i)
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
    default: crash(__FILE__, __LINE__, i); return x86::regs::xmm0;
    }
}

x86::Vec yreg(int i)
{
    switch(i) {
    case 0: return x86::regs::ymm0;
    case 1: return x86::regs::ymm1;
    case 2: return x86::regs::ymm2;
    case 3: return x86::regs::ymm3;
    case 4: return x86::regs::ymm4;
    case 5: return x86::regs::ymm5;
    case 6: return x86::regs::ymm6;
    case 7: return x86::regs::ymm7;
    case 8: return x86::regs::ymm8;
    case 9: return x86::regs::ymm9;
    case 10: return x86::regs::ymm10;
    case 11: return x86::regs::ymm11;
    case 12: return x86::regs::ymm12;
    case 13: return x86::regs::ymm13;
    case 14: return x86::regs::ymm14;
    case 15: return x86::regs::ymm15;
    default: crash(__FILE__, __LINE__, i); return x86::regs::ymm0;
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
// RBP:64 = EBP:32 =(_:16,BP:16)
// RSI:64 = ESI:32 =(_:16,SI:16)
// RDI:64 = EDI:32 =(_:16,DI:16)
// RSP:64 = ESP:32 =(_:16,SP:16)
// R8:64 = (_:32, R8d:32 = (_:16,R8w:16=(_:8,R8b:8)))
// ..
// R15:64=(_:32, R15d:32 = (_:16,R15w:16=(_:8,R15b:8)))

x86::Gp reg(int i)
{
    switch(i) {
    case 0: return x86::regs::r8;   // CALLER
    case 1: return x86::regs::r9;   // CALLER
    case 2: return x86::regs::r10;  // CALLER
    case 3: return x86::regs::r11;  // CALLER
    case 4: return x86::regs::r12;  // +
    case 5: return x86::regs::r13;  // +
    case 6: return x86::regs::r14;  // +
    case 7: return x86::regs::r15;  // +
    case 8: return x86::regs::rax;  // CALLER ( return value )
    case 9: return x86::regs::rbx;  // +
    case 10: return x86::regs::rcx; // CALLER
    case 11: return x86::regs::rdx; // CALLER
    case 12: return x86::regs::rbp; // +      ( base pointer )
    case 13: return x86::regs::rsi; // CALLER
    case 14: return x86::regs::rdi; // CALLER
    case 15: return x86::regs::rsp; // CALLER ( stack pointer )
    default: crash(__FILE__, __LINE__, i); return x86::regs::rax;
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
    // caller save registers
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

#ifdef unused
static void emit_inc(ZAssembler &a, uint8_t type, int dst)
{
    switch(type) {
    case UINT8:
    case INT8:    a.inc(reg(dst).r8()); break;
    case UINT16:
    case INT16:   a.inc(reg(dst).r16()); break;
    case UINT32:
    case INT32:   a.inc(reg(dst).r32()); break;
    case UINT64:
    case INT64:   a.inc(reg(dst).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;	
    }    
}
#endif

static void emit_dec(ZAssembler &a, uint8_t type, int dst)
{
    switch(type) {
    case UINT8:
    case INT8:    a.dec(reg(dst).r8()); break;
    case UINT16:
    case INT16:   a.dec(reg(dst).r16()); break;
    case UINT32:
    case INT32:   a.dec(reg(dst).r32()); break;
    case UINT64:
    case INT64:   a.dec(reg(dst).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_movi(ZAssembler &a, uint8_t type, int dst, int16_t imm)
{
    switch(type) {
    case UINT8:
    case INT8:    a.mov(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.mov(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.mov(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.rex().mov(reg(dst).r64(), imm); break;
    case FLOAT32:
	a.add_dirty_reg(reg(TMPREG));	
	a.mov(reg(TMPREG).r32(), imm);
	a.cvtsi2ss(xreg(dst).xmm(), reg(TMPREG).r32());
	break;
    case FLOAT64:
	a.add_dirty_reg(reg(TMPREG));	
	a.mov(reg(TMPREG).r32(), imm);
	a.cvtsi2sd(xreg(dst).xmm(), reg(TMPREG).r32());
	break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}


static void emit_slli(ZAssembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));
    switch(type) {
    case UINT8:
    case INT8:    a.shl(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.shl(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.shl(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.shl(reg(dst).r64(), imm); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_srli(ZAssembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));    
    switch(type) {
    case UINT8:
    case INT8:    a.shr(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.shr(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.shr(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.shr(reg(dst).r64(), imm); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}



static void emit_zero(ZAssembler &a, uint8_t type, int dst)
{
    switch(type) {
    case UINT8:
    case INT8:    a.xor_(reg(dst).r8(), reg(dst).r8()); break;
    case UINT16:
    case INT16:   a.xor_(reg(dst).r16(), reg(dst).r16()); break;
    case UINT32:
    case INT32:   a.xor_(reg(dst).r32(), reg(dst).r32()); break;
    case UINT64:
    case INT64:   a.xor_(reg(dst).r64(), reg(dst).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

#ifdef unused
static void emit_one(ZAssembler &a, uint8_t type, int dst)
{
    emit_movi(a, type, dst, 1);
}
#endif

static void emit_neg_dst(ZAssembler &a, uint8_t type, int dst)
{
    switch(type) {
    case UINT8:	
    case INT8:       a.neg(reg(dst).r8()); break;
    case UINT16:	
    case INT16:      a.neg(reg(dst).r16()); break;
    case UINT32:	
    case INT32:      a.neg(reg(dst).r32()); break;
    case UINT64:	
    case INT64:      a.neg(reg(dst).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_movr(ZAssembler &a, uint8_t type, int dst, int src)
{
    switch(type) {
    case UINT8:
    case INT8:    a.mov(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:
    case INT16:   a.mov(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:
    case INT32:   a.mov(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:
    case INT64:   a.mov(reg(dst).r64(), reg(src).r64()); break;
    case FLOAT32: a.movss(xreg(dst).xmm(),xreg(src).xmm()); break;
    case FLOAT64: a.movsd(xreg(dst).xmm(),xreg(src).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_src(ZAssembler &a, int dst, int src)
{
    if (src != dst) {
	a.mov(reg(dst), reg(src));
    }
}

static void emit_vmov(ZAssembler &a, uint8_t type,
		      int dst, int src)
{
    a.add_dirty_reg(xreg(dst).xmm());
    if (IS_FLOAT_TYPE(type))
	a.movaps(xreg(dst).xmm(), xreg(src).xmm());  // dst = src1;
    else
	a.movdqa(xreg(dst).xmm(), xreg(src).xmm());  // dst = src1;    
}

// dst = src
static void emit_vmovr(ZAssembler &a, uint8_t type, int dst, int src)
{
    if (src != dst) {
	emit_vmov(a, type, dst, src);
    }
}

// reduce two sources into one! operation does not depend on order!
static int emit_one_src(ZAssembler &a, uint8_t type, int dst,
			int src1, int src2)
{
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	return dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	return src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	return src1;
    else {
	emit_movr(a, type, dst, src1); // dst = src1;
	return src2;
    }
}

static int emit_one_vsrc(ZAssembler &a, uint8_t type, int dst,
			 int src1, int src2)
{
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	return dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	return src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	return src1;
    else {
	emit_vmovr(a, type, dst, src1); // dst = src1;
	return src2;
    }
}

// ordered src  (use tmpvreg1 if needed)
static int emit_one_ord_vsrc(ZAssembler &a, uint8_t type, int dst,
			     int src1, int src2)
{
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	return dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	return src2;
    else if (src2 == dst) { // dst = src1 + dst
	emit_vmov(a, type, TMPVREG1, dst); // TMP=dst
	emit_vmov(a, type, dst, src1);     // dst=src1
	return TMPVREG1;
    }
    else {
	emit_vmovr(a, type, dst, src1); // dst = src1;
	return src2;
    }
}


// move src to dst if condition code (matching cmp) is set
// else set dst=0
static void emit_movecc(ZAssembler &a, int cmp, uint8_t type, int dst)
{
    Label Skip = a.newLabel();
    emit_movi(a, type, dst, 0);
    switch(cmp) {
    case CMP_EQ:
	a.jne(Skip); break;
    case CMP_NEQ:
	a.je(Skip); break;
    case CMP_LT:
	if (get_base_type(type) == UINT)
	    a.jae(Skip);
	else
	    a.jge(Skip);
	break;
    case CMP_LE:
	if (get_base_type(type) == UINT)
	    a.ja(Skip);
	else
	    a.jg(Skip);
	break;
    case CMP_GT:
	if (get_base_type(type) == UINT)
	    a.jbe(Skip);
	else
	    a.jle(Skip);
	break;
    case CMP_GE:
	if (get_base_type(type) == UINT)
	    a.jb(Skip);
	else
	    a.jl(Skip);
	break;
    default: crash(__FILE__, __LINE__, type); break;
    }
    emit_dec(a, type, dst);
    a.bind(Skip);
}

// above without jump
#ifdef not_used
static void emit_movecc_dst(ZAssembler &a, int cmp, uint8_t type, int dst)
{
    if ((type != UINT8) && (type != INT8))
	emit_movi(a, type, dst, 0); // clear if dst is 16,32,64
    // set byte 0|1
    switch(cmp) {
    case CMP_EQ: a.seteq(reg(dst).r8()); break;	
    case CMP_NEQ: a.setne(reg(dst).r8()); break;
    case CMP_LT: a.setl(reg(dst).r8()); break;
    case CMP_LE: a.setle(reg(dst).r8()); break;
    case CMP_GT: a.setg(reg(dst).r8()); break;
    case CMP_GE: a.setge(reg(dst).r8()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
    // negate to set all bits (optimise before conditional jump)
    emit_neg_dst(a, type, dst);    
}
#endif

// set dst = 0
static void emit_vzero(ZAssembler &a, int dst)
{
    a.add_dirty_reg(xreg(dst).xmm());    
    a.pxor(xreg(dst).xmm(), xreg(dst).xmm());
}


static void emit_vone(ZAssembler &a, int dst)
{
    a.pcmpeqb(xreg(dst).xmm(), xreg(dst).xmm());
}


// dst = -src  = 0 - src
static void emit_neg(ZAssembler &a, uint8_t type, int dst, int src)
{
    emit_src(a, dst, src);
    emit_neg_dst(a, type, dst);
}

// dst = ~src 
static void emit_bnot(ZAssembler &a, uint8_t type, int dst, int src)
{
    emit_src(a, dst, src);
    switch(type) {
    case UINT8:	
    case INT8:       a.not_(reg(dst).r8()); break;
    case UINT16:	
    case INT16:      a.not_(reg(dst).r16()); break;
    case UINT32:	
    case INT32:      a.not_(reg(dst).r32()); break;
    case UINT64:	
    case INT64:      a.not_(reg(dst).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_add(ZAssembler &a, uint8_t type, int dst, int src1, int src2)
{
    int src = emit_one_src(a, type, dst, src1, src2);
    switch(type) {
    case UINT8:	
    case INT8:       a.add(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.add(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.add(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.add(reg(dst).r64(), reg(src).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_addi(ZAssembler &a, uint8_t type, int dst, int src,
		      int imm8)
{
    emit_src(a, dst, src);  // move src to dst (unless src == dst)
    switch(type) {
    case UINT8:
    case INT8:    a.add(reg(dst).r8(), imm8); break;
    case UINT16:
    case INT16:   a.add(reg(dst).r16(), imm8); break;
    case UINT32:
    case INT32:   a.add(reg(dst).r32(), imm8); break;
    case UINT64:
    case INT64:   a.add(reg(dst).r64(), imm8); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}


static void emit_sub(ZAssembler &a, uint8_t type,
		     int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) && (dst == src2)) { // dst = dst - dst : dst = 0
	emit_zero(a, type, dst);
	return;
    }
    else if (src1 == dst) {   // dst = dst - src2 : dst -= src2
	src = src2;
    }
    else if (src2 == dst) { // dst = src - dst; dst = src1 + (0 - dst)
	emit_neg(a, type, dst, dst);
	emit_add(a, type, dst, src1, dst);
	return;
    }
    else {
	emit_movr(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case UINT8:	
    case INT8:       a.sub(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.sub(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.sub(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.sub(reg(dst).r64(), reg(src).r64()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

// dst = src - imm 
static void emit_subi(ZAssembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    emit_src(a, dst, src);  // move src to dst (unless src == dst)    
    switch(type) {
    case UINT8:
    case INT8:    a.sub(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.sub(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.sub(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.sub(reg(dst).r64(), imm); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_rsub(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    emit_sub(a, type, dst, src2, src1);
}

// dst = imm - src
static void emit_rsubi(ZAssembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    emit_neg(a, type, dst, src);
    emit_addi(a, type, dst, dst, imm);
}



static void emit_mul(ZAssembler &a, uint8_t type, int dst, int src1, int src2)
{
    int src = emit_one_src(a, type, dst, src1, src2);
    switch(type) {
    case UINT8:	
    case INT8: // FIXME: make code that do not affect high byte!
	a.and_(reg(dst).r16(), 0x00ff);
	a.imul(reg(dst).r16(), reg(src).r16());
	break;
    case UINT16:	
    case INT16:      a.imul(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.imul(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.imul(reg(dst).r64(), reg(src).r64()); break;
    case FLOAT32: // fixme: how 
    case FLOAT64: // fixme: how
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_muli(ZAssembler &a, uint8_t type, int dst, int src, int imm8)
{
    emit_src(a, dst, src);  // move src to dst (unless src == dst)        
    switch(type) {
    case UINT8:	
    case INT8: // FIXME: make code that do not affect high byte!
	a.and_(reg(dst).r16(), 0x00ff);
	a.imul(reg(dst).r16(), imm8);
	break;
    case UINT16:	
    case INT16:      a.imul(reg(dst).r16(), imm8); break; // max imm16
    case UINT32:	
    case INT32:      a.imul(reg(dst).r32(), imm8); break; // max imm32
    case UINT64:	
    case INT64:      a.imul(reg(dst).r64(), imm8); break; // max imm32
    default: crash(__FILE__, __LINE__, type); break;		
    }
}


static void emit_sll(ZAssembler &a, uint8_t type, int dst, int src1, int src2)
{
    a.add_dirty_reg(x86::regs::cl);    
    a.mov(x86::regs::cl, reg(src2).r8()); // setup shift value
    if (src1 != dst)
	a.mov(reg(dst), reg(src1));
    switch(type) {
    case UINT8:
    case INT8:    a.shl(reg(dst).r8(), x86::regs::cl); break;
    case UINT16:
    case INT16:   a.shl(reg(dst).r16(), x86::regs::cl); break;
    case UINT32:
    case INT32:   a.shl(reg(dst).r32(), x86::regs::cl); break;
    case UINT64:
    case INT64:   a.shl(reg(dst).r64(), x86::regs::cl); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_srl(ZAssembler &a, uint8_t type, int dst, int src1, int src2)
{
    a.add_dirty_reg(x86::regs::cl);    
    a.mov(x86::regs::cl, reg(src2).r8()); // setup shift value
    if (src1 != dst)
	a.mov(reg(dst), reg(src1));
    switch(type) {
    case UINT8:
    case INT8:    a.shr(reg(dst).r8(), x86::regs::cl); break;
    case UINT16:
    case INT16:   a.shr(reg(dst).r16(), x86::regs::cl); break;
    case UINT32:
    case INT32:   a.shr(reg(dst).r32(), x86::regs::cl); break;
    case UINT64:
    case INT64:   a.shr(reg(dst).r64(), x86::regs::cl); break;
    default: crash(__FILE__, __LINE__, type); break;
    }        
}

static void emit_sra(ZAssembler &a, uint8_t type, int dst, int src1, int src2)
{
    a.add_dirty_reg(x86::regs::cl);
    a.mov(x86::regs::cl, reg(src2).r8()); // setup shift value
    if (src1 != dst)
	a.mov(reg(dst), reg(src1));
    switch(type) {
    case UINT8:
    case INT8:    a.sar(reg(dst).r8(), x86::regs::cl); break;
    case UINT16:
    case INT16:   a.sar(reg(dst).r16(), x86::regs::cl); break;
    case UINT32:
    case INT32:   a.sar(reg(dst).r32(), x86::regs::cl); break;
    case UINT64:
    case INT64:   a.sar(reg(dst).r64(), x86::regs::cl); break;
    default: crash(__FILE__, __LINE__, type); break;
    }        
}

static void emit_srai(ZAssembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    emit_src(a, dst, src);    
    switch(type) {
    case UINT8:
    case INT8:    a.sar(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.sar(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.sar(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.sar(reg(dst).r64(), imm); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_band(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src = emit_one_src(a, type, dst, src1, src2);
    switch(type) {
    case UINT8:	
    case INT8:       a.and_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.and_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.and_(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.and_(reg(dst).r64(), reg(src).r64()); break;
    case FLOAT32:    a.andps(xreg(dst).xmm(), xreg(src).xmm()); break;
    case FLOAT64:    a.andpd(xreg(dst).xmm(), xreg(src).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;	
    }    
}

// dst = src & imm => dst = imm; dst = src & src;
// dst = dst & imm
static void emit_bandi(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    if (dst != src)
	emit_movi(a, type, dst, imm);
    else {
	a.add_dirty_reg(reg(TMPREG));
	emit_movi(a, type, TMPREG, imm);
	src = TMPREG;
    }
    switch(type) {
    case UINT8:	
    case INT8:       a.and_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.and_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.and_(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.and_(reg(dst).r64(), reg(src).r64()); break;
    case FLOAT32:    a.andps(xreg(dst).xmm(), xreg(src).xmm()); break;
    case FLOAT64:    a.andpd(xreg(dst).xmm(), xreg(src).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;	
    }    
}

static void emit_bor(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{    
    int src = emit_one_src(a, type, dst, src1, src2);
    switch(type) {
    case UINT8:	
    case INT8:       a.or_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.or_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.or_(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.or_(reg(dst).r64(), reg(src).r64()); break;
    case FLOAT32:    a.orps(xreg(dst).xmm(), xreg(src).xmm()); break;
    case FLOAT64:    a.orpd(xreg(dst).xmm(), xreg(src).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;
    }        
}

// dst = src | imm => dst = imm; dst = dst | src;
// dst = dst | imm
static void emit_bori(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    if (dst != src)
	emit_movi(a, type, dst, imm);
    else {
	a.add_dirty_reg(reg(TMPREG));
	emit_movi(a, type, TMPREG, imm);
	src = TMPREG;
    }    
    switch(type) {
    case UINT8:	
    case INT8:       a.or_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.or_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.or_(reg(dst).r32(), reg(dst).r32()); break;
    case UINT64:	
    case INT64:      a.or_(reg(dst).r64(), reg(dst).r64()); break;
    case FLOAT32:    a.orps(xreg(dst).xmm(), xreg(src).xmm()); break;
    case FLOAT64:    a.orpd(xreg(dst).xmm(), xreg(src).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;	
    }    
}

static void emit_bxor(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    int src = emit_one_src(a, type, dst, src1, src2);    
    switch(type) {
    case UINT8:	
    case INT8:       a.xor_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.xor_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.xor_(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.xor_(reg(dst).r64(), reg(src).r64()); break;
    case FLOAT32:    a.xorps(xreg(dst).xmm(), xreg(src).xmm()); break;
    case FLOAT64:    a.xorpd(xreg(dst).xmm(), xreg(src).xmm()); break;	
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_bxori(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    if (dst != src)
	emit_movi(a, type, dst, imm);
    else {
	a.add_dirty_reg(reg(TMPREG));
	emit_movi(a, type, TMPREG, imm);
	src = TMPREG;
    }        
    switch(type) {
    case UINT8:	
    case INT8:       a.xor_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.xor_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.xor_(reg(dst).r32(), reg(dst).r32()); break;
    case UINT64:	
    case INT64:      a.xor_(reg(dst).r64(), reg(dst).r64()); break;
    case FLOAT32:    a.xorps(xreg(dst).xmm(), xreg(src).xmm()); break;
    case FLOAT64:    a.xorpd(xreg(dst).xmm(), xreg(src).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;	
    }    
}


static void emit_cmpi(ZAssembler &a, uint8_t type, int src1, int imm)
{
    switch(type) {
    case INT8:
    case UINT8:	a.cmp(reg(src1).r8(), imm); break;
    case INT16:
    case UINT16: a.cmp(reg(src1).r16(), imm); break;
    case INT32:	    
    case UINT32: a.cmp(reg(src1).r32(), imm); break;
    case INT64:	    
    case UINT64: a.cmp(reg(src1).r64(), imm); break;
    case FLOAT32:
	a.add_dirty_reg(reg(TMPREG));
	a.add_dirty_reg(xreg(TMPVREG1));	
	a.mov(reg(TMPREG).r32(), imm);
	a.cvtsi2ss(xreg(TMPVREG1).xmm(), reg(TMPREG).r32());
	a.comiss(xreg(src1).xmm(), xreg(TMPVREG1).xmm());
	break;
    case FLOAT64:
	a.add_dirty_reg(reg(TMPREG));
	a.add_dirty_reg(xreg(TMPVREG1));	
	a.mov(reg(TMPREG).r32(), imm);
	a.cvtsi2sd(xreg(TMPVREG1).xmm(),reg(TMPREG).r32());
	a.comisd(xreg(src1).xmm(), xreg(TMPVREG1).xmm());
	break;
    default: crash(__FILE__, __LINE__, type); break;
    }	
}

static void emit_cmp(ZAssembler &a, uint8_t type, int src1, int src2)
{
    switch(type) {
    case INT8:
    case UINT8:	a.cmp(reg(src1).r8(), reg(src2).r8()); break;
    case INT16:
    case UINT16: a.cmp(reg(src1).r16(), reg(src2).r16()); break;
    case INT32:
    case UINT32: a.cmp(reg(src1).r32(), reg(src2).r32()); break;
    case INT64:
    case UINT64: a.cmp(reg(src1).r64(), reg(src2).r64()); break;
    case FLOAT32: a.comiss(xreg(src1).xmm(), xreg(src2).xmm()); break;
    case FLOAT64: a.comisd(xreg(src1).xmm(), xreg(src2).xmm()); break;
    default: crash(__FILE__, __LINE__, type); break;	
    }	
}

static void emit_cmpeq(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmp(a, type, src1, src2);
    emit_movecc(a, CMP_EQ, type, dst);
}

static void emit_cmpeqi(ZAssembler &a, uint8_t type,
			int dst, int src1, int8_t imm)
{
    emit_cmpi(a, type, src1, imm);
    emit_movecc(a, CMP_EQ, type, dst);
}


// emit dst = src1 > src2
static void emit_cmpgt(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmp(a, type, src1, src2);
    emit_movecc(a, CMP_GT, type, dst);
}

static void emit_cmpgti(ZAssembler &a, uint8_t type,
		       int dst, int src1, int8_t imm)
{
    emit_cmpi(a, type, src1, imm);
    emit_movecc(a, CMP_GT, type, dst);
}


// emit dst = src1 >= src2
static void emit_cmpge(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmp(a, type, src1, src2);
    emit_movecc(a, CMP_GE, type, dst);
}

static void emit_cmpgei(ZAssembler &a, uint8_t type,
		       int dst, int src1, int8_t imm)
{
    emit_cmpi(a, type, src1, imm);
    emit_movecc(a, CMP_GE, type, dst);
}

static void emit_cmplt(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmp(a, type, src1, src2);
    emit_movecc(a, CMP_LT, type, dst);
}

static void emit_cmplti(ZAssembler &a, uint8_t type,
		       int dst, int src1, int8_t imm)
{
    emit_cmpi(a, type, src1, imm);
    emit_movecc(a, CMP_LT, type, dst);
}

static void emit_cmple(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmp(a, type, src1, src2);
    emit_movecc(a, CMP_LE, type, dst);
}

static void emit_cmplei(ZAssembler &a, uint8_t type,
		       int dst, int src1, int8_t imm)
{
    emit_cmpi(a, type, src1, imm);
    emit_movecc(a, CMP_LE, type, dst);
}


static void emit_cmpne(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmp(a, type, src1, src2);
    emit_movecc(a, CMP_NEQ, type, dst);
}

static void emit_cmpnei(ZAssembler &a, uint8_t type,
		       int dst, int src1, int8_t imm)
{
    emit_cmpi(a, type, src1, imm);
    emit_movecc(a, CMP_NEQ, type, dst);
}



#define DST xreg(dst).xmm()
#define SRC xreg(src).xmm()
#define SRC1 xreg(src1).xmm()
#define SRC2 xreg(src2).xmm()
#define T0  xreg(TMPVREG0).xmm()
#define T1  xreg(TMPVREG1).xmm()

static void emit_vneg_avx(ZAssembler &a, uint8_t type, int dst, int src)
{
    int zero = TMPVREG1;
    emit_vzero(a, zero);
    switch(type) {  // dst = -src; dst = 0 - src
    case INT8:
    case UINT8:   a.vpsubb(DST, xreg(zero).xmm(), SRC); break;
    case INT16:	    
    case UINT16:  a.vpsubw(DST, xreg(zero).xmm(), SRC); break;
    case INT32:
    case UINT32:  a.vpsubd(DST, xreg(zero).xmm(), SRC); break;
    case INT64:
    case UINT64:  a.vpsubq(DST, xreg(zero).xmm(), SRC); break;
    case FLOAT32: a.vsubps(DST, xreg(zero).xmm(), SRC); break;
    case FLOAT64: a.vsubpd(DST, xreg(zero).xmm(), SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_vneg_sse2(ZAssembler &a, uint8_t type, int dst, int src)
{
    if (src == dst) { // dst = -dst;
	src = TMPVREG1;
	emit_vmov(a, type, src, dst); // copy dst to xmm15
    }
    emit_vzero(a, dst);
    switch(type) {  // dst = src - dst
    case INT8:
    case UINT8:   a.psubb(DST, SRC); break;
    case INT16:	    
    case UINT16:  a.psubw(DST, SRC); break;
    case INT32:
    case UINT32:  a.psubd(DST, SRC); break;
    case INT64:
    case UINT64:  a.psubq(DST, SRC); break;
    case FLOAT32: a.subps(DST, SRC); break;
    case FLOAT64: a.subpd(DST, SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

// dst = -src  = 0 - src
static void emit_vneg(ZAssembler &a, uint8_t type, int dst, int src)
{
    if (a.use_avx())
	emit_vneg_avx(a, type, dst, src);
    else if (a.use_sse2())
	emit_vneg_sse2(a, type, dst, src);
    else
	emit_neg(a, type, dst, src);
}

// broadcast integer value imm12 into element uses TMPREG
static void emit_vmovi(ZAssembler &a, uint8_t type, int dst, int16_t imm12)
{
    switch(type) {
    case INT8:
    case UINT8:
	a.add_dirty_reg(reg(TMPREG));
	a.mov(reg(TMPREG).r32(), imm12);
	a.movd(DST, reg(TMPREG).r32());
	a.punpcklbw(DST,DST);
	a.punpcklwd(DST,DST);
	a.pshufd(DST,DST,0);
	break;
    case INT16:
    case UINT16:
	a.add_dirty_reg(reg(TMPREG));
	a.mov(reg(TMPREG).r32(), imm12);
	a.movd(DST, reg(TMPREG).r32());
	a.punpcklwd(DST,DST);
	a.pshufd(DST,DST,0);
	break;
    case INT32:
    case UINT32:
	a.add_dirty_reg(reg(TMPREG));	
	a.mov(reg(TMPREG).r32(), imm12);
	a.movd(DST, reg(TMPREG).r32());
	a.pshufd(DST,DST,0);
	break;
    case INT64:
    case UINT64:
	a.add_dirty_reg(reg(TMPREG));
	a.rex().mov(reg(TMPREG).r64(), imm12);
	a.movq(DST, reg(TMPREG).r64());
	a.punpcklqdq(DST, DST);
	break;
    case FLOAT32:
	a.add_dirty_reg(reg(TMPREG));	
	a.mov(reg(TMPREG).r32(), imm12);
	a.cvtsi2ss(DST, reg(TMPREG).r32());	
	a.pshufd(DST,DST,0);
	break;
    case FLOAT64:
	a.add_dirty_reg(reg(TMPREG));	
	a.rex().mov(reg(TMPREG).r64(), imm12);
	a.cvtsi2sd(DST, reg(TMPREG).r32());
	a.punpcklqdq(DST, DST);
	break;
    default:
	break;
    }
}

static void emit_vslli(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm8)
{
    emit_vmovr(a, type, dst, src);
    switch(type) {
    case UINT8:
    case INT8:
	a.add_dirty_reg(T0);
	a.movdqa(T0, DST);
	// tmp = |FEDCBA98|76543210| imm=3
	// HIGH
 	a.psrlw(T0, 8);         // tmp = |00000000|FEDCBA98|
	a.psllw(T0, 8+imm8);    // tmp = |CBA98000|00000000|
	a.psrlw(T0, 8);         // tmp = |00000000|CBA98000|
	// LOW
	a.psllw(DST, 8+imm8);   // dst = |43210000|00000000|
	a.psrlw(DST, 8);        // dst = |00000000|43210000|
	a.packuswb(DST, T0);
	break;
    case UINT16:
    case INT16:   a.psllw(DST, imm8); break;
    case UINT32:
    case INT32:   a.pslld(DST, imm8); break;
    case UINT64:
    case INT64:   a.psllq(DST, imm8); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_vsrli(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm8)
{
    emit_vmovr(a, type, dst, src);    
    switch(type) {
    case UINT8:
    case INT8:
	a.movdqa(T1, DST);
	// tmp = |FEDCBA98|76543210| imm8=3
	// HIGH
 	a.psrlw(T1, 8+imm8); // tmp = |00000000|000FEDCB|
	// LOW
	a.psllw(DST, 8);          // dst = |76543210|00000000|
 	a.psrlw(DST, 8+imm8);      // dst = |00000000|00076543|	
	a.packuswb(DST, T1);
	break;	
    case UINT16:
    case INT16:   a.psrlw(DST, imm8); break;
    case UINT32:
    case INT32:   a.psrld(DST, imm8); break;
    case UINT64:
    case INT64:   a.psrlq(DST, imm8); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_vsrai(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm8)
{
    emit_vmovr(a, type, dst, src);
    switch(type) {
    case UINT8:
    case INT8:
	a.add_dirty_reg(T1);
	a.movdqa(T1, DST);
	// tmp = |FEDCBA98|76543210| imm8=4
	// HIGH
	a.psraw(T1, imm8);   // tmp = |FFEDCBA9|87654321|
	a.psrlw(T1, 8);     // tmp = |00000000|FFEDCBA9|
	// LOW
	a.psllw(DST, 8);          // dst = |76543210|00000000|
	a.psraw(DST, imm8);        // dst = |07654321|00000000|
	a.psrlw(DST, 8);          // tmp = |00000000|07654321|
	a.packuswb(DST, T1);
	break;		
    case UINT16:
    case INT16:   a.psraw(DST, imm8); break;
    case UINT32:
    case INT32:   a.psrad(DST, imm8); break;
    case UINT64:
    case INT64:
	a.add_dirty_reg(reg(TMPREG));
	a.add_dirty_reg(T0);
	a.movdqa(T0, DST);	
	// shift low
	a.movq(reg(TMPREG).r64(), DST);
	a.sar(reg(TMPREG).r64(), imm8);
	a.movq(DST, reg(TMPREG).r64());
	// shift high
	a.movhlps(T0, T0); // shift left 64
	a.movq(reg(TMPREG).r64(), T0);
	a.sar(reg(TMPREG).r64(), imm8);
	a.movq(T0, reg(TMPREG).r64());
	a.punpcklqdq(DST, T0);
	break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

// [V]PADDW dst,src  : dst = dst + src
// dst = src1 + src2  |  dst=src1; dst += src2;
// dst = dst + src2   |  dst += src2;
// dst = src1 + dst   |  dst += src1;
// dst = dst + dst    |  dst += dst;   == dst = 2*dst == dst = (dst<<1)

static void emit_vadd_avx(ZAssembler &a, uint8_t type,
			  int dst, int src1, int src2)
{
    switch(type) {
    case INT8:
    case UINT8: a.vpaddb(DST, SRC1, SRC2); break;
    case INT16:
    case UINT16:  a.vpaddw(DST, SRC1, SRC2); break;
    case INT32:	    
    case UINT32:  a.vpaddd(DST, SRC1, SRC2); break;
    case INT64:	    
    case UINT64:  a.vpaddq(DST, SRC1, SRC2); break;
    case FLOAT32: a.vaddps(DST, SRC1, SRC2); break;
    case FLOAT64: a.vaddpd(DST, SRC1, SRC2); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_vadd_sse2(ZAssembler &a, uint8_t type,
			   int dst, int src1, int src2)
{
    int src = emit_one_vsrc(a, type, dst, src1, src2);
    switch(type) {
    case INT8:
    case UINT8:  a.paddb(DST, SRC); break;
    case INT16:	    
    case UINT16:  a.paddw(DST, SRC); break;
    case INT32:	    
    case UINT32:  a.paddd(DST, SRC); break;
    case INT64:	    
    case UINT64:  a.paddq(DST, SRC); break;
    case FLOAT32: a.addps(DST, SRC); break;
    case FLOAT64: a.addpd(DST, SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_vadd(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    if (a.use_avx()) 
	emit_vadd_avx(a, type, dst, src1, src2);
    else if (a.use_sse2())
	emit_vadd_sse2(a, type, dst, src1, src2);
    else
	emit_add(a, type, dst, src1, src2);
}

static void emit_vaddi(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    a.add_dirty_reg(xreg(TMPVREG1));
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vadd(a, type, dst, src, TMPVREG1);
}

// SUB r0, r1, r2   (r2 = r0 - r1 )
// dst = src1 - src2 
// PADDW dst,src  : dst = src - dst ???

static void emit_vsub_avx(ZAssembler &a, uint8_t type,
			  int dst, int src1, int src2)
{
    switch(type) {
    case INT8:
    case UINT8: a.vpsubb(DST, SRC1, SRC2); break;
    case INT16:
    case UINT16:  a.vpsubw(DST, SRC1, SRC2); break;
    case INT32:	    
    case UINT32:  a.vpsubd(DST, SRC1, SRC2); break;
    case INT64:	    
    case UINT64:  a.vpsubq(DST, SRC1, SRC2); break;
    case FLOAT32: a.vsubps(DST, SRC1, SRC2); break;
    case FLOAT64: a.vsubpd(DST, SRC1, SRC2); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_vsub_sse2(ZAssembler &a, uint8_t type,
			  int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) &&
	(dst == src2)) { // dst = dst - dst : dst = 0
	emit_vzero(a, dst);
	return;
    }
    else if (src1 == dst) {   // dst = dst - src2 : dst -= src2
	src = src2;
    }
    else if (src2 == dst) { // dst = src - dst; dst = src1 + (0 - dst)
	emit_vneg_sse2(a, type, dst, dst);
	emit_vadd_sse2(a, type, dst, src1, dst);
	return;
    }
    else {
	emit_vmovr(a, type, dst, src1);  // dst != src1
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.psubb(DST, SRC); break;
    case INT16:	    
    case UINT16:  a.psubw(DST, SRC); break;
    case INT32:
    case UINT32:  a.psubd(DST, SRC); break;
    case INT64:	    
    case UINT64:  a.psubq(DST, SRC); break;
    case FLOAT32: a.subps(DST, SRC); break;
    case FLOAT64: a.subpd(DST, SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }    
}

static void emit_vsub(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    if (a.use_avx())
	emit_vsub_avx(a, type, dst, src1,src2);
    else if (a.use_sse2())
	emit_vsub_sse2(a, type, dst, src1,src2);	
    else
	emit_sub(a, type, dst, src1, src2);
}

static void emit_vsubi(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    a.add_dirty_reg(xreg(TMPVREG1));
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vsub(a, type, dst, src, TMPVREG1);
}

static void emit_vrsub(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_vsub(a, type, dst, src2, src1);
}

static void emit_vrsubi(ZAssembler &a, uint8_t type,
			int dst, int src, int8_t imm)
{
    emit_vneg(a, type, dst, src);
    emit_vaddi(a, type, dst, dst, imm);
}

static void emit_vmul_sse2(ZAssembler &a, uint8_t type,
			   int dst, int src1, int src2)
{
    int src = emit_one_vsrc(a, type, dst, src1, src2);    

    switch(type) {
    case INT8:
    case UINT8: {
	a.add_dirty_reg(T1);
	a.movdqa(T1, DST);
	a.pmullw(T1, SRC);
	a.psllw(T1, 8);
	a.psrlw(T1, 8);
    
	a.psrlw(DST, 8);
	a.psrlw(SRC, 8);  // FIXME: do not modify SRC!
	a.pmullw(DST, SRC);
	a.psllw(DST, 8);
	a.por(DST, T1);
	break;
    }
    case INT16:
    case UINT16: a.pmullw(DST, SRC); break;
    case INT32:	    
    case UINT32: a.pmulld(DST, SRC); break;

    case INT64:
    case UINT64: // need 2 temporary registers!
	a.add_dirty_reg(T1);
	a.add_dirty_reg(T0);	
	a.movdqa(T0, DST);     // T0=DST
	a.pmuludq(T0, SRC);    // T0=L(DST)*L(SRC)
	a.movdqa(T1, SRC);     // T1=SRC
	a.psrlq(T1, 32);       // T1=H(SRC)
	a.pmuludq(T1, DST);    // T1=H(SRC)*L(DST)
	a.psllq(T1,32);	       // T1=H(SRC)*L(DST)<<32
	a.paddq(T0, T1);       // T0+=H(SRC)*L(DST)<<32

	a.movdqa(T1, DST);     // T1=DST
	a.psrlq(T1, 32);       // T1=H(DST)
	a.pmuludq(T1, SRC);    // T1=H(DST)*L(SRC)
	a.psllq(T1,32);	       // T1=H(DST)*L(SRC)<<32
	a.paddq(T0, T1);       // T0+=H(DST)*L(SRC)<<32	

	a.movdqa(DST, T0);     // T0=DST	
	break;
	
    case FLOAT32: a.mulps(DST, SRC); break;	    
    case FLOAT64: a.mulpd(DST, SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_vmul_avx(ZAssembler &a, uint8_t type,
			  int dst, int src1, int src2)
{
    switch(type) {
    case INT8:
    case UINT8: {
	a.add_dirty_reg(T0);	
	a.add_dirty_reg(T1);
	a.vpmullw(T1, SRC2, SRC1);
	a.vpsllw(T1, T1, 8);
	a.vpsrlw(T1, T1, 8);
    
	a.vpsrlw(DST, SRC2, 8);
	a.vpsrlw(T0, SRC1, 8);
	a.vpmullw(DST, DST, T0);
	a.vpsllw(DST, SRC2, 8);
	a.vpor(DST, DST, T1);
	break;
    }
    case INT16:
    case UINT16: a.vpmullw(DST, SRC1, SRC2); break;
    case INT32:	    
    case UINT32: a.vpmulld(DST, SRC1, SRC2); break;

    case INT64:
    case UINT64: // need 2 temporary registers!
	a.add_dirty_reg(T1);
	a.add_dirty_reg(T0);
	
	a.vpmuludq(T0, SRC1, SRC2); // T0=L(SRC1)*L(SRC2)

	a.vpsrlq(T1, SRC1, 32);    // T1=H(SRC1)
	a.vpmuludq(T1, T1, SRC2);  // T1=H(SRC1)*L(SRC2)
	a.vpsllq(T1, T1, 32);     // T1=H(SRC1)*L(SRC2)<<32
	a.vpaddq(T0, T0, T1);     // T0+=H(SRC1)*L(SRC2)<<32

	a.vpsrlq(T1, SRC2, 32);
	a.vpmuludq(T1, T1, SRC1);   // T1=H(SRC2)*L(SRC1)
	a.vpsllq(T1, T1,32);	    // T1=H(SRC2)*L(SRC1)<<32
	a.vpaddq(DST, T0, T1);      // DST=T0+H(DST)*L(SRC)<<32	
	break;
	
    case FLOAT32: a.vmulps(DST, SRC1, SRC2); break;	    
    case FLOAT64: a.vmulpd(DST, SRC1, SRC2); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}



// dst = src1*src2
static void emit_vmul(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    if (a.use_avx())
	emit_vmul_avx(a, type, dst, src1, src2);
    else if (a.use_sse2())
	emit_vmul_sse2(a, type, dst, src1, src2);
    else
	emit_mul(a, type, dst, src1, src2);
}

static void emit_vsll(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src = emit_one_ord_vsrc(a, type, dst, src1, src2); // may use T1
    switch(type) {
    case UINT8:
    case INT8:   // DST=SRC1, SRC=SRC2|T0
	a.add_dirty_reg(T0);	
	a.movdqa(T0, DST);
 	a.psrlw(T0, 8);
	a.psllw(T0, 8);
	a.psllw(T0, SRC);
	a.psrlw(T0, 8);
	// LOW
	a.psllw(DST, 8);
	a.psllw(DST, SRC);
	a.psrlw(DST, 8);
	a.packuswb(DST, T0);
	break;
    case UINT16:
    case INT16:   a.psllw(DST, SRC); break;
    case UINT32:
    case INT32:   a.pslld(DST, SRC); break;
    case UINT64:
    case INT64:   a.psllq(DST, SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }            
}

static void emit_vsrl(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src = emit_one_ord_vsrc(a, type, dst, src1, src2);
    switch(type) {
    case UINT8:
    case INT8:
	a.add_dirty_reg(T0);	
	a.movdqa(T0, DST);
 	a.psrlw(T0, 8);
	a.psrlw(T0, SRC);
	// LOW
	a.psllw(DST, 8);          // dst = |76543210|00000000|
	a.psrlw(DST, 8);
	a.psrlw(DST, SRC);
	a.packuswb(DST, T0);
	break;		
    case UINT16:
    case INT16:   a.psrlw(DST, SRC); break;
    case UINT32:
    case INT32:   a.psrld(DST, SRC); break;
    case UINT64:
    case INT64:   a.psrlq(DST, SRC); break;
    default: crash(__FILE__, __LINE__, type); break;
    }            
}

static void emit_vsra(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src = emit_one_ord_vsrc(a, type, dst, src1, src2); 
    switch(type) {
    case UINT8:
    case INT8:
	a.add_dirty_reg(T0);
	a.movdqa(T0, DST);
	a.psraw(T0, SRC);   // tmp = |FFEDCBA9|87654321|
	a.psrlw(T0, 8);     // tmp = |00000000|FFEDCBA9|
	// LOW
	a.psllw(DST, 8);          // dst = |76543210|00000000|
	a.psraw(DST, SRC);        // dst = |07654321|00000000|
	a.psrlw(DST, 8);          // tmp = |00000000|07654321|
	a.packuswb(DST, T0);
	break;		
    case UINT16:
    case INT16:   a.psraw(DST, SRC); break;
    case UINT32:
    case INT32:   a.psrad(DST, SRC); break;
	
    case UINT64:
    case INT64:  // a.psraq(DST, SRC); break; DOES not exist
	a.add_dirty_reg(reg(TMPREG));
	a.add_dirty_reg(T0);
	a.movdqa(T0, DST);	
	// shift low
	a.movq(reg(TMPREG).r64(), DST);
	emit_sra(a, UINT64, TMPREG, TMPREG, src); // a.sar(reg(TMPREG).r64(), SRC);
	a.movq(DST, reg(TMPREG).r64());
	// shift high
	a.movhlps(T0, T0); // shift left 64 bit (in 128)
	a.movq(reg(TMPREG).r64(), T0);
	emit_sra(a, UINT64, TMPREG, TMPREG, src); //a.sar(reg(TMPREG).r64(), SRC);
	a.movq(T0, reg(TMPREG).r64());
	a.punpcklqdq(DST, T0);
	break;	
    default: crash(__FILE__, __LINE__, type); break;
    }                
}


static void emit_vmuli(ZAssembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vmul(a, type, dst, src, TMPVREG1); // FIXME: wont work INT64!
}



static void emit_vbor_sse2(ZAssembler &a, uint8_t type,
			   int dst, int src1, int src2)
{
    (void) type;
    int src = emit_one_vsrc(a, type, dst, src1, src2);    
    switch(type) {
    case FLOAT32:
	a.orps(DST, SRC);
	break;	
    case FLOAT64:
	a.orpd(DST, SRC);
	break;	
    default:
	a.por(DST, SRC);
	break;
    }
}

static void emit_vbor_avx(ZAssembler &a, uint8_t type,
			  int dst, int src1, int src2)
{
    switch(type) {
    case FLOAT32:
	a.vorps(DST, SRC1, SRC2);
	break;	
    case FLOAT64:
	a.vorpd(DST, SRC1, SRC2);
	break;	
    default:
	a.vpor(DST, SRC1, SRC2);
	break;
    }
}

static void emit_vbor(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    if (a.use_avx())
	emit_vbor_avx(a, type, dst, src1, src2);
    else if (a.use_sse2())
	emit_vbor_sse2(a, type, dst, src1, src2);
    else
	emit_bor(a, type, dst, src1, src2);
}

static void emit_vbori(ZAssembler &a, uint8_t type,
			int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vbor(a, type, dst, src, TMPVREG1);
}


static void emit_vband_sse2(ZAssembler &a, uint8_t type,
			   int dst, int src1, int src2)
{
    int src = emit_one_vsrc(a, type, dst, src1, src2);
    switch(type) {
    case FLOAT32:
	a.andps(DST, SRC);
	break;	
    case FLOAT64:
	a.andpd(DST, SRC);
	break;
    default:
	a.pand(DST, SRC);
	break;
    }
}

static void emit_vband_avx(ZAssembler &a, uint8_t type,
			   int dst, int src1, int src2)
{
    switch(type) {
    case FLOAT32:
	a.vandps(DST, SRC1, SRC2);
	break;	
    case FLOAT64:
	a.vandpd(DST, SRC1, SRC2);
	break;
    default:
	a.vpand(DST, SRC1, SRC2);
	break;
    }    
}

static void emit_vband(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    if (a.use_avx())
	emit_vband_avx(a, type, dst, src1, src2);
    else if (a.use_sse2())
	emit_vband_sse2(a, type, dst, src1, src2);
    else
	emit_band(a, type, dst, src1, src2);
}

static void emit_vbandi(ZAssembler &a, uint8_t type,
			int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vband(a, type, dst, src, TMPVREG1);
}

static void emit_vbxor_sse2(ZAssembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    (void) type;
    int src = emit_one_vsrc(a, type, dst, src1, src2);        
    switch(type) {
    case FLOAT32:
	a.xorps(DST, SRC);
	break;
    case FLOAT64:
	a.xorpd(DST, SRC);
	break;
    default:
	a.pxor(DST, SRC);
	break;
    }
}

static void emit_vbxor_avx(ZAssembler &a, uint8_t type,
			  int dst, int src1, int src2)
{
    switch(type) {
    case FLOAT32:
	a.vxorps(DST, SRC1, SRC2);
	break;	
    case FLOAT64:
	a.vxorpd(DST, SRC1, SRC2);
	break;	
    default:
	a.vpxor(DST, SRC1, SRC2);
	break;
    }
}

static void emit_vbxor(ZAssembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    if (a.use_avx())
	emit_vbxor_avx(a, type, dst, src1, src2);
    else if (a.use_sse2())
	emit_vbxor_sse2(a, type, dst, src1, src2);
    else
	emit_bxor(a, type, dst, src1, src2);
}

static void emit_vbxori(ZAssembler &a, uint8_t type,
			int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vbxor(a, type, dst, src, TMPVREG1);
}

static void emit_vbnot(ZAssembler &a, uint8_t type, int dst, int src)
{
    assert(src != TMPVREG0);
    emit_vmov(a, type, dst, src);  // dst = src
    emit_vone(a, TMPVREG0);
    emit_vbxor(a, type, dst, dst, TMPVREG0);
}

// dst = dst == src
static void emit_vcmpeq1(ZAssembler &a, uint8_t type, int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:   a.pcmpeqb(DST, SRC); break;
    case INT16:	    
    case UINT16:  a.pcmpeqw(DST, SRC); break;
    case INT32:	    
    case UINT32:  a.pcmpeqd(DST, SRC); break;
    case INT64:	    
    case UINT64:  a.pcmpeqq(DST, SRC); break;  // FIXME: check for SSE4_1 !!!
    case FLOAT32: a.cmpps(DST, SRC, CMP_EQ); break;
    case FLOAT64: a.cmppd(DST, SRC, CMP_EQ); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

// dst = dst != src
static void emit_vcmpne1(ZAssembler &a, uint8_t type, int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:
	a.pcmpeqb(DST, SRC);
	emit_vbnot(a, type, dst, dst);
	break;
    case INT16:
    case UINT16:
	a.pcmpeqw(DST, SRC);
	emit_vbnot(a, type, dst, dst);
	break;
    case INT32:	    
    case UINT32:
	a.pcmpeqd(DST, SRC);
	emit_vbnot(a, type, dst, dst);	
	break;
    case INT64:	    
    case UINT64:
	a.pcmpeqq(DST, SRC);
	emit_vbnot(a, type, dst, dst);
	break;
    case FLOAT32: a.cmpps(DST, SRC, CMP_NEQ); break;
    case FLOAT64: a.cmppd(DST, SRC, CMP_NEQ); break;
    default: crash(__FILE__, __LINE__, type); break;
    }
    
}

// dst = dst > src
static void emit_vcmpgt1(ZAssembler &a, uint8_t type,
			 int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:   a.pcmpgtb(DST, SRC); break;
    case INT16:
    case UINT16:  a.pcmpgtw(DST, SRC); break;
    case INT32:	    
    case UINT32:  a.pcmpgtd(DST, SRC); break;
    case INT64:	    
    case UINT64:  a.pcmpgtq(DST, SRC); break;
    case FLOAT32: a.cmpps(DST, SRC, CMP_GT); break;
    case FLOAT64: a.cmppd(DST, SRC, CMP_GT); break;	
    default: crash(__FILE__, __LINE__, type); break;
    }
}

// dst = dst < src | dst = src > dst |
static void emit_vcmplt1(ZAssembler &a, uint8_t type,
			 int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:
	emit_vmov(a, type, TMPVREG1, dst);
	emit_vmov(a, type, dst, src);
	a.pcmpgtb(DST, T1);  // SSE2
	break;
    case INT16:
    case UINT16:
	emit_vmov(a, type, TMPVREG1, dst);
	emit_vmov(a, type, dst, src);	
	a.pcmpgtw(DST, T1);  // SSE2
	break;
    case INT32:	    
    case UINT32:
	emit_vmov(a, type, TMPVREG1, dst);
	emit_vmov(a, type, dst, src);	
	a.pcmpgtd(DST,T1); // SSE2
	break;
    case INT64:	    
    case UINT64:
	emit_vmov(a, type, TMPVREG1, dst);
	emit_vmov(a, type, dst, src);
	a.pcmpgtq(DST, T1); // SSE4.2
	break;
    case FLOAT32: a.cmpps(DST, SRC, CMP_LT); return;
    case FLOAT64: a.cmppd(DST, SRC, CMP_LT); return;
    default: crash(__FILE__, __LINE__, type); break;
    }
}

// dst = dst <= src // dst = !(dst > src) = !(src < dst)
static void emit_vcmple1(ZAssembler &a, uint8_t type,
			 int dst, int src)
{
    // assert dst != src
    switch(type) {
    case INT8:
    case UINT8:
	emit_vmov(a, type, TMPVREG1, dst);
	a.pcmpgtb(T1, SRC);
	emit_vbnot(a, type, dst, TMPVREG1);
	break;
    case INT16:
    case UINT16:
	emit_vmov(a, type, TMPVREG1, dst);
	a.pcmpgtw(T1, SRC);
	emit_vbnot(a, type, dst, TMPVREG1);
	break;	
    case INT32:	    
    case UINT32:
	emit_vmov(a, type, TMPVREG1, dst);
	a.pcmpgtd(T1, SRC);
	emit_vbnot(a, type, dst, TMPVREG1);	
	break;		
    case INT64:	    
    case UINT64:
	emit_vmov(a, type, TMPVREG1, dst);
	a.pcmpgtq(T1, SRC);
	emit_vbnot(a, type, dst, TMPVREG1);
	break;			
    case FLOAT32: a.cmpps(DST, SRC, CMP_LE); break;
    case FLOAT64: a.cmppd(DST, SRC, CMP_LE); break;	
    default: crash(__FILE__, __LINE__, type); break;
    }
}

static void emit_vcmpeq(ZAssembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    int src = emit_one_src(a, type, dst, src1, src2);
    if (src1 == src2)
	emit_vone(a, dst);
    else
	emit_vcmpeq1(a, type, dst, src);
}

static void emit_vcmpeqi(ZAssembler &a, uint8_t type,
			 int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vcmpeq(a, type, dst, src, TMPVREG1);
}

static void emit_vcmpne(ZAssembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    int src = emit_one_src(a, type, dst, src1, src2);
    if (src1 == src2)
	emit_vzero(a, dst);
    else
	emit_vcmpne1(a, type, dst, src);
}

static void emit_vcmpnei(ZAssembler &a, uint8_t type,
			 int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vcmpne(a, type, dst, src, TMPVREG1);
}

// emit dst = src1 > src2
static void emit_vcmpgt(ZAssembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    if ((dst == src1) && (dst == src2)) { // dst = dst > dst
	emit_vzero(a, dst);
	return;
    }
    else if (src1 == dst) { // DST = DST > SRC2
	emit_vcmpgt1(a, type, dst, src2);
    }
    else if (src2 == dst) {  // DST = SRC1 > DST => DST = DST < SRC1
	emit_vcmplt1(a, type, dst, src1);
	return;
    }
    else {
	// DST = SRC1 > SRC2 : DST=SRC1,  DST = DST > SRC2
	emit_vmov(a, type, dst, src1);
	emit_vcmpgt1(a, type, dst, src2);
    }
}

static void emit_vcmpgti(ZAssembler &a, uint8_t type,
			 int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vcmpgt(a, type, dst, src, TMPVREG1);
}



// emit dst = src1 >= src2
static void emit_vcmpge(ZAssembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    int src;

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
	    a.cmpps(DST, SRC, cmp);
	else 
	    a.cmppd(DST, SRC, cmp);
    }
    else {
	if ((dst == src1) && (dst == src2)) {
            // DST = DST >= DST (TRUE)
	    emit_vone(a, dst);
	}
	else if (src1 == dst) { 
            // DST = DST >= SRC2 => DST == !(DST < SRC2)
	    emit_vcmplt1(a, type, dst, src2);
	    emit_vbnot(a, type, dst, dst);
	}
	else if (src2 == dst) {
	    // DST = SRC1 >= DST; DST = (DST <= SRC1)
	    emit_vcmple1(a, type, dst, src1);
	}
	else {
	    // DST = (SRC1 >= SRC2); DST = !(SRC1 < SRC2); DST = !(SRC2 > SRC1)
	    emit_vmov(a, type, dst, src2);     // DST = SRC2
	    emit_vcmpgt1(a, type, dst, src1);  // DST = (SRC2 > SRC1)
	    emit_vbnot(a, type, dst, dst);     // DST = !(SRC2 > SRC1)
	}
    }
}

static void emit_vcmpgei(ZAssembler &a, uint8_t type,
			 int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vcmpge(a, type, dst, src, TMPVREG1);
}


static void emit_vcmplt(ZAssembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    emit_vcmpgt(a, type, dst, src2, src1);
}

static void emit_vcmplti(ZAssembler &a, uint8_t type,
			 int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vcmplt(a, type, dst, src, TMPVREG1);
}

static void emit_vcmple(ZAssembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    emit_vcmpge(a, type, dst, src2, src1);
}

static void emit_vcmplei(ZAssembler &a, uint8_t type,
			 int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG1, imm);
    emit_vcmple(a, type, dst, src, TMPVREG1);
}


// Helper function to generate instructions based on type and operation
void emit_instruction(ZAssembler &a, instr_t* p, x86::Gp ret)
{
    switch(p->op) {
    case OP_NOP: a.nop(); break;
    case OP_VNOP: a.nop(); break;
    case OP_RET: a.mov(x86::ptr(ret), reg(p->rd)); break;
    case OP_VRET: a.movdqu(x86::ptr(ret), xreg(p->rd).xmm()); break;
	
    case OP_MOV: emit_movr(a, p->type, p->rd, p->ri); break;
    case OP_MOVI: emit_movi(a, p->type, p->rd, p->imm12); break;
    case OP_VMOV: emit_vmovr(a, p->type, p->rd, p->ri); break;
    case OP_VMOVI: emit_vmovi(a, p->type, p->rd, p->imm12); break;
	
    case OP_NEG: emit_neg(a, p->type, p->rd,p->ri); break;
    case OP_VNEG: emit_vneg(a, p->type, p->rd, p->ri); break;

    case OP_BNOT: emit_bnot(a, p->type, p->rd, p->ri); break;
    case OP_VBNOT: emit_vbnot(a, p->type, p->rd, p->ri); break;		
	
    case OP_ADD: emit_add(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_ADDI: emit_addi(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VADD: emit_vadd(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VADDI: emit_vaddi(a, p->type, p->rd, p->ri, p->imm8); break;
	
    case OP_SUB: emit_sub(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_SUBI: emit_subi(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VSUB: emit_vsub(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VSUBI: emit_vsubi(a, p->type, p->rd, p->ri, p->imm8); break;

    case OP_RSUB: emit_rsub(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_RSUBI: emit_rsubi(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VRSUB: emit_vrsub(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VRSUBI: emit_vrsubi(a, p->type, p->rd, p->ri, p->imm8); break;	
	
    case OP_MUL: emit_mul(a, p->type, p->rd, p->ri, p->rj); break;	
    case OP_MULI: emit_muli(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VMUL: emit_vmul(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VMULI: emit_vmuli(a, p->type, p->rd, p->ri, p->imm8); break;	
	
    case OP_SLL: emit_sll(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_SLLI: emit_slli(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VSLL: emit_vsll(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VSLLI: emit_vslli(a, p->type, p->rd, p->ri, p->imm8); break;
	
    case OP_SRL: emit_srl(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_SRLI: emit_srli(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VSRL: emit_vsrl(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VSRLI: emit_vsrli(a, p->type, p->rd, p->ri, p->imm8); break;
	
    case OP_SRA: emit_sra(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_SRAI: emit_srai(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VSRA: emit_vsra(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VSRAI: emit_vsrai(a, p->type, p->rd, p->ri, p->imm8); break;
	
    case OP_BAND: emit_band(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_BANDI: emit_bandi(a, p->type, p->rd, p->ri, p->imm8); break;	
    case OP_VBAND: emit_vband(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VBANDI: emit_vbandi(a, p->type, p->rd, p->ri, p->imm8); break;	
	
    case OP_BOR:  emit_bor(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_BORI: emit_bori(a, p->type, p->rd, p->ri, p->imm8); break;		
    case OP_VBOR: emit_vbor(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_VBORI: emit_vbori(a, p->type, p->rd, p->ri, p->imm8); break;

    case OP_BXOR: emit_bxor(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_BXORI: emit_bxori(a, p->type, p->rd, p->ri, p->imm8); break;	
    case OP_VBXOR: emit_vbxor(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VBXORI: emit_vbxori(a,p->type,p->rd,p->ri,p->imm8); break;	
	
    case OP_CMPLT: emit_cmplt(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_CMPLTI: emit_cmplti(a, p->type, p->rd, p->ri, p->imm8); break;	
    case OP_VCMPLT: emit_vcmplt(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VCMPLTI: emit_vcmplti(a,p->type,p->rd,p->ri,p->imm8); break;	
	
    case OP_CMPLE: emit_cmple(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_CMPLEI: emit_cmplei(a, p->type, p->rd, p->ri, p->imm8); break;	
    case OP_VCMPLE: emit_vcmple(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VCMPLEI: emit_vcmplei(a,p->type,p->rd,p->ri,p->imm8); break;	
	
    case OP_CMPEQ: emit_cmpeq(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_CMPEQI: emit_cmpeqi(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VCMPEQ: emit_vcmpeq(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VCMPEQI: emit_vcmpeqi(a,p->type,p->rd,p->ri,p->imm8); break;
	
    case OP_CMPGT: emit_cmpgt(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_CMPGTI: emit_cmpgti(a, p->type, p->rd, p->ri, p->imm8); break;	
    case OP_VCMPGT: emit_vcmpgt(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VCMPGTI: emit_vcmpgti(a,p->type,p->rd,p->ri,p->imm8); break;	

    case OP_CMPGE: emit_cmpge(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_CMPGEI: emit_cmpgei(a, p->type, p->rd, p->ri, p->imm8); break;	
    case OP_VCMPGE: emit_vcmpge(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VCMPGEI: emit_vcmpgei(a,p->type,p->rd,p->ri,p->imm8); break;
	
    case OP_CMPNE:  emit_cmpne(a, p->type, p->rd, p->ri, p->rj); break;
    case OP_CMPNEI: emit_cmpnei(a, p->type, p->rd, p->ri, p->imm8); break;
    case OP_VCMPNE: emit_vcmpne(a,p->type,p->rd,p->ri,p->rj); break;
    case OP_VCMPNEI: emit_vcmpnei(a, p->type, p->rd, p->ri, p->imm8); break;	
	
    default: crash(__FILE__, __LINE__, p->type); break;
    }
}

void add_dirty_regs(ZAssembler &a, instr_t* code, size_t n)
{
    while (n--) {
	if ((code->op == OP_JMP) ||
	    (code->op == OP_NOP) ||
	    (code->op == OP_JNZ) ||
	    (code->op == OP_JZ)) {
	}
	if (code->op & OP_VEC) {
	    a.add_dirty_reg(xreg(code->rd));
	}
	else {
	    a.add_dirty_reg(reg(code->rd));
	}
	code++;
    }
}

void assemble(ZAssembler &a, const Environment &env,
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
    a.set_func_frame(&frame);
    add_dirty_regs(a, code, n);

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

    // Setup all labels
    Label lbl[n];    // potential landing positions
    int i;

    for (i = 0; i < (int) n; i++)
	lbl[i].reset();

    for (i = 0; i < (int) n; i++) {
	if ((code[i].op == OP_JMP) ||
	    (code[i].op == OP_JZ) ||
	    (code[i].op == OP_JNZ)) {
	    int j = (i+1)+code[i].imm12;
	    if (lbl[j].id() == Globals::kInvalidId) //?
		lbl[j] = a.newLabel();
	}
    }
    
    // assemble all code
    for (i = 0; i < (int)n; i++) {
	if (lbl[i].id() != Globals::kInvalidId)
	    a.bind(lbl[i]);
	if (code[i].op == OP_JMP) {
	    int j = (i+1)+code[i].imm12;
	    a.jmp(lbl[j]);
	}
	else if (code[i].op == OP_JNZ) {
	    int j = (i+1)+code[i].imm12;
	    switch(code[i].type) {
	    case INT8:
	    case UINT8:	a.cmp(reg(code[i].rd).r8(), 0); break;
	    case INT16:
	    case UINT16: a.cmp(reg(code[i].rd).r16(), 0); break;
	    case INT32:	    
	    case UINT32: a.cmp(reg(code[i].rd).r32(), 0); break;
	    case INT64:	    
	    case UINT64: a.cmp(reg(code[i].rd).r64(), 0); break;
#ifdef FIXME
	    case FLOAT32:
		a.cmpps(reg(code[i].rd),0.0); break;
	    case FLOAT64:
		a.cmppd(reg(code[i].rd), 0.0); break;
#endif
	    default: crash(__FILE__, __LINE__, code[i].type); break;
	    }
	    a.jnz(lbl[j]);
	}
	else if (code[i].op == OP_JZ) {
	    int j = (i+1)+code[i].imm12;
	    switch(code[i].type) {
	    case INT8:
	    case UINT8:	a.cmp(reg(code[i].rd).r8(), 0); break;
	    case INT16:
	    case UINT16: a.cmp(reg(code[i].rd).r16(), 0); break;
	    case INT32:	    
	    case UINT32: a.cmp(reg(code[i].rd).r32(), 0); break;
	    case INT64:	    
	    case UINT64: a.cmp(reg(code[i].rd).r64(), 0); break;
#ifdef FIXME
	    case FLOAT32:
		a.cmpps(reg(code[i].rd),0.0); break;
	    case FLOAT64:
		a.cmppd(reg(code[i].rd), 0.0); break;
#endif
	    default: crash(__FILE__, __LINE__, code[i].type); break;
	    }
	    a.jz(lbl[j]);
	}	
	else {
	    emit_instruction(a, &code[i], p_dst);
	}
    }
    a.emitEpilog(frame);              // Emit function epilog and return.
}
