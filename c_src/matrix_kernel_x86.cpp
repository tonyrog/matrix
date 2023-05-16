#include <asmjit/x86.h>
#include <iostream>

using namespace asmjit;

#include "matrix_types.h"
#include "matrix_kernel.h"

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

struct Assembly
{
    x86::Assembler* a;
    Zone*      z;
    ConstPool* pool;
    Label pool_label;

    Assembly(x86::Assembler& aa, size_t zone_size) {
	a = &aa;
	z = new Zone(zone_size);
	pool = new ConstPool(z);
	pool_label = a->newLabel();
    }
    
    x86::Mem add_constant(void* data, size_t len) {
	size_t offset;
	pool->add(data, len, offset);
	return x86::ptr(pool_label, offset);
    }
    
    x86::Mem add_double(double value) {
	return add_constant(&value, sizeof(value));
    }

    x86::Mem add_int8(int8_t value) {
	return add_constant(&value, sizeof(value));
    }

    x86::Mem add_vint8(int8_t value) {
	int8_t vec[16];
	memset(vec, value, 16);
	return add_constant(vec, sizeof(vec));
    }
};

#define TMPVREG0 14
#define TMPVREG  15

static void emit_vbxor(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2);

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
static void emit_inc(x86::Assembler &a, uint8_t type, int dst)
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
    default: break;
    }    
}
#endif

static void emit_dec(x86::Assembler &a, uint8_t type, int dst)
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
    default: break;
    }    
}

static void emit_movi(x86::Assembler &a, uint8_t type, int dst,
		      int16_t imm12)
{
    switch(type) {
    case UINT8:
    case INT8:    a.mov(reg(dst).r8(), imm12); break;
    case UINT16:
    case INT16:   a.mov(reg(dst).r16(), imm12); break;
    case UINT32:
    case INT32:   a.mov(reg(dst).r32(), imm12); break;
    case UINT64:
    case INT64:   a.mov(reg(dst).r64(), imm12); break;
    default: break;
    }    
}

static void emit_addi(x86::Assembler &a, uint8_t type, int dst, int src, int imm)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));    
    switch(type) {
    case UINT8:
    case INT8:    a.add(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.add(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.add(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.add(reg(dst).r64(), imm); break;
    default: break;
    }    
}

static void emit_subi(x86::Assembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));    
    switch(type) {
    case UINT8:
    case INT8:    a.sub(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.sub(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.sub(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.sub(reg(dst).r64(), imm); break;
    default: break;
    }    
}


static void emit_slli(x86::Assembler &a, uint8_t type, int dst, int src, int8_t imm)
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
    default: break;
    }    
}

static void emit_srli(x86::Assembler &a, uint8_t type, int dst, int src, int8_t imm)
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
    default: break;
    }    
}

static void emit_srai(x86::Assembler &a, uint8_t type, int dst, int src, int8_t imm)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));    
    switch(type) {
    case UINT8:
    case INT8:    a.sar(reg(dst).r8(), imm); break;
    case UINT16:
    case INT16:   a.sar(reg(dst).r16(), imm); break;
    case UINT32:
    case INT32:   a.sar(reg(dst).r32(), imm); break;
    case UINT64:
    case INT64:   a.sar(reg(dst).r64(), imm); break;
    default: break;
    }    
}


static void emit_zero(x86::Assembler &a, uint8_t type, int dst)
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
    default: break;
    }    
}

#ifdef unused
static void emit_one(x86::Assembler &a, uint8_t type, int dst)
{
    emit_movi(a, type, dst, 1);
}
#endif

static void emit_neg_dst(x86::Assembler &a, uint8_t type, int dst)
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
    default: break;
    }
}

static void emit_movr(x86::Assembler &a, uint8_t type, int dst, int src)
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
    default: break;
    }    
}

// move src to dst if condition code (matching cmp) is set
// else set dst=0
static void emit_movecc(x86::Assembler &a, int cmp, uint8_t type, int dst)
{
    Label Skip = a.newLabel();
    emit_movi(a, type, dst, 0);
    switch(cmp) {
    case CMP_EQ:
	a.jne(Skip); break;
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
    default: break;
    }
    emit_dec(a, type, dst);
    a.bind(Skip);
}

// above without jump
#ifdef not_used
static void emit_movecc_dst(x86::Assembler &a, int cmp, uint8_t type, int dst)
{
    if ((type != UINT8) && (type != INT8))
	emit_movi(a, type, dst, 0); // clear if dst is 16,32,64
    // set byte 0|1
    switch(cmp) {
    case CMP_EQ: a.setne(reg(dst).r8()); break;
    case CMP_LT: a.setl(reg(dst).r8()); break;
    case CMP_LE: a.setle(reg(dst).r8()); break;
    case CMP_GT: a.setg(reg(dst).r8()); break;
    case CMP_GE: a.setge(reg(dst).r8()); break;
    default: break;
    }
    // negate to set all bits (optimise before conditional jump)
    emit_neg_dst(a, type, dst);    
}
#endif

// set dst = 0
static void emit_vzero(x86::Assembler &a, int dst)
{
    a.pxor(vreg(dst).xmm(), vreg(dst).xmm());
}


static void emit_vone(x86::Assembler &a, int dst)
{
    a.pcmpeqb(vreg(dst).xmm(), vreg(dst).xmm());
}

static void emit_vmov(x86::Assembler &a, uint8_t type,
		      int dst, int src)
{
    if (IS_FLOAT_TYPE(type))
	a.movaps(vreg(dst).xmm(), vreg(src).xmm());  // dst = src1;
    else
	a.movdqa(vreg(dst).xmm(), vreg(src).xmm());  // dst = src1;    
}

// dst = src
static void emit_vmovr(x86::Assembler &a, uint8_t type,
		       int dst, int src)
{
    if (src == dst) { // dst = dst;
	return;
    }
    else {
	emit_vmov(a, type, dst, src); // dst = src
    }
}

// dst = -src  = 0 - src
static void emit_neg(x86::Assembler &a, uint8_t type, int dst, int src)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));
    emit_neg_dst(a, type, dst);
}

// dst = ~src 
static void emit_bnot(x86::Assembler &a, uint8_t type, int dst, int src)
{
    if (src != dst)
	a.mov(reg(dst), reg(src));
    switch(type) {
    case UINT8:	
    case INT8:       a.not_(reg(dst).r8()); break;
    case UINT16:	
    case INT16:      a.not_(reg(dst).r16()); break;
    case UINT32:	
    case INT32:      a.not_(reg(dst).r32()); break;
    case UINT64:	
    case INT64:      a.not_(reg(dst).r64()); break;
    default: break;
    }
}


static void emit_add(x86::Assembler &a, uint8_t type, int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) && (dst == src2)) { // dst = dst + dst : dst += dst
	src = dst;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_movr(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case UINT8:	
    case INT8:       a.add(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.add(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.add(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.add(reg(dst).r64(), reg(src).r64()); break;
    default: break;
    }
}

static void emit_sub(x86::Assembler &a, uint8_t type,
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
    default: break;
    }
}

static void emit_mul(x86::Assembler &a, uint8_t type, int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) && (dst == src2)) { // dst = dst * dst : dst *= dst
	src = dst;
    }
    else if (src1 == dst) // dst = dst * src2 : dst *= src2
	src = src2;
    else if (src2 == dst) // dst = src1 * dst : dst *= src1
	src = src1;
    else {
	emit_movr(a, type, dst, src1);
	src = src2;
    }
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
    default: break;
    }
}


static void emit_band(x86::Assembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_movr(a, type, dst, src1); // dst = src1;
	src = src2;
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
    default: break;
    }    
}

static void emit_bor(x86::Assembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_movr(a, type, dst, src1);  // dst = src1;
	src = src2;
    }
    switch(type) {
    case UINT8:	
    case INT8:       a.or_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.or_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.or_(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.or_(reg(dst).r64(), reg(src).r64()); break;
    default: break;
    }        
}

static void emit_bxor(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    int src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_movr(a, type, dst, src1);  // dst = src1;
	src = src2;
    }
    switch(type) {
    case UINT8:	
    case INT8:       a.xor_(reg(dst).r8(), reg(src).r8()); break;
    case UINT16:	
    case INT16:      a.xor_(reg(dst).r16(), reg(src).r16()); break;
    case UINT32:	
    case INT32:      a.xor_(reg(dst).r32(), reg(src).r32()); break;
    case UINT64:	
    case INT64:      a.xor_(reg(dst).r64(), reg(src).r64()); break;
    default: break;
    }            
}

static void emit_cmp(x86::Assembler &a, uint8_t type, int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:	a.cmp(reg(dst).r8(), reg(src).r8()); break;
    case INT16:
    case UINT16: a.cmp(reg(dst).r16(), reg(src).r16()); break;
    case INT32:	    
    case UINT32: a.cmp(reg(dst).r32(), reg(src).r32()); break;
    case INT64:	    
    case UINT64: a.cmp(reg(dst).r64(), reg(src).r64()); break;
#ifdef FIXME
    case FLOAT32:
	a.cmpps(dst.xmm(), src.xmm(), cmp); break;
    case FLOAT64:
	a.cmppd(dst.xmm(), src.xmm(), cmp); break;
#endif
    default: break;
    }	
}

static void emit_cmpeq(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    int src;
    int cmp = CMP_EQ;  // EQ
    
    if ((dst == src1) && (dst == src2)) {  // dst = dst == dst
	emit_movi(a, type, dst, -1);
	return;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_movr(a, type, dst, src1); // dst = src1;
	src = src2;
    }
    emit_cmp(a, type, dst, src);
    emit_movecc(a, cmp, type, dst);
}

// emit dst = src1 > src2
static void emit_cmpgt(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    int src;
    int cmp = CMP_GT;

    if (((dst == src1) && (dst == src2)) || (src1==src2)) { // dst = dst > dst || dst = srcx > srcx
	emit_zero(a, type, dst);
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
	emit_movr(a, type, dst, src1);
	src = src2;
    }
    emit_cmp(a, type, dst, src);
    emit_movecc(a, cmp, type, dst);
}


// emit dst = src1 >= src2
static void emit_cmpge(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    int src;

    if (IS_FLOAT_TYPE(type)) {
	// int cmp = CMP_GE;
	if ((dst == src1) && (dst == src2)) { // dst = dst >= dst (TRUE!)
	    emit_movi(a, type, dst, -1);
	    return;
	}
	else if (src1 == dst)   // dst = dst >= src2
	    src = src2;
	else if (src2 == dst) { // dst = src1 >= dst => dst = dst <= src1
	    // cmp = CMP_LE;
	    src = src1;
	}
	else {
	    emit_movr(a, type, dst, src1); // dst = (src1 >= src2)
	    src = src2;
	}
#ifdef FIXME
	if (type == FLOAT32)
	    a.cmpps(dst.xmm(), src.xmm(), cmp);
	else 
	    a.cmppd(dst.xmm(), src.xmm(), cmp);
#endif
    }
    else {
	int cmp = CMP_GE;
	if (((dst == src1) && (dst == src2)) || (src1 == src2)) { // dst = dst >= dst (TRUE)
	    emit_movi(a, type, dst, -1);
	    return;
	}
	else if (src1 == dst)   // dst = dst >= src2
	    src = src2;
	else if (src2 == dst) {   // dst = src1 >= dst
	    cmp = CMP_LE;
	    src = src1;
	}
	else {
	    emit_movr(a, type, dst, src1); // dst = (src1 >= src2)
	    src = src2;
	}
	emit_cmp(a, type, dst, src);
	emit_movecc(a, cmp, type, dst);
    }
}

static void emit_cmplt(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmpgt(a, type, dst, src2, src1);
}

static void emit_cmple(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    emit_cmpge(a, type, dst, src2, src1);
}


// dst = -src  = 0 - src
static void emit_vneg(x86::Assembler &a, uint8_t type, int dst, int src)
{
    if (src == dst) { // dst = -dst;
	src = 15; // x86::regs::xmm15;
	emit_vmov(a, type, src, dst); // copy dst to xmm15
    }
    emit_vzero(a, dst);    
    switch(type) {  // dst = src - dst
    case INT8:
    case UINT8:   a.psubb(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT16:	    
    case UINT16:  a.psubw(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT32:
    case UINT32:  a.psubd(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT64:
    case UINT64:  a.psubq(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT32: a.subps(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT64: a.subpd(vreg(dst).xmm(), vreg(src).xmm()); break;
    default: break;
    }    
}

// broadcast integer value imm12 into element
static void emit_vmovi(x86::Assembler &a, uint8_t type,
		       int dst, int16_t imm12)
{

}

static void emit_vslli(x86::Assembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    if (src != dst)
	emit_vmov(a, type, dst, src);
    switch(type) {
    case UINT8:
    case INT8:    a.psllw(vreg(dst).xmm(), imm); break;  // FIXME!!!
    case UINT16:
    case INT16:   a.psllw(vreg(dst).xmm(), imm); break;
    case UINT32:
    case INT32:   a.pslld(vreg(dst).xmm(), imm); break;
    case UINT64:
    case INT64:   a.psllq(vreg(dst).xmm(), imm); break;
    default: break;
    }    
}

static void emit_vsrli(x86::Assembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    if (src != dst)
	emit_vmov(a, type, dst, src);
    switch(type) {
    case UINT8:
    case INT8:    a.psrlw(vreg(dst).xmm(), imm); break;  // FIXME!!!
    case UINT16:
    case INT16:   a.psrlw(vreg(dst).xmm(), imm); break;
    case UINT32:
    case INT32:   a.psrld(vreg(dst).xmm(), imm); break;
    case UINT64:
    case INT64:   a.psrlq(vreg(dst).xmm(), imm); break;
    default: break;
    }    
}

static void emit_vsrai(x86::Assembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    if (src != dst)
	emit_vmov(a, type, dst, src);
    switch(type) {
    case UINT8:
    case INT8:    a.psraw(vreg(dst).xmm(), imm); break;  // FIXME!!
    case UINT16:
    case INT16:   a.psraw(vreg(dst).xmm(), imm); break;
    case UINT32:
    case INT32:   a.psrad(vreg(dst).xmm(), imm); break;
    case UINT64:
    case INT64:   a.psrad(vreg(dst).xmm(), imm); break;  // FIXME!!
    default: break;
    }    
}


static void emit_vbnot(x86::Assembler &a, uint8_t type, int dst, int src)
{
    (void) type;
    if (src == dst) {
	emit_vone(a, TMPVREG);
	emit_vbxor(a, type, dst, src, TMPVREG);
    }
    else {
	emit_vmov(a, type, dst, src);  // dst = src
	emit_vone(a, src);             // src = -1 (mask)
	emit_vbxor(a, type, dst, dst, src);	
    }
}

// PADDW dst,src  : dst = dst + src
// dst = src1 + src2  |  dst=src1; dst += src2;
// dst = dst + src2   |  dst += src2;
// dst = src1 + dst   |  dst += src1;
// dst = dst + dst    |  dst += dst;   == dst = 2*dst == dst = (dst<<1)

static void emit_vadd(x86::Assembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src;
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
    case UINT8:   a.paddb(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT16:	    
    case UINT16:  a.paddw(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT32:	    
    case UINT32:  a.paddd(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT64:	    
    case UINT64:  a.paddq(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT32: a.addps(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT64: a.addpd(vreg(dst).xmm(), vreg(src).xmm()); break;
    default: break;
    }
}

// SUB r0, r1, r2   (r2 = r0 - r1 )
// dst = src1 - src2 
// PADDW dst,src  : dst = src - dst ???
static void emit_vsub(x86::Assembler &a, uint8_t type,
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
	emit_vneg(a, type, dst, dst);
	emit_vadd(a, type, dst, src1, dst);
	return;
    }
    else {
	emit_vmovr(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.psubb(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT16:	    
    case UINT16:  a.psubw(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT32:
    case UINT32:  a.psubd(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT64:	    
    case UINT64:  a.psubq(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT32: a.subps(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT64: a.subpd(vreg(dst).xmm(), vreg(src).xmm()); break;
    default: break;
    }
}

// dst = src1*src2
static void emit_vmul(x86::Assembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    int src;
    
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
    case UINT8: {
	a.movdqa(vreg(TMPVREG).xmm(), vreg(dst).xmm());
	a.pmullw(vreg(TMPVREG).xmm(), vreg(src).xmm());
	a.psllw(vreg(TMPVREG).xmm(), 8);
	a.psrlw(vreg(TMPVREG).xmm(), 8);
    
	a.psrlw(vreg(dst).xmm(), 8);
	a.psrlw(vreg(src).xmm(), 8);
	a.pmullw(vreg(dst).xmm(), vreg(src).xmm());
	a.psllw(vreg(dst).xmm(), 8);
	a.por(vreg(dst).xmm(), vreg(TMPVREG).xmm());
	break;
    }
    case INT16:
    case UINT16: a.pmullw(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT32:	    
    case UINT32: a.pmulld(vreg(dst).xmm(), vreg(src).xmm()); break;
	
    case UINT64:
    case INT64: { // temp1 = xmm14, temp2 = xmm15
	a.movdqa(vreg(TMPVREG0).xmm(),  vreg(src).xmm());
	a.pmuludq(vreg(TMPVREG0).xmm(), vreg(dst).xmm()); // xmm14=AC = BA*DC
	a.movdqa(vreg(TMPVREG).xmm(),  vreg(src).xmm()); // B = BA
	a.psrlq(vreg(TMPVREG).xmm(), 32);    // B = BA >> 32
	a.pmuludq(vreg(TMPVREG).xmm(), vreg(dst).xmm()); // BC = xmm15 = B * (DC & 0xFFFFFFFF)
	a.psrlq(vreg(dst).xmm(), 32);           // D = DC >> 32
	a.pmuludq(vreg(dst).xmm(), vreg(src).xmm());        // DA = (BA & 0xFFFFFFFF) * D;
	a.paddq(vreg(dst).xmm(), vreg(TMPVREG).xmm());   // H = BC + DA
	a.psllq(vreg(dst).xmm(), 32);           // H <<= 32
	a.paddq(vreg(dst).xmm(), vreg(TMPVREG0).xmm());   // H + AC
	break;
    }
    case FLOAT32: a.mulps(vreg(dst).xmm(), vreg(src).xmm()); break;	    
    case FLOAT64: a.mulpd(vreg(dst).xmm(), vreg(src).xmm()); break;
    default: break;
    }
}


static void emit_vaddi(x86::Assembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG, imm);
    emit_vadd(a, type, dst, src, TMPVREG);
}

static void emit_vsubi(x86::Assembler &a, uint8_t type,
		       int dst, int src, int8_t imm)
{
    emit_vmovi(a, type, TMPVREG, imm);
    emit_vsub(a, type, dst, src, TMPVREG);
}


static void emit_vbor(x86::Assembler &a, uint8_t type,
		      int dst, int src1, int src2)
{
    (void) type;
    int src;
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
    switch(type) {
    case FLOAT32:
	a.orps(vreg(dst).xmm(), vreg(src).xmm());
	break;	
    case FLOAT64:
	a.orpd(vreg(dst).xmm(), vreg(src).xmm());
	break;	
    default:
	a.por(vreg(dst).xmm(), vreg(src).xmm());
	break;
    }
}

static void emit_vband(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    (void) type;
    int src;
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
    switch(type) {
    case FLOAT32:
	a.andps(vreg(dst).xmm(), vreg(src).xmm());
	break;	
    case FLOAT64:
	a.andpd(vreg(dst).xmm(), vreg(src).xmm());
	break;
    default:
	a.pand(vreg(dst).xmm(), vreg(src).xmm());
	break;
    }
}

static void emit_vbxor(x86::Assembler &a, uint8_t type,
		       int dst, int src1, int src2)
{
    (void) type;    
    int src;
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
    switch(type) {
    case FLOAT32:
	a.xorps(vreg(dst).xmm(), vreg(src).xmm());
	break;
    case FLOAT64:
	a.xorpd(vreg(dst).xmm(), vreg(src).xmm());
	break;
    default:
	a.pxor(vreg(dst).xmm(), vreg(src).xmm());
	break;
    }
}

// dst = dst == src
static void emit_vcmpeq1(x86::Assembler &a, uint8_t type,
			 int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:   a.pcmpeqb(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT16:	    
    case UINT16:  a.pcmpeqw(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT32:	    
    case UINT32:  a.pcmpeqd(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT64:	    
    case UINT64:  a.pcmpeqq(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT32: a.cmpps(vreg(dst).xmm(), vreg(src).xmm(), CMP_EQ); break;
    case FLOAT64: a.cmppd(vreg(dst).xmm(), vreg(src).xmm(), CMP_EQ); break;
    default: break;
    }
}

// dst = dst > src
static void emit_vcmpgt1(x86::Assembler &a, uint8_t type,
			 int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:   a.pcmpgtb(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT16:
    case UINT16:  a.pcmpgtw(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT32:	    
    case UINT32:  a.pcmpgtd(vreg(dst).xmm(), vreg(src).xmm()); break;
    case INT64:	    
    case UINT64:  a.pcmpgtq(vreg(dst).xmm(), vreg(src).xmm()); break;
    case FLOAT32: a.cmpps(vreg(dst).xmm(), vreg(src).xmm(), CMP_GT); break;
    case FLOAT64: a.cmppd(vreg(dst).xmm(), vreg(src).xmm(), CMP_GT); break;	
    default: break;
    }
}

// dst = dst < src | dst = src > dst |
static void emit_vcmplt1(x86::Assembler &a, uint8_t type,
			 int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:
	emit_vmov(a, type, TMPVREG, dst);
	emit_vmov(a, type, dst, src);
	a.pcmpgtb(vreg(dst).xmm(), vreg(TMPVREG).xmm());  // SSE2
	break;
    case INT16:
    case UINT16:
	emit_vmov(a, type, TMPVREG, dst);
	emit_vmov(a, type, dst, src);	
	a.pcmpgtw(vreg(dst).xmm(), vreg(TMPVREG).xmm());  // SSE2
	break;
    case INT32:	    
    case UINT32:
	emit_vmov(a, type, TMPVREG, dst);
	emit_vmov(a, type, dst, src);	
	a.pcmpgtd(vreg(dst).xmm(),vreg(TMPVREG).xmm()); // SSE2
	break;
    case INT64:	    
    case UINT64:
	emit_vmov(a, type, TMPVREG, dst);
	emit_vmov(a, type, dst, src);
	a.pcmpgtq(vreg(dst).xmm(), vreg(TMPVREG).xmm()); // SSE4.2
	break;
    case FLOAT32: a.cmpps(vreg(dst).xmm(), vreg(src).xmm(), CMP_LT); return;
    case FLOAT64: a.cmppd(vreg(dst).xmm(), vreg(src).xmm(), CMP_LT); return;
    default: break;
    }
}

// dst = dst <= src // dst = !(dst > src) = !(src < dst)
static void emit_vcmple1(x86::Assembler &a, uint8_t type,
			 int dst, int src)
{
    switch(type) {
    case INT8:
    case UINT8:
	emit_vmov(a, type, TMPVREG, dst);
	a.pcmpgtb(vreg(TMPVREG).xmm(), vreg(src).xmm());
	emit_vmov(a, type, dst, TMPVREG);
	emit_vone(a, TMPVREG);
	a.pandn(vreg(dst).xmm(), vreg(TMPVREG).xmm());
	break;
    case INT16:
    case UINT16:
	emit_vmov(a, type, TMPVREG, dst);
	a.pcmpgtw(vreg(TMPVREG).xmm(), vreg(src).xmm());
	emit_vmov(a, type, dst, TMPVREG);
	emit_vone(a, TMPVREG);
	a.pandn(vreg(dst).xmm(), vreg(TMPVREG).xmm());
	break;	
    case INT32:	    
    case UINT32:
	emit_vmov(a, type, TMPVREG, dst);
	a.pcmpgtd(vreg(TMPVREG).xmm(), vreg(src).xmm());
	emit_vmov(a, type, dst, TMPVREG);
	emit_vone(a, TMPVREG);
	a.pandn(vreg(dst).xmm(), vreg(TMPVREG).xmm());
	break;		
    case INT64:	    
    case UINT64:
	emit_vmov(a, type, TMPVREG, dst);
	a.pcmpgtq(vreg(TMPVREG).xmm(), vreg(src).xmm());
	emit_vmov(a, type, dst, TMPVREG);
	emit_vone(a, TMPVREG);
	a.pandn(vreg(dst).xmm(), vreg(TMPVREG).xmm());
	break;			
    case FLOAT32: a.cmpps(vreg(dst).xmm(), vreg(src).xmm(), CMP_LE); break;
    case FLOAT64: a.cmppd(vreg(dst).xmm(), vreg(src).xmm(), CMP_LE); break;	
    default: break;
    }
}


static void emit_vcmpeq(x86::Assembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    int src;
    // int cmp = CMP_EQ;  // EQ
    
    if ((dst == src1) && (dst == src2)) {  // dst = dst == dst
	emit_vone(a, dst);
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
    emit_vcmpeq1(a, type, dst, src);
}

// emit dst = src1 > src2
static void emit_vcmpgt(x86::Assembler &a, uint8_t type,
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


// emit dst = src1 >= src2
static void emit_vcmpge(x86::Assembler &a, uint8_t type,
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
	    a.cmpps(vreg(dst).xmm(), vreg(src).xmm(), cmp);
	else 
	    a.cmppd(vreg(dst).xmm(), vreg(src).xmm(), cmp);
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

static void emit_vcmplt(x86::Assembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    emit_vcmpgt(a, type, dst, src2, src1);
}

static void emit_vcmple(x86::Assembler &a, uint8_t type,
			int dst, int src1, int src2)
{
    emit_vcmpge(a, type, dst, src2, src1);
}

// Helper function to generate instructions based on type and operation
void emit_instruction(x86::Assembler &a, instr_t* optr, x86::Gp ret)
{
    switch(optr->op) {
    case OP_RET: a.mov(x86::ptr(ret), reg(optr->rd)); break;
    case OP_MOVI:   emit_movi(a, optr->type, optr->rd, optr->imm12); break;
    case OP_ADDI:   emit_addi(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_SUBI:   emit_subi(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_SLLI:   emit_slli(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_SRLI:   emit_srli(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_SRAI:   emit_srai(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_MOVR:   emit_movr(a, optr->type, optr->rd, optr->ri); break;
    case OP_NEG: emit_neg(a, optr->type, optr->rd,optr->ri); break;
    case OP_ADD: emit_add(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_SUB: emit_sub(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_MUL: emit_mul(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_BNOT: emit_bnot(a, optr->type, optr->rd, optr->ri); break;
    case OP_BAND: emit_band(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_BOR:  emit_bor(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_BXOR: emit_bxor(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_CMPEQ: emit_cmpeq(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_CMPLT: emit_cmplt(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_CMPLE: emit_cmple(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VRET:
	a.movdqu(x86::ptr(ret), vreg(optr->rd).xmm());
	break;
    case OP_VMOVI: emit_vmovi(a, optr->type, optr->rd, optr->imm12); break;
    case OP_VADDI: emit_vaddi(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_VSUBI: emit_vsubi(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_VSLLI: emit_vslli(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_VSRLI: emit_vsrli(a, optr->type, optr->rd, optr->ri, optr->imm8); break;
    case OP_VSRAI: emit_vsrai(a, optr->type, optr->rd, optr->ri, optr->imm8); break;				
    case OP_VMOVR:  emit_vmovr(a, optr->type, optr->rd, optr->ri); break;
    case OP_VNEG:   emit_vneg(a, optr->type, optr->rd, optr->ri); break;
    case OP_VADD:   emit_vadd(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VSUB:   emit_vsub(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VMUL:   emit_vmul(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VBNOT:  emit_vbnot(a, optr->type, optr->rd, optr->ri); break;	
    case OP_VBAND:  emit_vband(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VBOR:   emit_vbor(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VBXOR:  emit_vbxor(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VCMPEQ: emit_vcmpeq(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VCMPLT: emit_vcmplt(a, optr->type, optr->rd, optr->ri, optr->rj); break;
    case OP_VCMPLE: emit_vcmple(a, optr->type, optr->rd, optr->ri, optr->rj); break;	
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
	    if ((code->op == OP_JMP) || (code->op == OP_NOP)) { // nothing
	    }
	    else if ((code->op == OP_JNZ) ||
		     (code->op == OP_JZ) ||
		     (code->op == OP_RET))
		frame.addDirtyRegs(reg(code->rd));
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
    Assembly ay(a, 1024);

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
	    default: break;
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
	    default: break;
	    }
	    a.jz(lbl[j]);
	}	
	else {
	    emit_instruction(a, &code[i], p_dst);
	}
    }
    a.emitEpilog(frame);              // Emit function epilog and return.
}
