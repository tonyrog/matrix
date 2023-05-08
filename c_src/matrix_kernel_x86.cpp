#include <asmjit/x86.h>
#include <iostream>

using namespace asmjit;

#include "matrix_types.h"
#include "matrix_kernel.h"

static void emit_vadd(x86::Assembler &a, uint8_t type, x86::Xmm dst, x86::Xmm src1, x86::Xmm src2);


static x86::Xmm vreg(int i)
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

// set dst = 0
static void emit_vzero(x86::Assembler &a, x86::Xmm dst)
{
    a.pxor(dst, dst);
}

static void emit_vone(x86::Assembler &a, x86::Xmm dst)
{
    a.pcmpeqb(dst, dst);
}

static void emit_mov(x86::Assembler &a, uint8_t type,
		     x86::Xmm dst, x86::Xmm src)
{
    if (IS_FLOAT_TYPE(type))
	a.movaps(dst, src);  // dst = src1;
    else
	a.movdqa(dst, src);  // dst = src1;    
}

// dst = src
static void emit_vmovr(x86::Assembler &a, uint8_t type,
		       x86::Xmm dst, x86::Xmm src)
{
    if (src == dst) { // dst = dst;
	return;
    }
    else {
	emit_mov(a, type, dst, src); // dst = src
    }
}

// dst = -src  = 0 - src
static void emit_vneg(x86::Assembler &a, uint8_t type, x86::Xmm dst, x86::Xmm src)
{
    if (src == dst) { // dst = -dst;
	src = x86::regs::xmm15;
	emit_mov(a, type, src, dst);
    }
    emit_vzero(a, dst);    
    switch(type) {  // dst = src - dst
    case INT8:
    case UINT8:   a.psubb(dst, src); break;
    case INT16:	    
    case UINT16:  a.psubw(dst, src); break;
    case INT32:
    case UINT32:  a.psubd(dst, src); break;
    case INT64:
    case UINT64:  a.psubq(dst, src); break;
    case FLOAT32: a.subps(dst, src); break;
    case FLOAT64: a.subpd(dst, src); break;
    default: break;
    }    
}

static void emit_vbnot(x86::Assembler &a, uint8_t type, x86::Xmm dst, x86::Xmm src)
{
    (void) type;
    if (src == dst) {
	emit_vone(a, x86::regs::xmm15);  // pcmpeqb xmm15,xmm15
	if (IS_FLOAT_TYPE(type))
	    a.andnps(dst, x86::regs::xmm15);
	else
	    a.pandn(dst, x86::regs::xmm15);
    }
    else {
	emit_mov(a, type, dst, src);  // dst = src
	emit_vone(a, src);      // src = -1 (mask)
	if (IS_FLOAT_TYPE(type))
	    a.andnps(dst, src);  // dst = ~dst & src
	else
	    a.pandn(dst, src);
    }
}

// PADDW dst,src  : dst = dst + src
// dst = src1 + src2  |  dst=src1; dst += src2;
// dst = dst + src2   |  dst += src2;
// dst = src1 + dst   |  dst += src1;
// dst = dst + dst    |  dst += dst;   == dst = 2*dst == dst = (dst<<1)

static void emit_vadd(x86::Assembler &a, uint8_t type,
		      x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    x86::Xmm src;
    if ((dst == src1) && (dst == src2)) { // dst = dst + dst : dst += dst
	// emit_mov(a, type, x86::xmm15, dst); 
	// src = x86::xmm15;
	src = dst;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_mov(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.paddb(dst, src); break;
    case INT16:	    
    case UINT16:  a.paddw(dst, src); break;
    case INT32:	    
    case UINT32:  a.paddd(dst, src); break;
    case INT64:	    
    case UINT64:  a.paddq(dst, src); break;
    case FLOAT32: a.addps(dst, src); break;
    case FLOAT64: a.addpd(dst, src); break;
    default: break;
    }
}

// SUB r0, r1, r2   (r2 = r0 - r1 )
// dst = src1 - src2 
// PADDW dst,src  : dst = src - dst ???
static void emit_vsub(x86::Assembler &a, uint8_t type,
		      x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    x86::Xmm src;
    if ((dst == src1) && (dst == src2)) { // dst = dst - dst : dst = 0
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
	a.movaps(dst, src1);  // dst = src1;
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.psubb(dst, src); break;
    case INT16:	    
    case UINT16:  a.psubw(dst, src); break;
    case INT32:
    case UINT32:  a.psubd(dst, src); break;
    case INT64:	    
    case UINT64:  a.psubq(dst, src); break;
    case FLOAT32: a.subps(dst, src); break;
    case FLOAT64: a.subpd(dst, src); break;
    default: break;
    }
}

// dst = src1*src2
static void emit_vmul(x86::Assembler &a, uint8_t type,
		      x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    x86::Xmm src;
    
    if ((dst == src1) && (dst == src2)) // dst = dst * dst : dst *= dst
	src = dst;
    else if (src1 == dst)               // dst = dst * src2 : dst *= src2
	src = src2;
    else if (src2 == dst)               // dst = src1 * dst : dst *= src1
	src = src1;
    else {
	emit_mov(a, type, dst, src1);
	src = src2;
    }
    
    switch(type) {
    case INT8:
    case UINT8: {
	a.movdqa(x86::xmm15, dst);
	a.pmullw(x86::xmm15, src);
	a.psllw(x86::xmm15, 8);
	a.psrlw(x86::xmm15, 8);           
    
	a.psrlw(dst, 8);
	a.psrlw(src, 8);
	a.pmullw(dst, src);
	a.psllw(dst, 8);
	a.por(dst, x86::xmm15);
	break;
    }
    case INT16:
    case UINT16: a.pmullw(dst, src); break;
    case INT32:	    
    case UINT32: a.pmulld(dst, src); break;
	
    case UINT64:
    case INT64: {
	a.movdqa(x86::xmm14,  src);
	a.pmuludq(x86::xmm14, dst); // xmm14=AC = BA*DC
	a.movdqa(x86::xmm15,  src); // B = BA
	a.psrlq(x86::xmm15, 32);    // B = BA >> 32
	a.pmuludq(x86::xmm15, dst); // BC = xmm15 = B * (DC & 0xFFFFFFFF)
	a.psrlq(dst, 32);           // D = DC >> 32
	a.pmuludq(dst, src);        // DA = (BA & 0xFFFFFFFF) * D;
	a.paddq(dst, x86::xmm15);   // H = BC + DA
	a.psllq(dst, 32);           // H <<= 32
	a.paddq(dst, x86::xmm14);   // H + AC
	break;
    }
    case FLOAT32: a.mulps(dst, src); break;	    
    case FLOAT64: a.mulpd(dst, src); break;
    default: break;
    }
}

static void emit_vbor(x86::Assembler &a, uint8_t type,
		      x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    (void) type;
    x86::Xmm src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_mov(a, type, dst, src1);  // dst = src1;
	src = src2;
    }
    a.por(dst, src);
}

static void emit_vband(x86::Assembler &a, uint8_t type,
		       x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    (void) type;
    x86::Xmm src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_mov(a, type, dst, src1); // dst = src1;
	src = src2;
    }
    a.pand(dst, src);    
}

static void emit_vbxor(x86::Assembler &a, uint8_t type,
		       x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    (void) type;    
    x86::Xmm src;
    if ((dst == src1) && (dst == src2)) // dst = dst + dst : dst += dst
	src = dst;
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_mov(a, type, dst, src1);  // dst = src1;
	src = src2;
    }
    a.pxor(dst, src);    
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
			x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    x86::Xmm src;
    int cmp = CMP_EQ;  // EQ
    
    if ((dst == src1) && (dst == src2)) {  // dst = dst == dst
	emit_vone(a, dst);
	return;
    }
    else if (src1 == dst) // dst = dst + src2 : dst += src2
	src = src2;
    else if (src2 == dst) // dst = src1 + dst : dst += src1
	src = src1;
    else {
	emit_mov(a, type, dst, src1); // dst = src1;
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:   a.pcmpeqb(dst, src); break;
    case INT16:	    
    case UINT16:  a.pcmpeqw(dst, src); break;
    case INT32:	    
    case UINT32:  a.pcmpeqd(dst, src); break;
    case INT64:	    
    case UINT64:  a.pcmpeqq(dst, src); break;
    case FLOAT32: a.cmpps(dst, src, cmp); break;
    case FLOAT64: a.cmppd(dst, src, cmp); break;
    default: break;
    }
}

// emit dst = src1 > src2
static void emit_vcmpgt(x86::Assembler &a, uint8_t type,
			x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    x86::Xmm src;
    int cmp = CMP_GT;

    if ((dst == src1) && (dst == src2)) { // dst = dst > dst
	emit_vzero(a, dst);
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
	emit_mov(a, type, dst, src1);
	src = src2;
    }
    switch(type) {
    case INT8:
    case UINT8:
	a.pcmpgtb(dst, src);
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);
	break;
    case INT16:
    case UINT16:
	a.pcmpgtw(dst, src);
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);	
	break;
    case INT32:
    case UINT32:
	a.pcmpgtd(dst, src);
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);	
	break;
    case INT64:	    
    case UINT64:
	a.pcmpgtq(dst, src);
	if (cmp == CMP_LT)
	    emit_vbnot(a, type, dst, dst);		
	break;
    case FLOAT32: a.cmpps(dst, src, cmp); break;
    case FLOAT64: a.cmppd(dst, src, cmp); break;
    default: break;
    }
}


// emit dst = src1 >= src2
static void emit_vcmpge(x86::Assembler &a, uint8_t type,
			x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    x86::Xmm src;

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
	    emit_mov(a, type, dst, src1); // dst = (src1 >= src2)
	    src = src2;
	}
	if (type == FLOAT32)
	    a.cmpps(dst, src, cmp);
	else 
	    a.cmppd(dst, src, cmp);
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
	    emit_mov(a, type, dst, src2); // dst = (src1 >= src2)
	    src = src1;
	}
	switch(type) {
	case INT8:
	case UINT8:   a.pcmpgtb(dst, src); break;
	case INT16:
	case UINT16:  a.pcmpgtw(dst, src); break;
	case INT32:	    
	case UINT32:  a.pcmpgtd(dst, src); break;
	case INT64:	    
	case UINT64:  a.pcmpgtq(dst, src); break;
	default: break;
	}
	emit_vbnot(a, type, dst, dst);
    }
}

static void emit_vcmplt(x86::Assembler &a, uint8_t type,
			x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    emit_vcmpgt(a, type, dst, src2, src1);
}

static void emit_vcmple(x86::Assembler &a, uint8_t type,
			x86::Xmm dst, x86::Xmm src1, x86::Xmm src2)
{
    emit_vcmpge(a, type, dst, src2, src1);
}

// Helper function to generate instructions based on type and operation
void emit_instruction(x86::Assembler &a, instr_t* optr, x86::Gp ret)
{
    x86::Xmm src1, src2, src, dst;

    src1 = vreg(optr->ri);
    src2 = vreg(optr->rj);
    dst  = vreg(optr->rd);

    switch(optr->op) {
    case OP_VRET: a.movdqu(x86::ptr(ret), src1); break;
    case OP_MOVR:   emit_vmovr(a, optr->type, dst, src1); break;
    case OP_VNEG:   emit_vneg(a, optr->type, dst, src1); break;
    case OP_VADD:   emit_vadd(a, optr->type, dst, src1, src2); break;
    case OP_VSUB:   emit_vsub(a, optr->type, dst, src1, src2); break;
    case OP_VMUL:   emit_vmul(a, optr->type, dst, src1, src2); break;
    case OP_VBNOT:  emit_vbnot(a, optr->type, dst, src1); break;	
    case OP_VBAND:  emit_vband(a, optr->type, dst, src1, src2); break;
    case OP_VBOR:   emit_vbor(a, optr->type, dst, src1, src2); break;
    case OP_VBXOR:  emit_vbxor(a, optr->type, dst, src1, src2); break;
    case OP_VCMPEQ: emit_vcmpeq(a, optr->type, dst, src1, src2); break;
    case OP_VCMPLT: emit_vcmplt(a, optr->type, dst, src1, src2); break;
    case OP_VCMPLE: emit_vcmple(a, optr->type, dst, src1, src2); break;	
    default: break;
    }
}

// add all dirty register 
void add_dirty_regs(FuncFrame &frame, instr_t* code, size_t n)
{
    while (n--) {
	if (code->op == OP_RET)
	    frame.addDirtyRegs(vreg(code->ri));
	else if (code->op & OP_BIN) {
	    frame.addDirtyRegs(vreg(code->ri));
	    frame.addDirtyRegs(vreg(code->rj));
	    frame.addDirtyRegs(vreg(code->rd));
	    if (code->op & OP_CND)
		frame.addDirtyRegs(vreg(code->rc));
	}
	else {
	    frame.addDirtyRegs(vreg(code->ri));
	    frame.addDirtyRegs(vreg(code->rd));
	    if (code->op & OP_CND)
		frame.addDirtyRegs(vreg(code->rc));
	}
	code++;
    }
}

void assemble(x86::Assembler &a, const Environment &env, instr_t* code, size_t n)
{
    FuncDetail func;
    FuncFrame frame;
    
    x86::Gp ret   = a.zax();
    x86::Gp src_a = a.zcx();
    x86::Gp src_b = a.zdx();
    
    x86::Xmm vec0 = x86::xmm0;
    x86::Xmm vec1 = x86::xmm1;
    x86::Xmm vec2 = x86::xmm2;
    
    func.init(FuncSignatureT<void, vector_t*, const vector_t*, const vector_t*>(CallConvId::kHost), env);
    frame.init(func);

    frame.addDirtyRegs(x86::xmm0);
    frame.addDirtyRegs(x86::xmm1);
    add_dirty_regs(frame, code, n);

    FuncArgsAssignment args(&func);   // Create arguments assignment context.
    args.assignAll(ret, src_a, src_b);// Assign our registers to arguments.
    args.updateFuncFrame(frame);      // Reflect our args in FuncFrame.    
    frame.finalize();                 // Finalize the FuncFrame (updates it).

    //a.emitProlog(frame);              // Emit function prolog.
    a.emitArgsAssignment(frame, args);// Assign arguments to registers.
    // TESTING
    a.movdqu(vec2, x86::ptr(ret));  // vector from [ret] to XMM2.
    // LOAD 2 arguments
    a.movdqu(vec0, x86::ptr(src_a));  // vector from [src_a] to XMM0.
    a.movdqu(vec1, x86::ptr(src_b));  // vector from [src_b] to XMM1.    
    
    // assemble all code
    while (n--) {
	emit_instruction(a, code, ret);
	code++;
    }
    a.emitEpilog(frame);              // Emit function epilog and return.
}
