//
// Emulate Matrix kernel languge
// 

#include "matrix_types.h"
#include "matrix_kernel.h"

#define op_nop(x) (x)
#define op_neg(x) (-(x))
#define op_inv(x) (1.0/(x))
#define op_add(x,y) ((x)+(y))
#define op_sub(x,y) ((x)-(y))
#define op_mul(x,y) ((x)*(y))
#define op_bnot(x) (~(x))
#define op_bor(x,y) ((x)|(y))
#define op_band(x,y) ((x)&(y))
#define op_bxor(x,y) ((x)^(y))
#define op_cmplt(x,y)  (-((x)<(y)))
#define op_cmple(x,y) (-((x)<=(y)))
#define op_cmpeq(x,y)  (-((x)==(y)))

#define EMUid(fld,i,d,op) do {				\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(r[0].fld[0]);k++)	\
	    r[d].fld[k] = op(r[i].fld[k]);		\
  } while(0)

#define EMUijd(fld,i,j,d,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(r[0].fld[0]);k++)	\
	    r[d].fld[k] = op(r[i].fld[k],r[j].fld[k]);	\
  } while(0)

#define IEMUid(ifld,ofld,i,d,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(r[0].ofld[0]);k++)	\
	    r[d].ofld[k] = op(r[i].ifld[k]);		\
  } while(0)

#define IEMUijd(ifld,ofld,i,j,d,op) do {			\
	unsigned int k;						\
	for (k=0; k<VSIZE/sizeof(r[0].ofld[0]);k++)		\
	    r[d].ofld[k] = op(r[i].ifld[k],r[j].ifld[k]);	\
    } while(0)

#define FMUid(t,i,d,op) do {					\
    switch((t)) {						\
    case UINT8:   EMUid(vu8,(i),(d),op); break;			\
    case UINT16:  EMUid(vu16,(i),(d),op); break;		\
    case UINT32:  EMUid(vu32,(i),(d),op); break;		\
    case UINT64:  EMUid(vu64,(i),(d),op); break;		\
    case INT8:    EMUid(vi8,(i),(d),op); break;			\
    case INT16:   EMUid(vi16,(i),(d),op); break;		\
    case INT32:   EMUid(vi32,(i),(d),op); break;		\
    case INT64:   EMUid(vi64,(i),(d),op); break;		\
    case FLOAT16: EMUid(vf16,(i),(d),op); break;		\
    case FLOAT32: EMUid(vf32,(i),(d),op); break;		\
    case FLOAT64: EMUid(vf64,(i),(d),op); break;		\
    default: break;						\
    }								\
  } while(0)

#define FMUijd(t,i,j,d,op) do {					\
    switch((t)) {						\
    case UINT8:   EMUijd(vu8,(i),(j),(d),op); break;		\
    case UINT16:  EMUijd(vu16,(i),(j),(d),op); break;		\
    case UINT32:  EMUijd(vu32,(i),(j),(d),op); break;		\
    case UINT64:  EMUijd(vu64,(i),(j),(d),op); break;		\
    case INT8:    EMUijd(vi8,(i),(j),(d),op); break;		\
    case INT16:   EMUijd(vi16,(i),(j),(d),op); break;		\
    case INT32:   EMUijd(vi32,(i),(j),(d),op); break;		\
    case INT64:   EMUijd(vi64,(i),(j),(d),op); break;		\
    case FLOAT16: EMUijd(vf16,(i),(j),(d),op); break;		\
    case FLOAT32: EMUijd(vf32,(i),(j),(d),op); break;		\
    case FLOAT64: EMUijd(vf64,(i),(j),(d),op); break;		\
    default: break;						\
    }								\
  } while(0)

#define IFMUid(t,i,d,op) do {					\
    switch((t)) {						\
    case UINT8:   EMUid(vu8,vi8,(i),(d),op); break;		\
    case UINT16:  EMUid(vu16,vi16,(i),(d),op); break;		\
    case UINT32:  EMUid(vu32,vi32,(i),(d),op); break;		\
    case UINT64:  EMUid(vu64,vi64,(i),(d),op); break;		\
    case INT8:    EMUid(vi8,vi8,(i),(d),op); break;		\
    case INT16:   EMUid(vi16,vi16,(i),(d),op); break;		\
    case INT32:   EMUid(vi32,vi32,(i),(d),op); break;		\
    case INT64:   EMUid(vi64,vi64,(i),(d),op); break;		\
    case FLOAT16: EMUid(vf16,vi16,(i),(d),op); break;		\
    case FLOAT32: EMUid(vf32,vi32,(i),(d),op); break;		\
    case FLOAT64: EMUid(vf64,vi64,(i),(d),op); break;		\
    default: break;						\
    }								\
  } while(0)

#define IFMUijd(t,i,j,d,op) do {				\
    switch((t)) {						\
    case UINT8:   IEMUijd(vu8,vi8,(i),(j),(d),op); break;	\
    case UINT16:  IEMUijd(vu16,vi16,(i),(j),(d),op); break;	\
    case UINT32:  IEMUijd(vu32,vi32,(i),(j),(d),op); break;	\
    case UINT64:  IEMUijd(vu64,vi64,(i),(j),(d),op); break;	\
    case INT8:    IEMUijd(vi8,vi8,(i),(j),(d),op); break;	\
    case INT16:   IEMUijd(vi16,vi16,(i),(j),(d),op); break;	\
    case INT32:   IEMUijd(vi32,vi32,(i),(j),(d),op); break;	\
    case INT64:   IEMUijd(vi64,vi64,(i),(j),(d),op); break;	\
    case FLOAT16: IEMUijd(vf16,vi16,(i),(j),(d),op); break;	\
    case FLOAT32: IEMUijd(vf32,vi32,(i),(j),(d),op); break;	\
    case FLOAT64: IEMUijd(vf64,vi64,(i),(j),(d),op); break;	\
    default: break;						\
    }								\
  } while(0)
    
void emu_vneg(uint8_t type, vscalar0_t r[16], int i, int d)
{
    FMUid(type,i,d,op_neg);
}

void emu_vmov(uint8_t type, vscalar0_t r[16], int i, int d)
{
    FMUid(type,i,d,op_nop);    
}

void emu_vinv(uint8_t type, vscalar0_t r[16], int i, int d)
{
    switch(type) {
    case FLOAT16: break;
    case FLOAT32: EMUid(vf32,i,d,op_inv); break;
    case FLOAT64: EMUid(vf32,i,d,op_inv); break;
    default: break;
    }
}

void emu_vadd(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    FMUijd(type,i,j,d,op_add); 
}

void emu_vsub(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    FMUijd(type,i,j,d,op_sub); 
}

void emu_vmul(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    FMUijd(type,i,j,d,op_mul); 
}

void emu_vbnot(uint8_t type, vscalar0_t r[16], int i, int d)
{
    (void) type;    
    EMUid(vu64,i,d,op_bnot);
}

void emu_vbor(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    (void) type;
    EMUijd(vu64,i,j,d,op_bor);  // bit operation !!!
}

void emu_vband(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    (void) type;
    EMUijd(vu64,i,j,d,op_band);  // bit operation !!!    
}

void emu_vbxor(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    (void) type;    
    EMUijd(vu32,i,j,d,op_bxor);  // bit operation !!!        
}

void emu_vcmplt(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    IFMUijd(type,i,j,d,op_cmplt); 
}

void emu_vcmple(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    IFMUijd(type,i,j,d,op_cmple); 
}

void emu_vcmpeq(uint8_t type, vscalar0_t r[16], int i, int j, int d)
{
    IFMUijd(type,i,j,d,op_cmpeq); 
}

void emulate(vscalar0_t r[16], instr_t* code, size_t n, int* ret)
{
    instr_t* pc = code;
next:
    switch(pc->op) {
    case OP_VRET: *ret = pc->ri; return;
    case OP_VNEG: emu_vneg(pc->type, r, pc->ri, pc->rd); break;
    case OP_VBNOT: emu_vbnot(pc->type, r, pc->ri, pc->rd); break;
    case OP_VINV: emu_vinv(pc->type, r, pc->ri, pc->rd); break;	
    case OP_VMOVR: emu_vmov(pc->type, r, pc->ri, pc->rd); break;
    case OP_VADD: emu_vadd(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VSUB: emu_vsub(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VMUL: emu_vmul(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VBOR: emu_vbor(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VBAND: emu_vband(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VBXOR: emu_vbxor(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VCMPEQ: emu_vcmpeq(pc->type, r, pc->ri, pc->rj, pc->rd); break;    	
    case OP_VCMPLT: emu_vcmplt(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    case OP_VCMPLE: emu_vcmple(pc->type, r, pc->ri, pc->rj, pc->rd); break;
    default: break;	
    }
    pc++;
    if (pc == code + n)
	return;
    goto next;
}
