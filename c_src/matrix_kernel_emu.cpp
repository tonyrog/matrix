//
// Emulate Matrix kernel languge
// 

#include <stdio.h>
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
#define op_cmpeq(x,y) (-((x)==(y)))
#define op_cmplt(x,y) (-((x)<(y)))
#define op_cmple(x,y) (-((x)<=(y)))
#define op_sll(x,i) ((x)<<(i))
#define op_srl(x,i) ((x)>>(i))
#define op_sra(x,i) ((x)>>(i))

#define FRdimm12(fld,d,imm12,op) r[d].fld = op(imm12)
#define FRdiimm8(fld,d,i,imm,op) r[d].fld = op(r[i].fld,imm)
#define FRdi(fld,d,i,op) r[d].fld = op(r[i].fld)
#define FRdij(fld,d,i,j,op) r[d].fld = op(r[i].fld,r[j].fld)
#define FFRdi(ifld,ofld,d,i,op) r[d].ofld = op(r[i].ifld)
#define FFRdij(ifld,ofld,d,i,j,op) r[d].ofld = op(r[i].ifld,r[j].ifld)

#define FVdi(fld,d,i,op)        v[d].fld = op(v[i].fld)
#define FVdij(fld,d,i,j,op)     v[d].fld = op(v[i].fld,v[j].fld)
#define FFVdi(ifld,ofld,d,i,op) v[d].ofld = op(v[i].ifld)
#define FFVdij(ifld,ofld,d,i,j,op) v[d].ofld = op(v[i].ifld,v[j].ifld)

#define TFRdi(t,d,i,op) do {					\
	switch((t)) {						\
	case UINT8:   FRdi(u8,(d),(i),op); break;		\
	case UINT16:  FRdi(u16,(d),(i),op); break;		\
	case UINT32:  FRdi(u32,(d),(i),op); break;		\
	case UINT64:  FRdi(u64,(d),(i),op); break;		\
	case INT8:    FRdi(i8,(d),(i),op); break;		\
	case INT16:   FRdi(i16,(d),(i),op); break;		\
	case INT32:   FRdi(i32,(d),(i),op); break;		\
	case INT64:   FRdi(i64,(d),(i),op); break;		\
	case FLOAT16: FRdi(f16,(d),(i),op); break;		\
	case FLOAT32: FRdi(f32,(d),(i),op); break;		\
	case FLOAT64: FRdi(f64,(d),(i),op); break;		\
	default: break;						\
	}							\
    } while(0)

// MOVi

#define TFRdimm12(t,d,imm12,op) do {				\
	switch((t)) {						\
	case UINT8:   FRdimm12(u8,(d),(imm12),op); break;		\
	case UINT16:  FRdimm12(u16,(d),(imm12),op); break;		\
	case UINT32:  FRdimm12(u32,(d),(imm12),op); break;		\
	case UINT64:  FRdimm12(u64,(d),(imm12),op); break;		\
	case INT8:    FRdimm12(i8,(d),(imm12),op); break;		\
	case INT16:   FRdimm12(i16,(d),(imm12),op); break;		\
	case INT32:   FRdimm12(i32,(d),(imm12),op); break;		\
	case INT64:   FRdimm12(i64,(d),(imm12),op); break;		\
	case FLOAT16: FRdimm12(f16,(d),(imm12),op); break;		\
	case FLOAT32: FRdimm12(f32,(d),(imm12),op); break;		\
	case FLOAT64: FRdimm12(f64,(d),(imm12),op); break;		\
	default: break;						\
	}							\
    } while(0)

// SLLI / ADDI / SUBI
#define TFRdiimm8(t,d,i,imm,op) do {				\
	switch((t)) {						\
	case UINT8:   FRdiimm8(u8,(d),(i),(imm),op); break;	\
	case UINT16:  FRdiimm8(u16,(d),(i),(imm),op); break;		\
	case UINT32:  FRdiimm8(u32,(d),(i),(imm),op); break;		\
	case UINT64:  FRdiimm8(u64,(d),(i),(imm),op); break;		\
	case INT8:    FRdiimm8(i8,(d),(i),(imm),op); break;		\
	case INT16:   FRdiimm8(i16,(d),(i),(imm),op); break;		\
	case INT32:   FRdiimm8(i32,(d),(i),(imm),op); break;		\
	case INT64:   FRdiimm8(i64,(d),(i),(imm),op); break;		\
	default: break;						\
	}							\
    } while(0)

// SRLI
#define TFURdiimm8(t,d,i,imm,op) do {				\
	switch((t)) {						\
	case UINT8:   FRdiimm8(u8,(d),(i),(imm),op); break;	\
	case UINT16:  FRdiimm8(u16,(d),(i),(imm),op); break;		\
	case UINT32:  FRdiimm8(u32,(d),(i),(imm),op); break;		\
	case UINT64:  FRdiimm8(u64,(d),(i),(imm),op); break;		\
	case INT8:    FRdiimm8(u8,(d),(i),(imm),op); break;		\
	case INT16:   FRdiimm8(u16,(d),(i),(imm),op); break;		\
	case INT32:   FRdiimm8(u32,(d),(i),(imm),op); break;		\
	case INT64:   FRdiimm8(u64,(d),(i),(imm),op); break;		\
	default: break;						\
	}							\
    } while(0)

// SRA
#define TFIRdiimm8(t,d,i,imm,op) do {					\
	switch((t)) {							\
	case UINT8:   FRdiimm8(i8,(d),(i),(imm),op); break;		\
	case UINT16:  FRdiimm8(i16,(d),(i),(imm),op); break;		\
	case UINT32:  FRdiimm8(i32,(d),(i),(imm),op); break;		\
	case UINT64:  FRdiimm8(i64,(d),(i),(imm),op); break;		\
	case INT8:    FRdiimm8(i8,(d),(i),(imm),op); break;		\
	case INT16:   FRdiimm8(i16,(d),(i),(imm),op); break;		\
	case INT32:   FRdiimm8(i32,(d),(i),(imm),op); break;		\
	case INT64:   FRdiimm8(i64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// ADD / SUB / MUL 
#define TFRdij(t,d,i,j,op) do {					\
	switch((t)) {						\
	case UINT8:   FRdij(u8,(d),(i),(j),op); break;		\
	case UINT16:  FRdij(u16,(d),(i),(j),op); break;		\
	case UINT32:  FRdij(u32,(d),(i),(j),op); break;		\
	case UINT64:  FRdij(u64,(d),(i),(j),op); break;		\
	case INT8:    FRdij(i8,(d),(i),(j),op); break;		\
	case INT16:   FRdij(i16,(d),(i),(j),op); break;		\
	case INT32:   FRdij(i32,(d),(i),(j),op); break;		\
	case INT64:   FRdij(i64,(d),(i),(j),op); break;		\
	case FLOAT16: FRdij(f16,(d),(i),(j),op); break;		\
	case FLOAT32: FRdij(f32,(d),(i),(j),op); break;		\
	case FLOAT64: FRdij(f64,(d),(i),(j),op); break;		\
	default: break;						\
	}							\
    } while(0)

// BNOT 
#define TFURdi(t,d,i,op) do {					\
	switch((t)) {						\
	case UINT8:   FRdi(u8,(d),(i),op); break;		\
	case UINT16:  FRdi(u16,(d),(i),op); break;		\
	case UINT32:  FRdi(u32,(d),(i),op); break;		\
	case UINT64:  FRdi(u64,(d),(i),op); break;		\
	case INT8:    FRdi(u8,(d),(i),op); break;		\
	case INT16:   FRdi(u16,(d),(i),op); break;		\
	case INT32:   FRdi(u32,(d),(i),op); break;		\
	case INT64:   FRdi(u64,(d),(i),op); break;		\
	case FLOAT16: FRdi(u16,(d),(i),op); break;		\
	case FLOAT32: FRdi(u32,(d),(i),op); break;		\
	case FLOAT64: FRdi(u64,(d),(i),op); break;		\
	default: break;						\
	}							\
    } while(0)

// BOR / BAND / BXOR
#define TFURdij(t,d,i,j,op) do {				\
    switch((t)) {						\
    case UINT8:   FRdij(u8,(d),(i),(j),op); break;		\
    case UINT16:  FRdij(u16,(d),(i),(j),op); break;		\
    case UINT32:  FRdij(u32,(d),(i),(j),op); break;		\
    case UINT64:  FRdij(u64,(d),(i),(j),op); break;		\
    case INT8:    FRdij(u8,(d),(i),(j),op); break;		\
    case INT16:   FRdij(u16,(d),(i),(j),op); break;		\
    case INT32:   FRdij(u32,(d),(i),(j),op); break;		\
    case INT64:   FRdij(u64,(d),(i),(j),op); break;		\
    case FLOAT16: FRdij(u16,(d),(i),(j),op); break;		\
    case FLOAT32: FRdij(u32,(d),(i),(j),op); break;		\
    case FLOAT64: FRdij(u64,(d),(i),(j),op); break;		\
    default: break;						\
    }								\
  } while(0)

// CMPLT / CMPLE / CMPEQ
#define TFIRdij(t,d,i,j,op) do {					\
	switch((t)) {							\
	case UINT8:   FFRdij(u8,i8,(d),(i),(j),op); break;		\
	case UINT16:  FFRdij(u16,i16,(d),(i),(j),op); break;		\
	case UINT32:  FFRdij(u32,i32,(d),(i),(j),op); break;		\
	case UINT64:  FFRdij(u64,i64,(d),(i),(j),op); break;		\
	case INT8:    FFRdij(i8,i8,(d),(i),(j),op); break;		\
	case INT16:   FFRdij(i16,i16,(d),(i),(j),op); break;		\
	case INT32:   FFRdij(i32,i32,(d),(i),(j),op); break;		\
	case INT64:   FFRdij(i64,i64,(d),(i),(j),op); break;		\
	case FLOAT16: FFRdij(f16,i16,(d),(i),(j),op); break;		\
	case FLOAT32: FFRdij(f32,i32,(d),(i),(j),op); break;		\
	case FLOAT64: FFRdij(f64,i64,(d),(i),(j),op); break;		\
	default: break;							\
	}								\
    } while(0)

#define KFVdimm12(fld,d,imm12,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op(imm12);			\
    } while(0)

#define KFVdiimm8(fld,d,i,imm,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op(v[i].fld[k],imm);		\
    } while(0)

#define KFVdi(fld,d,i,op) do {				\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op(v[i].fld[k]);		\
    } while(0)

#define KFVdij(fld,d,i,j,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op(v[i].fld[k],v[j].fld[k]);	\
    } while(0)

#define KFFVdi(ifld,ofld,d,i,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].ofld[0]);k++)	\
	    v[d].ofld[k] = op(v[i].ifld[k]);		\
  } while(0)

#define KFFVdij(ifld,ofld,d,i,j,op) do {			\
	unsigned int k;						\
	for (k=0; k<VSIZE/sizeof(v[0].ofld[0]);k++)		\
	    v[d].ofld[k] = op(v[i].ifld[k],v[j].ifld[k]);	\
    } while(0)

#define TKVdi(t,d,i,op) do {					\
    switch((t)) {						\
    case UINT8:   KFVdi(vu8,(d),(i),op); break;			\
    case UINT16:  KFVdi(vu16,(d),(i),op); break;		\
    case UINT32:  KFVdi(vu32,(d),(i),op); break;		\
    case UINT64:  KFVdi(vu64,(d),(i),op); break;		\
    case INT8:    KFVdi(vi8,(d),(i),op); break;			\
    case INT16:   KFVdi(vi16,(d),(i),op); break;		\
    case INT32:   KFVdi(vi32,(d),(i),op); break;		\
    case INT64:   KFVdi(vi64,(d),(i),op); break;		\
    case FLOAT16: KFVdi(vf16,(d),(i),op); break;		\
    case FLOAT32: KFVdi(vf32,(d),(i),op); break;		\
    case FLOAT64: KFVdi(vf64,(d),(i),op); break;		\
    default: break;						\
    }								\
  } while(0)

#define TKVdij(t,d,i,j,op) do {					\
    switch((t)) {						\
    case UINT8:   KFVdij(vu8,(d),(i),(j),op); break;		\
    case UINT16:  KFVdij(vu16,(d),(i),(j),op); break;		\
    case UINT32:  KFVdij(vu32,(d),(i),(j),op); break;		\
    case UINT64:  KFVdij(vu64,(d),(i),(j),op); break;		\
    case INT8:    KFVdij(vi8,(d),(i),(j),op); break;		\
    case INT16:   KFVdij(vi16,(d),(i),(j),op); break;		\
    case INT32:   KFVdij(vi32,(d),(i),(j),op); break;		\
    case INT64:   KFVdij(vi64,(d),(i),(j),op); break;		\
    case FLOAT16: KFVdij(vf16,(d),(i),(j),op); break;		\
    case FLOAT32: KFVdij(vf32,(d),(i),(j),op); break;		\
    case FLOAT64: KFVdij(vf64,(d),(i),(j),op); break;		\
    default: break;						\
    }								\
  } while(0)

// VMOVI
#define TKVdimm12(t,d,imm12,op) do {					\
	switch((t)) {							\
	case UINT8:   KFVdimm12(vu8,(d),(imm12),op); break;		\
	case UINT16:  KFVdimm12(vu16,(d),(imm12),op); break;		\
	case UINT32:  KFVdimm12(vu32,(d),(imm12),op); break;		\
	case UINT64:  KFVdimm12(vu64,(d),(imm12),op); break;		\
	case INT8:    KFVdimm12(vi8,(d),(imm12),op); break;		\
	case INT16:   KFVdimm12(vi16,(d),(imm12),op); break;		\
	case INT32:   KFVdimm12(vi32,(d),(imm12),op); break;		\
	case INT64:   KFVdimm12(vi64,(d),(imm12),op); break;		\
	default: break;							\
	}								\
    } while(0)


// VSLLI /  VADDI / VSUBI
#define TKVdiimm8(t,d,i,imm,op) do {					\
	switch((t)) {							\
	case UINT8:   KFVdiimm8(vu8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdiimm8(vu16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdiimm8(vu32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdiimm8(vu64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdiimm8(vi8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdiimm8(vi16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdiimm8(vi32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdiimm8(vi64,(d),(i),(imm),op); break;		\
	default: break;						\
	}							\
    } while(0)

// VSRLI
#define TKUVdiimm8(t,d,i,imm,op) do { 				\
	switch((t)) {						\
	case UINT8:   KFVdiimm8(vu8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdiimm8(vu16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdiimm8(vu32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdiimm8(vu64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdiimm8(vu8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdiimm8(vu16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdiimm8(vu32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdiimm8(vu64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// VSRAI
#define TKIVdiimm8(t,d,i,imm,op) do { 				\
	switch((t)) {						\
	case UINT8:   KFVdiimm8(vi8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdiimm8(vi16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdiimm8(vi32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdiimm8(vi64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdiimm8(vi8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdiimm8(vi16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdiimm8(vi32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdiimm8(vi64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// ????
#define TKIVdi(t,d,i,op) do {					\
    switch((t)) {						\
    case UINT8:   KFFVdi(vu8,vi8,(d),(i),op); break;		\
    case UINT16:  KFFVdi(vu16,vi16,(d),(i),op); break;		\
    case UINT32:  KFFVdi(vu32,vi32,(d),(i),op); break;		\
    case UINT64:  KFFVdi(vu64,vi64,(d),(i),op); break;		\
    case INT8:    KFFVdi(vi8,vi8,(d),(i),op); break;		\
    case INT16:   KFFVdi(vi16,vi16,(d),(i),op); break;		\
    case INT32:   KFFVdi(vi32,vi32,(d),(i),op); break;		\
    case INT64:   KFFVdi(vi64,vi64,(d),(i),op); break;		\
    case FLOAT16: KFFVdi(vf16,vi16,(d),(i),op); break;		\
    case FLOAT32: KFFVdi(vf32,vi32,(d),(i),op); break;		\
    case FLOAT64: KFFVdi(vf64,vi64,(d),(i),op); break;		\
    default: break;						\
    }								\
  } while(0)

// VCMPLT / VCMPLE / VCMPEQ 
#define TKIVdij(t,d,i,j,op) do {				\
    switch((t)) {						\
    case UINT8:   KFFVdij(vu8,vi8,(d),(i),(j),op); break;	\
    case UINT16:  KFFVdij(vu16,vi16,(d),(i),(j),op); break;	\
    case UINT32:  KFFVdij(vu32,vi32,(d),(i),(j),op); break;	\
    case UINT64:  KFFVdij(vu64,vi64,(d),(i),(j),op); break;	\
    case INT8:    KFFVdij(vi8,vi8,(d),(i),(j),op); break;	\
    case INT16:   KFFVdij(vi16,vi16,(d),(i),(j),op); break;	\
    case INT32:   KFFVdij(vi32,vi32,(d),(i),(j),op); break;	\
    case INT64:   KFFVdij(vi64,vi64,(d),(i),(j),op); break;	\
    case FLOAT16: KFFVdij(vf16,vi16,(d),(i),(j),op); break;	\
    case FLOAT32: KFFVdij(vf32,vi32,(d),(i),(j),op); break;	\
    case FLOAT64: KFFVdij(vf64,vi64,(d),(i),(j),op); break;	\
    default: break;						\
    }								\
  } while(0)


void emu_mov(uint8_t type, scalar0_t r[16], int d, int i)
{
    TFRdi(type,d,i,op_nop);
}

void emu_movi(uint8_t type, scalar0_t r[16], int d, int16_t imm12)
{
    TFRdimm12(type,d,imm12,op_nop);
}

void emu_addi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm8)
{
    TFRdiimm8(type,d,i,imm8,op_add);
}

void emu_subi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm8)
{
    TFRdiimm8(type,d,i,imm8,op_sub);    
}

void emu_slli(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm8)
{
    TFRdiimm8(type,d,i,imm8,op_sll);
}

void emu_srli(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm8)
{
    TFURdiimm8(type,d,i,imm8,op_srl);
}

void emu_srai(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm8)
{
    TFIRdiimm8(type,d,i,imm8,op_sra);
}

void emu_neg(uint8_t type, scalar0_t r[16], int d, int i)
{
    TFRdi(type,d,i,op_neg);
}

void emu_inv(uint8_t type, scalar0_t r[16], int i, int d)
{
    switch(type) {
    case FLOAT16: break;
    case FLOAT32: FRdi(f32,d,i,op_inv); break;
    case FLOAT64: FRdi(f64,d,i,op_inv); break;
    default: break;
    }
}

void emu_add(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,i,j,op_add); 
}

void emu_sub(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,i,j,op_sub); 
}

void emu_mul(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,i,j,op_mul); 
}

void emu_bnot(uint8_t type, scalar0_t r[16], int d, int i)
{
    TFURdi(type,d,i,op_bnot);
}

void emu_bor(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFURdij(type,d,i,j,op_bor);
}

void emu_band(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFURdij(type,d,i,j,op_band);
}

void emu_bxor(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFURdij(type,d,i,j,op_bxor);
}

void emu_cmplt(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmplt); 
}

void emu_cmple(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmple); 
}

void emu_cmpeq(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmpeq); 
}

// Vector version

void emu_vneg(uint8_t type, vscalar0_t v[16], int d, int i)
{
    TKVdi(type,d,i,op_neg);
}

void emu_vmov(uint8_t type, vscalar0_t v[16], int d, int i)
{
    TKVdi(type,d,i,op_nop);
}

void emu_vinv(uint8_t type, vscalar0_t v[16], int d, int i)
{
    switch(type) {
    case FLOAT16: break;
    case FLOAT32: KFVdi(vf32,d,i,op_inv); break;
    case FLOAT64: KFVdi(vf64,d,i,op_inv); break;
    default: break;
    }
}

void emu_vmovi(uint8_t type, vscalar0_t v[16], int d, int imm12)
{
    TKVdimm12(type,d,imm12,op_nop);
}

void emu_vaddi(uint8_t type, vscalar0_t v[16], int d, int i, int imm8)
{
    TKVdiimm8(type,d,i,imm8,op_add); 
}

void emu_vsubi(uint8_t type, vscalar0_t v[16], int d, int i, int imm8)
{
    TKVdiimm8(type,d,i,imm8,op_sub);    
}

void emu_vslli(uint8_t type, vscalar0_t v[16], int d, int i, int imm8)
{
    TKVdiimm8(type,d,i,imm8,op_sll);    
}

void emu_vsrli(uint8_t type, vscalar0_t v[16], int d, int i, int imm8)
{
    TKUVdiimm8(type,d,i,imm8,op_srl);
}

void emu_vsrai(uint8_t type, vscalar0_t v[16], int d, int i, int imm8)
{
    TKIVdiimm8(type,d,i,imm8,op_sra);
}


void emu_vadd(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    TKVdij(type,d,i,j,op_add); 
}

void emu_vsub(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    TKVdij(type,d,i,j,op_sub); 
}

void emu_vmul(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    TKVdij(type,d,i,j,op_mul); 
}

void emu_vbnot(uint8_t type, vscalar0_t v[16], int d, int i)
{
    (void) type;
    KFFVdi(vu64,vu64,d,i,op_bnot);
}

void emu_vbor(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    (void) type;
    KFFVdij(vu64,vu64,d,i,j,op_bor);
}

void emu_vband(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    (void) type;
    KFFVdij(vu64,vu64,d,i,j,op_band);    
}

void emu_vbxor(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    (void) type;
    KFFVdij(vu64,vu64,d,i,j,op_bxor);
}

void emu_vcmplt(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{    
    TKIVdij(type,d,i,j,op_cmplt); 
}

void emu_vcmple(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    TKIVdij(type,d,i,j,op_cmple); 
}

void emu_vcmpeq(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    TKIVdij(type,d,i,j,op_cmpeq); 
}

void emulate(scalar0_t r[16], vscalar0_t v[16],
	     instr_t* code, size_t n, int* ret)
{
    instr_t* pc = code;
    // instr_t* code_end = code + n;
next:
    switch(pc->op) {
    case OP_NOP:  break;
    case OP_NEG: emu_neg(pc->type, r, pc->rd, pc->ri); break;
    case OP_BNOT: emu_bnot(pc->type, r, pc->rd, pc->ri); break;
    case OP_INV: emu_inv(pc->type, r, pc->rd, pc->ri); break;	
    case OP_MOVR: emu_mov(pc->type, r, pc->rd, pc->ri); break;
    case OP_MOVI: emu_movi(pc->type, r, pc->rd, pc->imm12); break;
    case OP_ADDI: emu_addi(pc->type, r, pc->rd, pc->ri, pc->imm8); break;
    case OP_SUBI: emu_subi(pc->type, r, pc->rd, pc->ri, pc->imm8); break;
    case OP_SLLI: emu_slli(pc->type, r, pc->rd, pc->ri, pc->imm8); break;	
    case OP_SRLI: emu_srli(pc->type, r, pc->rd, pc->ri, pc->imm8); break;
    case OP_SRAI: emu_srai(pc->type, r, pc->rd, pc->ri, pc->imm8); break;
    case OP_ADD: emu_add(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_SUB: emu_sub(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_MUL: emu_mul(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_BOR: emu_bor(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_BAND: emu_band(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_BXOR: emu_bxor(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_CMPEQ: emu_cmpeq(pc->type, r, pc->rd, pc->ri, pc->rj); break;    	
    case OP_CMPLT: emu_cmplt(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_CMPLE: emu_cmple(pc->type, r, pc->rd, pc->ri, pc->rj); break;
    case OP_JNZ:
	switch(pc->type) {
	case INT8:
	case UINT8:  if (r[pc->rd].u8 == 0) goto cont; break;
	case FLOAT16:
	case INT16:
	case UINT16: if (r[pc->rd].u16 == 0) goto cont; break;
	case FLOAT32:	    
	case INT32:
	case UINT32: if (r[pc->rd].u32 == 0) goto cont; break;
	case FLOAT64:
	case INT64:
	case UINT64: if (r[pc->rd].u64 == 0) goto cont; break;
	default: goto cont;
	}
	pc += pc->imm12;
	break;
    case OP_JZ:
	switch(pc->type) {
	case INT8:
	case UINT8:  if (r[pc->rd].u8 != 0) goto cont; break;
	case FLOAT16:
	case INT16:
	case UINT16: if (r[pc->rd].u16 != 0) goto cont; break;
	case FLOAT32:	    
	case INT32:
	case UINT32: if (r[pc->rd].u32 != 0) goto cont; break;
	case FLOAT64:
	case INT64:
	case UINT64: if (r[pc->rd].u64 != 0) goto cont; break;
	default: goto cont;
	}
	pc += pc->imm12;
	break;	
    case OP_JMP: pc += pc->imm12; break;
	
    case OP_RET: *ret = pc->rd; return;

    case OP_VRET: *ret = pc->rd; return;

    case OP_VMOVI: emu_vmovi(pc->type, v, pc->rd, pc->imm12); break;	
    case OP_VADDI: emu_vaddi(pc->type, v, pc->rd, pc->ri, pc->imm8); break;
    case OP_VSUBI: emu_vsubi(pc->type, v, pc->rd, pc->ri, pc->imm8); break;
    case OP_VSLLI: emu_vslli(pc->type, v, pc->rd, pc->ri, pc->imm8); break;	
    case OP_VSRLI: emu_vsrli(pc->type, v, pc->rd, pc->ri, pc->imm8); break;
    case OP_VSRAI: emu_vsrai(pc->type, v, pc->rd, pc->ri, pc->imm8); break;
	
    case OP_VNEG: emu_vneg(pc->type, v, pc->rd, pc->ri); break;
    case OP_VBNOT: emu_vbnot(pc->type, v, pc->rd, pc->ri); break;
    case OP_VINV: emu_vinv(pc->type, v, pc->rd, pc->ri); break;	
    case OP_VMOVR: emu_vmov(pc->type, v, pc->rd, pc->ri); break;
    case OP_VADD: emu_vadd(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VSUB: emu_vsub(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VMUL: emu_vmul(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VBOR: emu_vbor(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VBAND: emu_vband(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VBXOR: emu_vbxor(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VCMPEQ: emu_vcmpeq(pc->type, v, pc->rd, pc->ri, pc->rj); break;    	
    case OP_VCMPLT: emu_vcmplt(pc->type, v, pc->rd, pc->ri, pc->rj); break;
    case OP_VCMPLE: emu_vcmple(pc->type, v, pc->rd, pc->ri, pc->rj); break;

    default: break;
    }
cont:
    pc++;
    if (pc == code + n)
	return;
    goto next;
}
