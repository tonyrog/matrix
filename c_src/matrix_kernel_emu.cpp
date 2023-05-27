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
#define op_cmpne(x,y) (-((x)!=(y)))
#define op_cmpgt(x,y) (-((x)>(y)))
#define op_cmpge(x,y) (-((x)>=(y)))
#define op_sll(x,y) ((x)<<(y))
#define op_srl(x,y) ((x)>>(y))
#define op_sra(x,y) ((x)>>(y))

#define FRdimm12(fld,d,imm12,op) r[d].fld = op(imm12)
#define FRdi8(fld,d,i,imm,op) r[d].fld = op(r[i].fld,(imm))
#define FRd8i(fld,d,imm,i,op) r[d].fld = op((imm),r[i].fld)
#define FRdi(fld,d,i,op) r[d].fld = op(r[i].fld)
#define FRdij(fld,d,i,j,op) r[d].fld = op(r[i].fld,r[j].fld)
#define FFRdi(ifld,ofld,d,i,op) r[d].ofld = op(r[i].ifld)
#define FFRdij(ifld,ofld,d,i,j,op) r[d].ofld = op(r[i].ifld,r[j].ifld)
#define FFRdi8(ifld,ofld,d,i,imm,op) r[d].ofld = op(r[i].ifld,(imm))

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
#define TFRdi8(t,d,i,imm,op) do {				\
	switch((t)) {						\
	case UINT8:   FRdi8(u8,(d),(i),(imm),op); break;	\
	case UINT16:  FRdi8(u16,(d),(i),(imm),op); break;	\
	case UINT32:  FRdi8(u32,(d),(i),(imm),op); break;	\
	case UINT64:  FRdi8(u64,(d),(i),(imm),op); break;	\
	case INT8:    FRdi8(i8,(d),(i),(imm),op); break;	\
	case INT16:   FRdi8(i16,(d),(i),(imm),op); break;	\
	case INT32:   FRdi8(i32,(d),(i),(imm),op); break;	\
	case INT64:   FRdi8(i64,(d),(i),(imm),op); break;	\
	default: break;						\
	}							\
    } while(0)

#define TFRd8i(t,d,imm,i,op) do {				\
	switch((t)) {						\
	case UINT8:   FRd8i(u8,(d),(imm),(i),op); break;	\
	case UINT16:  FRd8i(u16,(d),(imm),(i),op); break;	\
	case UINT32:  FRd8i(u32,(d),(imm),(i),op); break;	\
	case UINT64:  FRd8i(u64,(d),(imm),(i),op); break;	\
	case INT8:    FRd8i(i8,(d),(imm),(i),op); break;	\
	case INT16:   FRd8i(i16,(d),(imm),(i),op); break;	\
	case INT32:   FRd8i(i32,(d),(imm),(i),op); break;	\
	case INT64:   FRd8i(i64,(d),(imm),(i),op); break;	\
	default: break;						\
	}							\
    } while(0)

// SLL/SRL (logical)
#define TUFURdij(t,d,i,j,op) do {					\
    switch((t)) {							\
    case UINT8:   FRdij(u8,(d),(i),(j),op); break;			\
    case UINT16:  FRdij(u16,(d),(i),(j),op); break;			\
    case UINT32:  FRdij(u32,(d),(i),(j),op); break;			\
    case UINT64:  FRdij(u64,(d),(i),(j),op); break;			\
    case INT8:    FRdij(u8,(d),(i),(j),op); break;			\
    case INT16:   FRdij(u16,(d),(i),(j),op); break;			\
    case INT32:   FRdij(u32,(d),(i),(j),op); break;			\
    case INT64:   FRdij(u64,(d),(i),(j),op); break;			\
    default: break;							\
    }									\
    } while(0)

// SRA
#define TIFIRdij(t,d,i,j,op) do {					\
    switch((t)) {							\
    case UINT8:   FRdij(i8,(d),(i),(j),op); break;			\
    case UINT16:  FRdij(i16,(d),(i),(j),op); break;			\
    case UINT32:  FRdij(i32,(d),(i),(j),op); break;			\
    case UINT64:  FRdij(i64,(d),(i),(j),op); break;			\
    case INT8:    FRdij(i8,(d),(i),(j),op); break;			\
    case INT16:   FRdij(i16,(d),(i),(j),op); break;			\
    case INT32:   FRdij(i32,(d),(i),(j),op); break;			\
    case INT64:   FRdij(i64,(d),(i),(j),op); break;			\
    default: break;							\
    }									\
    } while(0)

// SRLI
#define TFURdi8(t,d,i,imm,op) do {				\
	switch((t)) {						\
	case UINT8:   FRdi8(u8,(d),(i),(imm),op); break;	\
	case UINT16:  FRdi8(u16,(d),(i),(imm),op); break;		\
	case UINT32:  FRdi8(u32,(d),(i),(imm),op); break;		\
	case UINT64:  FRdi8(u64,(d),(i),(imm),op); break;		\
	case INT8:    FRdi8(u8,(d),(i),(imm),op); break;		\
	case INT16:   FRdi8(u16,(d),(i),(imm),op); break;		\
	case INT32:   FRdi8(u32,(d),(i),(imm),op); break;		\
	case INT64:   FRdi8(u64,(d),(i),(imm),op); break;		\
	default: break;						\
	}							\
    } while(0)

// SRA
#define TFIRdi8(t,d,i,imm,op) do {					\
	switch((t)) {							\
	case UINT8:   FRdi8(i8,(d),(i),(imm),op); break;		\
	case UINT16:  FRdi8(i16,(d),(i),(imm),op); break;		\
	case UINT32:  FRdi8(i32,(d),(i),(imm),op); break;		\
	case UINT64:  FRdi8(i64,(d),(i),(imm),op); break;		\
	case INT8:    FRdi8(i8,(d),(i),(imm),op); break;		\
	case INT16:   FRdi8(i16,(d),(i),(imm),op); break;		\
	case INT32:   FRdi8(i32,(d),(i),(imm),op); break;		\
	case INT64:   FRdi8(i64,(d),(i),(imm),op); break;		\
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
	case FLOAT8:  FRdij(f8,(d),(i),(j),op); break;		\
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

// CMPxxI
#define srx_di8(t,d,i,imm,op) do {					\
	switch((t)) {							\
	case UINT8:   FFRdi8(u8,i8,(d),(i),((uint8_t)(imm)),op); break; \
	case UINT16:  FFRdi8(u16,i16,(d),(i),((uint16_t)(imm)),op); break; \
	case UINT32:  FFRdi8(u32,i32,(d),(i),((uint32_t)(imm)),op); break; \
	case UINT64:  FFRdi8(u64,i64,(d),(i),((uint64_t)(imm)),op); break; \
	case INT8:    FFRdi8(i8,i8,(d),(i),(imm),op); break;		\
	case INT16:   FFRdi8(i16,i16,(d),(i),(imm),op); break;		\
	case INT32:   FFRdi8(i32,i32,(d),(i),(imm),op); break;		\
	case INT64:   FFRdi8(i64,i64,(d),(i),(imm),op); break;		\
	case FLOAT16: FFRdi8(f16,i16,(d),(i),(imm),op); break;		\
	case FLOAT32: FFRdi8(f32,i32,(d),(i),(imm),op); break;		\
	case FLOAT64: FFRdi8(f64,i64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

#define KFVdimm12(fld,d,imm12,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op(imm12);			\
    } while(0)

#define KFVdi8(fld,d,i,imm,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op(v[i].fld[k],(imm));	\
    } while(0)

#define KFVd8i(fld,d,imm,i,op) do {			\
	unsigned int k;					\
	for (k=0; k<VSIZE/sizeof(v[0].fld[0]);k++)	\
	    v[d].fld[k] = op((imm),v[i].fld[k]);	\
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

#define KFFVdi8(ifld,ofld,d,i,imm,op) do {			\
	unsigned int k;						\
	for (k=0; k<VSIZE/sizeof(v[0].ofld[0]);k++)		\
	    v[d].ofld[k] = op((int)v[i].ifld[k],(imm));		\
    } while(0)

// vx - all types
// vi - integers
// vs - siged
// vu - unsiged
// vf - float
// svx - all types return signed integer (compare etc)
//
// VSLL/VSRL/VSRA
#define vi_dij(t,d,i,j,op) do {					\
    switch((t)) {							\
    case UINT8:   KFVdij(vu8,(d),(i),(j),op); break;			\
    case UINT16:  KFVdij(vu16,(d),(i),(j),op); break;			\
    case UINT32:  KFVdij(vu32,(d),(i),(j),op); break;			\
    case UINT64:  KFVdij(vu64,(d),(i),(j),op); break;			\
    case INT8:    KFVdij(vi8,(d),(i),(j),op); break;			\
    case INT16:   KFVdij(vi16,(d),(i),(j),op); break;			\
    case INT32:   KFVdij(vi32,(d),(i),(j),op); break;			\
    case INT64:   KFVdij(vi64,(d),(i),(j),op); break;			\
    default: break;							\
    }									\
    } while(0)

// VMOV/VNEG/
#define vx_di(t,d,i,op) do {					\
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

// VADD/VSUB/VRSUB/VMUL
#define vx_dij(t,d,i,j,op) do {					\
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
	case FLOAT32: KFVdimm12(vf32,(d),(imm12),op); break;		\
	case FLOAT64: KFVdimm12(vf64,(d),(imm12),op); break;		\
	default: break;							\
	}								\
    } while(0)



// VBANDI / VBORI / VBXORI / VSLLI
#define vi_di8(t,d,i,imm,op) do {					\
	switch((t)) {							\
	case UINT8:   KFVdi8(vu8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdi8(vu16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdi8(vu32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdi8(vu64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdi8(vi8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdi8(vi16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdi8(vi32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdi8(vi64,(d),(i),(imm),op); break;		\
	case FLOAT32: KFVdi8(vu32,(d),(i),(imm),op); break;		\
	case FLOAT64: KFVdi8(vu64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// VADDI / VSUBI / VMULI 
#define vx_di8(t,d,i,imm,op) do {					\
	switch((t)) {							\
	case UINT8:   KFVdi8(vu8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdi8(vu16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdi8(vu32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdi8(vu64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdi8(vi8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdi8(vi16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdi8(vi32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdi8(vi64,(d),(i),(imm),op); break;		\
	case FLOAT32: KFVdi8(vf32,(d),(i),(imm),op); break;		\
	case FLOAT64: KFVdi8(vf64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// VRSUBI
#define vx_d8i(t,d,imm,i,op) do {					\
	switch((t)) {							\
	case UINT8:   KFVd8i(vu8,(d),(imm),(i),op); break;		\
	case UINT16:  KFVd8i(vu16,(d),(imm),(i),op); break;		\
	case UINT32:  KFVd8i(vu32,(d),(imm),(i),op); break;		\
	case UINT64:  KFVd8i(vu64,(d),(imm),(i),op); break;		\
	case INT8:    KFVd8i(vi8,(d),(imm),(i),op); break;		\
	case INT16:   KFVd8i(vi16,(d),(imm),(i),op); break;		\
	case INT32:   KFVd8i(vi32,(d),(imm),(i),op); break;		\
	case INT64:   KFVd8i(vi64,(d),(imm),(i),op); break;		\
	case FLOAT32: KFVd8i(vf32,(d),(imm),(i),op); break;		\
	case FLOAT64: KFVd8i(vf64,(d),(imm),(i),op); break;		\
	default: break;							\
	}								\
    } while(0)

// VSRLI
#define TKUVdi8(t,d,i,imm,op) do { 				\
	switch((t)) {						\
	case UINT8:   KFVdi8(vu8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdi8(vu16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdi8(vu32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdi8(vu64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdi8(vu8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdi8(vu16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdi8(vu32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdi8(vu64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// VSRAI
#define TKIVdi8(t,d,i,imm,op) do { 				\
	switch((t)) {						\
	case UINT8:   KFVdi8(vi8,(d),(i),(imm),op); break;		\
	case UINT16:  KFVdi8(vi16,(d),(i),(imm),op); break;		\
	case UINT32:  KFVdi8(vi32,(d),(i),(imm),op); break;		\
	case UINT64:  KFVdi8(vi64,(d),(i),(imm),op); break;		\
	case INT8:    KFVdi8(vi8,(d),(i),(imm),op); break;		\
	case INT16:   KFVdi8(vi16,(d),(i),(imm),op); break;		\
	case INT32:   KFVdi8(vi32,(d),(i),(imm),op); break;		\
	case INT64:   KFVdi8(vi64,(d),(i),(imm),op); break;		\
	default: break;							\
	}								\
    } while(0)

// VCMPx
#define svx_dij(t,d,i,j,op) do {				\
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

// VCMPxxI
#define svx_di8(t,d,i,imm,op) do {				\
    switch((t)) {						\
    case UINT8:   KFFVdi8(vu8,vi8,(d),(i),(imm),op); break;	\
    case UINT16:  KFFVdi8(vu16,vi16,(d),(i),(imm),op); break;	\
    case UINT32:  KFFVdi8(vu32,vi32,(d),(i),(imm),op); break;	\
    case UINT64:  KFFVdi8(vu64,vi64,(d),(i),(imm),op); break;	\
    case INT8:    KFFVdi8(vi8,vi8,(d),(i),(imm),op); break;	\
    case INT16:   KFFVdi8(vi16,vi16,(d),(i),(imm),op); break;	\
    case INT32:   KFFVdi8(vi32,vi32,(d),(i),(imm),op); break;	\
    case INT64:   KFFVdi8(vi64,vi64,(d),(i),(imm),op); break;	\
    case FLOAT16: KFFVdi8(vf16,vi16,(d),(i),(imm),op); break;	\
    case FLOAT32: KFFVdi8(vf32,vi32,(d),(i),(imm),op); break;	\
    case FLOAT64: KFFVdi8(vf64,vi64,(d),(i),(imm),op); break;	\
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



void emu_vmov(uint8_t type, vscalar0_t v[16], int d, int i)
{
    vx_di(type,d,i,op_nop);
}

void emu_vmovi(uint8_t type, vscalar0_t v[16], int d, int imm12)
{
    TKVdimm12(type,d,imm12,op_nop);
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

void emu_vinv(uint8_t type, vscalar0_t v[16], int d, int i)
{
    switch(type) {
    case FLOAT16: break;
    case FLOAT32: KFVdi(vf32,d,i,op_inv); break;
    case FLOAT64: KFVdi(vf64,d,i,op_inv); break;
    default: break;
    }
}

void emu_neg(uint8_t type, scalar0_t r[16], int d, int i)
{
    TFRdi(type,d,i,op_neg);
}

void emu_vneg(uint8_t type, vscalar0_t v[16], int d, int i)
{
    vx_di(type,d,i,op_neg);
}


void emu_bnot(uint8_t type, scalar0_t r[16], int d, int i)
{
    TFURdi(type,d,i,op_bnot);
}

void emu_vbnot(uint8_t type, vscalar0_t v[16], int d, int i)
{
    (void) type;
    KFFVdi(vu64,vu64,d,i,op_bnot);
}

void emu_add(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,i,j,op_add); 
}

void emu_addi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRdi8(type,d,i,imm,op_add);
}

void emu_vaddi(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vx_di8(type,d,i,imm,op_add); 
}

void emu_vadd(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vx_dij(type,d,i,j,op_add); 
}

void emu_sub(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,i,j,op_sub); 
}

void emu_subi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRdi8(type,d,i,imm,op_sub);    
}

void emu_vsub(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vx_dij(type,d,i,j,op_sub);
}

void emu_vsubi(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vx_di8(type,d,i,imm,op_sub);    
}

// Reverse subtract

void emu_rsub(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,j,i,op_sub); 
}

void emu_rsubi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRd8i(type,d,imm,i,op_sub);
}

void emu_vrsub(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vx_dij(type,d,j,i,op_sub); 
}

void emu_vrsubi(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vx_d8i(type,d,imm,i,op_sub);
}

void emu_mul(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFRdij(type,d,i,j,op_mul); 
}

void emu_muli(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRdi8(type,d,i,imm,op_mul);
}

void emu_vmul(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vx_dij(type,d,i,j,op_mul); 
}

void emu_vmuli(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vx_di8(type,d,i,imm,op_mul); 
}

void emu_sll(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TUFURdij(type,d,i,j,op_sll);
}

void emu_slli(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFIRdi8(type,d,i,imm,op_sll);
    // TFRdi8(type,d,i,imm,op_sll);
}

void emu_vsll(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vi_dij(type,d,i,j,op_sll); 
}

void emu_vslli(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vi_di8(type,d,i,imm,op_sll);    
}

void emu_srl(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TUFURdij(type,d,i,j,op_srl);   
}

void emu_srli(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFURdi8(type,d,i,imm,op_srl);
}

void emu_vsrl(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vi_dij(type,d,i,j,op_srl); 
}

void emu_vsrli(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    TKUVdi8(type,d,i,imm,op_srl);
}

void emu_sra(uint8_t type, scalar0_t r[16], int d, int i, int j)
{    
    TIFIRdij(type,d,i,j,op_sra);
}

void emu_srai(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFIRdi8(type,d,i,imm,op_sra);
}

void emu_vsra(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    vi_dij(type,d,i,j,op_sra);     
}

void emu_vsrai(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    TKIVdi8(type,d,i,imm,op_sra);
}

void emu_bor(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFURdij(type,d,i,j,op_bor);
}


void emu_band(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFURdij(type,d,i,j,op_band);
}

void emu_bandi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRdi8(type,d,i,imm,op_band);
}

void emu_vband(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    (void) type;
    KFFVdij(vu64,vu64,d,i,j,op_band);    
}

void emu_vbandi(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vi_di8(type,d,i,imm,op_band);    
}

void emu_bori(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRdi8(type,d,i,imm,op_bor);
}

void emu_vbor(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    (void) type;
    KFFVdij(vu64,vu64,d,i,j,op_bor);
}

void emu_vbori(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vi_di8(type,d,i,imm,op_bor);    
}

void emu_bxor(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFURdij(type,d,i,j,op_bxor);
}

void emu_bxori(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    TFRdi8(type,d,i,imm,op_bxor);
}

void emu_vbxor(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    (void) type;
    KFFVdij(vu64,vu64,d,i,j,op_bxor);
}

void emu_vbxori(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{
    vi_di8(type,d,i,imm,op_bxor);    
}

void emu_cmplt(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmplt); 
}

void emu_cmplti(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    srx_di8(type,d,i,imm,op_cmplt); 
}

void emu_vcmplt(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{    
    svx_dij(type,d,i,j,op_cmplt); 
}

void emu_vcmplti(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{    
    svx_di8(type,d,i,imm,op_cmplt); 
}

void emu_cmple(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmple); 
}

void emu_cmplei(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    srx_di8(type,d,i,imm,op_cmple);
}

void emu_vcmple(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    svx_dij(type,d,i,j,op_cmple); 
}

void emu_vcmplei(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{    
    svx_di8(type,d,i,imm,op_cmple); 
}

void emu_cmpeq(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmpeq); 
}

void emu_cmpeqi(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    srx_di8(type,d,i,imm,op_cmpeq); 
}

void emu_vcmpeq(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    svx_dij(type,d,i,j,op_cmpeq); 
}

void emu_vcmpeqi(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{    
    svx_di8(type,d,i,imm,op_cmpeq); 
}

void emu_cmpgt(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmpgt); 
}

void emu_cmpgti(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    // printf("cmpgti: r[%d].u16=%u > %d\n", i, r[i].u16, (uint16_t) imm); 
    srx_di8(type,d,i,imm,op_cmpgt);
    // printf(" r[%d].i16 = %d\n", d, r[d].i16);    
}

void emu_vcmpgt(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{    
    svx_dij(type,d,i,j,op_cmpgt); 
}

void emu_vcmpgti(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{    
    svx_di8(type,d,i,imm,op_cmpgt); 
}

void emu_cmpge(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmpge); 
}

void emu_cmpgei(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    srx_di8(type,d,i,imm,op_cmpge);

}

void emu_vcmpge(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    svx_dij(type,d,i,j,op_cmpge); 
}

void emu_vcmpgei(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{    
    svx_di8(type,d,i,imm,op_cmpge); 
}

void emu_cmpne(uint8_t type, scalar0_t r[16], int d, int i, int j)
{
    TFIRdij(type,d,i,j,op_cmpne); 
}

void emu_cmpnei(uint8_t type, scalar0_t r[16], int d, int i, int8_t imm)
{
    srx_di8(type,d,i,imm,op_cmpne); 
}

void emu_vcmpne(uint8_t type, vscalar0_t v[16], int d, int i, int j)
{
    svx_dij(type,d,i,j,op_cmpne); 
}

void emu_vcmpnei(uint8_t type, vscalar0_t v[16], int d, int i, int8_t imm)
{    
    svx_di8(type,d,i,imm,op_cmpne); 
}


void emulate(scalar0_t r[16], vscalar0_t v[16],
	     instr_t* code, size_t n, int* ret)
{
    instr_t* p = code;
    // instr_t* code_end = code + n;
next:
    switch(p->op) {
    case OP_NOP:  break;
    case OP_MOVI: emu_movi(p->type, r, p->rd, p->imm12); break;	
    case OP_MOV: emu_mov(p->type, r, p->rd, p->ri); break;
    case OP_VMOV: emu_vmov(p->type, v, p->rd, p->ri); break;	
    case OP_VMOVI: emu_vmovi(p->type, v, p->rd, p->imm12); break;	
	
    case OP_NEG:  emu_neg(p->type, r, p->rd, p->ri); break;
    case OP_VNEG: emu_vneg(p->type, v, p->rd, p->ri); break;

    case OP_BNOT: emu_bnot(p->type, r, p->rd, p->ri); break;
    case OP_VBNOT: emu_vbnot(p->type, v, p->rd, p->ri); break;

    case OP_INV:  emu_inv(p->type, r, p->rd, p->ri); break;	
    case OP_VINV: emu_vinv(p->type, v, p->rd, p->ri); break;			

    case OP_ADD: emu_add(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_ADDI: emu_addi(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VADD: emu_vadd(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VADDI: emu_vaddi(p->type, v, p->rd, p->ri, p->imm8); break;

    case OP_SUB: emu_sub(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_SUBI: emu_subi(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VSUB: emu_vsub(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VSUBI: emu_vsubi(p->type, v, p->rd, p->ri, p->imm8); break;

    case OP_RSUB: emu_rsub(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_RSUBI: emu_rsubi(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VRSUB: emu_vrsub(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VRSUBI: emu_vrsubi(p->type, v, p->rd, p->ri, p->imm8); break;	
	
    case OP_MUL: emu_mul(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_MULI: emu_muli(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VMUL: emu_vmul(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VMULI: emu_vmuli(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_SLL: emu_sll(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_SLLI: emu_slli(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VSLL: emu_vsll(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VSLLI: emu_vslli(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_SRL: emu_srl(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_SRLI: emu_srli(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VSRL: emu_vsrl(p->type, v, p->rd, p->ri, p->rj); break;	
    case OP_VSRLI: emu_vsrli(p->type, v, p->rd, p->ri, p->imm8); break;		

    case OP_SRA: emu_sra(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_SRAI: emu_srai(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VSRA: emu_vsra(p->type, v, p->rd, p->ri, p->rj); break;	
    case OP_VSRAI: emu_vsrai(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_BOR: emu_bor(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_BORI: emu_bori(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VBOR: emu_vbor(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VBORI: emu_vbori(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_BAND: emu_band(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_BANDI: emu_bandi(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VBAND: emu_vband(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VBANDI: emu_vbandi(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_BXOR: emu_bxor(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_BXORI: emu_bxori(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VBXOR: emu_vbxor(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VBXORI: emu_vbxor(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_CMPLT: emu_cmplt(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_CMPLTI: emu_cmplti(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VCMPLT: emu_vcmplt(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VCMPLTI: emu_vcmplti(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_CMPLE: emu_cmple(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_CMPLEI: emu_cmplei(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VCMPLE: emu_vcmple(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VCMPLEI: emu_vcmplei(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_CMPEQ: emu_cmpeq(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_CMPEQI: emu_cmpeqi(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VCMPEQ: emu_vcmpeq(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VCMPEQI: emu_vcmpeqi(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_CMPGT: emu_cmpgt(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_CMPGTI: emu_cmpgti(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VCMPGT: emu_vcmpgt(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VCMPGTI: emu_vcmpgti(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_CMPGE: emu_cmpge(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_CMPGEI: emu_cmpgei(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VCMPGE: emu_vcmpge(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VCMPGEI: emu_vcmpgei(p->type, v, p->rd, p->ri, p->imm8); break;
	
    case OP_CMPNE: emu_cmpne(p->type, r, p->rd, p->ri, p->rj); break;
    case OP_CMPNEI: emu_cmpnei(p->type, r, p->rd, p->ri, p->imm8); break;
    case OP_VCMPNE: emu_vcmpne(p->type, v, p->rd, p->ri, p->rj); break;
    case OP_VCMPNEI: emu_vcmpnei(p->type, v, p->rd, p->ri, p->imm8); break;

    case OP_JNZ:
	switch(p->type) {
	case INT8:
	case UINT8:  if (r[p->rd].u8 == 0) goto cont; break;
	case FLOAT16:
	case INT16:
	case UINT16: if (r[p->rd].u16 == 0) goto cont; break;
	case FLOAT32:	    
	case INT32:
	case UINT32: if (r[p->rd].u32 == 0) goto cont; break;
	case FLOAT64:
	case INT64:
	case UINT64: if (r[p->rd].u64 == 0) goto cont; break;
	default: goto cont;
	}
	p += p->imm12;
	break;
    case OP_JZ:
	switch(p->type) {
	case INT8:
	case UINT8:  if (r[p->rd].u8 != 0) goto cont; break;
	case FLOAT16:
	case INT16:
	case UINT16: if (r[p->rd].u16 != 0) goto cont; break;
	case FLOAT32:	    
	case INT32:
	case UINT32: if (r[p->rd].u32 != 0) goto cont; break;
	case FLOAT64:
	case INT64:
	case UINT64: if (r[p->rd].u64 != 0) goto cont; break;
	default: goto cont;
	}
	p += p->imm12;
	break;
	
    case OP_JMP: p += p->imm12; break;
    case OP_RET: *ret = p->rd; return;
    case OP_VRET: *ret = p->rd; return;


    default: break;
    }
cont:
    p++;
    if (p == code + n)
	return;
    goto next;
}
