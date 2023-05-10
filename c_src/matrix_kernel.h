#ifndef __MATRIX_KERNEL_H__
#define __MATRIX_KERNEL_H__

#define OP_VEC   0x80  // vector version of operation
#define OP_BIN   0x40  // binary operation / else unary
#define OP_CND   0x20  // use condition register

// FIXME: make register machine 8 vector regs?

#define    OP_RET  0     // return from function
#define    OP_MOVR 1     // move register to register
#define    OP_MOVA 2     // move argument to register
#define    OP_MOVC 3     // move const to register
#define    OP_MOVI 4     // move imm:16 to register {mov.i8, rd, 17}
#define    OP_SLLI 4     // shift left logical {slli, rd, rdi, 8}
#define    OP_SRLI 5     // shift right logical {srli, rd, rdi, 8}
#define    OP_NEG  5     // negate
#define    OP_BNOT 6     // bitwise negate
#define    OP_INV  7     // recipocal

#define    OP_ADD   (OP_BIN|1)   // add
#define    OP_SUB   (OP_BIN|2)   // subtract
#define    OP_MUL   (OP_BIN|3)   // mul
#define    OP_BAND  (OP_BIN|4)   // bitwise and
#define    OP_BOR   (OP_BIN|5)   // bitwise or
#define    OP_BXOR  (OP_BIN|6)   // bitwise xor
#define    OP_CMPLT (OP_BIN|7)   // less
#define    OP_CMPLE (OP_BIN|8)   // less or equal
#define    OP_CMPEQ (OP_BIN|9)   // equal    
// unary
#define    OP_VRET  (OP_VEC|OP_RET)
#define    OP_VMOVR (OP_VEC|OP_MOVR)
#define    OP_VMOVA (OP_VEC|OP_MOVA)
#define    OP_VMOVC (OP_VEC|OP_MOVC)
#define    OP_VMOVI (OP_VEC|OP_MOVI)
#define    OP_VNEG  (OP_VEC|OP_NEG)
#define    OP_VBNOT (OP_VEC|OP_BNOT)
#define    OP_VINV  (OP_VEC|OP_INV)
//binary
#define    OP_VADD  (OP_VEC|OP_ADD)
#define    OP_VSUB  (OP_VEC|OP_SUB)
#define    OP_VMUL  (OP_VEC|OP_MUL)
#define    OP_VBAND (OP_VEC|OP_BAND)
#define    OP_VBOR  (OP_VEC|OP_BOR)
#define    OP_VBXOR (OP_VEC|OP_BXOR)
#define    OP_VCMPLT (OP_VEC|OP_CMPLT)
#define    OP_VCMPLE (OP_VEC|OP_CMPLE)
#define    OP_VCMPEQ (OP_VEC|OP_CMPEQ)

// 32bit
// base_exp_size: 0 => 2^(3+0) = 8
// base_exp_size: 1 => 2^(3+1) = 16
// base_exp_size: 2 => 2^(3+2) = 32
// base_exp_size: 3 => 2^(3+3) = 64
// base_exp_size: 4 => 2^(3+3) = 128
// base_exp_size: 5 => 2^(3+3) = 256
// base_exp_size: 6 => 2^(3+3) = 512
// base_exp_size: 7 => 2^(3+3) = 1024

// base_type: 0 => UINT
// base_type: 1 => INT
// base_type: 2 => FLOAT01 (not used)
// base_type: 3 => FLOAT

// NEW vector_size coding (3 bit!)
// vector_size: 0  => 1  (scalar!)
// vector_size: 1  => 2  (pair x,y)
// vector_size: 2  => 3  (tripe x,y,z)
// vector_size: 3  => 4  (quad x,y,z,w)
// vector_size: 4  => 8  (8 elements)
// vector_size: 5  => 16 (16 elements)
// vector_size: 6  => 32 (32 elements)
// vector_size: 7  => 64 (64 elements)


typedef struct {
    unsigned op:8;   // CND|BIN|VEC|<op>
    unsigned type:8; // type <<arr_size:3,base_exp_size:3,base_type:2>>
    unsigned ri:4;   // src1 r<i> | v<i>    
    union {
	struct {
	    unsigned rj:4;   // src2 r<j> | v<j>
	    unsigned pad:4;
	};
	int imm:8;
    };
    unsigned rd:4;   // dst  r<d> | v<d>

} instr_t;

#endif
