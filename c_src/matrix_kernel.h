#ifndef __MATRIX_KERNEL_H__
#define __MATRIX_KERNEL_H__

#define OP_VEC   0x80  // vector version of operation
#define OP_BIN   0x40  // binary operation / else unary
#define OP_IMM   0x20  // Rj is replaced with imm8 / imm12 (mov)
#define OP_MASK  0x1F  // 1..31 BIN 1..31 BIN|VEC etc
// FIXME: make register machine 8 vector regs?

// Unary ops

#define    OP_NOP   0
#define    OP_VNOP  (OP_NOP|OP_VEC)  // same as nop

#define    OP_RET   1
#define    OP_VRET  (OP_RET|OP_VEC)  // return vector!

#define    OP_MOV  2
#define    OP_MOVI  (OP_MOV|OP_IMM)
#define    OP_VMOV  (OP_MOV|OP_VEC)
#define    OP_VMOVI  (OP_VMOV|OP_IMM)

#define    OP_NEG  3
#define    OP_VNEG  (OP_NEG|OP_VEC)

#define    OP_BNOT 4
#define    OP_VBNOT (OP_BNOT|OP_VEC)

#define    OP_INV   5
#define    OP_VINV  (OP_INV|OP_VEC)

#define    OP_JMP  (6|OP_IMM)  // imm12 (relative)
#define    OP_JNZ  (7|OP_IMM)  // imm12 (relative)
#define    OP_JZ   (8|OP_IMM)  // imm12 (relative)

// Add
#define    OP_ADD   (OP_BIN|1)
#define    OP_ADDI  (OP_ADD|OP_IMM)
#define    OP_VADD  (OP_ADD|OP_VEC)
#define    OP_VADDI (OP_VADD|OP_IMM)
// Subtract
#define    OP_SUB   (OP_BIN|2)
#define    OP_SUBI  (OP_SUB|OP_IMM)
#define    OP_VSUB  (OP_SUB|OP_VEC)
#define    OP_VSUBI (OP_VSUB|OP_IMM)

// Multiply
#define    OP_MUL   (OP_BIN|3)
#define    OP_MULI  (OP_MUL|OP_IMM)
#define    OP_VMUL  (OP_MUL|OP_VEC)
#define    OP_VMULI (OP_VMUL|OP_IMM)
// Shift Left Logical
#define    OP_SLL   (OP_BIN|4)   
#define    OP_SLLI  (OP_SLL|OP_IMM)
#define    OP_VSLL  (OP_SLL|OP_VEC)
#define    OP_VSLLI (OP_VSLL|OP_IMM)
// Shift Right Logical
#define    OP_SRL   (OP_BIN|5)   
#define    OP_SRLI  (OP_SRL|OP_IMM)
#define    OP_VSRL  (OP_SRL|OP_VEC)
#define    OP_VSRLI (OP_VSRL|OP_IMM)
// Shift Right Arithmentical
#define    OP_SRA   (OP_BIN|6)
#define    OP_SRAI  (OP_SRA|OP_IMM)
#define    OP_VSRA  (OP_SRA|OP_VEC)
#define    OP_VSRAI (OP_VSRA|OP_IMM)
// Bitwise And
#define    OP_BAND  (OP_BIN|7)
#define    OP_BANDI  (OP_BAND|OP_IMM)
#define    OP_VBAND (OP_BAND|OP_VEC)
#define    OP_VBANDI (OP_VBAND|OP_IMM)
// Bitwise Or
#define    OP_BOR   (OP_BIN|8)   
#define    OP_BORI   (OP_BOR|OP_IMM)
#define    OP_VBOR  (OP_BOR|OP_VEC)
#define    OP_VBORI  (OP_VBOR|OP_IMM)
// Bitwise Xor
#define    OP_BXOR  (OP_BIN|9)   
#define    OP_BXORI  (OP_BXOR|OP_IMM)
#define    OP_VBXOR (OP_BXOR|OP_VEC)
#define    OP_VBXORI (OP_VBXOR|OP_IMM)
// Compare Less Than
#define    OP_CMPLT (OP_BIN|10)
#define    OP_CMPLTI (OP_CMPLT|OP_IMM)
#define    OP_VCMPLT (OP_CMPLT|OP_VEC)
#define    OP_VCMPLTI (OP_VCMPLT|OP_IMM)
// Compare Less than or Equal
#define    OP_CMPLE (OP_BIN|11)
#define    OP_CMPLEI (OP_CMPLE|OP_IMM)
#define    OP_VCMPLE (OP_CMPLE|OP_VEC)
#define    OP_VCMPLEI (OP_VCMPLE|OP_IMM)
// Compare Equal
#define    OP_CMPEQ (OP_BIN|12)
#define    OP_CMPEQI (OP_CMPEQ|OP_IMM)
#define    OP_VCMPEQ (OP_CMPEQ|OP_VEC)
#define    OP_VCMPEQI (OP_VCMPEQ|OP_IMM)
// Compare Greater Than
#define    OP_CMPGT (OP_BIN|13)
#define    OP_CMPGTI (OP_CMPGT|OP_IMM)
#define    OP_VCMPGT (OP_CMPGT|OP_VEC)
#define    OP_VCMPGTI (OP_VCMPGT|OP_IMM)
// Compare Greater then or Equal
#define    OP_CMPGE (OP_BIN|14)
#define    OP_CMPGEI (OP_CMPGE|OP_IMM)
#define    OP_VCMPGE (OP_CMPGE|OP_VEC)
#define    OP_VCMPGEI (OP_VCMPGE|OP_IMM)
// Compare Not Equal
#define    OP_CMPNE (OP_BIN|15)
#define    OP_CMPNEI (OP_CMPNE|OP_IMM)
#define    OP_VCMPNE (OP_CMPNE|OP_VEC)
#define    OP_VCMPNEI (OP_VCMPNE|OP_IMM)

// Reverse Subtract
#define    OP_RSUB   (OP_BIN|16)
#define    OP_RSUBI  (OP_RSUB|OP_IMM)
#define    OP_VRSUB  (OP_RSUB|OP_VEC)
#define    OP_VRSUBI (OP_VRSUB|OP_IMM)

// base_type: 0 => UINT
// base_type: 1 => INT
// base_type: 2 => FLOAT01 (?use me)
// base_type: 3 => FLOAT

// NEW vector_size coding (3 bit!) implement me?
// vector_size: 0  => 1  (scalar!)
// vector_size: 1  => 2  (pair x,y)
// vector_size: 2  => 3  (tripe x,y,z)
// vector_size: 3  => 4  (quad x,y,z,w)
// vector_size: 4  => 8  (8 elements)
// vector_size: 5  => 16 (16 elements)
// vector_size: 6  => 32 (32 elements)
// vector_size: 7  => 64 (64 elements)


typedef struct {
    unsigned op:8;   // BIN|VEC|IMM <op>
    unsigned type:8; // type <<arr_size:3,base_exp_size:3,base_type:2>>
    unsigned rd:4;   // dst  r<d> | v<d> (return, cond jump)
    union {
	struct {
	    unsigned ri:4;   // src1 r<i> | v<i>
	    union {
		struct {
		    unsigned rj:4;   // src2 r<j> | v<j>
		    unsigned pad:4;
		};
		int8_t imm8;    // addi,subi,slli,srli,slai
	    };
	};
	int imm12:12;  // jmp,jnz,jz relative offset
    };
} instr_t;

#endif
