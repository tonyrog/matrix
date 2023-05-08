#ifndef __MATRIX_KERNEL_H__
#define __MATRIX_KERNEL_H__

#define OP_VEC   0x80  // vector version of operation
#define OP_BIN   0x40  // binary operation / else unary
#define OP_CND   0x20  // use condition register

// FIXME: make register machine 8 vector regs?

#define    OP_RET  0     // return from function
#define    OP_MOVR 1     // move register to register
#define    OP_MOVA 2     // move argument to register
#define    OP_MOVC 3     // move constant to register
#define    OP_NEG  4     // negate
#define    OP_BNOT 5     // bitwise negate
#define    OP_INV  6     // recipocal

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
typedef struct {
    unsigned op:8;   // CND|BIN|VEC|<op>
    unsigned type:8; // element type <<base_exp_size:3,base_type:2>>
    unsigned ri:4;   // src1 r<i> | v<i>
    unsigned rj:4;   // src2 r<j> | v<j>
    unsigned rd:4;   // dst  r<d> | v<d>
    unsigned rc:4;   // mask register one bit for each element (16/32) bit
} instr_t;

#endif
