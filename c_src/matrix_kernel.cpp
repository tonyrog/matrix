// Test of matrix kernel

#include <asmjit/x86.h>
#include <iostream>

using namespace asmjit;

#include "matrix_types.h"
#include "matrix_kernel.h"

extern void assemble(x86::Assembler &a, const Environment &env,
		     x86::Reg dst, x86::Reg src1, x86::Reg src2,
		     instr_t* code, size_t n);
extern void emulate(scalar0_t r[16], vscalar0_t v[16], instr_t* code,
		    size_t n, int* ret);

extern void sprint(FILE* f,uint8_t type, scalar0_t v);
extern int  scmp(uint8_t type, scalar0_t v1, scalar0_t v2);

extern x86::Vec vreg(int i);
extern x86::Gp reg(int i);


extern void vprint(FILE* f, uint8_t type, vector_t v);
extern int  vcmp(uint8_t type, vector_t v1, vector_t v2);

extern void set_element_int64(matrix_type_t type, vector_t &r, int i, int64_t v);
extern void set_element_float64(matrix_type_t type, vector_t &r, int i, float64_t v);

#define OPdij(o,d,i,j) \
    {.op = (o),.type=INT64,.rd=(d),.rj=(j),.pad=0,.ri=(i)}
#define OPdi(o,d,i) \
    {.op = (o),.type=INT64,.rd=(d),.rj=0,.pad=0,.ri=(i),}
#define OPd(o,d) \
    {.op = (o),.type=INT64,.rd=(d),.rj=0,.pad=0,.ri=0,}
#define OPdiimm8(o,d,i,imm)				\
    {.op = (o),.type=INT64,.rd=(d),.imm8=(imm),.ri=(i)}

#define OPimm12d(o,d,rel)				\
    {.op = (o),.type=INT64,.rd=(d),.imm12=(rel)}
#define OPimm12(o,rel)					\
    {.op = (o),.type=INT64,.rd=(0),.imm12=(rel)}

static const char* asm_opname(uint8_t op)
{
    switch(op) {
    case OP_NOP:  return "nop";
    case OP_JMP:  return "jmp";
    case OP_JNZ:  return "jnz";
    case OP_JZ:   return "jz";		
    case OP_RET:  return "ret";
    case OP_MOVR:  return "mov";
    case OP_MOVI:  return "movi";
    case OP_ADDI:  return "addi";
    case OP_SUBI:  return "subi";
    case OP_SLLI:  return "slli";
    case OP_SRLI:  return "srli";
    case OP_SRAI:  return "srai";		
    case OP_NEG:  return "neg";
    case OP_BNOT:  return "bnot";
    case OP_INV:  return "inv";
    case OP_ADD:  return "add";
    case OP_SUB:  return "sub";
    case OP_MUL:  return "mul";
    case OP_BAND:  return "band";
    case OP_BOR:  return "bor";
    case OP_BXOR:  return "bxor";
    case OP_CMPLT:  return "cmplt";
    case OP_CMPLE:  return "cmple";
    case OP_CMPEQ:  return "cmpeq";
    case OP_VRET:  return "vret";
    case OP_VMOVR:  return "vmov";
    case OP_VMOVI:  return "vmovi";	
    case OP_VNEG:  return "vneg";	
    case OP_VBNOT:  return "vbnot";
    case OP_VADDI:  return "vaddi";
    case OP_VSUBI:  return "vsubi";
    case OP_VSLLI:  return "vslli";
    case OP_VSRLI:  return "vsrli";
    case OP_VSRAI:  return "vsrai";
    case OP_VINV:  return "vinv";
    case OP_VADD:  return "vadd";
    case OP_VSUB:  return "vsub";
    case OP_VMUL:  return "vmul";
    case OP_VBAND:  return "vband";
    case OP_VBOR:  return "vbor";
    case OP_VBXOR:  return "vbxor";
    case OP_VCMPLT:  return "vcmplt";
    case OP_VCMPLE:  return "vcmple";
    case OP_VCMPEQ:  return "vcmpeq";
    default: return "?????";
    }
}

static const char* asm_typename(uint8_t type)
{
    switch(type) {
    case UINT8:  return "u8"; 
    case UINT16: return "u16"; 
    case UINT32: return "u32"; 
    case UINT64: return "u64"; 
    case INT8:   return "i8"; 
    case INT16:  return "i16"; 
    case INT32:  return "i32";  
    case INT64:  return "i64"; 
    case FLOAT16: return "f16"; 
    case FLOAT32: return "f32"; 
    case FLOAT64: return "f64";
    default: return "??";
    }
}

static const char* asm_regname(uint8_t op, uint8_t r)
{
    if (op & OP_VEC) {
	switch(r) {
	case 0: return "v0";
	case 1: return "v1";
	case 2: return "v2";
	case 3: return "v3";
	case 4: return "v4";
	case 5: return "v5";
	case 6: return "v6";
	case 7: return "v7";
	case 8: return "v8";
	case 9: return "v9";
	case 10: return "v10";
	case 11: return "v11";
	case 12: return "v12";
	case 13: return "v13";
	case 14: return "v14";
	case 15: return "v15";
	default: return "v?";
	}
    }
    else {
	switch(r) {
	case 0: return "r0";
	case 1: return "r1";
	case 2: return "r2";
	case 3: return "r3";
	case 4: return "r4";
	case 5: return "r5";
	case 6: return "r6";
	case 7: return "r7";
	case 8: return "r8";
	case 9: return "r9";
	case 10: return "r10";
	case 11: return "r11";
	case 12: return "r12";
	case 13: return "r13";
	case 14: return "r14";
	case 15: return "r15";
	default: return "r?";
	}
    }
}

// set all instructions to the same type!
void set_type(uint8_t type, instr_t* code, size_t n)
{
    while (n--) {
	code->type = type;
	code++;
    }
}

size_t code_size(instr_t* code)
{
    size_t len = 0;
    
    while((code->op != OP_RET) && (code->op != OP_VRET)) {
	len++;
	code++;
    }
    return len+1;
}

typedef void (*vecfun2_t)(vector_t* dst, const vector_t* a, const vector_t* b);
typedef void (*fun2_t)(scalar0_t* dst, const scalar0_t* a, const scalar0_t* b);


int64_t value_u8_0[16] =
{ 0,1,2,5,10,20,50,70,127,128,130,150,180,253,254,255};
int64_t value_u8_1[16] =
{ 0,2,2,100,20,30,30,71,8,128,2,255,180,254,20,9};
int64_t value_u8_2[16] =
{ 0,3,4,200,10,150,200,2,3,128,170,3,5,8};

int64_t value_i8_0[16] =
{ 0,1,2,5,10,20,50,70,127,-128,-126,-106,-56,-3,-2,-1};
int64_t value_i8_1[16] =
{ 0,2,2,100,20,30,30,71,8,128,2,255,-76,-3,20,9};
int64_t value_i8_2[16] =
{ 0,3,4,-56,10,-106,-56,2,3,-128,-86,100,3,5,8};

int64_t value_u16_0[8] =
{ 0,2,50,32767,1300,180,253,65535};
int64_t value_u16_1[8] =
{ 2,1000,71,128,65535,100,20,9};
int64_t value_u16_2[8] =
{ 3,200,1500,2,128,100,5, 8};

int64_t value_i16_0[8] =
{ -32768,-1260,-106,-76,-56,-3,-2,-1};
int64_t value_i16_1[8] =
{ 0,2,2,1000,200,30,71,8 };
int64_t value_i16_2[8] =
{ 0,4,-56,100,-32768,3,5,8};

int64_t value_u32_0[4] =
{ 0,500,130,2530 };
int64_t value_u32_1[4] =
{ 100,12800,100,9};
int64_t value_u32_2[4] =
{ 200,2,1000, 8};

int64_t value_i32_0[4] =
{ -106,-3,-2,-1};
int64_t value_i32_1[4] =
{ 20,30,71,8 };
int64_t value_i32_2[4] =
{ 0,-56,10,-32768};

int64_t value_u64_0[2] =
{ 130,2530 };
int64_t value_u64_1[2] =
{ 12800,100};
int64_t value_u64_2[2] =
{ 2,1000 };

int64_t value_i64_0[2] =
{ -106,-56};
int64_t value_i64_1[2] =
{ 20,30 };
int64_t value_i64_2[2] =
{ -56-32768};

    
int setup_vinput(matrix_type_t type, vector_t& a0, vector_t& a1, vector_t& a2)
{
    int i;    
    switch(type) {
    case UINT8:
	for (i = 0; i < 16; i++) {
	    set_element_int64(type, a0, i, value_u8_0[i]);
	    set_element_int64(type, a1, i, value_u8_1[i]);
	    set_element_int64(type, a2, i, value_u8_2[i]);
	}
	break;
    case INT8:
	for (i = 0; i < 16; i++) {
	    set_element_int64(type, a0, i, value_i8_0[i]);
	    set_element_int64(type, a1, i, value_i8_1[i]);
	    set_element_int64(type, a2, i, value_i8_2[i]);
	}
	break;
    case UINT16:
	for (i = 0; i < 8; i++) {
	    set_element_int64(type, a0, i, value_u16_0[i]);
	    set_element_int64(type, a1, i, value_u16_1[i]);
	    set_element_int64(type, a2, i, value_u16_2[i]);
	}
	break;
    case INT16:
	for (i = 0; i < 8; i++) {
	    set_element_int64(type, a0, i, value_i16_0[i]);
	    set_element_int64(type, a1, i, value_i16_1[i]);
	    set_element_int64(type, a2, i, value_i16_2[i]);	    
	}
	break;
    case UINT32:
	for (i = 0; i < 4; i++) {
	    set_element_int64(type, a0, i, value_u32_0[i]);
	    set_element_int64(type, a1, i, value_u32_1[i]);
	    set_element_int64(type, a2, i, value_u32_2[i]);	    
	}
	break;
    case FLOAT32:
    case INT32:
	for (i = 0; i < 4; i++) {
	    set_element_int64(type, a0, i, value_i32_0[i]);
	    set_element_int64(type, a1, i, value_i32_1[i]);
	    set_element_int64(type, a2, i, value_i32_2[i]);	    	    
	}
	break;
    case UINT64:
	for (i = 0; i < 2; i++) {
	    set_element_int64(type, a0, i, value_u64_0[i]);
	    set_element_int64(type, a1, i, value_u64_1[i]);
	    set_element_int64(type, a2, i, value_u64_2[i]);	    	    
	}
	break;
    case INT64:
    case FLOAT64:
	for (i = 0; i < 2; i++) {
	    set_element_int64(type, a0, i, value_i64_0[i]);
	    set_element_int64(type, a1, i, value_i64_1[i]);
	    set_element_int64(type, a2, i, value_i64_2[i]);
	}
	break;
    default:
	break;
    }
    return 0;
}


int setup_input(matrix_type_t type, int i, scalar0_t& a0, scalar0_t& a1, scalar0_t& a2)
{
    switch(type) {
    case UINT8:
	a0.u8 = value_u8_0[i & 0xf];
	a1.u8 = value_u8_1[i & 0xf];
	a2.u8 = value_u8_2[i & 0xf];
	break;
    case INT8:
	a0.i8 = value_i8_0[i & 0xf];
	a1.i8 = value_i8_1[i & 0xf];
	a2.i8 = value_i8_2[i & 0xf];
	break;
    case UINT16:
	a0.u16 = value_u16_0[i & 0x7];
	a1.u16 = value_u16_1[i & 0x7];
	a2.u16 = value_u16_2[i & 0x7];
	break;
    case INT16:
	a0.i16 = value_i16_0[i & 0x7];
	a1.i16 = value_i16_1[i & 0x7];
	a2.i16 = value_i16_2[i & 0x7];
	break;
    case UINT32:
	a0.u32 = value_u32_0[i & 0x3];
	a1.u32 = value_u32_1[i & 0x3];
	a2.u32 = value_u32_2[i & 0x3];
	break;
    case INT32:
	a0.i32 = value_i32_0[i & 0x3];
	a1.i32 = value_i32_1[i & 0x3];
	a2.i32 = value_i32_2[i & 0x3];
	break;
    case UINT64:
	a0.u64 = value_u64_0[i & 0x1];
	a1.u64 = value_u64_1[i & 0x1];
	a2.u64 = value_u64_2[i & 0x1];
	break;
    case INT64:
	a0.i64 = value_i64_0[i & 0x1];
	a1.i64 = value_i64_1[i & 0x1];
	a2.i64 = value_i64_2[i & 0x1];
	break;
    default:
	break;
    }
    return 0;
}


void print_instr(FILE* f,instr_t* pc)
{
    switch(pc->op) {
    case OP_JMP:
	fprintf(f, "%s %d", asm_opname(pc->op), pc->imm12);
	break;
    case OP_RET:
	fprintf(f, "%s.%s %s",
		asm_opname(pc->op),
		asm_typename(pc->type),
		asm_regname(pc->op,pc->rd));
	break;
    case OP_VADDI:
    case OP_VSUBI:
    case OP_VSLLI:
    case OP_VSRLI:
    case OP_VSRAI:
    case OP_ADDI:
    case OP_SUBI:
    case OP_SLLI:
    case OP_SRLI:
    case OP_SRAI:
	    fprintf(f, "%s.%s, %s, %s, %d",
		    asm_opname(pc->op),
		    asm_typename(pc->type),
		    asm_regname(pc->op,pc->rd),
		    asm_regname(pc->op,pc->ri),
		    pc->imm8);
	    break;
    case OP_MOVI:
    case OP_VMOVI:	
    case OP_JNZ:
    case OP_JZ:
	fprintf(f, "%s.%s %s, %d",
		asm_opname(pc->op),
		asm_typename(pc->type),
		asm_regname(pc->op,pc->rd), pc->imm12);
	break;	
    default:
	if (pc->op & OP_BIN) {
	    fprintf(f, "%s.%s, %s, %s, %s",
		    asm_opname(pc->op),
		    asm_typename(pc->type),
		    asm_regname(pc->op,pc->rd),
		    asm_regname(pc->op,pc->ri),
		    asm_regname(pc->op,pc->rj));
	}
	else {
	    fprintf(f, "%s.%s, %s, %s",
		    asm_opname(pc->op),
		    asm_typename(pc->type),
		    asm_regname(pc->op,pc->rd),
		    asm_regname(pc->op,pc->ri));
	}
	break;
    }
}

void print_code(FILE* f, instr_t* code, size_t len)
{
    int i;

    for (i = 0; i < (int)len; i++) {
	print_instr(f, &code[i]);
	fprintf(f, "\n");
    }
}


static int verbose = 1;
static int debug   = 0;

int test_icode(matrix_type_t itype, matrix_type_t otype, int i, instr_t* icode, size_t code_len)
{
    JitRuntime rt;           // Runtime designed for JIT code execution
    CodeHolder code;         // Holds code and relocation information
    FileLogger logger(stderr);
    scalar0_t a0, a1, a2;
    
    // Initialize to the same arch as JIT runtime
    code.init(rt.environment(), rt.cpuFeatures());

    x86::Assembler a(&code);  // Create and attach x86::Assembler to `code`
    
    setup_input(itype, i, a0, a1, a2);

    if (verbose) {
	fprintf(stderr, "TEST ");
	icode[0].type = itype; // otherwise set by test_code!
	print_instr(stderr, icode);
	fprintf(stderr, " %s=", "r0");
	sprint(stderr, itype, a0);
	fprintf(stderr, " %s=", "r1");
	sprint(stderr, itype, a1);
	fprintf(stderr, " %s=", "r2");
	sprint(stderr, itype, a2);		
    }

    if (debug) {
	a.setLogger(&logger);
    }

    set_type(itype, icode, code_len);

    assemble(a, rt.environment(),
	     reg(2), reg(0), reg(1),
	     icode, code_len);

    fun2_t fn;
    Error err = rt.add(&fn, &code);   // Add the generated code to the runtime.
    if (err) {
	fprintf(stderr, "rt.add ERROR\n");
	return -1;               // Handle a possible error case.
    }

    // fprintf(stderr, "code is added %p\n", fn);

    scalar0_t r0, r1;
    scalar0_t  rr[16];
    vscalar0_t vr[16];

    memset(rr, 0, sizeof(rr));
    memset(vr, 0, sizeof(vr));

    // fprintf(stderr, "\nemulate fn\n");
    
    memcpy(&rr[0], &a0, sizeof(scalar0_t));
    memcpy(&rr[1], &a1, sizeof(scalar0_t));
    memcpy(&rr[2], &a2, sizeof(scalar0_t));

    memcpy(&r0, &a2, sizeof(scalar0_t));
    memcpy(&r1, &a2, sizeof(scalar0_t));
    
    emulate(rr, vr, icode, code_len, &i);
    memcpy(&r0, &rr[i], sizeof(scalar0_t));

    if (debug) {
	fprintf(stderr, "emu:r = "); sprint(stderr,otype, r0); fprintf(stderr,"\n");
    }

    // fprintf(stderr, "\ncall fn\n");

    fn((scalar0_t*)&r1, (scalar0_t*)&a0, (scalar0_t*)&a1);

//    fprintf(stderr, "\ndone fn\n");
    
    rt.release(fn);

    // compare output from emu and exe
    if (scmp(otype, r0, r1) != 0) {
	if (verbose)
	    fprintf(stderr, " FAIL\n");
	else {
	    print_code(stderr, icode, code_len);
	}
	fprintf(stderr, "a0 = "); sprint(stderr,itype, a0); fprintf(stderr,"\n");
	fprintf(stderr,"a1 = "); sprint(stderr,itype, a1); fprintf(stderr,"\n");
	fprintf(stderr,"a2 = "); sprint(stderr,itype, a2); fprintf(stderr,"\n");
	fprintf(stderr, "emu:r = "); sprint(stderr,otype, r0); fprintf(stderr,"\n");
	// reassemble with logging
	CodeHolder code1;         // Holds code and relocation information
	x86::Assembler a1(&code1); // Create and attach x86::Assembler to `code`
	a1.setLogger(&logger);
	
	assemble(a1, rt.environment(),
		 reg(2), reg(0), reg(1),
		 icode, code_len);
	

	fprintf(stderr, "exe:r = "); sprint(stderr, otype, r1); fprintf(stderr, "\n");
	return -1;
    }
    else {
	if (debug) {
	    print_code(stderr, icode, code_len);
	    fprintf(stderr, "a0 = "); sprint(stderr,itype, a0); fprintf(stderr,"\n");
	    fprintf(stderr,"a1 = "); sprint(stderr,itype, a1); fprintf(stderr,"\n");
	    fprintf(stderr,"a2 = "); sprint(stderr,itype, a2); fprintf(stderr,"\n");	    
	    fprintf(stderr, "exe:r = "); sprint(stderr, otype, r1); fprintf(stderr, "\n");
	}
	if (verbose) fprintf(stderr, " OK\n");
    }
    return 0;
}

int test_code(matrix_type_t itype, matrix_type_t otype, instr_t* icode, size_t code_len)
{
    int failed = 0;    
    size_t n = 16/(1 << get_scalar_exp_size(itype)); // 16,8,4,2
    int i;
    for (i = 0; i < (int)n; i++) {
	failed += test_icode(itype, otype, i, icode, code_len);
    }
    return failed;
}


int test_vcode(matrix_type_t itype, matrix_type_t otype,
	       instr_t* icode, size_t code_len)
{
    JitRuntime rt;           // Runtime designed for JIT code execution
    CodeHolder code;         // Holds code and relocation information
    int i;
    FileLogger logger(stderr);
    
    // Initialize to the same arch as JIT runtime
    code.init(rt.environment(), rt.cpuFeatures()); 

    x86::Assembler a(&code);  // Create and attach x86::Assembler to `code`

    if (verbose) {
	fprintf(stderr, "TEST ");
	icode[0].type = itype; // otherwise set by test_code!
	print_instr(stderr, icode);
    }

    if (debug)
	a.setLogger(&logger);

    set_type(itype, icode, code_len);

    assemble(a, rt.environment(),
	     vreg(2), vreg(0), vreg(1),
	     icode, code_len);

    vecfun2_t fn;
    Error err = rt.add(&fn, &code);   // Add the generated code to the runtime.
    if (err) {
	fprintf(stderr, "rt.add ERROR\n");
	return -1;               // Handle a possible error case.
    }

    // fprintf(stderr, "code is added %p\n", fn);

    vector_t a0, a1, a2, r0, r1;
    scalar0_t rr[16];
    vscalar0_t vr[16];

    memset(rr, 0, sizeof(rr));
    memset(vr, 0, sizeof(vr)); 

    setup_vinput(itype, a0, a1, a2);
    
    // fprintf(stderr, "\nemulate fn\n");
    memcpy(&vr[0], &a0, sizeof(vector_t));
    memcpy(&vr[1], &a1, sizeof(vector_t));
    memcpy(&vr[2], &a2, sizeof(vector_t));

    memcpy(&r0, &a2, sizeof(vector_t));
    memcpy(&r1, &a2, sizeof(vector_t));
    
    emulate(rr, vr, icode, code_len, &i);
    memcpy(&r0, &vr[i], sizeof(vector_t));

    // fprintf(stderr, "\ncalling fn\n");

    fn((vector_t*)&r1, (vector_t*)&a0, (vector_t*)&a1);
    rt.release(fn);

    // compare output from emu and exe
    if (vcmp(otype, r0, r1) != 0) {
	if (verbose)
	    printf(" FAIL\n");
	else {
	    print_code(stderr, icode, code_len);
	}

	fprintf(stderr, "a0 = "); vprint(stderr, itype, (vector_t)a0); fprintf(stderr,"\n");
	fprintf(stderr, "a1 = "); vprint(stderr, itype, (vector_t)a1); fprintf(stderr,"\n");
	fprintf(stderr, "a2 = "); vprint(stderr, itype, (vector_t)a2); fprintf(stderr,"\n");    	
	fprintf(stderr,"emu:r = "); vprint(stderr,otype, (vector_t)r0); fprintf(stderr,"\n");
	// reassemble with logging
	code.reset();
	x86::Assembler a1(&code);  // Create and attach x86::Assembler to `code`
	a1.setLogger(&logger);

	assemble(a1, rt.environment(),
		 vreg(2), vreg(0), vreg(1),
		 icode, code_len);
	
	fprintf(stderr,"exe:r = "); vprint(stderr,otype, (vector_t)r1); fprintf(stderr,"\n");
	return -1;
    }
    else {
	if (debug) {
	    print_code(stderr, icode, code_len);
	    fprintf(stderr, "a0 = "); vprint(stderr, itype, (vector_t)a0); fprintf(stderr,"\n");
	    fprintf(stderr, "a1 = "); vprint(stderr, itype, (vector_t)a1); fprintf(stderr,"\n");
	    fprintf(stderr, "a2 = "); vprint(stderr, itype, (vector_t)a2); fprintf(stderr,"\n");    		    
	    fprintf(stderr, "exe:r = "); vprint(stderr, otype, (vector_t) r1); fprintf(stderr, "\n");
	}
	if (verbose) printf(" OK\n");
    }
    return 0;
}

// convert type to integer type with the same size
static uint8_t int_type(uint8_t at)
{
    return ((at & ~BASE_TYPE_MASK) | INT);
}

#define CODE_LEN(code) (sizeof((code))/sizeof((code)[0]))

int test_ts_code(uint8_t* ts, int td, int vec, instr_t* code, size_t code_len)
{
    int failed = 0;
    
    while(*ts != VOID) {
	uint8_t otype;
	switch(td) {
	case INT: otype = int_type(*ts); break;
	default:  otype = *ts; break;
	}

	if (vec) {
	    if (test_vcode(*ts, otype, code, code_len) < 0)
		failed++;
	}
	else {
	    if (test_code(*ts, otype, code, code_len) < 0)
		failed++;
	}
	ts++;
    }
    return failed;
}

int test_binary(uint8_t op, uint8_t* ts, uint8_t otype)
{
    instr_t code[2];
    int i, j;
    int failed = 0;
    int vec = (op & OP_VEC) != 0;

    printf("+------------------------------\n");
    printf("| %s\n", asm_opname(op));
    printf("+------------------------------\n");    
    
    code[0].op = op;
    code[1].op = vec ? OP_VRET : OP_RET;
    code[1].rd = 2;
    code[0].rd = 2;
    
    for (i = 0; i <= 2; i++) {
	code[0].ri = i;
	for (j = 0; j <= 2; j++) {
	    code[0].rj = j;
	    failed += test_ts_code(ts, otype, vec, code, 2);
	}
    }
    return failed;
}

int test_unary(uint8_t op, uint8_t* ts, uint8_t otype)
{
    instr_t code[2];
    int i;
    int failed = 0;
    int vec = (op & OP_VEC) != 0;

    printf("+------------------------------\n");
    printf("| %s\n", asm_opname(op));
    printf("+------------------------------\n");
    
    code[0].op = op;
    code[1].op = vec ? OP_VRET : OP_RET;
    code[1].rd = 2;
    code[0].rd = 2;
    
    for (i = 0; i <= 2; i++) {
	code[0].ri = i;
	failed += test_ts_code(ts, otype, vec, code, 2);
    }
    return failed;
}

int test_imm8(uint8_t op, uint8_t* ts, uint8_t otype)
{
    instr_t code[2];
    int i, imm;
    int failed = 0;
    int vec = (op & OP_VEC) != 0;

    printf("+------------------------------\n");
    printf("| %s\n", asm_opname(op));
    printf("+------------------------------\n");
        
    code[0].op = op;
    code[1].op = vec ? OP_VRET : OP_RET;
    code[1].rd = 2;
    code[0].rd = 2;

    for (i = 0; i <= 2; i++) {
	code[0].ri = i;
	for (imm = -128; imm < 128; imm += 27) {
	    code[0].imm8 = imm;
	    failed += test_ts_code(ts, otype, vec, code, 2);
	}
    }
    return failed;
}

int test_imm12(uint8_t op, uint8_t* ts, uint8_t otype)
{
    instr_t code[2];
    int imm;
    int failed = 0;
    int vec = (op & OP_VEC) != 0;

    printf("+------------------------------\n");
    printf("| %s\n", asm_opname(op));
    printf("+------------------------------\n");
    
    code[0].op = op;
    code[1].op = vec ? OP_VRET : OP_RET;
    code[1].rd = 2;
    code[0].rd = 2;

    for (imm = -2048; imm < 2048; imm += 511) {
	code[0].imm12 = imm;
	failed += test_ts_code(ts, otype, vec, code, 2);
    }
    return failed;
}

int test_shift(uint8_t op, uint8_t* ts, uint8_t otype)
{
    instr_t code[2];
    int i;
    int failed = 0;
    int vec = (op & OP_VEC) != 0;

    printf("+------------------------------\n");
    printf("| %s\n", asm_opname(op));
    printf("+------------------------------\n");
    
    
    code[0].op = op;
    code[1].op = vec ? OP_VRET : OP_RET;
    code[1].rd = 2;
    code[0].rd = 2;

    for (i = 0; i <= 2; i++) {
	uint8_t* ts1 = ts;
	code[0].ri = i;
	while(*ts1 != VOID) {
	    int imm;
	    size_t n = (1 << get_scalar_exp_bit_size(*ts1));
	    for (imm = 0; imm < (int)n; imm++) {
		uint8_t tv[2];
		code[0].imm8 = imm;
		tv[0] = *ts1; tv[1] = VOID;
		failed += test_ts_code(tv, otype, vec, code, 2);	
	    }
	    ts1++;
	}	    
    }
    return failed;
}


int main()
{
    printf("VSIZE = %d\n", VSIZE);
    printf("sizeof(vector_t) = %ld\n", sizeof(vector_t));
    printf("sizeof(scalar0_t) = %ld\n", sizeof(scalar0_t));
    printf("sizeof(vscalar0_t) = %ld\n", sizeof(vscalar0_t));
    printf("sizeof(instr_t) = %ld\n", sizeof(instr_t));

    int failed = 0;  // number of failed cases
    
    instr_t code_sum[] = {
	OPimm12d(OP_MOVI, 0, 0),     // SUM=0 // OPdij(OP_BXOR, 0, 0, 0),
	OPimm12d(OP_MOVI, 1, 13),    // I=13  OPdij(OP_BXOR, 0, 0, 0),
	OPdij(OP_ADD, 0, 1, 0),      // SUM += I
	OPimm12d(OP_SUBI, 1, 1),    // I -= 1
	OPimm12d(OP_JNZ, 1, -3),
	OPd(OP_RET, 0)
    };
    
    uint8_t int_types[] =
	{ UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, VOID };
//    uint8_t float_types[] =
//	{ FLOAT32, FLOAT64, VOID };
    uint8_t all_types[] =
	{ UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64,
	  //FLOAT16
	  FLOAT32, FLOAT64, VOID };
    
    uint8_t int_type[] = { INT8, VOID };
   
//    debug = 1;
//    failed += test_binary(OP_MUL, int_type, VOID);
//    exit(0);
    
//    instr_t code1[] = { OPdij(OP_CMPLT,2,0,2), OPd(OP_RET, 2) };
//    code1[0].type = UINT64;
//    code1[1].type = UINT64;
//    test_code(UINT64, INT64, code1, 2);
//     instr_t code1[] = { OPdiimm8(OP_SRLI,2,0,7), OPd(OP_RET, 2) };
//     code1[0].type = INT16;
//     code1[1].type = INT16;
//     test_code(INT16, INT16, code1, 2);

//    instr_t code1[] = { OPdiimm8(OP_VADDI,2,0,7), OPd(OP_VRET, 2) };
//     code1[0].type = INT16;
//     code1[1].type = INT16;
//     test_vcode(INT16, INT16, code1, 2);        
//     exit(0);

    failed += test_unary(OP_NOP, int_types, VOID);    
    failed += test_imm12(OP_MOVI, int_types, INT);
    failed += test_unary(OP_MOVR, int_types, VOID); // fixme: float registers!
    failed += test_shift(OP_SLLI, int_types, INT);
    failed += test_shift(OP_SRLI, int_types, INT);
    failed += test_shift(OP_SRAI, int_types, INT);
    failed += test_unary(OP_NEG, int_types, VOID);
    failed += test_unary(OP_BNOT, int_types, INT);
    // failed += test_unary(OP_INV, int_types, INT);
    // OP_JMP/OP_JNZ/OP_JZ
    failed += test_imm8(OP_ADDI, int_types, INT);
    failed += test_imm8(OP_SUBI, int_types, INT);

    failed += test_binary(OP_ADD, int_types, VOID);
    failed += test_binary(OP_SUB, int_types, VOID);    
    failed += test_binary(OP_MUL, int_types, VOID);
    failed += test_binary(OP_BAND, int_types, INT);    
    failed += test_binary(OP_BOR, int_types, INT);        
    failed += test_binary(OP_BXOR, int_types, INT);
    
    failed += test_binary(OP_CMPLT, int_types, INT);
    failed += test_binary(OP_CMPLE, int_types, INT);    
    failed += test_binary(OP_CMPEQ, int_types, INT);

    // vectors
    failed += test_unary(OP_VMOVR, all_types, VOID);
    failed += test_imm12(OP_VMOVI, int_types, INT);
    failed += test_imm8(OP_VADDI, int_types, INT);
    failed += test_imm8(OP_VSUBI, int_types, INT);
    failed += test_shift(OP_VSLLI, int_types, INT);
    failed += test_shift(OP_VSRLI, int_types, INT);
    failed += test_shift(OP_VSRAI, int_types, INT);    
    failed += test_unary(OP_VNEG, all_types, VOID);    
    failed += test_unary(OP_VBNOT, all_types, INT);
    //failed += test_unary(OP_VINV, all_types, VOID); float types?
    
    failed += test_binary(OP_VADD, all_types, VOID);
    failed += test_binary(OP_VSUB, all_types, VOID);
    failed += test_binary(OP_VMUL, all_types, VOID);
    failed += test_binary(OP_VBAND, all_types, INT);
    failed += test_binary(OP_VBOR, all_types, INT);
    failed += test_binary(OP_VBXOR, all_types, INT);
    failed += test_binary(OP_VCMPEQ, all_types, INT);
    failed += test_binary(OP_VCMPLT, all_types, INT);
    failed += test_binary(OP_VCMPLE, all_types, INT);


    if (failed) {
	printf("ERROR: %d cases failed\n", failed);
	exit(1);
    }
    printf("OK\n");
    exit(0);
}
