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

extern void sprint(uint8_t type, scalar0_t v);
extern int  scmp(uint8_t type, scalar0_t v1, scalar0_t v2);

extern x86::Vec vreg(int i);
extern x86::Gp reg(int i);


extern void vprint(uint8_t type, vector_t v);
extern int  vcmp(uint8_t type, vector_t v1, vector_t v2);



extern void set_element_int64(matrix_type_t type, vector_t &r, int i, int64_t v);
extern void set_element_float64(matrix_type_t type, vector_t &r, int i, float64_t v);

#define OPdij(o,d,i,j) \
    {.op = (o),.type=INT64,.ri=(i),.rj=(j),.pad=0,.rd=(d)}
#define OPdi(o,d,i) \
    {.op = (o),.type=INT64,.ri=(i),.rj=0,.pad=0,.rd=(d)}
#define OPi(o,i) \
    {.op = (o),.type=INT64,.ri=(i),.rj=0,.pad=0,.rd=0}

static const char* asm_opname(uint8_t op)
{
    switch(op) {
    case OP_RET:  return "ret";
    case OP_MOVR:  return "mov";
    case OP_MOVA:  return "mov";
    case OP_MOVC:  return "mov";
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
    case OP_VMOVA:  return "vmov";
    case OP_VMOVC:  return "vmov";
    case OP_VNEG:  return "vneg";
    case OP_VBNOT:  return "vbnot";
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

int setup_vinput(matrix_type_t type, vector_t& a0, vector_t& a1, vector_t& a2)
{
    switch(VSIZE/get_scalar_size(type)) {
    case 16:
	set_element_int64(type, a0, 15,  2);
	set_element_int64(type, a0, 14, -3);
	set_element_int64(type, a0, 13,  4);
	set_element_int64(type, a0, 12, -5);
	set_element_int64(type, a0, 11,  6);
	set_element_int64(type, a0, 10, -7);
	set_element_int64(type, a0, 9,   8);
	set_element_int64(type, a0, 8,  -9);
	// fall through
    case 8:
	set_element_int64(type, a0, 7,  10);
	set_element_int64(type, a0, 6, -11);
	set_element_int64(type, a0, 5,  12);
	set_element_int64(type, a0, 4, -13);
        // fall through	
    case 4:
	set_element_int64(type, a0, 3,  23);
	set_element_int64(type, a0, 2, -24);
	// fall through	
    case 2:
	set_element_int64(type, a0, 0,  50);
	set_element_int64(type, a0, 1, -49);
	// fall through	
    default:
	break;
    }

    switch(VSIZE/get_scalar_size(type)) {
    case 16:
	set_element_int64(type, a1, 15,  3);
	set_element_int64(type, a1, 14, -3);
	set_element_int64(type, a1, 13,  4);
	set_element_int64(type, a1, 12, -4);
	set_element_int64(type, a1, 11,  5);
	set_element_int64(type, a1, 10, -5);
	set_element_int64(type, a1, 9,   6);
	set_element_int64(type, a1, 8,  -6);
	// fall through	
    case 8:
	set_element_int64(type, a1, 7,  10);
	set_element_int64(type, a1, 6, -10);
	set_element_int64(type, a1, 5,  13);
	set_element_int64(type, a1, 4, -14);
	// fall through	
    case 4:
	set_element_int64(type, a1, 3,  23);
	set_element_int64(type, a1, 2, -25);
	// fall through	
    case 2:
	set_element_int64(type, a1, 0,  51);
	set_element_int64(type, a1, 1, -50);
	// fall through	
    default:
	break;	
    }

    switch(VSIZE/get_scalar_size(type)) {
    case 16:
	set_element_int64(type, a2, 15,  100);
	set_element_int64(type, a2, 14,  50);
	set_element_int64(type, a2, 13,  25);
	set_element_int64(type, a2, 12,  12);
	set_element_int64(type, a2, 11,  6);
	set_element_int64(type, a2, 10,  3);
	set_element_int64(type, a2, 9,   2);
	set_element_int64(type, a2, 8,   1);
	// fall through	
    case 8:
	set_element_int64(type, a2, 7,  99);
	set_element_int64(type, a2, 6,  98);
	set_element_int64(type, a2, 5,  97);
	set_element_int64(type, a2, 4,  96);
	// fall through	
    case 4:
	set_element_int64(type, a2, 3,  20);
	set_element_int64(type, a2, 2,  19);
	// fall through	
    case 2:
	set_element_int64(type, a2, 0,  8);
	set_element_int64(type, a2, 1,  7);
	// fall through	
    default:
	break;
    }        
    return 0;
}


int setup_input(matrix_type_t type, scalar0_t& a0, scalar0_t& a1, scalar0_t& a2)
{
    switch(type) {
    case UINT8:  a0.u8 = 15; break;
    case UINT16: a0.u16 = 14; break;
    case UINT32: a0.u32 = 13; break;
    case UINT64: a0.u64 = 12; break;
    case INT8:   a0.i8 = 2; break;
    case INT16:  a0.i16 = -3; break;
    case INT32:  a0.i32 = 4; break;
    case INT64:  a0.i64 = -5; break;	
    case FLOAT16: a0.f16 = 7; break;
    case FLOAT32: a0.f32 = 8.0; break;
    case FLOAT64: a0.f64 = -9.0; break;
    default: break;	
    }

    switch(type) {
    case UINT8:  a1.u8 = 3; break;
    case UINT16: a1.u16 = 4; break;
    case UINT32: a1.u32 = 6; break;
    case UINT64: a1.u64 = 6; break;
    case INT8:   a1.i8 = 5; break;
    case INT16:  a1.i16 = -5; break;
    case INT32:  a1.i32 = 6; break;
    case INT64:  a1.i64 = -6; break;	
    case FLOAT16: a1.f16 = 10; break;
    case FLOAT32: a1.f32 = -10.0; break;
    case FLOAT64: a1.f64 = -14.0; break;
    default: break;	
    }

    switch(type) {
    case UINT8: a2.u8 = 100; break;
    case UINT16: a2.u16 = 50; break;
    case UINT32: a2.u32 = 25; break;
    case UINT64: a2.u64 = 12; break;
    case INT8: a2.i8 = 6; break;
    case INT16: a2.i16 = 3; break;
    case INT32: a2.i32 = 2; break;
    case INT64: a2.i64 = 1; break;	
    case FLOAT16: a2.f16 = 10; break;
    case FLOAT32: a2.f32 = 99.0; break;
    case FLOAT64: a2.f64 = 98.0; break;
    default: break;
    }
    return 0;
}


void print_instr(instr_t* pc)
{
    if (pc->op & OP_BIN) {
	printf("%s.%s, %s, %s, %s",
	       asm_opname(pc->op),
	       asm_typename(pc->type),
	       asm_regname(pc->op,pc->rd),
	       asm_regname(pc->op,pc->ri),
	       asm_regname(pc->op,pc->rj));
    }
    else {
	printf("%s.%s, %s, %s",
	       asm_opname(pc->op),
	       asm_typename(pc->type),
	       asm_regname(pc->op,pc->rd),
	       asm_regname(pc->op,pc->ri));
    }
}

static int verbose = 1;
static int debug   = 0;

int test_instr(matrix_type_t itype, matrix_type_t otype, instr_t* icode)
{
    JitRuntime rt;           // Runtime designed for JIT code execution
    CodeHolder code;         // Holds code and relocation information
    int i;
    size_t n;
    FileLogger logger(stdout);
    
    // Initialize to the same arch as JIT runtime
    code.init(rt.environment(), rt.cpuFeatures()); 

    x86::Assembler a(&code);  // Create and attach x86::Assembler to `code`

    if (debug) {
	a.setLogger(&logger);
    }

    n = code_size(icode);
    set_type(itype, icode, n);

    if (verbose) {
	printf("TEST ");
	print_instr(&icode[0]);
	if (debug) printf("\n");
    }
    
    assemble(a, rt.environment(),
	     reg(2), reg(0), reg(1),
	     icode, n);

    fun2_t fn;
    Error err = rt.add(&fn, &code);   // Add the generated code to the runtime.
    if (err) {
	printf("rt.add ERROR\n");
	return -1;               // Handle a possible error case.
    }

    // printf("code is added %p\n", fn);

    scalar0_t a0, a1, a2, r0, r1;
    scalar0_t reg[16];
    vscalar0_t vreg[16];

    memset(reg, 0, sizeof(reg));
    memset(vreg, 0, sizeof(vreg));

    setup_input(itype, a0, a1, a2);
    
    // printf("emulate fn\n");
    memcpy(&reg[0], &a0, sizeof(scalar0_t));
    memcpy(&reg[1], &a1, sizeof(scalar0_t));
    memcpy(&reg[2], &a2, sizeof(scalar0_t));

    memcpy(&r0, &a2, sizeof(scalar0_t));
    memcpy(&r1, &a2, sizeof(scalar0_t));
    
    emulate(reg, vreg, icode, n, &i);
    memcpy(&r0, &reg[i], sizeof(scalar0_t));

    // printf("calling fn\n");

    fn((scalar0_t*)&r1, (scalar0_t*)&a0, (scalar0_t*)&a1);
    rt.release(fn);

    // compare output from emu and exe
    if (scmp(otype, r0, r1) != 0) {
	if (verbose) printf(" FAIL\n");
	else print_instr(&icode[0]);
	printf("a0 = "); sprint(itype, a0); printf("\n");
	printf("a1 = "); sprint(itype, a1); printf("\n");
	printf("a2 = "); sprint(itype, a2); printf("\n");    	
	printf("emu:r = "); sprint(otype, r0); printf("\n");
	printf("exe:r = "); sprint(otype, r1); printf("\n");
	return 0;
    }
    else {
	if (verbose) printf(" OK\n");
    }
    return 1;
}


int test_vinstr(matrix_type_t itype, matrix_type_t otype, instr_t* icode)
{
    JitRuntime rt;           // Runtime designed for JIT code execution
    CodeHolder code;         // Holds code and relocation information
    int i;
    size_t n;
    FileLogger logger(stdout);
    
    // Initialize to the same arch as JIT runtime
    code.init(rt.environment(), rt.cpuFeatures()); 

    x86::Assembler a(&code);  // Create and attach x86::Assembler to `code`

    if (debug)
	a.setLogger(&logger);

    n = code_size(icode);
    set_type(itype, icode, n);

    if (verbose) {
	printf("TEST ");
	print_instr(&icode[0]);
	if (debug) printf("\n");
    }
    
    assemble(a, rt.environment(),
	     vreg(2), vreg(0), vreg(1),
	     icode, n);

    vecfun2_t fn;
    Error err = rt.add(&fn, &code);   // Add the generated code to the runtime.
    if (err) {
	printf("rt.add ERROR\n");
	return -1;               // Handle a possible error case.
    }

    // printf("code is added %p\n", fn);

    vector_t a0, a1, a2, r0, r1;
    scalar0_t reg[16];
    vscalar0_t vreg[16];

    memset(reg, 0, sizeof(reg));
    memset(vreg, 0, sizeof(vreg));    

    setup_vinput(itype, a0, a1, a2);
    
    // printf("emulate fn\n");
    memcpy(&vreg[0], &a0, sizeof(vector_t));
    memcpy(&vreg[1], &a1, sizeof(vector_t));
    memcpy(&vreg[2], &a2, sizeof(vector_t));

    memcpy(&r0, &a2, sizeof(vector_t));
    memcpy(&r1, &a2, sizeof(vector_t));
    
    emulate(reg, vreg, icode, n, &i);
    memcpy(&r0, &vreg[i], sizeof(vector_t));

    // printf("calling fn\n");

    fn((vector_t*)&r1, (vector_t*)&a0, (vector_t*)&a1);
    rt.release(fn);

    // compare output from emu and exe
    if (vcmp(otype, r0, r1) != 0) {
	if (verbose) printf(" FAIL\n");
	else print_instr(&icode[0]);
	printf("a0 = "); vprint(itype, (vector_t)a0); printf("\n");
	printf("a1 = "); vprint(itype, (vector_t)a1); printf("\n");
	printf("a2 = "); vprint(itype, (vector_t)a2); printf("\n");    	
	printf("emu:r = "); vprint(otype, (vector_t)r0); printf("\n");
	printf("exe:r = "); vprint(otype, (vector_t)r1); printf("\n");
	return 0;
    }
    else {
	if (verbose) printf(" OK\n");
    }
    return 1;
}

// convert float type to integer type with the same size
static uint8_t integer_type(uint8_t at)
{
    return ((at & ~BASE_TYPE_MASK) | INT);
}

// convert float type to integer type with the same size
static uint8_t unsigned_type(uint8_t at)
{
    return ((at & ~BASE_TYPE_MASK) | INT);
}

int main()
{
    printf("VSIZE = %d\n", VSIZE);
    printf("sizeof(vector_t) = %ld\n", sizeof(vector_t));
    printf("sizeof(scalar0_t) = %ld\n", sizeof(scalar0_t));
    printf("sizeof(vscalar0_t) = %ld\n", sizeof(vscalar0_t));
    printf("sizeof(instr_t) = %ld\n", sizeof(instr_t));

    // scalar ops
    instr_t code_neg_0[]  = { OPdi(OP_NEG,   2, 0), OPi(OP_RET, 2) };
    instr_t code_neg_2[]  = { OPdi(OP_NEG,   2, 2), OPi(OP_RET, 2) };

    instr_t code_bnot_0[]  = { OPdi(OP_BNOT, 2, 0), OPi(OP_RET, 2) };
    instr_t code_bnot_2[]  = { OPdi(OP_BNOT, 2, 2), OPi(OP_RET, 2) };    
    
    instr_t code_add_00[]  = { OPdij(OP_ADD,   2, 0, 0), OPi(OP_RET, 2) };
    instr_t code_add_01[]  = { OPdij(OP_ADD,   2, 0, 1), OPi(OP_RET, 2) };
    instr_t code_add_10[]  = { OPdij(OP_ADD,   2, 1, 0), OPi(OP_RET, 2) };
    instr_t code_add_11[]  = { OPdij(OP_ADD,   2, 1, 1), OPi(OP_RET, 2) };
    instr_t code_add_02[]  = { OPdij(OP_ADD,   2, 0, 2), OPi(OP_RET, 2) };
    instr_t code_add_21[]  = { OPdij(OP_ADD,   2, 2, 1), OPi(OP_RET, 2) };
    instr_t code_add_12[]  = { OPdij(OP_ADD,   2, 1, 2), OPi(OP_RET, 2) };
    instr_t code_add_22[]  = { OPdij(OP_ADD,   2, 2, 2), OPi(OP_RET, 2) };
    
    // vector ops    
    instr_t code_vneg_0[]  = { OPdi(OP_VNEG,   2, 0), OPi(OP_VRET, 2) };
    instr_t code_vneg_2[]  = { OPdi(OP_VNEG,   2, 2), OPi(OP_VRET, 2) };

    instr_t code_vbnot_0[]  = { OPdi(OP_VBNOT,   2, 0), OPi(OP_VRET, 2) };
    instr_t code_vbnot_2[]  = { OPdi(OP_VBNOT,   2, 2), OPi(OP_VRET, 2) };    
    
    instr_t code_vadd_00[]  = { OPdij(OP_VADD,   2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vadd_01[]  = { OPdij(OP_VADD,   2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vadd_10[]  = { OPdij(OP_VADD,   2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vadd_11[]  = { OPdij(OP_VADD,   2, 1, 1), OPi(OP_VRET, 2) };
    instr_t code_vadd_02[]  = { OPdij(OP_VADD,   2, 0, 2), OPi(OP_VRET, 2) };
    instr_t code_vadd_21[]  = { OPdij(OP_VADD,   2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vadd_12[]  = { OPdij(OP_VADD,   2, 1, 2), OPi(OP_VRET, 2) };
    instr_t code_vadd_22[]  = { OPdij(OP_VADD,   2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vsub_00[]  = { OPdij(OP_VSUB,   2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vsub_01[]  = { OPdij(OP_VSUB,   2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vsub_10[]  = { OPdij(OP_VSUB,   2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vsub_11[]  = { OPdij(OP_VSUB,   2, 1, 1), OPi(OP_VRET, 2) };
    instr_t code_vsub_02[]  = { OPdij(OP_VSUB,   2, 0, 2), OPi(OP_VRET, 2) };
    instr_t code_vsub_12[]  = { OPdij(OP_VSUB,   2, 1, 2), OPi(OP_VRET, 2) };
    instr_t code_vsub_21[]  = { OPdij(OP_VSUB,   2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vsub_22[]  = { OPdij(OP_VSUB,   2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vmul_00[]  = { OPdij(OP_VMUL,   2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vmul_01[]  = { OPdij(OP_VMUL,   2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vmul_10[]  = { OPdij(OP_VMUL,   2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vmul_02[]  = { OPdij(OP_VMUL,   2, 0, 2), OPi(OP_VRET, 2) };
    instr_t code_vmul_12[]  = { OPdij(OP_VMUL,   2, 1, 2), OPi(OP_VRET, 2) };
    instr_t code_vmul_21[]  = { OPdij(OP_VMUL,   2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vmul_22[]  = { OPdij(OP_VMUL,   2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vlt_00[]   = { OPdij(OP_VCMPLT, 2, 0, 0), OPi(OP_VRET, 2) }; 
    instr_t code_vlt_01[]   = { OPdij(OP_VCMPLT, 2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vlt_10[]   = { OPdij(OP_VCMPLT, 2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vlt_21[]   = { OPdij(OP_VCMPLT, 2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vlt_22[]   = { OPdij(OP_VCMPLT, 2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vle_00[]   = { OPdij(OP_VCMPLE, 2, 0, 0), OPi(OP_VRET, 2) };   
    instr_t code_vle_01[]   = { OPdij(OP_VCMPLE, 2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vle_10[]   = { OPdij(OP_VCMPLE, 2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vle_21[]   = { OPdij(OP_VCMPLE, 2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vle_22[]   = { OPdij(OP_VCMPLE, 2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_veq_00[]   = { OPdij(OP_VCMPEQ, 2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_veq_01[]   = { OPdij(OP_VCMPEQ, 2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_veq_10[]   = { OPdij(OP_VCMPEQ, 2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_veq_21[]   = { OPdij(OP_VCMPEQ, 2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_veq_22[]   = { OPdij(OP_VCMPEQ, 2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vband_00[]   = { OPdij(OP_VBAND, 2, 0, 0), OPi(OP_VRET, 2) };  
    instr_t code_vband_01[]   = { OPdij(OP_VBAND, 2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vband_21[]   = { OPdij(OP_VBAND, 2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vband_22[]   = { OPdij(OP_VBAND, 2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vbor_01[]   = { OPdij(OP_VBOR, 2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vbor_00[]   = { OPdij(OP_VBOR, 2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vbor_21[]   = { OPdij(OP_VBOR, 2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vbor_22[]   = { OPdij(OP_VBOR, 2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vbxor_01[]   = { OPdij(OP_VBXOR, 2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vbxor_00[]   = { OPdij(OP_VBXOR, 2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vbxor_21[]   = { OPdij(OP_VBXOR, 2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vbxor_22[]   = { OPdij(OP_VBXOR, 2, 2, 2), OPi(OP_VRET, 2) };
    
    int t;
    uint8_t int_types[] =
	{ UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64, VOID };
    uint8_t float_types[] =
	{ FLOAT32, FLOAT64, VOID };
    uint8_t all_types[] =
	{ UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64,
	  //FLOAT16
	  FLOAT32, FLOAT64, VOID };
/*
    debug = 1;
//    test_vinstr(INT8,    INT8,    code_vmul_01);    
//    test_vinstr(FLOAT32, FLOAT32, code_vadd_22);
//    test_vinstr(INT8,    INT8,    code_vneg_0);    
//    test_vinstr(INT8,    INT8,    code_vsub_02);
    exit(0);
*/
    printf("+------------------------------\n");
    printf("| neg\n");
    printf("+------------------------------\n");

    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_neg_0);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_neg_2);

    printf("+------------------------------\n");
    printf("| bnot\n");
    printf("+------------------------------\n");

    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], unsigned_type(int_types[t]), code_bnot_0);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], unsigned_type(int_types[t]), code_bnot_2);

    printf("+------------------------------\n");
    printf("| add\n");
    printf("+------------------------------\n");
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_00);    
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_01);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_10);
    for (t = 0; int_types[t] != VOID; t++) 
	test_instr(int_types[t], int_types[t], code_add_11);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_02);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_21);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_12);
    for (t = 0; int_types[t] != VOID; t++)
	test_instr(int_types[t], int_types[t], code_add_22);	
    

    printf("+------------------------------\n");
    printf("| vneg\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vneg_0);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vneg_2);

    printf("+------------------------------\n");
    printf("| vbnot\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbnot_0);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbnot_2);

    printf("+------------------------------\n");
    printf("| vadd\n");
    printf("+------------------------------\n");
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_00);    
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_01);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_10);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vadd_11);    
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_02);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_21);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_12);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vadd_22);	

    printf("+------------------------------\n");
    printf("| vsub\n");
    printf("+------------------------------\n");
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vsub_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vsub_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vsub_10);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vsub_11);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vsub_02);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], all_types[t], code_vsub_21);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], all_types[t], code_vsub_12);    
    for (t = 0; all_types[t] != VOID; t++)	
	test_vinstr(all_types[t], all_types[t], code_vsub_22);	
    
    printf("+------------------------------\n");
    printf("| veq\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_veq_00);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_veq_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_veq_10);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_veq_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_veq_22);

    printf("+------------------------------\n");
    printf("| vcmplt\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vlt_00);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vlt_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vlt_10);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vlt_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vlt_22);    
    
    printf("+------------------------------\n");
    printf("| vcmple\n");
    printf("+------------------------------\n");
    
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vle_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vle_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vle_10);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vle_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vle_22);        
    printf("+------------------------------\n");
    printf("| vband\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vband_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vband_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vband_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vband_22);
    
    printf("+------------------------------\n");
    printf("| vbor\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbor_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbor_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbor_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbor_22);
    
    printf("+------------------------------\n");
    printf("| vbxor\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbxor_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbxor_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbxor_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vbxor_22);

    printf("+------------------------------\n");
    printf("| vmul\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_00);    
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_01);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_10);    
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_02);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_12);    
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_21);
    for (t = 0; all_types[t] != VOID; t++)
	test_vinstr(all_types[t], unsigned_type(all_types[t]), code_vmul_22);
    return 0;    
}
