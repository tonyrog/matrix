// Test of matrix kernel

#include <asmjit/x86.h>
#include <iostream>

using namespace asmjit;

#include "matrix_types.h"
#include "matrix_kernel.h"

extern void assemble(x86::Assembler &a, const Environment &env, instr_t* code, size_t n);
extern void emulate(vscalar0_t r[16], instr_t* code, size_t n, int* ret);
extern void vprint(uint8_t type, vector_t v);
extern int  vcmp(uint8_t type, vector_t v1, vector_t v2);

extern void set_element_int64(matrix_type_t type, vector_t &r, int i, int64_t v);
extern void set_element_float64(matrix_type_t type, vector_t &r, int i, float64_t v);

#define OPdij(o,d,i,j) \
    {.op = (o),.type=INT64,.ri=(i),.rj=(j),.rd=(d),.rc=0}
#define OPdi(o,d,i) \
    {.op = (o),.type=INT64,.ri=(i),.rj=0,.rd=(d),.rc=0}
#define OPi(o,i) \
    {.op = (o),.type=INT64,.ri=(i),.rj=0,.rd=0,.rc=0}

const char* vasm_opname(uint8_t op)
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

const char* vasm_typename(uint8_t type)
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

int setup_input(matrix_type_t type, vector_t& a0, vector_t& a1, vector_t& a2)
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

void print_instr(instr_t* pc)
{
    if (pc->op & OP_BIN)
	printf("%s.%s, v%d, v%d, v%d",
	       vasm_opname(pc->op),
	       vasm_typename(pc->type),
	       pc->rd, pc->ri, pc->rj);
    else
	printf("%s.%s, v%d, v%d",
	       vasm_opname(pc->op),
	       vasm_typename(pc->type),
	       pc->rd, pc->ri);
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

    if (debug)
	a.setLogger(&logger);

    n = code_size(icode);
    set_type(itype, icode, n);

    if (verbose) {
	printf("TEST ");
	print_instr(&icode[0]);
    }
    
    assemble(a, rt.environment(), icode, n);

    vecfun2_t fn;
    Error err = rt.add(&fn, &code);   // Add the generated code to the runtime.
    if (err) {
	printf("rt.add ERROR\n");
	return -1;               // Handle a possible error case.
    }

    // printf("code is added %p\n", fn);

    vector_t a0, a1, a2, r0, r1;
    vscalar0_t reg[16];

    setup_input(itype, a0, a1, a2);
    
    // printf("emulate fn\n");
    memcpy(&reg[0], &a0, sizeof(vector_t));
    memcpy(&reg[1], &a1, sizeof(vector_t));
    memcpy(&reg[2], &a2, sizeof(vector_t));

    memcpy(&r0, &a2, sizeof(vector_t));
    memcpy(&r1, &a2, sizeof(vector_t));
    
    emulate(reg, icode, n, &i);
    memcpy(&r0, &reg[i], sizeof(vector_t));

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

    instr_t code_vneg_0[]  = { OPdi(OP_VNEG,   2, 0), OPi(OP_VRET, 2) };
    instr_t code_vneg_2[]  = { OPdi(OP_VNEG,   2, 2), OPi(OP_VRET, 2) };

    instr_t code_vbnot_0[]  = { OPdi(OP_VBNOT,   2, 0), OPi(OP_VRET, 2) };
    instr_t code_vbnot_2[]  = { OPdi(OP_VBNOT,   2, 2), OPi(OP_VRET, 2) };    
    
    instr_t code_vadd_00[]  = { OPdij(OP_VADD,   2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vadd_01[]  = { OPdij(OP_VADD,   2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vadd_10[]  = { OPdij(OP_VADD,   2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vadd_02[]  = { OPdij(OP_VADD,   2, 0, 2), OPi(OP_VRET, 2) };
    instr_t code_vadd_21[]  = { OPdij(OP_VADD,   2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vadd_12[]  = { OPdij(OP_VADD,   2, 1, 2), OPi(OP_VRET, 2) };
    instr_t code_vadd_22[]  = { OPdij(OP_VADD,   2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vsub_00[]  = { OPdij(OP_VSUB,   2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vsub_01[]  = { OPdij(OP_VSUB,   2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vsub_10[]  = { OPdij(OP_VSUB,   2, 1, 0), OPi(OP_VRET, 2) };
    instr_t code_vsub_02[]  = { OPdij(OP_VSUB,   2, 0, 2), OPi(OP_VRET, 2) };
    instr_t code_vsub_12[]  = { OPdij(OP_VSUB,   2, 1, 2), OPi(OP_VRET, 2) };
    instr_t code_vsub_21[]  = { OPdij(OP_VSUB,   2, 2, 1), OPi(OP_VRET, 2) };
    instr_t code_vsub_22[]  = { OPdij(OP_VSUB,   2, 2, 2), OPi(OP_VRET, 2) };

    instr_t code_vmul_00[]  = { OPdij(OP_VMUL,   2, 0, 0), OPi(OP_VRET, 2) };
    instr_t code_vmul_01[]  = { OPdij(OP_VMUL,   2, 0, 1), OPi(OP_VRET, 2) };
    instr_t code_vmul_10[]  = { OPdij(OP_VMUL,   2, 1, 0), OPi(OP_VRET, 2) };       instr_t code_vmul_02[]  = { OPdij(OP_VMUL,   2, 0, 2), OPi(OP_VRET, 2) };
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
    uint8_t all_types[] =
	{ UINT8, UINT16, UINT32, UINT64, INT8, INT16, INT32, INT64,
	  //FLOAT16
	  FLOAT32, FLOAT64, VOID };

/*
    debug = 1;
//    test_instr(INT8,    INT8,    code_vmul_01);    
//    test_instr(FLOAT32, FLOAT32, code_vadd_22);
//    test_instr(INT8,    INT8,    code_vneg_0);    
//    test_instr(INT8,    INT8,    code_vsub_02);
    exit(0);
*/

    printf("+------------------------------\n");
    printf("| vneg\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vneg_0);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vneg_2);

    printf("+------------------------------\n");
    printf("| vbnot\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbnot_0);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbnot_2);

    printf("+------------------------------\n");
    printf("| vadd\n");
    printf("+------------------------------\n");
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_00);    
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_01);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_10);    
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_02);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_21);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_12);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vadd_22);	

    printf("+------------------------------\n");
    printf("| vsub\n");
    printf("+------------------------------\n");
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], all_types[t], code_vsub_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], all_types[t], code_vsub_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], all_types[t], code_vsub_10);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], all_types[t], code_vsub_02);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], all_types[t], code_vsub_21);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], all_types[t], code_vsub_12);    
    for (t = 0; all_types[t] != VOID; t++)	
	test_instr(all_types[t], all_types[t], code_vsub_22);	
    
    printf("+------------------------------\n");
    printf("| veq\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_veq_00);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_veq_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_veq_10);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_veq_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_veq_22);

    printf("+------------------------------\n");
    printf("| vcmplt\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vlt_00);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vlt_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vlt_10);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vlt_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vlt_22);    
    
    printf("+------------------------------\n");
    printf("| vcmple\n");
    printf("+------------------------------\n");
    
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vle_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vle_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vle_10);    
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vle_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vle_22);        
    printf("+------------------------------\n");
    printf("| vband\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vband_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vband_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vband_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vband_22);          
    
    printf("+------------------------------\n");
    printf("| vbor\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbor_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbor_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbor_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbor_22);
    
    printf("+------------------------------\n");
    printf("| vbxor\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbxor_01);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbxor_00);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbxor_21);
    for (t = 0; all_types[t] != VOID; t++) 
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vbxor_22);

    
    printf("+------------------------------\n");
    printf("| vmul\n");
    printf("+------------------------------\n");

    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_00);    
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_01);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_10);    
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_02);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_12);    
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_21);
    for (t = 0; all_types[t] != VOID; t++)
	test_instr(all_types[t], unsigned_type(all_types[t]), code_vmul_22);
    return 0;    
}
