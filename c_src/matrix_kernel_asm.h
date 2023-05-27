#ifndef __MATRIX_KERNEL_ASM__
#define __MATRIX_KERNEL_ASM__

#include <asmjit/x86.h>

#include "matrix_types.h"

using namespace asmjit;

#define VEC_TYPE_MMX    (1 << 0)
#define VEC_TYPE_SSE    (1 << 1)
#define VEC_TYPE_SSE2   (1 << 2)
#define VEC_TYPE_SSE3   (1 << 3)
#define VEC_TYPE_SSE4_1 (1 << 4)
#define VEC_TYPE_AVX    (1 << 5)
#define VEC_TYPE_AVX2   (1 << 6)
// all vector flags
#define VEC_TYPE_VEC    (0x7f)

class ZAssembler : public x86::Assembler {
    Zone*      z_;
    ConstPool* pool_;
    Label pool_label_;
    FuncFrame* frame_;
    unsigned vec_available;
    unsigned vec_enabled;
public:
    ZAssembler(CodeHolder* code, size_t zone_size) : x86::Assembler(code) {
	z_ = new Zone(zone_size);
	pool_ = new ConstPool(z_);
	pool_label_ = newLabel();
	frame_ = NULL;
	vec_available = 0;
	if (code->cpuFeatures().x86().hasMMX())
	    vec_available |= VEC_TYPE_AVX;
	if (code->cpuFeatures().x86().hasSSE())
	    vec_available |= VEC_TYPE_SSE;
	if (code->cpuFeatures().x86().hasSSE2())
	    vec_available |= VEC_TYPE_SSE2;
	if (code->cpuFeatures().x86().hasSSE3())
	    vec_available |= VEC_TYPE_SSE3;
	if (code->cpuFeatures().x86().hasSSE4_1())
	    vec_available |= VEC_TYPE_SSE4_1;
	if (code->cpuFeatures().x86().hasAVX())
	    vec_available |= VEC_TYPE_AVX;
	if (code->cpuFeatures().x86().hasAVX2())
	    vec_available |= VEC_TYPE_AVX2;
	vec_enabled = vec_available;
    }

    ~ZAssembler() {
	delete pool_;
	delete z_;
	// what to delete?
    }

    bool has_vec() { return (vec_available & VEC_TYPE_VEC) != 0; }
    bool has_mmx() { return (vec_available & VEC_TYPE_MMX) != 0; }    
    bool has_sse() { return (vec_available & VEC_TYPE_SSE) != 0; }
    bool has_sse2() { return (vec_available & VEC_TYPE_SSE2) != 0; }
    bool has_sse3() { return (vec_available & VEC_TYPE_SSE3) != 0; }
    bool has_sse4_1() { return (vec_available & VEC_TYPE_SSE4_1) != 0; }
    bool has_avx() { return (vec_available & VEC_TYPE_AVX) != 0; }
    bool has_avx2() { return (vec_available & VEC_TYPE_AVX2) != 0; }

    bool use_vec() { return (vec_enabled & VEC_TYPE_VEC) != 0; }
    bool use_mmx() { return (vec_enabled & VEC_TYPE_MMX) != 0; }    
    bool use_sse() { return (vec_enabled & VEC_TYPE_SSE) != 0; }
    bool use_sse2() { return (vec_enabled & VEC_TYPE_SSE2) != 0; }
    bool use_sse3() { return (vec_enabled & VEC_TYPE_SSE3) != 0; }
    bool use_sse4_1() { return (vec_enabled & VEC_TYPE_SSE4_1) != 0; }
    bool use_avx() { return (vec_enabled & VEC_TYPE_AVX) != 0; }
    bool use_avx2() { return (vec_enabled & VEC_TYPE_AVX2) != 0; }        

    void disable_vec() { vec_enabled &= ~(VEC_TYPE_VEC); }        
    void disable_mmx() { vec_enabled &= ~(VEC_TYPE_MMX); }    
    void disable_avx() { vec_enabled &= ~(VEC_TYPE_AVX); }
    void disable_avx2() { vec_enabled &= ~(VEC_TYPE_AVX2); }
    void disable_sse() { vec_enabled &= ~(VEC_TYPE_SSE); }
    void disable_sse2() { vec_enabled &= ~(VEC_TYPE_SSE2); }
    void disable_sse3() { vec_enabled &= ~(VEC_TYPE_SSE3); }    
    void disable_sse4_1() { vec_enabled &= ~(VEC_TYPE_SSE4_1); }

    void enable_vec() { vec_enabled |= (vec_available & VEC_TYPE_VEC); }        
    void enable_mmx() { vec_enabled |= (vec_available & VEC_TYPE_MMX); }        
    void enable_avx() { vec_enabled |= (vec_available & VEC_TYPE_AVX); }        
    void enable_avx2() {vec_enabled |= (vec_available & VEC_TYPE_AVX2); }
    void enable_sse() { vec_enabled |= (vec_available & VEC_TYPE_SSE); }    
    void enable_sse2() {vec_enabled |= (vec_available & VEC_TYPE_SSE2); }
    void enable_sse3() {vec_enabled |= (vec_available & VEC_TYPE_SSE3); }
    void enable_sse4_1() {vec_enabled |= (vec_available & VEC_TYPE_SSE4_1); }

    void set_func_frame(FuncFrame* frame) {
	frame_ = frame;
    }

    void add_dirty_reg(const BaseReg& reg) {
	frame_->addDirtyRegs(reg);
    }
    
    x86::Mem add_constant(void* data, size_t len) {
	size_t offset;
	pool_->add(data, len, offset);
	return x86::ptr(pool_label_, offset);
    }
    
    x86::Mem add_constant(float64_t value) {
	return add_constant(&value, sizeof(value));
    }
    
    x86::Mem add_constant(float32_t value) {
	return add_constant(&value, sizeof(value));
    }
    x86::Mem add_constant(int8_t value) {
	return add_constant(&value, sizeof(value));
    }
    x86::Mem add_constant(int16_t value) {
	return add_constant(&value, sizeof(value));
    }            
    x86::Mem add_constant(int32_t value) {
	return add_constant(&value, sizeof(value));
    }
    x86::Mem add_constant(int64_t value) {
	return add_constant(&value, sizeof(value));
    }

    x86::Mem add_vint8(int8_t value) {
	vint8_t vvalue = vint8_t_const(value);
	return add_constant(&vvalue, sizeof(vvalue));
    }
};

#endif
