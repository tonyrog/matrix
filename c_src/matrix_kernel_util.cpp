#include <stdio.h>
#include "matrix_types.h"
#include "matrix_kernel.h"

void set_vuint8(vuint8_t &r, int i, uint8_t v) { r[i] = v; }
void set_vuint16(vuint16_t &r, int i, uint16_t v) { r[i] = v; }
void set_vuint32(vuint32_t &r, int i, uint32_t v) { r[i] = v; }
void set_vuint64(vuint64_t &r, int i, uint64_t v) { r[i] = v; }
void set_vint8(vint8_t &r, int i, int8_t v) { r[i] = v; }
void set_vint16(vint16_t &r, int i, int16_t v) { r[i] = v; }
void set_vint32(vint32_t &r, int i, int32_t v) { r[i] = v; }
void set_vint64(vint64_t &r, int i, int64_t v) { r[i] = v; }
void set_vfloat32(vfloat32_t &r, int i, float32_t v) { r[i] = v; }
void set_vfloat64(vfloat64_t &r, int i, float64_t v) { r[i] = v; }

void set_element_uint64(matrix_type_t type, vector_t &r, int i, uint64_t v)
{
    switch(type) {
    case UINT8: set_vuint8((vuint8_t&)r, i, (uint8_t) v); break;
    case UINT16: set_vuint16((vuint16_t&)r, i, (uint16_t) v); break;
    case UINT32: set_vuint32((vuint32_t&)r, i, (uint32_t) v); break;
    case UINT64: set_vuint64((vuint64_t&)r, i, (uint64_t) v); break;
    case INT8: set_vint8((vint8_t&)r, i, (int8_t) v); break;
    case INT16: set_vint16((vint16_t&)r, i, (int16_t) v); break;
    case INT32: set_vint32((vint32_t&)r, i, (int32_t) v); break;
    case INT64: set_vint64((vint64_t&)r, i, (int64_t) v); break;
    case FLOAT32: set_vfloat32((vfloat32_t&)r, i, (float32_t) v); break;
    case FLOAT64: set_vfloat64((vfloat64_t&)r, i, (float64_t) v); break;
    default: break;
    }
}

void set_element_int64(matrix_type_t type, vector_t &r, int i, int64_t v)
{
    switch(type) {
    case UINT8:   set_vuint8((vuint8_t&)r,   i, (uint8_t) v); break;
    case UINT16:  set_vuint16((vuint16_t&)r, i, (uint16_t) v); break;
    case UINT32:  set_vuint32((vuint32_t&)r, i, (uint32_t) v); break;
    case UINT64:  set_vuint64((vuint64_t&)r, i, (uint64_t) v); break;
    case INT8:    set_vint8((vint8_t&)r,     i, (int8_t) v); break;
    case INT16:   set_vint16((vint16_t&)r,   i, (int16_t) v); break;
    case INT32:   set_vint32((vint32_t&)r,   i, (int32_t) v); break;
    case INT64:   set_vint64((vint64_t&)r,   i, (int64_t) v); break;
    case FLOAT32: set_vfloat32((vfloat32_t&)r, i, (float32_t) v); break;
    case FLOAT64: set_vfloat64((vfloat64_t&)r, i, (float64_t) v); break;
    default: break;
    }
}

void set_element_float64(matrix_type_t type, vector_t &r, int i, float64_t v)
{
    switch(type) {
    case UINT8: set_vuint8((vuint8_t&)r, i, (uint8_t) v); break;
    case UINT16: set_vuint16((vuint16_t&)r, i, (uint16_t) v); break;
    case UINT32: set_vuint32((vuint32_t&)r, i, (uint32_t) v); break;
    case UINT64: set_vuint64((vuint64_t&)r, i, (uint64_t) v); break;
    case INT8: set_vint8((vint8_t&)r, i, (int8_t) v); break;
    case INT16: set_vint16((vint16_t&)r, i, (int16_t) v); break;
    case INT32: set_vint32((vint32_t&)r, i, (int32_t) v); break;
    case INT64: set_vint64((vint64_t&)r, i, (int64_t) v); break;
    case FLOAT32: set_vfloat32((vfloat32_t&)r, i, (float32_t) v); break;
    case FLOAT64: set_vfloat64((vfloat64_t&)r, i, (float64_t) v); break;
    default: break;
    }
}

uint8_t   get_vuint8(vuint8_t r, int i)     { return r[i]; }
uint16_t  get_vuint16(vuint16_t r, int i)   { return r[i]; }
uint32_t  get_vuint32(vuint32_t r, int i)   { return r[i]; }
uint64_t  get_vuint64(vuint64_t r, int i)   { return r[i]; }
int8_t    get_vint8(vint8_t r, int i)       { return r[i]; }
int16_t   get_vint16(vint16_t r, int i)     { return r[i]; }
int32_t   get_vint32(vint32_t r, int i)     { return r[i]; }
int64_t   get_vint64(vint64_t r, int i)     { return r[i]; }
float32_t get_vfloat32(vfloat32_t r, int i) { return r[i]; }
float64_t get_vfloat64(vfloat64_t r, int i) { return r[i]; }

int64_t get_element_int64(matrix_type_t type, vector_t r, int i)
{
    switch(type) {
    case UINT8: return   get_vuint8((vuint8_t)r, i); 
    case UINT16: return  get_vuint16((vuint16_t)r, i);
    case UINT32: return  get_vuint32((vuint32_t)r, i);
    case UINT64: return  get_vuint64((vuint64_t)r, i);
    case INT8: return    get_vint8((vint8_t)r, i);
    case INT16: return   get_vint16((vint16_t)r, i);
    case INT32: return   get_vint32((vint32_t)r, i);
    case INT64: return   get_vint64((vint64_t)r, i);
    case FLOAT32: return get_vfloat32((vfloat32_t)r, i);
    case FLOAT64: return get_vfloat64((vfloat64_t)r, i);
    default: return 0;
    }
}

uint64_t get_element_uint64(matrix_type_t type, vector_t r, int i)
{
    switch(type) {
    case UINT8: return get_vuint8((vuint8_t)r, i); 
    case UINT16: return get_vuint16((vuint16_t)r, i);
    case UINT32: return  get_vuint32((vuint32_t)r, i);
    case UINT64: return get_vuint64((vuint64_t)r, i);
    case INT8: return get_vint8((vint8_t)r, i);
    case INT16: return get_vint16((vint16_t)r, i);
    case INT32: return get_vint32((vint32_t)r, i);
    case INT64: return get_vint64((vint64_t)r, i);
    case FLOAT32: return get_vfloat32((vfloat32_t)r, i);
    case FLOAT64: return get_vfloat64((vfloat64_t)r, i);
    default: return 0;	
    }
}

float64_t get_elemen_float64(matrix_type_t type, vector_t r, int i)
{
    switch(type) {
    case UINT8: return get_vuint8((vuint8_t)r, i); 
    case UINT16: return get_vuint16((vuint16_t)r, i);
    case UINT32: return  get_vuint32((vuint32_t)r, i);
    case UINT64: return get_vuint64((vuint64_t)r, i);
    case INT8: return get_vint8((vint8_t)r, i);
    case INT16: return get_vint16((vint16_t)r, i);
    case INT32: return get_vint32((vint32_t)r, i);
    case INT64: return get_vint64((vint64_t)r, i);
    case FLOAT32: return get_vfloat32((vfloat32_t)r, i);
    case FLOAT64: return get_vfloat64((vfloat64_t)r, i);
    default: return 0;
    }
}

void print_vuint8(FILE* f,vuint8_t r)
{
    fprintf(f,"{%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x,%x}",
	   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
	   r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]);
}

void print_vuint16(FILE* f,vuint16_t r)
{
    fprintf(f,"{%x,%x,%x,%x,%x,%x,%x,%x}",
	   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

void print_vuint32(FILE* f,vuint32_t r)
{
    fprintf(f,"{%x,%x,%x,%x}",
	   r[0], r[1], r[2], r[3]);
}

void print_vuint64(FILE* f,vuint64_t r)
{
    fprintf(f,"{%lx,%lx}",
	   r[0], r[1]);
}

void print_vint8(FILE* f,vint8_t r)
{
    fprintf(f,"{%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}",
	   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7],
	   r[8], r[9], r[10], r[11], r[12], r[13], r[14], r[15]);
}

void print_vint16(FILE* f,vint16_t r)
{
    fprintf(f,"{%d,%d,%d,%d,%d,%d,%d,%d}",
	   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
}

void print_vint32(FILE* f,vint32_t r)
{
    fprintf(f,"{%d,%d,%d,%d}",
	   r[0], r[1], r[2], r[3]);
}

void print_vint64(FILE* f,vint64_t r)
{
    fprintf(f,"{%ld,%ld}",
	   r[0], r[1]);
}

void print_vfloat32(FILE* f,vfloat32_t r)
{
    fprintf(f,"{%f,%f,%f,%f}",
	   r[0], r[1], r[2], r[3]);
}

void print_vfloat64(FILE* f,vfloat64_t r)
{
    fprintf(f,"{%f,%f}",
	   r[0], r[1]);
}

void vprint(FILE* f, uint8_t type, vector_t v)
{
    switch(type) {
    case UINT8: print_vuint8(f,(vuint8_t) v); break;
    case UINT16: print_vuint16(f,(vuint16_t) v); break;
    case UINT32: print_vuint32(f,(vuint32_t) v); break;
    case UINT64: print_vuint64(f,(vuint64_t) v); break;	
    case INT8: print_vint8(f,(vint8_t) v); break;
    case INT16: print_vint16(f,(vint16_t) v); break;
    case INT32: print_vint32(f,(vint32_t) v); break;
    case INT64: print_vint64(f,(vint64_t) v); break;
    case FLOAT32: print_vfloat32(f,(vfloat32_t) v); break;
    case FLOAT64: print_vfloat64(f,(vfloat64_t) v); break;
    default: break;
    }
}

void sprint(FILE* f, uint8_t type, scalar0_t v)
{
    switch(type) {
    case UINT8: fprintf(f, "%x", v.u8); break;
    case UINT16: fprintf(f, "%x", v.u16); break;
    case UINT32: fprintf(f, "%x", v.u32); break;
    case UINT64: fprintf(f, "%lx", v.u64); break;
    case INT8: fprintf(f, "%d", v.i8); break;
    case INT16: fprintf(f, "%d", v.i16); break;
    case INT32: fprintf(f, "%d", v.i32); break;
    case INT64: fprintf(f, "%ld", v.i64); break;
    case FLOAT32: fprintf(f, "%f", v.f32);break;
    case FLOAT64: fprintf(f, "%f", v.f64);break;
    default: break;
    }
}

#define VCMP_BODY(v1,v2) do { \
	unsigned k;				\
	for (k=0; k<VSIZE/sizeof(v1[0]); k++) {	\
	    if (v1[k] < v2[k]) return -1;	\
	    else if (v1[k] > v2[k]) return 1;	\
	    else return 0;			\
	}					\
    } while(0)

int cmp_vuint8(vuint8_t v1, vuint8_t v2)    { VCMP_BODY(v1,v2); return 0; }
int cmp_vuint16(vuint16_t v1, vuint16_t v2) { VCMP_BODY(v1,v2); return 0; }
int cmp_vuint32(vuint32_t v1, vuint32_t v2) { VCMP_BODY(v1,v2); return 0; }
int cmp_vuint64(vuint64_t v1, vuint64_t v2) { VCMP_BODY(v1,v2); return 0; }
int cmp_vint8(vint8_t v1, vint8_t v2)       { VCMP_BODY(v1,v2); return 0; }
int cmp_vint16(vint16_t v1, vint16_t v2)    { VCMP_BODY(v1,v2); return 0; }
int cmp_vint32(vint32_t v1, vint32_t v2)    { VCMP_BODY(v1,v2); return 0; }
int cmp_vint64(vint64_t v1, vint64_t v2)    { VCMP_BODY(v1,v2); return 0; }
int cmp_vfloat32(vfloat32_t v1, vfloat32_t v2) { VCMP_BODY(v1,v2); return 0; }
int cmp_vfloat64(vfloat64_t v1, vfloat64_t v2) { VCMP_BODY(v1,v2); return 0; }

int vcmp(uint8_t type, vector_t v1, vector_t v2)
{
    switch(type) {
    case UINT8:  return cmp_vuint8((vuint8_t) v1, (vuint8_t) v2);
    case UINT16: return cmp_vuint16((vuint16_t) v1, (vuint16_t) v2);
    case UINT32: return cmp_vuint32((vuint32_t) v1, (vuint32_t) v2);
    case UINT64: return cmp_vuint64((vuint64_t) v1, (vuint64_t) v2);
    case INT8:   return cmp_vint8((vint8_t) v1, (vint8_t) v2);
    case INT16:  return cmp_vint16((vint16_t) v1, (vint16_t) v2);
    case INT32:  return cmp_vint32((vint32_t) v1, (vint32_t) v2);
    case INT64:  return cmp_vint64((vint64_t) v1, (vint64_t) v2);
    case FLOAT32: return cmp_vfloat32((vfloat32_t) v1, (vfloat32_t) v2);
    case FLOAT64: return cmp_vfloat64((vfloat64_t) v1, (vfloat64_t) v2);
    default: break;
    }
    return -1;
}

#define CMP(a,b) (((a)<(b)) ? -1 : ( ((a)>(b)) ? 1 : 0))

int scmp(uint8_t type, scalar0_t v1, scalar0_t v2)
{
    switch(type) {
    case UINT8:  return CMP(v1.u8, v2.u8);
    case UINT16: return CMP(v1.u16, v2.u16);
    case UINT32: return CMP(v1.u32, v2.u32);
    case UINT64: return CMP(v1.u64, v2.u64); 
    case INT8:   return CMP(v1.i8, v2.i8);
    case INT16:  return CMP(v1.i16, v2.i16);
    case INT32:  return CMP(v1.i32, v2.i32);
    case INT64:  return CMP(v1.i64, v2.i64); 
    case FLOAT32: return CMP(v1.f32, v2.f32); 
    case FLOAT64: return CMP(v1.f64, v2.f64); 
    default: break;
    }
    return -1;
}

