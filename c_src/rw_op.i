
static int64_t read_int64_int8(byte_t* ptr)
{
    return (int64_t) *((int8_t*)ptr);
}

static int64_t read_int64_int16(byte_t* ptr)
{
    return (int64_t) *((int16_t*)ptr);
}

static int64_t read_int64_int32(byte_t* ptr)
{
    return (int64_t) *((int32_t*)ptr);
}

static int64_t read_int64_int64(byte_t* ptr)
{
    return (int64_t) *((int64_t*)ptr);
}

static int64_t read_int64_float32(byte_t* ptr)
{
    return (int64_t) *((float32_t*)ptr);
}

static int64_t read_int64_float64(byte_t* ptr)
{
    return (int64_t) *((float64_t*)ptr);
}

static int64_t read_int64_complex64(byte_t* ptr)
{
    return (int64_t) crealf(*((complex64_t*)ptr));
}

static int64_t read_int64_complex128(byte_t* ptr)
{
    return (int64_t) creal(*((complex128_t*)ptr));
}

static float64_t read_float64_int8(byte_t* ptr)
{
    return (float64_t) *((int8_t*)ptr);
}

static float64_t read_float64_int16(byte_t* ptr)
{
    return (float64_t) *((int16_t*)ptr);
}

static float64_t read_float64_int32(byte_t* ptr)
{
    return (float64_t) *((int32_t*)ptr);
}

static float64_t read_float64_int64(byte_t* ptr)
{
    return (float64_t) *((int64_t*)ptr);
}

static float64_t read_float64_float32(byte_t* ptr)
{
    return (float64_t) *((float32_t*)ptr);
}

static float64_t read_float64_float64(byte_t* ptr)
{
    return (float64_t) *((float64_t*)ptr);
}

static float64_t read_float64_complex64(byte_t* ptr)
{
    return (float64_t) crealf(*((complex64_t*)ptr));
}

static float64_t read_float64_complex128(byte_t* ptr)
{
    return (float64_t) creal(*((complex128_t*)ptr));
}

static complex128_t read_complex128_int8(byte_t* ptr)
{
    return CMPLX(*((int8_t*)ptr),0.0);
}

static complex128_t read_complex128_int16(byte_t* ptr)
{
    return CMPLX(*((int16_t*)ptr),0.0);
}

static complex128_t read_complex128_int32(byte_t* ptr)
{
    return CMPLX(*((int32_t*)ptr),0.0);
}

static complex128_t read_complex128_int64(byte_t* ptr)
{
    return CMPLX(*((int64_t*)ptr),0.0);
}

static complex128_t read_complex128_float32(byte_t* ptr)
{
    return CMPLX(*((float32_t*)ptr),0.0);
}

static complex128_t read_complex128_float64(byte_t* ptr)
{
    return CMPLX(*((float64_t*)ptr),0.0);
}

static complex128_t read_complex128_complex64(byte_t* ptr)
{
    return (complex128_t) *((complex64_t*)ptr);
}

static complex128_t read_complex128_complex128(byte_t* ptr)
{
    return (complex128_t) *((complex128_t*)ptr);
}

static void write_int64_int8(byte_t* ptr, int64_t v)
{
      *((int8_t*)ptr) = (int8_t) v;
}

static void write_int64_int16(byte_t* ptr, int64_t v)
{
      *((int16_t*)ptr) = (int16_t) v;
}

static void write_int64_int32(byte_t* ptr, int64_t v)
{
      *((int32_t*)ptr) = (int32_t) v;
}

static void write_int64_int64(byte_t* ptr, int64_t v)
{
      *((int64_t*)ptr) = (int64_t) v;
}

static void write_int64_float32(byte_t* ptr, int64_t v)
{
      *((float32_t*)ptr) = (float32_t) v;
}

static void write_int64_float64(byte_t* ptr, int64_t v)
{
      *((float64_t*)ptr) = (float64_t) v;
}

static void write_int64_complex64(byte_t* ptr, int64_t v)
{
    *((complex64_t*)ptr) = CMPLXF((float32_t) v, 0.0);
}

static void write_int64_complex128(byte_t* ptr, int64_t v)
{
    *((complex128_t*)ptr) = CMPLX((float64_t) v, 0.0);
}

static void write_float64_int8(byte_t* ptr, float64_t v)
{
      *((int8_t*)ptr) = (int8_t) v;
}

static void write_float64_int16(byte_t* ptr, float64_t v)
{
      *((int16_t*)ptr) = (int16_t) v;
}

static void write_float64_int32(byte_t* ptr, float64_t v)
{
      *((int32_t*)ptr) = (int32_t) v;
}

static void write_float64_int64(byte_t* ptr, float64_t v)
{
      *((int64_t*)ptr) = (int64_t) v;
}

static void write_float64_float32(byte_t* ptr, float64_t v)
{
      *((float32_t*)ptr) = (float32_t) v;
}

static void write_float64_float64(byte_t* ptr, float64_t v)
{
      *((float64_t*)ptr) = (float64_t) v;
}

static void write_float64_complex64(byte_t* ptr, float64_t v)
{
    *((complex64_t*)ptr) = CMPLXF((float32_t) v, 0.0);
}

static void write_float64_complex128(byte_t* ptr, float64_t v)
{
    *((complex128_t*)ptr) = CMPLX((float64_t) v, 0.0);
}

static void write_complex128_int8(byte_t* ptr, complex128_t v)
{
      *((int8_t*)ptr) = (int8_t) creal(v);
}

static void write_complex128_int16(byte_t* ptr, complex128_t v)
{
      *((int16_t*)ptr) = (int16_t) creal(v);
}

static void write_complex128_int32(byte_t* ptr, complex128_t v)
{
      *((int32_t*)ptr) = (int32_t) creal(v);
}

static void write_complex128_int64(byte_t* ptr, complex128_t v)
{
      *((int64_t*)ptr) = (int64_t) creal(v);
}

static void write_complex128_float32(byte_t* ptr, complex128_t v)
{
      *((float32_t*)ptr) = (float32_t) creal(v);
}

static void write_complex128_float64(byte_t* ptr, complex128_t v)
{
      *((float64_t*)ptr) = (float64_t) creal(v);
}

static void write_complex128_complex64(byte_t* ptr, complex128_t v)
{
  *((complex64_t*)ptr) = (complex64_t) v;
}

static void write_complex128_complex128(byte_t* ptr, complex128_t v)
{
    *((complex128_t*)ptr) = (complex128_t) v;
}

static int64_t (*read_int64_func[NUM_TYPES])(byte_t*) = {
  [INT8] = read_int64_int8,
  [INT16] = read_int64_int16,
  [INT32] = read_int64_int32,
  [INT64] = read_int64_int64,
  [FLOAT32] = read_int64_float32,
  [FLOAT64] = read_int64_float64,
  [COMPLEX64] = read_int64_complex64,
  [COMPLEX128] = read_int64_complex128,
};

static void (*write_int64_func[NUM_TYPES])(byte_t*,int64_t) = {
  [INT8] = write_int64_int8,
  [INT16] = write_int64_int16,
  [INT32] = write_int64_int32,
  [INT64] = write_int64_int64,
  [FLOAT32] = write_int64_float32,
  [FLOAT64] = write_int64_float64,
  [COMPLEX64] = write_int64_complex64,
  [COMPLEX128] = write_int64_complex128,
};

static float64_t (*read_float64_func[NUM_TYPES])(byte_t*) = {
  [INT8] = read_float64_int8,
  [INT16] = read_float64_int16,
  [INT32] = read_float64_int32,
  [INT64] = read_float64_int64,
  [FLOAT32] = read_float64_float32,
  [FLOAT64] = read_float64_float64,
  [COMPLEX64] = read_float64_complex64,
  [COMPLEX128] = read_float64_complex128,
};

static void (*write_float64_func[NUM_TYPES])(byte_t*,float64_t) = {
  [INT8] = write_float64_int8,
  [INT16] = write_float64_int16,
  [INT32] = write_float64_int32,
  [INT64] = write_float64_int64,
  [FLOAT32] = write_float64_float32,
  [FLOAT64] = write_float64_float64,
  [COMPLEX64] = write_float64_complex64,
  [COMPLEX128] = write_float64_complex128,
};


static complex128_t (*read_complex128_func[NUM_TYPES])(byte_t*) = {
  [INT8] = read_complex128_int8,
  [INT16] = read_complex128_int16,
  [INT32] = read_complex128_int32,
  [INT64] = read_complex128_int64,
  [FLOAT32] = read_complex128_float32,
  [FLOAT64] = read_complex128_float64,
  [COMPLEX64] = read_complex128_complex64,
  [COMPLEX128] = read_complex128_complex128,
};

static void (*write_complex128_func[NUM_TYPES])(byte_t*,complex128_t) = {
  [INT8] = write_complex128_int8,
  [INT16] = write_complex128_int16,
  [INT32] = write_complex128_int32,
  [INT64] = write_complex128_int64,
  [FLOAT32] = write_complex128_float32,
  [FLOAT64] = write_complex128_float64,
  [COMPLEX64] = write_complex128_complex64,
  [COMPLEX128] = write_complex128_complex128,
};
