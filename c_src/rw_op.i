
static int64_t read_int64_uint8(byte_t* ptr)
{
    return (int64_t) *((uint8_t*)ptr);
}

static int64_t read_int64_uint16(byte_t* ptr)
{
    return (int64_t) *((uint16_t*)ptr);
}

static int64_t read_int64_uint32(byte_t* ptr)
{
    return (int64_t) *((uint32_t*)ptr);
}

static int64_t read_int64_uint64(byte_t* ptr)
{
    return (int64_t) *((uint64_t*)ptr);
}

static int64_t read_int64_uint128(byte_t* ptr)
{
    return (int64_t) ((uint128_t*)ptr)->lo;
}

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

static int64_t read_int64_int128(byte_t* ptr)
{
    return (int64_t) ((int128_t*)ptr)->lo;
}

static int64_t read_int64_float32(byte_t* ptr)
{
    return (int64_t) *((float32_t*)ptr);
}

static int64_t read_int64_float64(byte_t* ptr)
{
    return (int64_t) *((float64_t*)ptr);
}

static uint64_t read_uint64_int8(byte_t* ptr)
{
    return (uint64_t) *((uint8_t*)ptr);
}

static uint64_t read_uint64_int16(byte_t* ptr)
{
    return (uint64_t) *((uint16_t*)ptr);
}

static uint64_t read_uint64_int32(byte_t* ptr)
{
    return (uint64_t) *((uint32_t*)ptr);
}

static uint64_t read_uint64_int64(byte_t* ptr)
{
    return (uint64_t) *((uint64_t*)ptr);
}

static uint64_t read_uint64_int128(byte_t* ptr)
{
    return (uint64_t) ((int128_t*)ptr)->lo;
}

static uint64_t read_uint64_uint8(byte_t* ptr)
{
    return (uint64_t) *((uint8_t*)ptr);
}

static uint64_t read_uint64_uint16(byte_t* ptr)
{
    return (uint64_t) *((uint16_t*)ptr);
}

static uint64_t read_uint64_uint32(byte_t* ptr)
{
    return (uint64_t) *((uint32_t*)ptr);
}

static uint64_t read_uint64_uint64(byte_t* ptr)
{
    return (uint64_t) *((uint64_t*)ptr);
}

static uint64_t read_uint64_uint128(byte_t* ptr)
{
    return (uint64_t) ((uint128_t*)ptr)->lo;
}


static uint64_t read_uint64_float32(byte_t* ptr)
{
    return (uint64_t) *((float32_t*)ptr);
}

static uint64_t read_uint64_float64(byte_t* ptr)
{
    return (uint64_t) *((float64_t*)ptr);
}

static float64_t read_float64_uint8(byte_t* ptr)
{
    return (float64_t) *((uint8_t*)ptr);
}

static float64_t read_float64_uint16(byte_t* ptr)
{
    return (float64_t) *((uint16_t*)ptr);
}

static float64_t read_float64_uint32(byte_t* ptr)
{
    return (float64_t) *((uint32_t*)ptr);
}

static float64_t read_float64_uint64(byte_t* ptr)
{
    return (float64_t) *((uint64_t*)ptr);
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

static float64_t read_float64_float16(byte_t* ptr)
{
    uint16_t val16 = *((uint16_t*) ptr);
    int s = (val16>>15);
    int e = ((val16>>10) & 0x1f)-((1<<4)-1);
    int m = val16 & 0x3ff;
    union {
	uint32_t u32;
	float32_t f32;
    } uf;
    // now make a float32_t
    uf.u32 = (s<<31)|((e+((1<<7)-1))<<23)|((m)<<(23-10));
    return (float64_t) uf.f32;
}

static float64_t read_float64_float32(byte_t* ptr)
{
    return (float64_t) *((float32_t*)ptr);
}

static float64_t read_float64_float64(byte_t* ptr)
{
    return (float64_t) *((float64_t*)ptr);
}

static void write_int64_uint8(byte_t* ptr, int64_t v)
{
    *((uint8_t*)ptr) = (uint8_t) v;
}

static void write_int64_uint16(byte_t* ptr, int64_t v)
{
    *((uint16_t*)ptr) = (uint16_t) v;
}

static void write_int64_uint32(byte_t* ptr, int64_t v)
{
    *((uint32_t*)ptr) = (uint32_t) v;
}

static void write_int64_uint64(byte_t* ptr, int64_t v)
{
    *((uint64_t*)ptr) = (uint64_t) v;
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

static void write_float64_uint8(byte_t* ptr, float64_t v)
{
    *((uint8_t*)ptr) = (uint8_t) v;
}

static void write_float64_uint16(byte_t* ptr, float64_t v)
{
    *((uint16_t*)ptr) = (uint16_t) v;
}

static void write_float64_uint32(byte_t* ptr, float64_t v)
{
    *((uint32_t*)ptr) = (uint32_t) v;
}

static void write_float64_uint64(byte_t* ptr, float64_t v)
{
    *((uint64_t*)ptr) = (uint64_t) v;
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

static void write_float64_float16(byte_t* ptr, float64_t v)
{
    int s, e, m;
    union {
	uint32_t u32;
	float32_t f32;
    } uf;
    uint32_t val32;
    uint16_t val16;
    uf.f32 = (float32_t) v;
    val32 = uf.u32;
    s = (val32>>31);
    e = ((val32>>23) & 0xff)-((1<<7)-1);
    m = val32 & 0x7fffff;
    val16 = (s<<15)|((e+((1<<4)-1))<<10)|((m)>>(23-10));    
    *((uint16_t*)ptr) = val16;
}


static void write_float64_float32(byte_t* ptr, float64_t v)
{
      *((float32_t*)ptr) = (float32_t) v;
}

static void write_float64_float64(byte_t* ptr, float64_t v)
{
      *((float64_t*)ptr) = (float64_t) v;
}

static uint64_t (*read_uint64_func[NUM_TYPES])(byte_t*) = {
    [UINT8] = read_uint64_uint8,
    [UINT16] = read_uint64_uint16,
    [UINT32] = read_uint64_uint32,
    [UINT64] = read_uint64_uint64,
    [UINT128] = read_uint64_uint128,
    [INT8] = read_uint64_int8,
    [INT16] = read_uint64_int16,
    [INT32] = read_uint64_int32,
    [INT64] = read_uint64_int64,
    [INT128] = read_uint64_int128,
    [FLOAT32] = read_uint64_float32,
    [FLOAT64] = read_uint64_float64,
};

static int64_t (*read_int64_func[NUM_TYPES])(byte_t*) = {
    [UINT8] = read_int64_uint8,
    [UINT16] = read_int64_uint16,
    [UINT32] = read_int64_uint32,
    [UINT64] = read_int64_uint64,
    [UINT128] = read_int64_uint128,
    [INT8] = read_int64_int8,
    [INT16] = read_int64_int16,
    [INT32] = read_int64_int32,
    [INT64] = read_int64_int64,
    [INT128] = read_int64_int128,
    [FLOAT32] = read_int64_float32,
    [FLOAT64] = read_int64_float64,
};

static void (*write_int64_func[NUM_TYPES])(byte_t*,int64_t) = {
    [UINT8] = write_int64_uint8,
    [UINT16] = write_int64_uint16,
    [UINT32] = write_int64_uint32,
    [UINT64] = write_int64_uint64,
    [INT8] = write_int64_int8,
    [INT16] = write_int64_int16,
    [INT32] = write_int64_int32,
    [INT64] = write_int64_int64,
    [FLOAT32] = write_int64_float32,
    [FLOAT64] = write_int64_float64,
};

static float64_t (*read_float64_func[NUM_TYPES])(byte_t*) = {
    [UINT8] = read_float64_uint8,
    [UINT16] = read_float64_uint16,
    [UINT32] = read_float64_uint32,
    [UINT64] = read_float64_uint64,

    [INT8] = read_float64_int8,
    [INT16] = read_float64_int16,
    [INT32] = read_float64_int32,
    [INT64] = read_float64_int64,

    [FLOAT16] = read_float64_float16,    
    [FLOAT32] = read_float64_float32,
    [FLOAT64] = read_float64_float64,
};

static void (*write_float64_func[NUM_TYPES])(byte_t*,float64_t) = {
    [UINT8] = write_float64_uint8,
    [UINT16] = write_float64_uint16,
    [UINT32] = write_float64_uint32,
    [UINT64] = write_float64_uint64,

    [INT8] = write_float64_int8,
    [INT16] = write_float64_int16,
    [INT32] = write_float64_int32,
    [INT64] = write_float64_int64,

    [FLOAT16] = write_float64_float16,
    [FLOAT32] = write_float64_float32,
    [FLOAT64] = write_float64_float64,
};

