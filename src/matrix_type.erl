-module(matrix_type).

%% "kernel" type code/decode
-compile(export_all).

-record(kernel_t,
	{
	 vsize = [1,2,3,4,8,16],
	 type  = [uint,int,float],
	 bsize = [8,16,32,64,128]
	}).

type_to_kernel(Type)                                           ->
    case Type of
	uint8 ->  #kernel_t{vsize=[1],type=[uint],bsize=[8]};
	uint16 -> #kernel_t{vsize=[1],type=[uint],bsize=[16]};
	uint32 -> #kernel_t{vsize=[1],type=[uint],bsize=[32]};
	uint64 -> #kernel_t{vsize=[1],type=[uint],bsize=[64]};
	uint128 -> #kernel_t{vsize=[1],type=[uint],bsize=[128]};

	int8 ->  #kernel_t{vsize=[1],type=[int],bsize=[8]};
	int16 -> #kernel_t{vsize=[1],type=[int],bsize=[16]};
	int32 -> #kernel_t{vsize=[1],type=[int],bsize=[32]};
	int64 -> #kernel_t{vsize=[1],type=[int],bsize=[64]};
	int128 -> #kernel_t{vsize=[1],type=[int],bsize=[128]};

	float16 -> #kernel_t{vsize=[1],type=[float],bsize=[16]};
	float32 -> #kernel_t{vsize=[1],type=[float],bsize=[32]};
	float64 -> #kernel_t{vsize=[1],type=[float],bsize=[64]};

	int8x2 -> #kernel_t{vsize=[2],type=[int],bsize=[8]};
	int8x3 -> #kernel_t{vsize=[3],type=[int],bsize=[8]};
	int8x4 -> #kernel_t{vsize=[4],type=[int],bsize=[8]};
	int8x8 -> #kernel_t{vsize=[8],type=[int],bsize=[8]};
	int8x16 -> #kernel_t{vsize=[16],type=[int],bsize=[8]};

	int16x2 -> #kernel_t{vsize=[2],type=[int],bsize=[16]};
	int16x3 -> #kernel_t{vsize=[3],type=[int],bsize=[16]};
	int16x4 -> #kernel_t{vsize=[4],type=[int],bsize=[16]};
	int16x8 -> #kernel_t{vsize=[8],type=[int],bsize=[16]};
	int16x16 -> #kernel_t{vsize=[16],type=[int],bsize=[16]};

	int32x2 -> #kernel_t{vsize=[2],type=[int],bsize=[32]};
	int32x3 -> #kernel_t{vsize=[3],type=[int],bsize=[32]};
	int32x4 -> #kernel_t{vsize=[4],type=[int],bsize=[32]};
	int32x8 -> #kernel_t{vsize=[8],type=[int],bsize=[32]};
	int32x16 -> #kernel_t{vsize=[16],type=[int],bsize=[32]};

	int64x2 -> #kernel_t{vsize=[2],type=[int],bsize=[64]};
	int64x3 -> #kernel_t{vsize=[3],type=[int],bsize=[64]};
	int64x4 -> #kernel_t{vsize=[4],type=[int],bsize=[64]};
	int64x8 -> #kernel_t{vsize=[8],type=[int],bsize=[64]};
	int64x16 -> #kernel_t{vsize=[16],type=[int],bsize=[64]};

	uint8x2 -> #kernel_t{vsize=[2],type=[uint],bsize=[8]};
	uint8x3 -> #kernel_t{vsize=[3],type=[uint],bsize=[8]};
	uint8x4 -> #kernel_t{vsize=[4],type=[uint],bsize=[8]};
	uint8x8 -> #kernel_t{vsize=[8],type=[uint],bsize=[8]};
	uint8x16 -> #kernel_t{vsize=[16],type=[uint],bsize=[8]};

	uint16x2 -> #kernel_t{vsize=[2],type=[uint],bsize=[16]};
	uint16x3 -> #kernel_t{vsize=[3],type=[uint],bsize=[16]};
	uint16x4 -> #kernel_t{vsize=[4],type=[uint],bsize=[16]};
	uint16x8 -> #kernel_t{vsize=[8],type=[uint],bsize=[16]};
	uint16x16 -> #kernel_t{vsize=[16],type=[uint],bsize=[16]};

	uint32x2 -> #kernel_t{vsize=[2],type=[uint],bsize=[32]};
	uint32x3 -> #kernel_t{vsize=[3],type=[uint],bsize=[32]};
	uint32x4 -> #kernel_t{vsize=[4],type=[uint],bsize=[32]};
	uint32x8 -> #kernel_t{vsize=[8],type=[uint],bsize=[32]};
	uint32x16 -> #kernel_t{vsize=[16],type=[uint],bsize=[32]};

	uint64x2 -> #kernel_t{vsize=[2],type=[uint],bsize=[64]};
	uint64x3 -> #kernel_t{vsize=[3],type=[uint],bsize=[64]};
	uint64x4 -> #kernel_t{vsize=[4],type=[uint],bsize=[64]};
	uint64x8 -> #kernel_t{vsize=[8],type=[uint],bsize=[64]};
	uint64x16 -> #kernel_t{vsize=[16],type=[uint],bsize=[64]};

	float16x2 -> #kernel_t{vsize=[2],type=[float],bsize=[16]};
	float16x3 -> #kernel_t{vsize=[3],type=[float],bsize=[16]};
	float16x4 -> #kernel_t{vsize=[4],type=[float],bsize=[16]};
	float16x8 -> #kernel_t{vsize=[8],type=[float],bsize=[16]};
	float16x16 -> #kernel_t{vsize=[16],type=[float],bsize=[16]};

	float32x2 -> #kernel_t{vsize=[2],type=[float],bsize=[32]};
	float32x3 -> #kernel_t{vsize=[3],type=[float],bsize=[32]};
	float32x4 -> #kernel_t{vsize=[4],type=[float],bsize=[32]};
	float32x8 -> #kernel_t{vsize=[8],type=[float],bsize=[32]};
	float32x16 -> #kernel_t{vsize=[16],type=[float],bsize=[32]};

	float64x2 -> #kernel_t{vsize=[2],type=[float],bsize=[64]};
	float64x3 -> #kernel_t{vsize=[3],type=[float],bsize=[64]};
	float64x4 -> #kernel_t{vsize=[4],type=[float],bsize=[64]};
	float64x8 -> #kernel_t{vsize=[8],type=[float],bsize=[64]};
	float64x16 -> #kernel_t{vsize=[16],type=[float],bsize=[64]}
    end.

kernel_to_type(T) ->
    case T of
	#kernel_t{vsize=[1],type=[uint],bsize=[8]} -> uint8;
	#kernel_t{vsize=[1],type=[uint],bsize=[16]} -> uint16;
	#kernel_t{vsize=[1],type=[uint],bsize=[32]} -> uint32;
	#kernel_t{vsize=[1],type=[uint],bsize=[64]} -> uint64;
	#kernel_t{vsize=[1],type=[uint],bsize=[128]} -> uint128;

	#kernel_t{vsize=[1],type=[int],bsize=[8]} -> int8;
	#kernel_t{vsize=[1],type=[int],bsize=[16]} -> int16;
	#kernel_t{vsize=[1],type=[int],bsize=[32]} -> int32;
	#kernel_t{vsize=[1],type=[int],bsize=[64]} -> int64;
	#kernel_t{vsize=[1],type=[int],bsize=[128]} -> int128;

	#kernel_t{vsize=[1],type=[float],bsize=[16]} -> float16;
	#kernel_t{vsize=[1],type=[float],bsize=[32]} -> float32;
	#kernel_t{vsize=[1],type=[float],bsize=[64]} -> float64;

	#kernel_t{vsize=[2],type=[int],bsize=[8]} -> int8x2;
	#kernel_t{vsize=[3],type=[int],bsize=[8]} -> int8x3;
	#kernel_t{vsize=[4],type=[int],bsize=[8]} -> int8x4;
	#kernel_t{vsize=[8],type=[int],bsize=[8]} -> int8x8;
	#kernel_t{vsize=[16],type=[int],bsize=[8]} -> int8x16;

	#kernel_t{vsize=[2],type=[int],bsize=[16]} -> int16x2;
	#kernel_t{vsize=[3],type=[int],bsize=[16]} -> int16x3;
	#kernel_t{vsize=[4],type=[int],bsize=[16]} -> int16x4;
	#kernel_t{vsize=[8],type=[int],bsize=[16]} -> int16x8;
	#kernel_t{vsize=[16],type=[int],bsize=[16]} -> int16x16;

	#kernel_t{vsize=[2],type=[int],bsize=[32]} -> int32x2;
	#kernel_t{vsize=[3],type=[int],bsize=[32]} -> int32x3;
	#kernel_t{vsize=[4],type=[int],bsize=[32]} -> int32x4;
	#kernel_t{vsize=[8],type=[int],bsize=[32]} -> int32x8;
	#kernel_t{vsize=[16],type=[int],bsize=[32]} -> int32x16;

	#kernel_t{vsize=[2],type=[int],bsize=[64]} -> int64x2;
	#kernel_t{vsize=[3],type=[int],bsize=[64]} -> int64x3;
	#kernel_t{vsize=[4],type=[int],bsize=[64]} -> int64x4;
	#kernel_t{vsize=[8],type=[int],bsize=[64]} -> int64x8;
	#kernel_t{vsize=[16],type=[int],bsize=[64]} -> int64x16;

	#kernel_t{vsize=[2],type=[uint],bsize=[8]} -> uint8x2;
	#kernel_t{vsize=[3],type=[uint],bsize=[8]} -> uint8x3;
	#kernel_t{vsize=[4],type=[uint],bsize=[8]} -> uint8x4;
	#kernel_t{vsize=[8],type=[uint],bsize=[8]} -> uint8x8;
	#kernel_t{vsize=[16],type=[uint],bsize=[8]} -> uint8x16;

	#kernel_t{vsize=[2],type=[uint],bsize=[16]} -> uint16x2;
	#kernel_t{vsize=[3],type=[uint],bsize=[16]} -> uint16x3;
	#kernel_t{vsize=[4],type=[uint],bsize=[16]} -> uint16x4;
	#kernel_t{vsize=[8],type=[uint],bsize=[16]} -> uint16x8;
	#kernel_t{vsize=[16],type=[uint],bsize=[16]} -> uint16x16;

	#kernel_t{vsize=[2],type=[uint],bsize=[32]} -> uint32x2;
	#kernel_t{vsize=[3],type=[uint],bsize=[32]} -> uint32x3;
	#kernel_t{vsize=[4],type=[uint],bsize=[32]} -> uint32x4;
	#kernel_t{vsize=[8],type=[uint],bsize=[32]} -> uint32x8;
	#kernel_t{vsize=[16],type=[uint],bsize=[32]} -> uint32x16;

	#kernel_t{vsize=[2],type=[uint],bsize=[64]} -> uint64x2;
	#kernel_t{vsize=[3],type=[uint],bsize=[64]} -> uint64x3;
	#kernel_t{vsize=[4],type=[uint],bsize=[64]} -> uint64x4;
	#kernel_t{vsize=[8],type=[uint],bsize=[64]} -> uint64x8;
	#kernel_t{vsize=[16],type=[uint],bsize=[64]} -> uint64x16;

	#kernel_t{vsize=[2],type=[float],bsize=[16]} -> float16x2;
	#kernel_t{vsize=[3],type=[float],bsize=[16]} -> float16x3;
	#kernel_t{vsize=[4],type=[float],bsize=[16]} -> float16x4;
	#kernel_t{vsize=[8],type=[float],bsize=[16]} -> float16x8;
	#kernel_t{vsize=[16],type=[float],bsize=[16]} -> float16x16;

	#kernel_t{vsize=[2],type=[float],bsize=[32]} -> float32x2;
	#kernel_t{vsize=[3],type=[float],bsize=[32]} -> float32x3;
	#kernel_t{vsize=[4],type=[float],bsize=[32]} -> float32x4;
	#kernel_t{vsize=[8],type=[float],bsize=[32]} -> float32x8;
	#kernel_t{vsize=[16],type=[float],bsize=[32]} -> float32x16;

	#kernel_t{vsize=[2],type=[float],bsize=[64]} -> float64x2;
	#kernel_t{vsize=[3],type=[float],bsize=[64]} -> float64x3;
	#kernel_t{vsize=[4],type=[float],bsize=[64]} -> float64x4;
	#kernel_t{vsize=[8],type=[float],bsize=[64]} -> float64x8;
	#kernel_t{vsize=[16],type=[float],bsize=[64]} -> float64x16
    end.    
    
