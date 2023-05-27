#@BEGIN-C_SRC-DEFAULT-RULES@
ERL=$(shell readlink -f `which erl`)
ERL_TOP=$(ERL:%/bin/erl=%)
OS=$(shell uname -s)
CFLAGS=-MMD -MP -MF .$<.d -I$(ERL_TOP)/usr/include
#@END-C_SRC-DEFAULT-RULES@

CXX=g++

APP=matrix
CFLAGS+= -Wall -Wextra -Wswitch-enum -Wswitch-default -fno-common -O2 # --std=gnu11
CFLAGS+= -msse3 # -mavx2  # -msse3
# raspberry pi
# CFLAGS+=-mfloat-abi=softfp -mfpu=neon -flax-vector-conversions
# CFLAGS+=-mfpu=neon -flax-vector-conversions
CXXFLAGS+= -Wall -Wextra -Wswitch-enum -Wswitch-default -fno-common -g #-O2
CXXFLAGS+= -msse3

LDFLAGS+=-shared

OBJS = $(APP)_nif.o
LIBS =
KOBJS = matrix_kernel_x86.o matrix_kernel_emu.o matrix_kernel_util.o \
	matrix_kernel.o
KLIBS = -lasmjit

all: ../priv/$(APP)_nif.so

kernel: ../priv/kernel

../priv/$(APP)_nif.so: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) $(LIBS) -o $@

../priv/kernel:	$(KOBJS)
	$(CXX) $(KOBJS) $(KLIBS) -g -o $ $@

matrix_kernel.o:	matrix_kernel.h matrix_types.h matrix_kernel_asm.h
matrix_kernel_util.o:   matrix_kernel.h matrix_types.h
matrix_kernel_emu.o:   matrix_kernel.h matrix_types.h
matrix_kernel_x86.o:	matrix_kernel.h matrix_types.h matrix_kernel_asm.h

%.o:	%.c
	$(CC) $(CFLAGS) -c -fPIC $<

%.o:	%.cpp
	$(CXX) $(CXXFLAGS) -c -fPIC $<

.%.d:	;

-include .*.d
