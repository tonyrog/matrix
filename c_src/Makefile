#@BEGIN-C_SRC-DEFAULT-RULES@
ERL=$(shell readlink -f `which erl`)
ERL_TOP=$(ERL:%/bin/erl=%)
OS=$(shell uname -s)
CFLAGS=-MMD -MP -MF .$<.d -I$(ERL_TOP)/usr/include
#@END-C_SRC-DEFAULT-RULES@

APP=matrix
CFLAGS+=-Wall -Wextra -Wswitch-enum -Wswitch-default -fno-common -O3
CFLAGS+=-msse3 -mavx2
# raspberry pi
# CFLAGS+=-mfloat-abi=softfp -mfpu=neon -flax-vector-conversions
# CFLAGS+=-mfpu=neon -flax-vector-conversions
LDFLAGS+=-shared

OBJS = $(APP)_nif.o
LIBS = 

all: ../priv/$(APP)_nif.so

../priv/$(APP)_nif.so: $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) $(LIBS) -o ../priv/$(APP)_nif.so

%.o:	%.c
	$(CC) $(CFLAGS) -c -fPIC $<

.%.d:	;

-include .*.d
