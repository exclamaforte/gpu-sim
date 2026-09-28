# Makefile for libfakegpu.so — the LD_PRELOAD interposer.
#
# Build:
#   make
#
# Run a CUDA binary against the fake GPU:
#   LD_PRELOAD=./libfakegpu.so FAKEGPU_NUM_DEVICES=2 ./my_cuda_binary
#
# Self-test (builds a tiny workload against shim/cuda_runtime.h):
#   make check

CC ?= gcc
NVCC ?= nvcc
CFLAGS  = -O2 -g -fPIC -Wall -Wextra -Wno-unused-parameter -fvisibility=default
LDFLAGS = -shared
INCLUDES = -Iinclude -Ishim

LIB = libfakegpu.so
SRCS = src/fakegpu_config.c src/fakegpu_random.c src/fakegpu_pool.c \
       src/fakegpu_latency.c src/fakegpu_cuda.c src/fakegpu_nccl.c
OBJS = $(SRCS:.c=.o)

TEST_BIN = build/fake_client
VEC_BIN  = build/vector_add
RING_BIN = build/ring_allreduce

.PHONY: all check clean

all: $(LIB)

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -ldl -lpthread

src/%.o: src/%.c include/fakegpu.h shim/cuda_runtime.h shim/nccl.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

tests/%: tests/%.c $(LIB) shim/cuda_runtime.h shim/nccl.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< -L. -lfakegpu -Wl,-rpath,$(CURDIR)

check: $(LIB)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $(TEST_BIN) tests/fake_client.c -L. -lfakegpu -Wl,-rpath,$(CURDIR)
	$(TEST_BIN)
	@echo "--- reference workloads under LD_PRELOAD ---"
	$(CC) $(CFLAGS) $(INCLUDES) -o $(VEC_BIN)  tests/vector_add.c     -L. -lfakegpu -Wl,-rpath,$(CURDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(RING_BIN) tests/ring_allreduce.c -L. -lfakegpu -Wl,-rpath,$(CURDIR)
	LD_PRELOAD=$(CURDIR)/$(LIB) FAKEGPU_NUM_DEVICES=4 $(VEC_BIN)
	LD_PRELOAD=$(CURDIR)/$(LIB) FAKEGPU_NUM_DEVICES=4 $(RING_BIN)
	@echo "--- -cudart=none fake path, no LD_PRELOAD (gpu-sim-u2o.6.1.3) ---"
	@bash tests/test_cudart_none.sh
	@echo "--- real-layout device-property overlay (gpu-sim-u2o.19) ---"
	@$(NVCC) -cudart=none -arch=sm_120 -Xlinker -rpath -Xlinker $(CURDIR) tests/test_device_prop_overlay.cu -o build/test_device_prop_overlay -L. -lfakegpu
	@./build/test_device_prop_overlay

clean:
	rm -f $(OBJS) $(LIB) src/*.o $(TEST_BIN) $(VEC_BIN) $(RING_BIN)
