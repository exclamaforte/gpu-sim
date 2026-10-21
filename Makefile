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
#
# TREE-WIDE POLICY (gpu-sim-xy8): -Werror=return-type is in CFLAGS below.
# Falling off the end of a non-void function is UB no test can reliably
# catch (gpu-sim-urr's miss test PASSED on the broken code -- rax luck;
# only the flag's rc=2 build failure caught it). A clean full rebuild from
# zero .o files produces 0 warnings, which is what makes adopting it free.
# Deliberately NOT blanket -Werror: one warning class as an error has a
# known blast radius; blanket suppression would hide unrelated warnings
# people may care about. If you add a C file, it builds under this flag.

CC ?= gcc
NVCC ?= nvcc
CFLAGS  = -O2 -g -fPIC -Wall -Wextra -Wno-unused-parameter -fvisibility=default -Werror=return-type
# -Werror=return-type (gpu-sim-urr rework): falling off the end of a non-void
# function is UB the test suite cannot reliably catch (a negative control
# showed the miss test PASSING on the broken code -- rax residue luck). The
# flag makes the class a build failure instead. Verified clean on all SRCS.
LDFLAGS = -shared -lm
INCLUDES = -Iinclude -Ishim -I/home/gabe/.local/cuda-sysroot/opt/cuda/targets/x86_64-linux/include

LIB = libfakegpu.so
SRCS = src/fakegpu_config.c src/fakegpu_random.c src/fakegpu_pool.c \
       src/fakegpu_latency.c src/fakegpu_cuda.c src/fakegpu_nccl.c \
       src/fakegpu_topology.c
OBJS = $(SRCS:.c=.o)

TEST_BIN = build/fake_client
VEC_BIN  = build/vector_add
RING_BIN = build/ring_allreduce
FABRIC_TEST_BIN = build/test_fabric

.PHONY: all check clean

all: $(LIB) libfakecuda.so

$(LIB): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $(OBJS) -ldl -lpthread

src/%.o: src/%.c include/fakegpu.h shim/cuda_runtime.h shim/nccl.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

tests/%: tests/%.c $(LIB) shim/cuda_runtime.h shim/nccl.h
	@mkdir -p build
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $< -L. -lfakegpu -Wl,-rpath,$(CURDIR)

# gpu-sim-u2o.20: the build inputs must be TRACKED, or `make check` below is
# evidence about one working tree rather than about the repository. A clean
# clone could not build libfakegpu.so at all until the five untracked SRCS
# files, the header and the three test programs were committed, so every
# "make check passes" claim made before that -- including the reviewer's own
# validation of gpu-sim-8dw -- was true here and unreproducible elsewhere.
check: guard-build-inputs $(LIB)
	@python3 tests/test_model_fault.py
	@python3 tests/test_model_agreement.py
	@python3 scripts/check_guard_uniqueness.py
	@mkdir -p build
	@$(CC) $(CFLAGS) $(INCLUDES) -o build/test_nccl_refusal tests/test_nccl_refusal.c -L. -lfakegpu -Wl,-rpath,$(CURDIR) -lpthread -lrt -ldl -lm
	@build/test_nccl_refusal
	@python3 scripts/gen_growth_list.py --check
	@python3 experiments/gpu-sim-6kp/test_net_plugin.py
	@mkdir -p build
	@$(CC) $(CFLAGS) -Iinclude -o build/test_retransmit \
	    experiments/gpu-sim-4tw/test_retransmit.c src/fakegpu_fabric.c -lpthread
	@build/test_retransmit
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
	@echo "--- shim device-property sizeof (gpu-sim-m64) ---"
	@$(CC) $(CFLAGS) $(INCLUDES) -o build/test_shim_prop_size tests/test_shim_prop_size.c
	@./build/test_shim_prop_size
	@echo "--- resolver attachment (gpu-sim-urr) ---"
	@$(CC) $(CFLAGS) $(INCLUDES) -o build/test_resolver_attach tests/test_resolver_attach.c -L. -lfakegpu -Wl,-rpath,$(CURDIR)
	@./build/test_resolver_attach

.PHONY: guard-build-inputs
# The guard runs FIRST, as a prerequisite: a red guard stops the suite before
# it can go green on untracked state. (An earlier form of this wiring replaced
# `check`'s recipe with the guard alone, so `make check` passed without
# building or testing anything -- the vacuous gate this bead exists to kill.
# The 2026-10-04 rebase caught it because the clean-clone log held only the
# guard's 14 lines.)
guard-build-inputs:
	bash tests/assert_build_inputs_tracked.sh

# No `check-only` (suite without the guard) is offered: a guard-free gate
# alongside the guarded one is how `make check` silently stops meaning
# anything the next time someone reaches for the shorter name.

.PHONY: test-fabric
test-fabric: $(FABRIC_TEST_BIN)
	@echo "--- fabric multi-process tests ---"
	@for t in basic barrier8 killpeer stale size; do \
	  ./$(FABRIC_TEST_BIN) $$t || exit 1; \
	done
	@echo "fabric tests PASS"

$(FABRIC_TEST_BIN): tests/test_fabric.c src/fakegpu_fabric.c include/fakegpu_fabric.h
	@mkdir -p build
	$(CC) $(CFLAGS) -Iinclude -o $@ tests/test_fabric.c src/fakegpu_fabric.c -lpthread


libfakecuda.so: src/fakecuda.o src/fakegpu_config.o src/fakegpu_random.o src/fakegpu_pool.o src/fakegpu_latency.o src/fakegpu_topology.o
	$(CC) $(LDFLAGS) -o $@ $^ -ldl -lpthread -lm

src/fakecuda.o: src/fakecuda.c include/fakegpu.h include/fakegpu_topology.h
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) $(LIB) src/*.o $(TEST_BIN) $(VEC_BIN) $(RING_BIN) libfakecuda.so
