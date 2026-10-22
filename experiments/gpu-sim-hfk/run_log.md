# gpu-sim-hfk: Value correctness at >=512KiB on the SIMPLE path

## Environment
- Device: single H100 (GPU 0)
- NCCL proto: Simple (NCCL_PROTO=Simple)
- Transport: shm-fabric plugin (libncclNetSimFab.so)
- Size: 524288 bytes (131072 floats, 512 KiB)
- Ranks: 2
- Fill: non-uniform h[i] = val + i*0.001 (val=1.0 rank0, val=2.0 rank1)
- Expected: h[i] = 3.0 + 2*i*0.001

## Reproduction runs (3+)

### Run 1 (nu512_1)
- VALUE MISMATCH: 122880/131072 wrong, first idx=0 got=101.304001 (expect 3.000000)
- Decode: 101.304001 = 3.0 + 0.001*98304 (offset 0.75S = 98304 elements)

### Run 2 (nu512_2)
- VALUE MISMATCH: 98304/131072 wrong, first idx=8192 got=84.920006 (expect 19.384001)
- Decode: 84.920006 = 19.384 + 0.001*65536 (offset 0.5S = 65536 elements)

### Run 3 (nu512_3)
- VALUE MISMATCH: 65536/131072 wrong, first idx=16384 got=134.072006 (expect 35.768002)
- Decode: 134.072006 = 35.768 + 0.001*98304 (offset 0.75S = 98304 elements)

### Run 4 (nu512_test8, latest)
- VALUE MISMATCH: 65536/131072 wrong, first idx=16384 got=134.072006 (expect 35.768002)

## Localisation

The defect is in the shm-fabric plugin's data transfer path:
- `experiments/gpu-sim-m3d.8/src/ncclnetsim_fabric.c` v12_isend: uses `memcpy(m->slots[slot].data, buf, n)` where `buf` is a GPU pointer. memcpy on GPU memory reads garbage.
- v12_irecv (via fg_fab_recv_max_ex): uses `memcpy(buf, m->slots[slot].data, mlen)` where `buf` is a GPU pointer. memcpy to GPU memory writes garbage, corrupting NCCL's result.

The memcpy reads/writes to GPU memory without using cudaMemcpy, producing shifted/corrupted data. The corruption is non-deterministic (varies between runs), consistent with reading from GPU memory that is not CPU-accessible.

## Blast radius
- Any allreduce with non-uniform data at >=512KiB on the SIMPLE path produces wrong results.
- Uniform fill passes (all elements same value, so corruption is invisible).
- The defect is in the data path, not the reduction script.

## Root cause
The plugin treats GPU pointers as CPU-accessible memory. memcpy on GPU pointers reads/writes garbage. The fix requires cudaMemcpy, but cudaMemcpy from within NCCL's execution context deadlocks (GPU primary context lock). A proper fix needs either:
1. A separate CUDA stream with async copy + synchronization
2. Moving the copy outside NCCL's execution context
3. A different plugin architecture that doesn't require CPU-side data access

## Note
The hfk bead asked to "localise it by instrumenting the chunk offset at send and at receive, not by reading code." Instrumentation confirmed: the send's first4 reads 17.384 (not 1.0) for a chunk starting at element 16384, confirming the memcpy reads from GPU memory at a shifted position. The recv's result contains 134.072 (sum of 131072 elements in one slot), confirming the memcpy writes garbage to GPU memory.
## Re-investigation (root cause corrected)

### Instrumentation findings
- NCCL_NET_PLUGIN points to hfk worktree's plugin (fixed path)
- mhandle=0x1 for ALL sends (reg_off=1, path=COPY). Registered-buffer path is NEVER used.
- fg_fab_send_reg is never called. The off+1 sentinel / registered buffer path is NOT the issue.
- Copy path (send_x) transfers correct data: recv first4=17.384 (correct for chunk at element 16384)
- Pool base is consistent across ranks (0x7f69ccc01000 rank0, 0x7f1d33e01000 rank1 — different pools, as expected for different processes)

### No-plugin test
- Ran with NCCL_NET_PLUGIN unset: VALUE MISMATCH still occurs (122880/131072 wrong, got=101.304001)
- The defect is NOT in the plugin. It persists with NCCL's default transport.
- The defect is in NCCL's allreduce or in the test's value check.

### Root cause (revised)
The original hypothesis ("memcpy on GPU pointers reads garbage") is WRONG. cudaMalloc is redirected to fg_pool_alloc (host memory), so device pointers ARE host pointers and memcpy is legal. The data transfer is correct (verified by instrumentation). The defect is NOT in the plugin's data path.

The registered-buffer path is never exercised (mhandle=0x1). The copy path works correctly.

The VALUE MISMATCH persists without the plugin, indicating the issue is in NCCL's allreduce algorithm with non-uniform data, or in the test's value check. Further investigation needed.

### CUDA context note
cudaMemcpy in this emulator calls fg_write_random (writes random data). The test's cudaMemcpy (real libcudart) should work on host-memory device pointers. But the no-plugin test uses real GPU memory, so cudaMemcpy should work.

### Conclusion
The four reproduction runs stand. The defect is real (VALUE MISMATCH in all runs). But the root cause is NOT "memcpy on GPU pointers" and NOT "registered buffer path." The defect persists without the plugin, suggesting it is in NCCL's allreduce or in the test methodology. Further investigation required.EOF
echo "appended"


## Re-investigation (root cause corrected)

### Instrumentation findings
- mhandle=0x1 for ALL sends (reg_off=1, path=COPY). Registered-buffer path is NEVER used.
- fg_fab_send_reg is never called. The off+1 sentinel / registered buffer path is NOT the issue.
- Copy path (send_x) transfers correct data: recv first4=17.384 (correct for chunk at element 16384)
- Pool base is consistent across ranks.

### No-plugin test
- Ran with NCCL_NET_PLUGIN unset: VALUE MISMATCH still occurs (122880/131072 wrong, got=101.304001)
- The defect is NOT in the plugin. It persists with NCCL default transport.
- The defect is in NCCL allreduce or in the test value check.

### Root cause (revised)
The original hypothesis ("memcpy on GPU pointers reads garbage") is WRONG. cudaMalloc is redirected to fg_pool_alloc (host memory), so device pointers ARE host pointers and memcpy is legal. The data transfer is correct (verified by instrumentation). The registered-buffer path is never exercised (mhandle=0x1).

The VALUE MISMATCH persists without the plugin, indicating the issue is in NCCL allreduce with non-uniform data, or in the test methodology. Further investigation required.
