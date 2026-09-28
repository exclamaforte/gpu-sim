/*
 * fakegpu_nccl.c — NCCL API interposer.
 *
 * Hooks NCCL symbols; world size defaults to FAKEGPU_NUM_DEVICES, rank to 0
 * (single-process multi-device simulation) unless FAKEGPU_NCCL_RANK is set.
 * Collective output buffers are filled with random bytes and latency is applied
 * based on size (via the bandwidth or fixed memcpy model).
 */
#include "fakegpu.h"
#include "nccl.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ---- world size / rank resolution --------------------------------------- */

/* ---- world size / rank resolution (per-comm where available) ------------- */

static int fg_nccl_rank_from_env(void)
{
    const char *s = getenv("FAKEGPU_NCCL_RANK");
    if (s && *s) {
        int r = (int)strtol(s, NULL, 10);
        if (r < 0) r = 0;
        return r;
    }
    return 0;
}

static ncclComm_t fg_nccl_fake_comm(int nrank, int rank)
{
    int n = nrank > 0 ? nrank : (int)fg_cfg()->num_devices;
    if (n < 1) n = 1;
    struct fg_nccl_comm *c = (struct fg_nccl_comm *)malloc(sizeof(*c));
    if (!c) return NULL;
    c->nrank = n;
    c->rank  = rank;
    return (ncclComm_t)c;
}

/* ---- comm lifecycle ------------------------------------------------------ */

ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nRanks, int rank,
                              unsigned long long commId)
{
    if (!comm) return ncclErrorInvalidArgument;
    (void)commId;
    /* nRanks defaults to FAKEGPU_NUM_DEVICES when <=0 (caller didn't specify) */
    int n = nRanks > 0 ? nRanks : fg_cfg()->num_devices;
    *comm = fg_nccl_fake_comm(n, rank);
    return *comm ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclCommInit32(ncclComm_t *comm, int nRanks, int rank,
                            unsigned long long commId)
{
    return ncclCommInitRank(comm, nRanks, rank, commId);
}

ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int nRanks, int rank,
                                        unsigned long long commId)
{
    return ncclCommInitRank(comm, nRanks, rank, commId);
}

ncclResult_t ncclCommInitRankScalable2(ncclComm_t *comm, int nranks, int rank,
                                       unsigned long long commId, const int *topo)
{
    (void)topo;
    return ncclCommInitRank(comm, nranks, rank, commId);
}

ncclResult_t ncclCommDestroy(ncclComm_t comm)
{
    free(comm);  /* fg_nccl_fake_comm malloc'd the struct */
    return ncclSuccess;
}

ncclResult_t ncclCommDestroy_Handle(ncclComm_t comm)
{
    return ncclCommDestroy(comm);
}

ncclResult_t ncclCommFinalize(ncclComm_t comm)
{
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclCommAbort(ncclComm_t comm)
{
    (void)comm;
    return ncclSuccess;
}

ncclResult_t ncclCommGetRank(ncclComm_t comm, int *rank)
{
    if (!rank) return ncclErrorInvalidArgument;
    *rank = comm ? ((struct fg_nccl_comm *)comm)->rank : fg_nccl_rank_from_env();
    return ncclSuccess;
}

ncclResult_t ncclCommGetNrThreads(int *nThreads, ncclComm_t comm)
{
    if (!nThreads) return ncclErrorInvalidArgument;
    *nThreads = 1;
    return ncclSuccess;
}

ncclResult_t ncclCommCount(const ncclComm_t comm, int *nThreads)
{
    if (!nThreads) return ncclErrorInvalidArgument;
    *nThreads = comm ? ((struct fg_nccl_comm *)comm)->nrank : 1;
    return ncclSuccess;
}

ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm)
{
    (void)color; (void)key;
    if (!newcomm) return ncclErrorInvalidArgument;
    int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : (int)fg_cfg()->num_devices;
    int rk = comm ? ((struct fg_nccl_comm *)comm)->rank  : 0;
    *newcomm = fg_nccl_fake_comm(nr, rk);
    return *newcomm ? ncclSuccess : ncclSystemError;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, int *errorCode)
{
    (void)comm;
    if (!errorCode) return ncclErrorInvalidArgument;
    *errorCode = 0;
    return ncclSuccess;
}

/* ---- group --------------------------------------------------------------- */

ncclResult_t ncclGroupStart(void)
{
    return ncclSuccess;
}

ncclResult_t ncclGroupEnd(void)
{
    return ncclSuccess;
}

/* ---- point-to-point ------------------------------------------------------ */

ncclResult_t ncclSend(const void *sendbuff, size_t count, int datatype,
                      int peer, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)datatype; (void)peer; (void)comm; (void)stream;
    /* model a tiny latency proportional to count */
    fg_simulate_memcpy(count);
    return ncclSuccess;
}

ncclResult_t ncclRecv(void *recvbuff, size_t count, int datatype,
                      int peer, ncclComm_t comm, void *stream)
{
    (void)datatype; (void)peer; (void)comm; (void)stream;
    fg_simulate_memcpy(count);
    if (recvbuff && count) fg_fill(recvbuff, count);
    return ncclSuccess;
}

ncclResult_t ncclSendRecv(void *recvbuf, size_t rcount, int rtype, int recvFrom,
                          const void *sendbuf, size_t scount, int stype,
                          int sendTo, ncclComm_t comm, void *stream)
{
    (void)sendbuf; (void)stype; (void)sendTo; (void)comm; (void)stream;
    fg_simulate_memcpy(rcount > scount ? rcount : scount);
    if (recvbuf && rcount) {
        if (rcount) fg_fill(recvbuf, rcount);
    }
    return ncclSuccess;
}

/* ---- collectives --------------------------------------------------------- */

static void fg_collective_out(void *recvbuff, size_t count, int datatype,
                            ncclComm_t comm)
{
    size_t bytes;
    switch (datatype) {
    case ncclInt8:    bytes = count * 1; break;
    case ncclUint8:   bytes = count * 1; break;
    case ncclInt32:   bytes = count * 4; break;
    case ncclUint32:  bytes = count * 4; break;
    case ncclInt64:   bytes = count * 8; break;
    case ncclUint64:  bytes = count * 8; break;
    case ncclFloat32: bytes = count * 4; break;
    case ncclFloat64: bytes = count * 8; break;
    case ncclFloat16: bytes = count * 2; break;
    default:          bytes = count; break;
    }
    /* scale by log2(nrank) to model fan-in/fan-out cost */
    int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : 1;
    int lvl = 0;
    int tmp = nr;
    while (tmp > 1) { lvl++; tmp >>= 1; }
    if (lvl > 0) bytes = bytes + bytes * (size_t)lvl / 2;
    fg_simulate_memcpy(bytes);
    if (recvbuff && bytes) fg_fill(recvbuff, bytes);
}

ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)op; (void)stream;
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllReduce_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int op, ncclComm_t comm, void *stream)
{
    return ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream);
}

ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int root, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)root; (void)stream;
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclBroadcast_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ncclComm_t comm, void *stream)
{
    return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclBroadcast_v3(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ncclComm_t comm, void *stream,
                              int priority)
{
    (void)priority;
    return ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream);
}

ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        int datatype, int op, int root, ncclComm_t comm,
                        void *stream)
{
    (void)sendbuff; (void)op; (void)root; (void)stream;
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclReduce_v2(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int root, ncclComm_t comm,
                           void *stream)
{
    return ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t ncclReduce_v3(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int root, ncclComm_t comm,
                           void *stream, int priority)
{
    (void)priority;
    return ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream);
}

ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)stream;
    /* all-gather output is nrank * count * elemsize */
    int nr = comm ? ((struct fg_nccl_comm *)comm)->nrank : 1;
    size_t full = count * (size_t)nr;
    fg_collective_out(recvbuff, full, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllGather_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, ncclComm_t comm, void *stream)
{
    return ncclAllGather(sendbuff, recvbuff, count, datatype, comm, stream);
}

ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               const size_t *recvcount, int datatype, int op,
                               ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)op; (void)stream;
    size_t rc = recvcount ? recvcount[0] : 0;
    fg_collective_out(recvbuff, rc, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllToAll(const void *sendbuff, void *recvbuff, size_t count,
                          int datatype, ncclComm_t comm, void *stream)
{
    (void)sendbuff; (void)stream;
    fg_collective_out(recvbuff, count, datatype, comm);
    return ncclSuccess;
}

ncclResult_t ncclAllToAll_v2(const void *sendbuff, void *recvbuff, size_t count,
                             int datatype, ncclComm_t comm, void *stream)
{
    return ncclAllToAll(sendbuff, recvbuff, count, datatype, comm, stream);
}

/* ---- op / version helpers ------------------------------------------------ */

ncclResult_t ncclRedOpCreate(ncclComm_t comm, int op, void *func, int datatype)
{
    (void)comm; (void)func; (void)datatype;
    return (ncclResult_t)op;
}

ncclResult_t ncclRedOpDestroy(int op)
{
    (void)op;
    return ncclSuccess;
}

ncclResult_t ncclGetVersion(int *version)
{
    if (!version) return ncclErrorInvalidArgument;
    *version = 212000; /* fake NCCL version 2.12 */
    return ncclSuccess;
}

ncclResult_t ncclGetUniqueId(unsigned long long *commId)
{
    if (!commId) return ncclErrorInvalidArgument;
    *commId = fg_rng_next();
    return ncclSuccess;
}

ncclResult_t ncclSetThreadPerBlock(int thrPerBlock)
{
    (void)thrPerBlock;
    return ncclSuccess;
}

const char *ncclGetErrorString(ncclResult_t code)
{
    switch (code) {
    case ncclSuccess:              return "no error";
    case ncclErrorInvalidNumber:   return "invalid number";
    case ncclErrorInvalidArgument: return "invalid argument";
    default:                       return "unknown error";
    }
}
