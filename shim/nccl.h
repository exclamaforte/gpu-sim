#ifndef FAKEGPU_NCCL_H
#define FAKEGPU_NCCL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* fakegpu's NCCL shim: subset of the NCCL API the interposer provides.
   Workloads for CPU-only CI include this header instead of <nccl.h>. */

typedef int ncclResult_t;
typedef void *ncclComm_t;
typedef void *ncclStreamCtx_t;

#define ncclSuccess              ((ncclResult_t)0)
#define ncclErrorInvalidNumber   ((ncclResult_t)1)
#define ncclErrorInvalidArgument ((ncclResult_t)2)
#define ncclErrorUnsupported     ((ncclResult_t)3)
#define ncclSystemError          ((ncclResult_t)4)
#define ncclErrorInternal        ((ncclResult_t)5)

typedef int ncclDataType_t;

/* These values MUST match real NCCL's ncclDataType_t exactly. torch passes the
 * constants it compiled against, so a permuted enum silently mislabels every
 * buffer size: this header previously had Float32=6, Float64=7, Float16=8, while
 * real NCCL has Float16=6, Float32=7, Float64=8. The result was float32 counted
 * as 8 bytes/element and float16 as 4 -- every reported collective volume for the
 * dominant dtype was exactly 2x too large, and the error was invisible because
 * the numbers were self-consistent. Measured with a 1000-element probe: 4000 B of
 * float32 traced as 8000 B.
 *
 * Reference: real nccl.h -- Int8=0, Uint8=1, Int32=2, Uint32=3, Int64=4, Uint64=5,
 * Float16=6/Half=6, Float32=7/Float=7, Float64=8/Double=8, Bfloat16=9. */
#define ncclInt8       ((ncclDataType_t)0)
#define ncclChar       ((ncclDataType_t)0)
#define ncclUint8      ((ncclDataType_t)1)
#define ncclInt32      ((ncclDataType_t)2)
#define ncclInt        ((ncclDataType_t)2)
#define ncclUint32     ((ncclDataType_t)3)
#define ncclInt64      ((ncclDataType_t)4)
#define ncclLong       ((ncclDataType_t)4)
#define ncclUint64     ((ncclDataType_t)5)
#define ncclFloat16    ((ncclDataType_t)6)
#define ncclHalf       ((ncclDataType_t)6)
#define ncclFloat32    ((ncclDataType_t)7)
#define ncclFloat      ((ncclDataType_t)7)
#define ncclFloat64    ((ncclDataType_t)8)
#define ncclDouble     ((ncclDataType_t)8)
#define ncclbfloat16   ((ncclDataType_t)9)
#define ncclBfloat16   ((ncclDataType_t)9)

typedef int ncclRedOp_t;

#define ncclSum     ((ncclRedOp_t)0)
#define ncclProd    ((ncclRedOp_t)1)
#define ncclMax     ((ncclRedOp_t)2)
#define ncclMin     ((ncclRedOp_t)3)
#define ncclAvg     ((ncclRedOp_t)4)
#define ncclRedOpMinMax   ((ncclRedOp_t)2)
#define ncclRedOpSum      ((ncclRedOp_t)0)
#define ncclRedOpProd     ((ncclRedOp_t)1)
#define ncclRedOpMax      ((ncclRedOp_t)2)
#define ncclRedOpMin      ((ncclRedOp_t)3)

typedef void *ncclComm;
typedef void *ncclComm_t_;
typedef int (*ncclPlan_t);

/* opaque handle wrapper for comm init */
typedef ncclComm_t ncclCommOpaque_t;

/* comm lifecycle */
ncclResult_t ncclCommInitRank(ncclComm_t *comm, int nRanks, int rank,
                              unsigned long long commId);
ncclResult_t ncclCommInit32(ncclComm_t *comm, int nRanks, int rank,
                            unsigned long long commId);
ncclResult_t ncclCommInitRankScalable(ncclComm_t *comm, int nRanks, int rank,
                                        unsigned long long commId);
ncclResult_t ncclCommInitRankScalable2(ncclComm_t *comm, int nranks, int rank,
                                       unsigned long long commId,
                                       const int *topo);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommDestroy_Handle(ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommGetRank(ncclComm_t comm, int *rank);
ncclResult_t ncclCommGetNrThreads(int *nThreads, ncclComm_t comm);
ncclResult_t ncclCommCount(const ncclComm_t comm, int *nThreads);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key,
                           ncclComm_t *newcomm);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, int *errorCode);

/* group */
ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

/* point-to-point */
ncclResult_t ncclSend(const void *sendbuff, size_t count, int datatype,
                      int peer, ncclComm_t comm, void *stream);
ncclResult_t ncclRecv(void *recvbuff, size_t count, int datatype,
                      int peer, ncclComm_t comm, void *stream);
ncclResult_t ncclSendRecv(void *recvbuf, size_t rcount, int rtype, int recvFrom,
                          const void *sendbuf, size_t scount, int stype,
                          int sendTo, ncclComm_t comm, void *stream);

/* collectives — base + _v2 + _v3 variants */
ncclResult_t ncclAllReduce(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, ncclComm_t comm, void *stream);
ncclResult_t ncclAllReduce_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int op, ncclComm_t comm, void *stream);
ncclResult_t ncclBroadcast(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int root, ncclComm_t comm, void *stream);
ncclResult_t ncclBroadcast_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ncclComm_t comm, void *stream);
ncclResult_t ncclBroadcast_v3(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, int root, ncclComm_t comm, void *stream,
                              int priority);
ncclResult_t ncclReduce(const void *sendbuff, void *recvbuff, size_t count,
                        int datatype, int op, int root, ncclComm_t comm,
                        void *stream);
ncclResult_t ncclReduce_v2(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int root, ncclComm_t comm,
                           void *stream);
ncclResult_t ncclReduce_v3(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int root, ncclComm_t comm,
                           void *stream, int priority);
ncclResult_t ncclAllGather(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, ncclComm_t comm, void *stream);
ncclResult_t ncclAllGather_v2(const void *sendbuff, void *recvbuff, size_t count,
                              int datatype, ncclComm_t comm, void *stream);
ncclResult_t ncclReduceScatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, int datatype, int op,
                               ncclComm_t comm, void *stream);
ncclResult_t ncclAllToAll(const void *sendbuff, void *recvbuff, size_t count,
                          int datatype, ncclComm_t comm, void *stream);
ncclResult_t ncclAllToAll_v2(const void *sendbuff, void *recvbuff, size_t count,
                             int datatype, ncclComm_t comm, void *stream);

/* custom reduction ops */
ncclResult_t ncclRedOpCreate(ncclComm_t comm, int op, void *func, int datatype);
ncclResult_t ncclRedOpDestroy(int op);

/* version / info */
ncclResult_t ncclGetVersion(int *version);
ncclResult_t ncclGetUniqueId(unsigned long long *commId);
ncclResult_t ncclSetThreadPerBlock(int thrPerBlock);

/* error string */
const char *ncclGetErrorString(ncclResult_t code);

#ifdef __cplusplus
}
#endif

#endif /* FAKEGPU_NCCL_H */
