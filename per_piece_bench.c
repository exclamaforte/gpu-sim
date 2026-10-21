/* per_piece_bench.c — measure the fabric's per-piece host cost.
 * Copies a 1 MiB payload into and out of a shm mailbox, 200 iterations.
 * Reports copy-in + copy-out time per piece.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#define PIECE_SIZE (1 << 20)  /* 1 MiB */
#define ITERS 200

int main() {
    /* allocate a shm region */
    char *buf = mmap(NULL, PIECE_SIZE, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (buf == MAP_FAILED) { perror("mmap"); return 1; }
    char *src = malloc(PIECE_SIZE);
    char *dst = malloc(PIECE_SIZE);
    for (size_t i = 0; i < PIECE_SIZE; i++) src[i] = (char)(i & 0xff);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = 0; i < ITERS; i++) {
        memcpy(buf, src, PIECE_SIZE);   /* copy in */
        memcpy(dst, buf, PIECE_SIZE);   /* copy out */
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double total_us = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
    double per_piece_us = total_us / ITERS;
    printf("per_piece_copy_us=%.2f (total=%.0f us, iters=%d, piece=%d bytes)\n",
           per_piece_us, total_us, ITERS, PIECE_SIZE);
    munmap(buf, PIECE_SIZE);
    free(src); free(dst);
    return 0;
}
