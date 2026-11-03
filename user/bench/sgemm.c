#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <limits.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>

static double now(void) {
    static mach_timebase_info_data_t timebase;
    if (!timebase.denom)
        mach_timebase_info(&timebase);
    return mach_absolute_time() * (double)timebase.numer / timebase.denom / 1e9;
}

int main(int argc, char **argv) {
    double target = argc > 1 ? strtod(argv[1], NULL) : 0.25;
    int n = 1;
    double best = 0;
    if (!(target > 0))
        target = 0.25;
    for (;;) {
        size_t a_cells = (size_t)n * (size_t)n;
        float *a = malloc(a_cells * sizeof(*a));
        if (!a) {
            printf("{\"rows\":%d,\"inner\":%d,\"allocation_complete\":false,\"peak_gflops_s\":%.9g}\n", n, n, best);
            free(a);
            break;
        }
        for (size_t i = 0; i < a_cells; ++i)
            a[i] = (float)((i * 2654435761u) % 1000) / 1000.0f;
        int complete = 1;
        int square_repetitions = 0;
        for (int k = 1;; k *= 2) {
            size_t bc_cells = (size_t)n * (size_t)k;
            float *b = malloc(bc_cells * sizeof(*b));
            float *c = calloc(bc_cells, sizeof(*c));
            if (!b || !c) {
                printf("{\"rows\":%d,\"inner\":%d,\"columns\":%d,\"allocation_complete\":false,\"peak_gflops_s\":%.9g}\n", n, n, k, best);
                free(b);
                free(c);
                complete = 0;
                break;
            }
            for (size_t i = 0; i < bc_cells; ++i)
                b[i] = (float)((i * 2246822519u) % 1000) / 1000.0f;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n, k, n, 1.0f, a, n, b, k, 0.0f, c, k);
            size_t repetitions = 1;
            double elapsed;
            for (;;) {
                double started = now();
                for (size_t repetition = 0; repetition < repetitions; ++repetition)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                n, k, n, 1.0f, a, n, b, k, 0.0f, c, k);
                elapsed = now() - started;
                if (elapsed >= target || repetitions > SIZE_MAX / 2)
                    break;
                repetitions *= 2;
            }
            double flops = 2.0 * n * n * (double)k;
            double rate = elapsed > 0 ? flops * repetitions / elapsed / 1e9 : 0;
            if (rate > best)
                best = rate;
            printf("{\"rows\":%d,\"inner\":%d,\"columns\":%d,\"elapsed_s\":%.9g,\"repetitions\":%zu,\"seconds_per_multiply\":%.9g,\"gflops_s\":%.9g,\"peak_gflops_s\":%.9g,\"bytes\":%zu,\"target_s\":%.9g,\"allocation_complete\":true}\n",
                   n, n, k, elapsed, repetitions, elapsed / repetitions, rate,
                   best, (a_cells + 2 * bc_cells) * sizeof(float), target);
            fflush(stdout);
            free(b);
            free(c);
            if (k == n) {
                square_repetitions = repetitions;
                break;
            }
        }
        free(a);
        if (!complete || square_repetitions == 1 || n > INT_MAX / 2)
            break;
        n *= 2;
    }
    return 0;
}
