#include <stdint.h>
#include <stdio.h>

#ifndef M
#define M 1
#endif

#define N 50

static inline double __attribute__((annotate("scalar(range(0, 1) disabled)"))) fast_rand01(void) {
    static uint64_t state = 0xC0FFEE1234ULL;
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    uint64_t x = state * 2685821657736338717ULL;
    return (x >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

static inline float __attribute__((annotate("scalar(range(1.02, 1.2) final disabled)"))) rand_range(float min, float max) {
    return (float)(min + (max - min) * fast_rand01());
}

static inline float __attribute__((annotate("scalar(range(1.001, 1.301) final disabled)"))) rand_range_2(float min, float max) {
    return (float)(min + (max - min) * fast_rand01());
}

static inline float __attribute__((annotate("scalar(range(1, 2) final disabled)"))) rand_range_3(float min, float max) {
    return (float)(min + (max - min) * fast_rand01());
}

float arr[N] __attribute__((annotate("scalar()")));
float A[N] __attribute__((annotate("scalar()")));
float B[N] __attribute__((annotate("scalar()")));

float foo(float x, float y) {
    return x + y;
}

int main(int argc, char const *argv[])
{

    for (int i = 0; i < N; i++) {
        arr[i] = rand_range(1.02f, 1.2f);
        A[i] = rand_range(1.001f, 1.301f);
        B[i] = rand_range(1.0f, 2.0f);
    }

    for (int m = 0; m < M; ++m) {
        uint32_t cycles_high1 = 0;
        uint32_t cycles_high = 0;
        uint32_t cycles_low = 0;
        uint32_t cycles_low1 = 0;

        asm volatile("CPUID\n\t"
                    "RDTSC\n\t"
                    "mov %%edx, %0\n\t"
                    "mov %%eax, %1\n\t"
                    : "=r"(cycles_high), "=r"(cycles_low)::"%rax", "%rbx", "%rcx", "%rdx");

        for (int i = 1; i < N; i++) {
            float tmp = A[i] * arr[i - 1];
            arr[i] = tmp + B[i];
        }

        asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t"
                 : "=r"(cycles_high1), "=r"(cycles_low1)::"%rax", "%rbx", "%rcx", "%rdx");
        uint64_t end = (uint64_t) cycles_high1 << 32 | cycles_low1;
        uint64_t start = (uint64_t) cycles_high << 32 | cycles_low;
        if (end > start)
        printf("Cycles: %li\n", end - start);
    }

    printf("Values Begin\n");
    for (int j = 0; j < N; ++j)
        printf("%f\n", arr[j]);
    printf("Values End\n");

    return 0;
}
