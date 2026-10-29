#include "matmul.h"

void matmul(const float *A, const float *B, float *C,
            int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < K; k++) {
                C[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void matvec(const float *A, const float *x, float *y, int M, int K) {
    for (int i = 0; i < M; i++) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[i * K + k] * x[k];
        }
        y[i] = sum;
    }
}