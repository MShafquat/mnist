#include "matmul.h"

void matmul(ConstMatrix A, ConstMatrix B, Matrix C) {
    for (int i = 0; i < C.extent(0); i++) {
        for (int j = 0; j < C.extent(1); j++) {
            float sum = 0.0f;
            for (int k = 0; k < A.extent(1); k++) {
                sum += A[i, k] * B[k, j];
            }
            C[i, j] = sum;
        }
    }
}

void matvec(ConstMatrix A, std::span<const float> x, std::span<float> y) {
    for (int i = 0; i < A.extent(0); i++) {
        float sum = 0.0f;
        for (int k = 0; k < A.extent(1); k++) {
            sum += A[i, k] * x[k];
        }
        y[i] = sum;
    }
}
