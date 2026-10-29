#pragma once

// C = A · B, where A is M×K, B is K×N and C is M×N (all row-major).
void matmul(const float* A, const float* B, float* C,
    int M, int K, int N);

// y = A · x, where A is M×K (row-major) and x has K elements.
void matvec(const float* A, const float* x, float* y, int M, int K);