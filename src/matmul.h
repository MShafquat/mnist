#pragma once

#include <mdspan>
#include <span>

// Row-major 2D views over flat float buffers. Indexing is m[row, col].
using Matrix = std::mdspan<float, std::dextents<int, 2>>;
using ConstMatrix = std::mdspan<const float, std::dextents<int, 2>>;

// C = A · B, where A is M×K, B is K×N and C is M×N.
void matmul(ConstMatrix A, ConstMatrix B, Matrix C);

// y = A · x, where A is M×K, x has K elements and y has M.
void matvec(ConstMatrix A, std::span<const float> x, std::span<float> y);
