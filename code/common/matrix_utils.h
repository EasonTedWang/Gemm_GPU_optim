#pragma once

#include <cstddef>

void random_matrix(float* matrix, int rows, int cols);
void fill_matrix(float* matrix, int rows, int cols, float value);
void print_matrix(const float* matrix, int rows, int cols, int maxPrint = 8);
