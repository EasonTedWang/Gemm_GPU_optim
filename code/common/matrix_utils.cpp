#include "matrix_utils.h"
#include <cstdlib>
#include <iostream>

void random_matrix(float* matrix, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }
}

void fill_matrix(float* matrix, int rows, int cols, float value)
{
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = value;
    }
}

void print_matrix(const float* matrix, int rows, int cols, int maxPrint)
{
    int r = std::min(rows, maxPrint);
    int c = std::min(cols, maxPrint);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
    if (rows > maxPrint || cols > maxPrint) {
        std::cout << "..." << std::endl;
    }
}
