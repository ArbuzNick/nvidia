#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>

__global__ void test() {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Test %d\n", idx);
}

int main(int argc, char const *argv[]) {
    int block_size = 1024;
    int grid_size = 1024;

    test<<<grid_size, block_size>>>();
    cudaDeviceSynchronize();


    return 0;
}
