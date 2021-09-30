#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
__glodal__ void transp(double *matrix[], int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx > idy){
        double tmp = matrix[idx * size + idy];
        matrix[idx * size + idy] = matrix[idy * size + idx];
        matrix[idy * size + idx] = tmp;
    }
}

int n;

int main(int argc, char const *argv[]) {
    sscanf(argv[1], "%d", &n);

    double* matrix[] = new(double[n][]);
    for(int i = 0; i < n; ++i){
         matrix[i] = new(double[n]);
    }
    char decision;
    std::cout << "Do you want to fill matrix by yourself? (Y/N)" << '\n';
	std::cin >> decision;
	switch(tolower(decision))
	{
		case 'y':
            for(int j = 0; j < n; ++j){
                std::cout << "Please, enter "<< j + 1 << " string: ";
    			for (int i = 0; i < n; ++i){
    				std::cin >> matrix[i][j];
    			}
            }
			break;
		case 'n':
			for(int i = 0; i < n; ++i){
                for(int j = 0; j < n; ++j){
                    matrix[i][j] = (double(rand()) / rand()) + rand();
                }
			}
			break;
		default:
			std::cout << "You have wrote something wrong." << '\n';
	}

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix[i][j];
        }
        std::cout << '\n';
    }

    int block_size = 1024;
    int grid_size = (n - 1) / block_size + 1;

    double *gpu_matrix;
    int bytes = n * n * sizeof(double);
    cudaMalloc(&gpu_matrix, bytes);

    for(int i = 0; i < n; ++i){
        cudaMemcpy(gpu_matrix + (n * i), matrix[i], cudaMemcpyHostToDevice);
    }

    auto start = std::chrono::steady_clock::now();
	transp<<<grid_size, block_size>>>(gpu_matrix, n);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for GPU: " << elapsed_seconds.count() << "s\n";

    for(int i = 0; i < n; ++i){
        cudaMemcpy(matrix[i], gpu_matrix + (n * i), cudaMemcpyDeviceToHost);
    }

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix[i][j];
        }
        std::cout << '\n';
    }


    return 0;
}
