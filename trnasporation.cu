#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iomanip>

__global__ void transp(double *matrix, int size){
    //Индекс текущего блока в гриде
    int blockIndex = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z*gridDim.y*gridDim.x;
    //Индекс треда внутри текущего блока
    int ThreadIndex = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.y*blockDim.x;

    //глобальный индекс нити
    int idx = blockIndex*blockDim.x*blockDim.y*blockDim.z + ThreadIndex;

    if (idx / size > idx % size){
        //std::cout << "[" << idx << ", " << idy << "] = " << matrix[idx][idy] << '\n';
        double tmp = matrix[(idx / size) * size + (idx % size)];
        matrix[(idx / size) * size + (idx % size)] = matrix[(idx % size) * size + (idx / size)];
        matrix[(idx % size) * size + (idx / size)] = tmp;
    }
}

int n;

bool transpose(double* matrix, double* res_gpu) //либо int matrix[][5], либо int (*matrix)[5]
{
    int t;
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < n * n; ++i)
    {
        if(i / n > i % n){
            t = matrix[(i / n) * n + (i % n)];
            matrix[(i / n) * n + (i % n)] = matrix[(i % n) * n + (i / n)];
            matrix[(i % n) * n + (i / n)] = t;
        }

    }
    auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for CPU: " << elapsed_seconds.count() << "s\n";
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix[i * n + j];
        }
        std::cout << '\n';
    }
    int num_of_err = 0;
	bool is_correct = 1;
	for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
    		if (matrix[i * n + j] != res_gpu[i * n + j]){
    			num_of_err++;
    			std::cout << "Error in " << i + 1 << ", " << j + 1 << " element;\nOn CPU " << matrix[i * n + j] << ", On GPU " << res_gpu[i * n + j] << ";\n";
    			is_correct = 0;
    		}
        }
	}
	if (is_correct){
		std::cout << "Everything is great, results are equal!" << '\n';

	} else{
		std::cout << "There are " << num_of_err << " errors. Hm... maybe we did something wrong..." << '\n';
	}
	return is_correct;
}


int main(int argc, char const *argv[]) {
    sscanf(argv[1], "%d", &n);
    int bytes = n * n * sizeof(double);
    double* matrix;
    matrix = (double*)malloc(bytes);
    /*for(int i = 0; i < n; ++i){
         matrix[i] = new(double[n]);
    }*/
    char decision;
    std::cout << "Do you want to fill matrix by yourself? (Y/N)" << '\n';
	std::cin >> decision;
	switch(tolower(decision))
	{
		case 'y':
            for(int j = 0; j < n; ++j){
                std::cout << "Please, enter "<< j + 1 << " string: ";
    			for (int i = 0; i < n; ++i){
    				std::cin >> matrix[i * n + j];
    			}
            }
			break;
		case 'n':
			for(int i = 0; i < n; ++i){
                for(int j = 0; j < n; ++j){
                    matrix[i * n + j] = /*(double(rand()) / rand()) + */int(rand() / 100000000);
                }
			}
			break;
		default:
			std::cout << "You have wrote something wrong." << '\n';
	}

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix[i * n + j];
        }
        std::cout << '\n';
    }

    int block_size = 1024;
    int grid_size = (n - 1) / block_size + 1;

    double *gpu_matrix;
    dim3 dimBlock(block_size, block_size, 1);
    dim3 dimGrid(grid_size, grid_size, 1);
    cudaMalloc(&gpu_matrix, bytes);

    cudaMemcpy(gpu_matrix, matrix, bytes, cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();
	transp<<<dimGrid, dimBlock>>>(gpu_matrix, n);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for GPU: " << elapsed_seconds.count() << "s\n";
    /*for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << gpu_matrix[i * n + j];
        }
        std::cout << '\n';
    }*/

    double* matrix_res;
    matrix_res = (double*)malloc(bytes);

    cudaMemcpy(matrix_res, gpu_matrix, bytes, cudaMemcpyDeviceToHost);

    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix_res[i * n + j];
        }
        std::cout << '\n';
    }

    transpose(matrix, matrix_res);
    cudaFree(gpu_matrix);
    free(matrix);
    return 0;
}
