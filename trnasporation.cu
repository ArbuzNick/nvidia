#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>
#include <iomanip>

__global__ void transp(double *matrix, int size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx > idy){
        //std::cout << "[" << idx << ", " << idy << "] = " << matrix[idx][idy] << '\n';
        double tmp = matrix[idx * size + idy];
        matrix[idx * size + idy] = matrix[idy * size + idx];
        matrix[idy * size + idx] = tmp;
    }
}

int n;

bool transpose(double** matrix, double** res_gpu) //либо int matrix[][5], либо int (*matrix)[5]
{
    int t;
    auto start = std::chrono::steady_clock::now();
    for(int i = 0; i < n; ++i)
    {
        for(int j = i; j < n; ++j)
        {
            t = matrix[i][j];
            matrix[i][j] = matrix[j][i];
            matrix[j][i] = t;
        }
    }
    auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for CPU: " << elapsed_seconds.count() << "s\n";
    /*for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix[i][j];
        }
        std::cout << '\n';
    }*/
    int num_of_err = 0;
	bool is_correct = 1;
	for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
    		if (matrix[i][j] != res_gpu[i][j]){
    			num_of_err++;
    			std::cout << "Error in " << i + 1 << ", " << j + 1 << " element;\nOn CPU " << matrix[i][j] << ", On GPU " << res_gpu[i][j] << ";\n";
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

    double** matrix;
    matrix = (double**)malloc(n * sizeof(double*));
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
                    matrix[i][j] = /*(double(rand()) / rand()) + */int(rand() / 100000000);
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
        cudaMemcpy(gpu_matrix + (n * i), matrix[i], n * sizeof(double), cudaMemcpyHostToDevice);
    }

    auto start = std::chrono::steady_clock::now();
	transp<<<grid_size, block_size>>>(gpu_matrix, n);
	cudaDeviceSynchronize();
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for GPU: " << elapsed_seconds.count() << "s\n";
    double** matrix_res;
    matrix_res = (double**)malloc(n * sizeof(double*));
    for(int i = 0; i < n; ++i){
         matrix_res[i] = new(double[n]);
    }

    for(int i = 0; i < n; ++i){
        cudaMemcpy(matrix_res[i], gpu_matrix + (n * i), n * sizeof(double), cudaMemcpyDeviceToHost);
    }
/*
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            std::cout << std::setw(8) << matrix[i][j];
        }
        std::cout << '\n';
    }*/
    transpose(matrix, matrix_res);
    cudaFree(gpu_matrix);
    for(int i = 0; i < n; ++i){
        delete(matrix[i]);
    }
    delete(matrix);
    return 0;
}
