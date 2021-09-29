#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
#include <chrono>

int n;

__global__ void vecAdd(double *A, double *B, double *C){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	C[idx] = A[idx] + B[idx];
}

bool checkResults(double *A, double *B, double *res_gpu){
	double *res_cpu = new double[n];
	auto start = std::chrono::steady_clock::now();
	for (int i = 0; i < n; ++i){
		res_cpu[i] = A[i] + B[i];
	}
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for CPU: " << elapsed_seconds.count() << "s\n";
	int num_of_err = 0;
	bool is_correct = 1;
	for (int i = 0; i < n; ++i){
		if (res_cpu[i] != res_gpu[i]){
			num_of_err++;
			std::cout << "Error in " << i + 1 << " element;\nOn CPU " << res_cpu[i] << ", On GPU " << res_gpu[i] << ";\n";
			is_correct = 0;
		}
	}
	if (is_correct){
		std::cout << "Everything is great, results are equal!" << '\n';

	} else{
		std::cout << "There are " << num_of_err << " errors. Hm... maybe we did something wrong..." << '\n';
	}
	return is_correct;
}

int main(int argc, char* argv[]){
	sscanf(argv[1], "%d", &n);
	size_t bytes = n * sizeof(double);
	double *A = (double*)malloc(bytes);
	double *B = (double*)malloc(bytes);
	double *res = (double*)malloc(bytes);
	char decision;
	std::cout << "Do you want to fill arrays by yourself? (Y/N)" << '\n';
	std::cin >> decision;
	switch(tolower(decision))
	{
		case 'y':
			std::cout << "Please, enter first vector: ";
			for (int i = 0; i < n; ++i){
				std::cin >> A[i];
			}
			std::cout << "Please, enter second vector: ";
			for (int i = 0; i < n; ++i){
				std::cin >> B[i];
			}
			break;
		case 'n':
			for (int i = 0; i < n; ++i){
				A[i] = (double(rand()) / rand()) + rand();
				B[i] = (double(rand()) / rand()) + rand();
			}
			break;
		default:
			std::cout << "You have wrote something wrong." << '\n';
	}

	double *A_gpu, *B_gpu, *res_gpu;


	cudaMalloc(&A_gpu, bytes);
	cudaMalloc(&B_gpu, bytes);
	cudaMalloc(&res_gpu, bytes);

	cudaMemcpy(A_gpu, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, bytes, cudaMemcpyHostToDevice);

	int block_size = 1024;
	int grid_size = (n - 1) / block_size + 1;

	auto start = std::chrono::steady_clock::now();
	vecAdd<<<grid_size, block_size>>>(A_gpu, B_gpu, res_gpu);
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double> elapsed_seconds = end-start;
 	std::cout << "Time for GPU: " << elapsed_seconds.count() << "s\n";
	cudaMemcpy(res, res_gpu, bytes, cudaMemcpyDeviceToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(res_gpu);

	checkResults(A, B, res);

	for(int i = 0; i < n; ++i){
		std::cout << res[i] << std::endl;
	}
	free(A);
	free(B);
	free(res);
}
