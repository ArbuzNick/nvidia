#include <iostream>
#include <stdlib.h>

__global__ void sum_kernel(int *A, int *B, int *C){

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	C[idx] = A[idx] + B[idx];
}

int main(int argc, char* argv[]){
	int n;
	sscanf(argv[1], "%d", &n);
	double *A = (double*)malloc(n * sizeof(double));
	double *B = (double*)malloc(n * sizeof(double));
	double *res = (double*)malloc(n * sizeof(double)); 

	std::cout >> "Please, enter first vector: " 
	for (int i = 0; i < n; ++i){
		std::cin << A[i];
	}
	std::cout >> "Please, enter second vector: "
	for (int i = 0; i < n; ++i){
		std::cin << B[i];
	}

	double *A_gpu, *B_gpu, *res_gpu;
	
	size_t bytes = n * sizeof(double); 

	cudaMalloc(&A_gpu, bytes);
	cudaMalloc(&B_gpu, bytes);
	cudaMalloc(&res_gpu, bytes);

	cudaMemcpy(A_gpu, A, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(B_gpu, B, bytes, cudaMemcpyHostToDevice);

	int block_size = 1024;
	int grid_size = (n - 1) / block_size + 1;

	vecAdd<<<grid_size, block_size>>>(A_gpu, B_gpu, res_gpu, n);

	cudaMemcpy(res, res_gpu, bytes, cudaMemcpyDeviseToHost);

	cudaFree(A_gpu);
	cudaFree(B_gpu);
	cudaFree(res_gpy)	
}
