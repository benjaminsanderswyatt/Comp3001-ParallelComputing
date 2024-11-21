#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void helloWorld() {
	int global_index = blockIdx.x * blockDim.x + threadIdx.x;
	printf("\n Hello from thread %d and block %d meaning global %d", threadIdx.x, blockIdx.x, global_index);
	

}



int main() {

	dim3 blocks(1, 1, 1); // grid size and dimensions, it consists of blocks
	dim3 threads(6, 1, 1); // block size and dimensions, it consists of threads


	helloWorld <<<blocks, threads >>> (); // MaxBlocks: , MaxThreads: 1024

	// Error message
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error));
	}



	cudaDeviceReset();

	return 0;
}