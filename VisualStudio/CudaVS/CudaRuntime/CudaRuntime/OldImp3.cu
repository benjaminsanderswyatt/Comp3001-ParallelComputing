#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 64  //arrays input size
#define TIMES 1 //times to run
#define ARITHMETICAL_OPS N*N*N*2

#define EPSILON 0.00001


__declspec(align(64)) float test[N*N], A[N*N], B[N*N], C[N*N];

__device__ float device_a[N * N];
__device__ float device_b[N * N];
__device__ float device_c[N * N];



unsigned short int compare(const float* C, const float* A, const float* B);
unsigned short int equal(float const x, float const y);

void init(float* C, float *A, float *B); // Init

void cuda_error();


// --------------------------------------- implementation #3 ---------------------------------------
__global__ void implementation_3(float* C, float* A, float* B) {

	printf("\nHello from thread %d of block %d", threadIdx.x, blockIdx.x);

	float tmp = 0.0;



	int i = blockIdx.x * blockDim.x + threadIdx.x; //i loop has been parallelized 

	int j = blockIdx.y * blockDim.y + threadIdx.y; //j loop has been parallelized 



	for (int k = 0; k < N; k++) {

		tmp += A[N * i + k] * B[N * k + j];

	}



	C[N * i + j] = tmp;
}




int main() {
	printf("\n------------- implementation #3 -------------\n");

	cudaError_t cudaStatus;

	//------create the cuda timers------
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float elapsed_time;

	int devId = 0;
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, devId);
	printf("\n Device: %s \n", prop.name);





	init(device_c, device_a, device_b); //initialize host arrays

	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(N / 16, N / 16, 1);

	cudaEventRecord(start, 0); //get timer value


	/* Copy the A array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_a, A, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		cuda_error();
		return -1;
	}

	/* Copy the B array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_b, B, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		cuda_error();
		return -1;
	}

	for (int i = 0; i < TIMES; i++) {
		implementation_3 <<<dimGrid, dimBlock>>> (C,A,B);

	}


	/* Copy back the result from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpyFromSymbol(C, device_c, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\ncudaMemcpy failed!");
		cuda_error();
		return -1;
	}


	cudaEventRecord(stop, 0);  //get timer value
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("\nElapsed time in msecs = %f", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	double flops = (double)((double)2 * N * N * N) / (elapsed_time / TIMES);
	printf("\nGflops achieved %f ", flops / 1000000);

	cuda_error();

	if (compare(C,A,B) == 0)
		printf("\nResult is ok\n");
	else
		printf("\nResult is FALSE\n");

	cudaDeviceReset();

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		return -1;
	}

	return 0;
}

// CUDA error
void cuda_error() {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("\nError: %s\n", cudaGetErrorString(error));
	}
}



// Init
void init(float *device_C, float * device_A, float * device_B) {

	int i;

	for (i = 0; i < (N * N); i++) {
		device_A[i] = (float)i;
		device_B[i] = (float)i + 1;
		device_C[i] = 0.0f;
	}

}



unsigned short int compare(const float* C, const float* A, const float* B) {

	int i, j, k;

	for (i = 0; i < (N * N); i++) {
		test[i] = 0.0f;
	}


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			for (k = 0; k < N; k++)
				test[N * i + j] += A[N * i + k] * B[N * k + j];

	for (j = 0; j < (N * N); j++)
		if (equal(C[j], test[j]) == 1) {
			printf("\n j=%d %f %f\n", j, test[j], C[j]);
			return -1;
		}

	return 0;
}


unsigned short int equal(float const x, float const y) {
	float temp = x - y;
	//printf("\n %f  %f", a, b);
	if (fabs(temp / y) < EPSILON)
		return 0; //success
	else
		return 1;
}