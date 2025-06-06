#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define N 4096  //arrays input size
#define TIMES 50 //times to run
#define TILE 16
#define ARITHMETICAL_OPS N*N*N*2 // Might be wrong as array size N*N???

#define EPSILON 0.00001


__declspec(align(64)) float test[N*N], A[N*N], B[N*N], C[N*N];



unsigned short int compare(const float* C, const float* A, const float* B);
unsigned short int equal(float const x, float const y);

void init(); // Init
void cuda_error();


// --------------------------------------- implementation #4 ---------------------------------------
__global__ void mmm_tiled(float* C, float* A, float* B) {



	__shared__ float aa[16][16];

	__shared__ float bb[16][16];

	float tmp = 0.0;

	int k, m;



	int row_A = 16 * blockIdx.y + threadIdx.y;

	int col_B = blockIdx.x * 16 + threadIdx.x;



	for (m = 0; m < N / 16; m++) {

		aa[threadIdx.y][threadIdx.x] = A[N * (row_A)+(m * 16 + threadIdx.x)];

		bb[threadIdx.y][threadIdx.x] = B[N * (m * TILE + threadIdx.y) + (col_B)];



		__syncthreads();



		for (k = 0; k < 16; k++) {

			tmp += aa[threadIdx.y][k] * bb[k][threadIdx.x];

		}

		__syncthreads();

	}

	C[N * row_A + col_B] = tmp;

}




int main() {
	printf("\n------------- implementation #4 -------------\n");

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

	init(); //initialize host arrays

	float* C_d, * A_d, * B_d;

	dim3 dimBlock(16, 16, 1);
	dim3 dimGrid(N / 16, N / 16, 1);

	cudaStatus = cudaMalloc((void**)&A_d, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\nCudaMalloc failed");
		cuda_error();
		cudaFree(C_d); cudaFree(A_d);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&B_d, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\nCudaMalloc failed");
		cuda_error();
		cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
		return -1;
	}

	cudaStatus = cudaMalloc((void**)&C_d, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\nCudaMalloc failed");
		cuda_error();
		cudaFree(C_d);
		return -1;
	}


	cudaEventRecord(start, 0); //get timer value

	for (int i = 0; i < TIMES; i++) {

		// copy arrays from host to device
		cudaStatus = cudaMemcpy(A_d, A, N * N * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
		if (cudaStatus != cudaSuccess) {
			printf("\ncuda copy failed");
			cuda_error();
			cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
			return -1;
		}

		cudaStatus = cudaMemcpy(B_d, B, N * N * sizeof(float), cudaMemcpyHostToDevice); //copy array from host to GPU
		if (cudaStatus != cudaSuccess) {
			cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
			cuda_error();
			printf("\ncuda copy failed");
			return -1;
		}



		mmm_tiled << <dimGrid, dimBlock >> > (C_d, A_d, B_d);




		cudaStatus = cudaMemcpy(C, C_d, N * N * sizeof(float), cudaMemcpyDeviceToHost); //copy array from GPU back to CPU
		if (cudaStatus != cudaSuccess) {
			printf("\ncuda copy failed");
			cuda_error();
			cudaFree(C_d); cudaFree(A_d); cudaFree(B_d);
			return -1;
		}

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
void init() {

	int i;

	for (i = 0; i < (N * N); i++) {
		A[i] = (float)i;
		B[i] = (float)i + 1;
		C[i] = 0.0f;
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