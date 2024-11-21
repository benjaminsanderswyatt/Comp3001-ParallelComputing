#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>




#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h> 
#include <stdio.h>
#include <time.h>
#include <chrono>
#include <iostream>

#define N 16  //arrays input size
#define TIMES 1 //times to run
#define ARITHMETICAL_OPS N*N*N*2

#define EPSILON 0.00001



float *C, *A, *B;
float test[N * N];


unsigned short int compare(const float* C, const float* A, const float* B);
unsigned short int equal(float const x, float const y);

void init(float* C, float *A, float *B); // Init


void implementation_1(float* C, float* A, float* B); // implementation 1
__global__ void implementation_3(float* C, float* A, float* B); // implementation 3


int main() {
	// --------------------------------------- Init ---------------------------------------
	A = (float*)_mm_malloc(N * N * sizeof(float), 64);
	B = (float*)_mm_malloc(N * N * sizeof(float), 64);
	C = (float*)_mm_malloc(N * N * sizeof(float), 64);

	if (A == NULL || B == NULL || C == NULL) {
		printf("\nMemory not allocated.\n");
		exit(0);
	}



	

	// --------------------------------------- implementation #1 ---------------------------------------
	printf("\n------------- implementation #1 -------------\n");

	init(C, A, B);

	auto start_implementation_1 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < TIMES; i++) {
		implementation_1(C, A, B);
	}

	auto finish_implementation_1 = std::chrono::high_resolution_clock::now();

	if (compare(C, A, B) == 0)
		printf("\nResult is ok\n");
	else
		printf("\nResult is FALSE\n");

	std::chrono::duration<double> elapsed_implementaion_1 = finish_implementation_1 - start_implementation_1;
	std::cout << "Elapsed time: " << elapsed_implementaion_1.count() << " s\n";


	// --------------------------------------- implementation #2 ---------------------------------------
	printf("\n------------- implementation #2 -------------\n");

	init(C, A, B);

	// --------------------------------------- implementation #3 ---------------------------------------
	printf("\n------------- implementation #3 -------------\n");

	init(C, A, B);

	dim3 dimBlock(16, 16, 1); 
	dim3 dimGrid(N/16, N/16, 1); 


	auto start_implementation_3 = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < TIMES; i++) {
		implementation_3 <<<dimGrid, dimBlock>>> (C,A,B);

	}

	auto finish_implementation_3 = std::chrono::high_resolution_clock::now();
	
	// Error message
	cudaError_t error_implementation_3 = cudaGetLastError();
	if (error_implementation_3 != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(error_implementation_3));
	}

	if (compare(C,A,B) == 0)
		printf("\nResult is ok\n");
	else
		printf("\nResult is FALSE\n");

	std::chrono::duration<double> elapsed_implementaion_3 = finish_implementation_3 - start_implementation_3;
	std::cout << "Elapsed time: " << elapsed_implementaion_3.count() << " s\n";



	cudaDeviceReset();



	// --------------------------------------- implementation #4 ---------------------------------------
	printf("\n------------- implementation #4 -------------\n");

	init(A, B, C);

	







	return 0;
}

// Init
void init(float *C, float *A, float *B) {

	int i;

	for (i = 0; i < (N * N); i++) {
		A[i] = (float)i;
		B[i] = (float)i + 1;
		C[i] = 0.0f;
	}

}


// --------------------------------------- implementation #1 ---------------------------------------
void implementation_1(float* C, float* A, float* B) {
	for (int i = 0; i < N; i++)

		for (int j = 0; j < N; j++)

			for (int k = 0; k < N; k++)

				C[N * i + j] += A[N * i + k] * B[N * k + j];


}


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
			return 1;
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