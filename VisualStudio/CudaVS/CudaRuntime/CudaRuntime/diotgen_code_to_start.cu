/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <cuda.h> 
#include <cuda_runtime.h> 
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>


#define N 256 //input size
#define ARITHMETICAL_OPS N*N*N*N*2


__declspec(align(64)) float test[N][N][N], sum[N][N][N], A[N][N][N], C[N][N]; 

__device__ float device_sum[N][N][N], device_A[N][N][N], device_C[N][N]; //allocate the device arrays statically (global GPU memory)

void init();
void cuda_error();
void default();
int Compare();
inline unsigned short int equal(float const a, float const b);


#define EPSILON 0.00001

#define MAX_NUMBER_OF_BLOCKS_PER_DIM 65535 //max number of blocks that our GPU can handle (for one dimension only)

#define TILE 8
#define TILE_x2 TILE*2

__global__ void diotgen_ver1_optimised() {
	__shared__ float shared_A_0[TILE][TILE][TILE];
	__shared__ float shared_C_0[TILE][TILE];

	__shared__ float shared_A_1[TILE][TILE][TILE];
	__shared__ float shared_C_1[TILE][TILE];

	__shared__ float shared_A_next_0[TILE][TILE][TILE];
	__shared__ float shared_C_next_0[TILE][TILE];

	__shared__ float shared_A_next_1[TILE][TILE][TILE];
	__shared__ float shared_C_next_1[TILE][TILE];

	int x = blockIdx.x * TILE_x2 + threadIdx.x;
	int y = blockIdx.y * TILE_x2 + threadIdx.y;
	int z = blockIdx.z * TILE + threadIdx.z;


	if (x < N && y < N && z < N) {

		float tempSum_0 = 0.0f;
		float tempSum_1 = 0.0f;
		float tempSum_2 = 0.0f;
		float tempSum_3 = 0.0f;


		shared_A_0[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y][threadIdx.x];
		shared_C_0[threadIdx.y][threadIdx.x] = device_C[threadIdx.y][x];

		shared_A_1[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y + TILE][threadIdx.x];
		shared_C_1[threadIdx.y][threadIdx.x] = device_C[threadIdx.y][x + TILE];

		__syncthreads();

		for (int m = 1; m < (N / TILE - 1); m += 2) {

			for (int s = 0; s < TILE; s++) {
				tempSum_0 += shared_A_0[threadIdx.z][threadIdx.y][s] * shared_C_0[s][threadIdx.x];
				tempSum_1 += shared_A_0[threadIdx.z][threadIdx.y][s] * shared_C_1[s][threadIdx.x];
				tempSum_2 += shared_A_1[threadIdx.z][threadIdx.y][s] * shared_C_0[s][threadIdx.x];
				tempSum_3 += shared_A_1[threadIdx.z][threadIdx.y][s] * shared_C_1[s][threadIdx.x];

			}

			shared_A_next_0[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y][m * TILE + threadIdx.x];
			shared_C_next_0[threadIdx.y][threadIdx.x] = device_C[m * TILE + threadIdx.y][x];
			shared_A_next_1[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y + TILE][m * TILE + threadIdx.x];
			shared_C_next_1[threadIdx.y][threadIdx.x] = device_C[m * TILE + threadIdx.y][x + TILE];

			__syncthreads();


			for (int s = 0; s < TILE; s++) {
				tempSum_0 += shared_A_next_0[threadIdx.z][threadIdx.y][s] * shared_C_next_0[s][threadIdx.x];
				tempSum_1 += shared_A_next_0[threadIdx.z][threadIdx.y][s] * shared_C_next_1[s][threadIdx.x];
				tempSum_2 += shared_A_next_1[threadIdx.z][threadIdx.y][s] * shared_C_next_0[s][threadIdx.x];
				tempSum_3 += shared_A_next_1[threadIdx.z][threadIdx.y][s] * shared_C_next_1[s][threadIdx.x];
			}

			shared_A_0[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y][(m + 1) * TILE + threadIdx.x];
			shared_C_0[threadIdx.y][threadIdx.x] = device_C[(m + 1) * TILE + threadIdx.y][x];
			shared_A_1[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y + TILE][(m + 1) * TILE + threadIdx.x];
			shared_C_1[threadIdx.y][threadIdx.x] = device_C[(m + 1) * TILE + threadIdx.y][x + TILE];

			__syncthreads();
		}


		for (int s = 0; s != TILE; s++) {
			tempSum_0 += shared_A_0[threadIdx.z][threadIdx.y][s] * shared_C_0[s][threadIdx.x];
			tempSum_1 += shared_A_0[threadIdx.z][threadIdx.y][s] * shared_C_1[s][threadIdx.x];
			tempSum_2 += shared_A_1[threadIdx.z][threadIdx.y][s] * shared_C_0[s][threadIdx.x];
			tempSum_3 += shared_A_1[threadIdx.z][threadIdx.y][s] * shared_C_1[s][threadIdx.x];
		}

		int m = N / TILE - 1;

		shared_A_next_0[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y][m * TILE + threadIdx.x];
		shared_C_next_0[threadIdx.y][threadIdx.x] = device_C[m * TILE + threadIdx.y][x];
		shared_A_next_1[threadIdx.z][threadIdx.y][threadIdx.x] = device_A[z][y + TILE][m * TILE + threadIdx.x];
		shared_C_next_1[threadIdx.y][threadIdx.x] = device_C[m * TILE + threadIdx.y][x + TILE];

		__syncthreads();

		for (int s = 0; s != TILE; s++) {
			tempSum_0 += shared_A_next_0[threadIdx.z][threadIdx.y][s] * shared_C_next_0[s][threadIdx.x];
			tempSum_1 += shared_A_next_0[threadIdx.z][threadIdx.y][s] * shared_C_next_1[s][threadIdx.x];
			tempSum_2 += shared_A_next_1[threadIdx.z][threadIdx.y][s] * shared_C_next_0[s][threadIdx.x];
			tempSum_3 += shared_A_next_1[threadIdx.z][threadIdx.y][s] * shared_C_next_1[s][threadIdx.x];
		}
		__syncthreads();

		device_sum[z][y][x] = tempSum_0;
		device_sum[z][y][x + TILE] = tempSum_1;
		device_sum[z][y + TILE][x] = tempSum_2;
		device_sum[z][y + TILE][x + TILE] = tempSum_3;


	}

}

int main()
{
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

	/* Copy the A array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_A, A, N * N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\nA cudaMemcpy failed!");
		cuda_error();
		return -1;
	}

	/* Copy the C array from the HOST memory to the DEVICE memory */
	cudaStatus = cudaMemcpyToSymbol(device_C, C, N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\nC cudaMemcpy failed!");
		cuda_error();
		return -1;
	}


	cudaEventRecord(start, 0); //get timer value

	dim3 dimBlock(TILE, TILE, TILE);
	dim3 dimGrid((N + TILE_x2 - 1) / TILE_x2, (N + TILE_x2 - 1) / TILE_x2, (N + TILE_x2 - 1) / TILE_x2);
	diotgen_ver1_optimised << <dimGrid, dimBlock >> > ( );

	cudaEventRecord(stop, 0);  //get timer value
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time, start, stop);
	printf("\nElapsed time in msecs = %f", elapsed_time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	/* Copy back the result from the DEVICE memory to the HOST memory */
	cudaStatus = cudaMemcpyFromSymbol(sum, device_sum, N * N * N * sizeof(float));
	if (cudaStatus != cudaSuccess) {
		printf("\nS cudaMemcpy failed!");
		cuda_error();
		return -1;
	}


	double flops = (double)((double)ARITHMETICAL_OPS) / (elapsed_time);
	printf("\nGflops achieved %f ", flops / 1000000);

	cuda_error();



	if (Compare() != 0)
		printf("\n---------WRONG OUTPUT---------------\n");
	else
		printf("\n---------OUTPUT IS OK---------------\n");


	/* Destroy all allocations and reset all state on the current device in the current process */
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		printf("\ncuda Reset failed!");
		cuda_error();
		return -1;
	}

	return 0;
}


void init() {

	float e = 0.12, p = 0.72;
	unsigned int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C[i][j] = (j % 9) + p;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			for (k = 0; k < N; k++) {
				sum[i][j][k] = 0.0;
				test[i][j][k] = 0.0;
				A[i][j][k] = (((i + j) % 99) + e);
			}
		}
	}


}

//this is the routine that you will parallelize 
void default() {

	for (int r = 0; r < N; r++)
		for (int q = 0; q < N; q++)
			for (int p = 0; p < N; p++)
				for (int s = 0; s < N; s++)
					test[r][q][p] = test[r][q][p] + A[r][q][s] * C[s][p];


}


unsigned short int equal(float const a, float const b) {
	float temp = a - b;
	//printf("\n %f  %f", a, b);
	if (fabs(temp/b) < EPSILON)
		return 0; //success
	else
		return 1;
}


int Compare() {


	default();


	for (int r = 0; r < N; r++)
		for (int q = 0; q < N; q++)
				for (int p = 0; p < N; p++)
					if (equal(sum[r][q][p], test[r][q][p]) == 1) {
				      printf("\n wrong at (%d,%d,%d)", r, q,p);
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