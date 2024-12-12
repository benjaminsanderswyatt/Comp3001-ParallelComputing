#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h> 
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <immintrin.h>
#include <chrono>

#define N 4096 //array size
#define TIMES 1 //times to run
#define ARITHMETICAL_OPS 2*N*N*N // FLOP 2*N^3

#define EPSILON 0.00001

unsigned short int equal(float const x, float const y);
unsigned short int compare(const float* C, const float* A, const float* B);

void init(float* C, float* A, float* B); // Init

float A[N * N], B[N * N], C[N * N], test[N * N];



void implementation_1(float* C, float* A, float* B) {

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			for (int k = 0; k < N; k++)
				C[N * i + j] += A[N * i + k] * B[N * k + j];

}


int main() {
	printf("\n------------- implementation #1 -------------\n");

	init(C, A, B);

	auto start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < TIMES; i++)
		implementation_1(C, A, B);
	
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	double flops = (double)ARITHMETICAL_OPS / (elapsed.count() / TIMES);
	printf("%f GigaFLOPS achieved\n", flops / 1000000000);

	if (compare(C, A, B) == 0)
		printf("\nCorrect Result\n");
	else
		printf("\nINcorrect Result\n");

	system("pause");

	return 0;

}


// Init
void init(float* C, float* A, float* B) {

	int i;

	for (i = 0; i < (N * N); i++) {
		A[i] = (float)(i % 99 + 0.1);
		B[i] = (float)(i % 65 + 0.1);
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
			return 1;
		}

	return 0;
}


unsigned short int equal(float const x, float const y) {
	float temp = x - y;
	if (fabs(temp / y) < EPSILON)
		return 0; //success
	else
		return 1;
}
