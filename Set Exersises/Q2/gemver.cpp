/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------COMPUTER SYSTEMS MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <stdio.h> //this library is needed for printf function
#include <stdlib.h> //this library is needed for rand() function
#include <windows.h> //this library is needed for pause() function
#include <time.h> //this library is needed for clock() function
#include <math.h> //this library is needed for abs()
#include <pmmintrin.h>
#include <process.h>
//#include <chrono>
#include <iostream>
#include <immintrin.h>

void initialize();
void initialize_again();
void slow_routine(float alpha, float beta);//I have optimized this routine
void fast_routine(float alpha, float beta);
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b) ;

#define N 8192 //input size 8192
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 150 //how many times the function will run
#define EPSILON 0.0001

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define TILE 4096

int main() {

float alpha=0.23f, beta=0.45f;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

	for (int i = 0; i < TIMES_TO_RUN; i++) { //this loop is needed to get an accurate ex.time value
		// slow_routine(alpha,beta);
		fast_routine(alpha, beta);
	}
	

end_1 = clock(); //end the timer 

printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));//print the ex.time

if (Compare(alpha, beta) == 0)
printf("\nCorrect Result\n");
else
printf("\nINcorrect Result\n");

system("pause"); //this command does not let the output window to close

return 0; //normally, by returning zero, we mean that the program ended successfully. 
}


void initialize() {

	unsigned int    i, j;

	//initialization
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = 1.1f;

		}

	for (i = 0; i < N; i++) {
		z[i] = (i % 9) * 0.8f;
		x[i] = 0.1f;
		u1[i] = (i % 9) * 0.2f;
		u2[i] = (i % 9) * 0.3f;
		v1[i] = (i % 9) * 0.4f;
		v2[i] = (i % 9) * 0.5f;
		w[i] = 0.0f;
		y[i] = (i % 9) * 0.7f;
	}

}

void initialize_again() {

	unsigned int    i, j;

	//initialization
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++) {
			A[i][j] = 1.1f;

		}

	for (i = 0; i < N; i++) {
		z[i] = (i % 9) * 0.8f;
		x[i] = 0.1f;
		test[i] = 0.0f;
		u1[i] = (i % 9) * 0.2f;
		u2[i] = (i % 9) * 0.3f;
		v1[i] = (i % 9) * 0.4f;
		v2[i] = (i % 9) * 0.5f;
		y[i] = (i % 9) * 0.7f;
	}

}

//you will optimize this routine
void slow_routine(float alpha, float beta) {

	unsigned int i, j;

	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]; // + * + * (4 * N * N)


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			x[i] = x[i] + beta * A[j][i] * y[j]; // + * * (3 * N * N)

	for (i = 0; i < N; i++)
		x[i] = x[i] + z[i]; // + (1 * N)


	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			w[i] = w[i] + alpha * A[i][j] * x[j]; // + * * (3 * N * N)
		}
	}

	// FLOP: 10N^2 + N
	// = 5497.56 Giga FLOP (to 2 decimal places)
}

// I have optimized this routine
void fast_routine(float alpha, float beta) {

	unsigned int i, j, jj;

	// 1st Loop Block
	for (i = 0; i <= N - 4; i += 4) {

		__m256 temp_u1_0 = _mm256_set1_ps(u1[i]); // u1  (0)
		__m256 temp_u2_0 = _mm256_set1_ps(u2[i]); // u2  (0)

		__m256 temp_u1_1 = _mm256_set1_ps(u1[i + 1]); // u1  (1)
		__m256 temp_u2_1 = _mm256_set1_ps(u2[i + 1]); // u2  (1)

		__m256 temp_u1_2 = _mm256_set1_ps(u1[i + 2]); // u1  (2)
		__m256 temp_u2_2 = _mm256_set1_ps(u2[i + 2]); // u2  (2)

		__m256 temp_u1_3 = _mm256_set1_ps(u1[i + 3]); // u1  (3)
		__m256 temp_u2_3 = _mm256_set1_ps(u2[i + 3]); // u2  (3)

		for (jj = 0; jj < N; jj += TILE) {
			for (j = jj; j <= MIN(N, jj + TILE) - 8; j += 8) {
				// A[i][j] = A[i][j] + temp_u1 * v1[j] + temp_u2 * v2[j];
				__m256 temp_v1 = _mm256_load_ps(&v1[j]); // v1
				__m256 temp_v2 = _mm256_load_ps(&v2[j]); // v2

				__m256 temp_A_0 = _mm256_load_ps(&A[i][j]); // A  (0)
				__m256 temp_A_1 = _mm256_load_ps(&A[i + 1][j]); // A  (1)
				__m256 temp_A_2 = _mm256_load_ps(&A[i + 2][j]); // A  (2)
				__m256 temp_A_3 = _mm256_load_ps(&A[i + 3][j]); // A  (3)

				__m256 temp_uvA_0 = _mm256_fmadd_ps(temp_u1_0, temp_v1, temp_A_0); // u1 * v1[j] + A[i][j]  (0)
				__m256 temp_uvAuv_0 = _mm256_fmadd_ps(temp_u2_0, temp_v2, temp_uvA_0); //  u2 * v2[j] + temp_uvA  (0)
				_mm256_store_ps(&A[i][j], temp_uvAuv_0); //  (0)

				__m256 temp_uvA_1 = _mm256_fmadd_ps(temp_u1_1, temp_v1, temp_A_1); // u1 * v1[j] + A[i][j]  (1)
				__m256 temp_uvAuv_1 = _mm256_fmadd_ps(temp_u2_1, temp_v2, temp_uvA_1); //  u2 * v2[j] + temp_uvA  (1)
				_mm256_store_ps(&A[i + 1][j], temp_uvAuv_1); //  (1)

				__m256 temp_uvA_2 = _mm256_fmadd_ps(temp_u1_2, temp_v1, temp_A_2); // u1 * v1[j] + A[i][j]  (2)
				__m256 temp_uvAuv_2 = _mm256_fmadd_ps(temp_u2_2, temp_v2, temp_uvA_2); //  u2 * v2[j] + temp_uvA  (2)
				_mm256_store_ps(&A[i + 2][j], temp_uvAuv_2); //  (2)

				__m256 temp_uvA_3 = _mm256_fmadd_ps(temp_u1_3, temp_v1, temp_A_3); // u1 * v1[j] + A[i][j]  (3)
				__m256 temp_uvAuv_3 = _mm256_fmadd_ps(temp_u2_3, temp_v2, temp_uvA_3); //  u2 * v2[j] + temp_uvA  (3)
				_mm256_store_ps(&A[i + 3][j], temp_uvAuv_3); //  (3)

			}
		}
		
		for (; j < N; j++) { // Leftovers j
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]; //  (0)
			A[i + 1][j] = A[i + 1][j] + u1[i + 1] * v1[j] + u2[i + 1] * v2[j]; //  (1)
			A[i + 2][j] = A[i + 2][j] + u1[i + 2] * v1[j] + u2[i + 2] * v2[j]; //  (2)
			A[i + 3][j] = A[i + 3][j] + u1[i + 3] * v1[j] + u2[i + 3] * v2[j]; //  (3)
		}
	}

	for (; i < N; i++) { // Leftovers i
		__m256 temp_u1 = _mm256_set1_ps(u1[i]);
		__m256 temp_u2 = _mm256_set1_ps(u2[i]);

		for (j = 0; j <= N - 8; j += 8) {
			__m256 temp_A = _mm256_load_ps(&A[i][j]);
			__m256 temp_v1 = _mm256_load_ps(&v1[j]);
			__m256 temp_v2 = _mm256_load_ps(&v2[j]);

			__m256 temp_uvA = _mm256_fmadd_ps(temp_u1, temp_v1, temp_A);
			__m256 temp_uvAuv = _mm256_fmadd_ps(temp_u2, temp_v2, temp_uvA);

			_mm256_store_ps(&A[i][j], temp_uvAuv);
		}


		for (; j < N; j++) { // Leftovers (i)j
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}
	














	// 2nd Loop Block
	for (i = 0; i <= N - 4; i += 4) {

		__m256 temp_y_0 = _mm256_set1_ps(beta * y[i]); // beta * y[i]  (0)
		__m256 temp_y_1 = _mm256_set1_ps(beta * y[i + 1]); // beta * y[i]  (1)
		__m256 temp_y_2 = _mm256_set1_ps(beta * y[i + 2]); // beta * y[i]  (2)
		__m256 temp_y_3 = _mm256_set1_ps(beta * y[i + 3]); // beta * y[i]  (3)

		for (j = 0; j <= N - 8; j += 8) {
			// x[j] = x[j] + A[i][j] * temp_y;
			__m256 temp_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_A_0 = _mm256_load_ps(&A[i][j]); // A  (0)
			temp_x = _mm256_fmadd_ps(temp_A_0, temp_y_0, temp_x); // x = A * y + x  (0)

			__m256 temp_A_1 = _mm256_load_ps(&A[i + 1][j]); // A  (1)
			temp_x = _mm256_fmadd_ps(temp_A_1, temp_y_1, temp_x); // x = A * y + x  (1)

			__m256 temp_A_2 = _mm256_load_ps(&A[i + 2][j]); // A  (2)
			temp_x = _mm256_fmadd_ps(temp_A_2, temp_y_2, temp_x); // x = A * y + x  (2)

			__m256 temp_A_3 = _mm256_load_ps(&A[i + 3][j]); // A  (3)
			temp_x = _mm256_fmadd_ps(temp_A_3, temp_y_3, temp_x); // x = A * y + x  (3)

			_mm256_store_ps(&x[j], temp_x);
		}

		for (; j < N; j++) { // Leftovers j
			x[j] += A[i][j] * y[i] * beta; //  (0)
			x[j] += A[i + 1][j] * y[i + 1] * beta; //  (1)
			x[j] += A[i + 2][j] * y[i + 2] * beta; //  (2)
			x[j] += A[i + 3][j] * y[i + 3] * beta; //  (3)
		}
	}

	for (; i < N; i++) { // Leftovers i
		__m256 temp_y = _mm256_set1_ps(beta * y[i]); // beta * y[i]

		for (j = 0; j <= N - 8; j += 8) {
			// x[j] = x[j] + A[i][j] * temp_y;
			__m256 temp_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_A = _mm256_load_ps(&A[i][j]); // A
			temp_x = _mm256_fmadd_ps(temp_A, temp_y, temp_x); // x = A * y + x

			_mm256_store_ps(&x[j], temp_x);
		}

		for (; j < N; j++) { // Leftovers (i)j
			x[j] += A[i][j] * y[i] * beta;
		}
	}













	

	// 3rd Loop Block
	for (i = 0; i <= N - 8; i += 8) {
		__m256 temp_x = _mm256_load_ps(&x[i]); // x
		__m256 temp_z = _mm256_load_ps(&z[i]); // z

		__m256 sum_x_z = _mm256_add_ps(temp_x, temp_z); //x + z

		_mm256_store_ps(&x[i], sum_x_z);
	}

	for (; i < N; i++) { // Leftovers
		x[i] += z[i];
	}











	// 4th Loop Block
	for (i = 0; i <= N - 4; i += 4) {

		__m256 temp_w_0 = _mm256_setzero_ps(); //  (0)
		__m256 temp_w_1 = _mm256_setzero_ps(); //  (1)
		__m256 temp_w_2 = _mm256_setzero_ps(); //  (2)
		__m256 temp_w_3 = _mm256_setzero_ps(); //  (3)

		__m256 temp_alpha = _mm256_set1_ps(alpha); // alpha

		for (jj = 0; jj < N; jj += TILE) {
			for (j = jj; j <= MIN(N, jj + TILE) - 8; j += 8) {
				// temp_w += alpha * A[i][j] * x[j];
				__m256 temp_x = _mm256_load_ps(&x[j]); // x

				__m256 temp_A_0 = _mm256_load_ps(&A[i][j]); // A  (0)
				__m256 temp_Ax_0 = _mm256_mul_ps(temp_A_0, temp_x); // A[i][j] * x[j]  (0)

				temp_w_0 = _mm256_fmadd_ps(temp_alpha, temp_Ax_0, temp_w_0); //  alpha * Ax + temp_w  (0)

				__m256 temp_A_1 = _mm256_load_ps(&A[i + 1][j]); // A  (1)
				__m256 temp_Ax_1 = _mm256_mul_ps(temp_A_1, temp_x); // A[i][j] * x[j]  (1)

				temp_w_1 = _mm256_fmadd_ps(temp_alpha, temp_Ax_1, temp_w_1); //  alpha * Ax + temp_w  (1)

				__m256 temp_A_2 = _mm256_load_ps(&A[i + 2][j]); // A  (2)
				__m256 temp_Ax_2 = _mm256_mul_ps(temp_A_2, temp_x); // A[i][j] * x[j]  (2)

				temp_w_2 = _mm256_fmadd_ps(temp_alpha, temp_Ax_2, temp_w_2); //  alpha * Ax + temp_w  (2)

				__m256 temp_A_3 = _mm256_load_ps(&A[i + 3][j]); // A  (3)
				__m256 temp_Ax_3 = _mm256_mul_ps(temp_A_3, temp_x); // A[i][j] * x[j]  (3)

				temp_w_3 = _mm256_fmadd_ps(temp_alpha, temp_Ax_3, temp_w_3); //  alpha * Ax + temp_w  (3)
			}
		}

		//w[i] = temp_w;  (0)
		__m128 low_0 = _mm256_extractf128_ps(temp_w_0, 0); // low temp_w
		__m128 high_0 = _mm256_extractf128_ps(temp_w_0, 1); // high temp_w

		__m128 sum_lh_0 = _mm_add_ps(low_0, high_0); // low + high

		sum_lh_0 = _mm_hadd_ps(sum_lh_0, sum_lh_0); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh_0 = _mm_hadd_ps(sum_lh_0, sum_lh_0); // (a0 + a1 + a2 + a3 , ...)

		_mm_store_ss((float*)&w[i], sum_lh_0); //  (0)


		//w[i] = temp_w;  (1)
		__m128 low_1 = _mm256_extractf128_ps(temp_w_1, 0); // low temp_w
		__m128 high_1 = _mm256_extractf128_ps(temp_w_1, 1); // high temp_w

		__m128 sum_lh_1 = _mm_add_ps(low_1, high_1); // low + high

		sum_lh_1 = _mm_hadd_ps(sum_lh_1, sum_lh_1); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh_1 = _mm_hadd_ps(sum_lh_1, sum_lh_1); // (a0 + a1 + a2 + a3 , ...)

		_mm_store_ss((float*)&w[i + 1], sum_lh_1); //  (1)

		//w[i] = temp_w;  (2)
		__m128 low_2 = _mm256_extractf128_ps(temp_w_2, 0); // low temp_w
		__m128 high_2 = _mm256_extractf128_ps(temp_w_2, 1); // high temp_w

		__m128 sum_lh_2 = _mm_add_ps(low_2, high_2); // low + high

		sum_lh_2 = _mm_hadd_ps(sum_lh_2, sum_lh_2); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh_2 = _mm_hadd_ps(sum_lh_2, sum_lh_2); // (a0 + a1 + a2 + a3 , ...)

		_mm_store_ss((float*)&w[i + 2], sum_lh_2); //  (2)

		//w[i] = temp_w;  (3)
		__m128 low_3 = _mm256_extractf128_ps(temp_w_3, 0); // low temp_w
		__m128 high_3 = _mm256_extractf128_ps(temp_w_3, 1); // high temp_w

		__m128 sum_lh_3 = _mm_add_ps(low_3, high_3); // low + high

		sum_lh_3 = _mm_hadd_ps(sum_lh_3, sum_lh_3); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh_3 = _mm_hadd_ps(sum_lh_3, sum_lh_3); // (a0 + a1 + a2 + a3 , ...)

		_mm_store_ss((float*)&w[i + 3], sum_lh_3); //  (3)

		for (; j < N; j++) { // Leftovers j
			w[i] += alpha * A[i][j] * x[j];
			w[i + 1] += alpha * A[i + 1][j] * x[j];
			w[i + 2] += alpha * A[i + 2][j] * x[j];
			w[i + 3] += alpha * A[i + 3][j] * x[j];
		}
	}
	
	
	for (; i < N; i++) { // Leftovers i
		__m256 temp_w = _mm256_setzero_ps();

		__m256 temp_alpha = _mm256_set1_ps(alpha); // alpha

		for (j = 0; j <= N - 8; j += 8) {
			// temp_w += alpha * A[i][j] * x[j];
			__m256 temp_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_A = _mm256_load_ps(&A[i][j]); // A
			__m256 temp_Ax = _mm256_mul_ps(temp_A, temp_x); // A[i][j] * x[j]

			temp_w = _mm256_fmadd_ps(temp_alpha, temp_Ax, temp_w); //  alpha * Ax + temp_w

		}

		//w[i] = temp_w;
		__m128 low = _mm256_extractf128_ps(temp_w, 0); // low temp_w
		__m128 high = _mm256_extractf128_ps(temp_w, 1); // high temp_w

		__m128 sum_lh = _mm_add_ps(low, high); // low + high

		sum_lh = _mm_hadd_ps(sum_lh, sum_lh); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh = _mm_hadd_ps(sum_lh, sum_lh); // (a0 + a1 + a2 + a3 , ...)

		_mm_store_ss((float*)&w[i], sum_lh);


		for (; j < N; j++) { // Leftovers (i)j
			w[i] += alpha * A[i][j] * x[j];
		}
	}
	
}



unsigned short int Compare(float alpha, float beta) {

unsigned int i,j;

initialize_again();


  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];


  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < N; i++)
    x[i] = x[i] + z[i];


  for (i = 0; i < N; i++){
    for (j = 0; j < N; j++){
     test[i]= test[i] + alpha * A[i][j] * x[j];
     } }



    for (j = 0; j < N; j++){
	if (equal(w[j],test[j]) == 1){
	  printf("\n %f %f",test[j], w[j]);
		return -1;
		}
		}

	return 0;
}




unsigned short int equal(float const a, float const b) {
	
	if (fabs(a-b)/fabs(a) < EPSILON)
		return 0; //success
	else
		return 1;
}



