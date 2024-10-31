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
void slow_routine(float alpha, float beta);//you will optimize this routine
void optimized_q2(float alpha, float beta);//--------------------------------------------------------------------
void reworked_q2(float alpha, float beta);//--------------------------------------------------------------------
void reg_blocking_optimized_q2(float alpha, float beta);//--------------------------------------------------------------------
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b) ;

#define N 8192 //input size 8192
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 100 //how many times the function will run
#define EPSILON 0.0001

int main() {

float alpha=0.23f, beta=0.45f;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

for (int i = 0; i < TIMES_TO_RUN; i++)//this loop is needed to get an accurate ex.time value
	// slow_routine(alpha,beta);
	//reworked_q2(alpha, beta);
	//optimized_q2(alpha,beta);
	reg_blocking_optimized_q2(alpha, beta);


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
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];


	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			x[i] = x[i] + beta * A[j][i] * y[j];

	for (i = 0; i < N; i++)
		x[i] = x[i] + z[i];


	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			w[i] = w[i] + alpha * A[i][j] * x[j];
		}
	}

}

void reworked_q2(float alpha, float beta) {

	unsigned int i, j;

	for (i = 0; i < N; i++) {

		float temp_u1 = u1[i];
		float temp_u2 = u2[i];

		for (j = 0; j < N; j++) {
			A[i][j] = A[i][j] + temp_u1 * v1[j] + temp_u2 * v2[j];
		}
	}

	for (i = 0; i < N; i++) {

		float temp_y = y[i] * beta;

		for (j = 0; j < N; j++) {
			x[j] = x[j] + A[i][j] * temp_y;
		}
	}


	for (i = 0; i < N; i++) {
		x[i] = x[i] + z[i];
	}


	for (i = 0; i < N; i++) {
		float temp_w = 0.0f;
		for (j = 0; j < N; j++) {
			temp_w += alpha * A[i][j] * x[j];
		}
		w[i] = temp_w;
	}


}

void optimized_q2(float alpha, float beta) {

	unsigned int i, j;
	
	// 1st Loop Block   -   loop tiling,
	for (i = 0; i < N; i++) {
		
		__m256 temp_u1 = _mm256_set1_ps(u1[i]); // u1
		__m256 temp_u2 = _mm256_set1_ps(u2[i]); // u2

		for (j = 0; j <= N - 8; j += 8) {
			// A[i][j] = A[i][j] + temp_u1 * v1[j] + temp_u2 * v2[j];
			__m256 temp_A = _mm256_load_ps(&A[i][j]); // A
			__m256 temp_v1 = _mm256_load_ps(&v1[j]); // v1
			__m256 temp_v2 = _mm256_load_ps(&v2[j]); // v2

			__m256 temp_uvA = _mm256_fmadd_ps(temp_u1, temp_v1, temp_A); // u1 * v1[j] + A[i][j]
			__m256 temp_uvAuv = _mm256_fmadd_ps(temp_u2, temp_v2, temp_uvA); //  u2 * v2[j] + temp_uvA

			_mm256_store_ps(&A[i][j], temp_uvAuv);
		}
		
		for (;j < N; j++) { // Leftovers
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
		}
	}
	
	// 2nd Loop Block
	for (i = 0; i < N; i++) {
		
		__m256 temp_y = _mm256_set1_ps(beta * y[i]); // beta * y[i]

		for (j = 0; j <= N - 8; j += 8) {
			// x[j] = x[j] + A[i][j] * temp_y;
			__m256 temp_x = _mm256_load_ps(&x[j]); // x
			__m256 temp_A = _mm256_load_ps(&A[i][j]); // A

			temp_x = _mm256_fmadd_ps(temp_A, temp_y, temp_x); // x = A * y + x

			_mm256_store_ps(&x[j], temp_x);
		}

		for (; j < N; j++) { // Leftovers
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
	
	// 4th Loop Block   -   loop tiling, Reduction
	for (i = 0; i < N; i++) {

		__m256 temp_w = _mm256_setzero_ps();
		__m256 temp_alpha = _mm256_set1_ps(alpha); // alpha

		for (j = 0; j <= N - 8; j += 8) {
			// temp_w += alpha * A[i][j] * x[j];
			__m256 temp_A = _mm256_load_ps(&A[i][j]); // A
			__m256 temp_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_Ax = _mm256_mul_ps(temp_A, temp_x); // A[i][j] * x[j]

			temp_w = _mm256_fmadd_ps(temp_alpha, temp_Ax, temp_w); //  alpha * Ax + temp_w
		}

		//w[i] = temp_w;
		__m128 low = _mm256_extractf128_ps(temp_w, 0); // low temp_w
		__m128 high = _mm256_extractf128_ps(temp_w, 1); // high temp_w

		__m128 sum_lh = _mm_add_ps(low, high); // low + high

		sum_lh = _mm_hadd_ps(sum_lh, sum_lh); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh = _mm_hadd_ps(sum_lh, sum_lh); // (a0 + a1 + a2 + a3 , ...)

		w[i] = _mm_cvtss_f32(sum_lh);

		// Leftovers
		for (; j < N; j++) {
			w[i] += alpha * A[i][j] * x[j];
		}
	}
	
}


void reg_blocking_optimized_q2(float alpha, float beta) {

	unsigned int i, j;

	// 1st Loop Block   -   loop tiling,
	for (i = 0; i <= N - 2; i+=2) {

		__m256 temp_u1_0 = _mm256_set1_ps(u1[i]); // u1  (0)
		__m256 temp_u2_0 = _mm256_set1_ps(u2[i]); // u2  (0)
		__m256 temp_u1_1 = _mm256_set1_ps(u1[i+1]); // u1  (1)
		__m256 temp_u2_1 = _mm256_set1_ps(u2[i+1]); // u2  (1)

		for (j = 0; j <= N - 8; j += 8) {
			// A[i][j] = A[i][j] + temp_u1 * v1[j] + temp_u2 * v2[j];
			__m256 temp_v1 = _mm256_load_ps(&v1[j]); // v1
			__m256 temp_v2 = _mm256_load_ps(&v2[j]); // v2

			__m256 temp_A_0 = _mm256_load_ps(&A[i][j]); // A  (0)
			__m256 temp_A_1 = _mm256_load_ps(&A[i + 1][j]); // A  (1)

			__m256 temp_uvA_0 = _mm256_fmadd_ps(temp_u1_0, temp_v1, temp_A_0); // u1 * v1[j] + A[i][j]  (0)
			__m256 temp_uvAuv_0 = _mm256_fmadd_ps(temp_u2_0, temp_v2, temp_uvA_0); //  u2 * v2[j] + temp_uvA  (0)

			__m256 temp_uvA_1 = _mm256_fmadd_ps(temp_u1_1, temp_v1, temp_A_1); // u1 * v1[j] + A[i][j]  (1)
			__m256 temp_uvAuv_1 = _mm256_fmadd_ps(temp_u2_1, temp_v2, temp_uvA_1); //  u2 * v2[j] + temp_uvA  (1)

			_mm256_store_ps(&A[i][j], temp_uvAuv_0); //  (0)
			_mm256_store_ps(&A[i+1][j], temp_uvAuv_1); //  (1)
		}

		for (; j < N; j++) { // Leftovers j
			A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j]; //  (0)
			A[i+1][j] = A[i+1][j] + u1[i+1] * v1[j] + u2[i+1] * v2[j]; //  (1)
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
	for (i = 0; i <= N - 2; i+=2) {

		__m256 temp_y_0 = _mm256_set1_ps(beta * y[i]); // beta * y[i]  (0)
		__m256 temp_y_1 = _mm256_set1_ps(beta * y[i+1]); // beta * y[i]  (1)

		for (j = 0; j <= N - 8; j += 8) {
			// x[j] = x[j] + A[i][j] * temp_y;
			__m256 temp_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_A_0 = _mm256_load_ps(&A[i][j]); // A  (0)
			temp_x = _mm256_fmadd_ps(temp_A_0, temp_y_0, temp_x); // x = A * y + x  (0)

			__m256 temp_A_1 = _mm256_load_ps(&A[i+1][j]); // A  (1)
			temp_x = _mm256_fmadd_ps(temp_A_1, temp_y_1, temp_x); // x = A * y + x  (1)


			_mm256_store_ps(&x[j], temp_x);
		}

		for (; j < N; j++) { // Leftovers j
			x[j] += A[i][j] * y[i] * beta;
			x[j] += A[i+1][j] * y[i+1] * beta;
		}
	}

	for (; i < N; i ++) { // Leftovers i

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
	for (i = 0; i <= N - 2; i+=2) {

		__m256 temp_w_0 = _mm256_setzero_ps(); //  (0)
		__m256 temp_w_1 = _mm256_setzero_ps(); //  (1)

		__m256 temp_alpha = _mm256_set1_ps(alpha); // alpha

		for (j = 0; j <= N - 8; j += 8) {
			// temp_w += alpha * A[i][j] * x[j];
			__m256 temp_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_A_0 = _mm256_load_ps(&A[i][j]); // A  (0)
			__m256 temp_Ax_0 = _mm256_mul_ps(temp_A_0, temp_x); // A[i][j] * x[j]  (0)

			temp_w_0 = _mm256_fmadd_ps(temp_alpha, temp_Ax_0, temp_w_0); //  alpha * Ax + temp_w  (0)

			__m256 temp_A_1 = _mm256_load_ps(&A[i+1][j]); // A  (1)
			__m256 temp_Ax_1 = _mm256_mul_ps(temp_A_1, temp_x); // A[i][j] * x[j]  (1)

			temp_w_1 = _mm256_fmadd_ps(temp_alpha, temp_Ax_1, temp_w_1); //  alpha * Ax + temp_w  (1)
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

		_mm_store_ss((float*)&w[i+1], sum_lh_1); //  (1)


		for (; j < N; j++) { // Leftovers j
			w[i] += alpha * A[i][j] * x[j];
			w[i+1] += alpha * A[i+1][j] * x[j];
		}
	}

	for (; i < N; i ++) { // Leftovers i

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








/*
for (ii = 0; ii < n; ii += TILE_SIZE_I) {
	for (jj = 0; jj < n; jj += TILE_SIZE_J) {
		for (i = ii; i < min(n, ii + TILE_SIZE_I); i++) {
			for (j = jj; j < min(n, jj + TILE_SIZE_J); j++) {
				c[i][j] = a[i] * b[j];
				printf("\n (%d i,%d j)", i, j);
			}
		}
	}
}
*/



// #define TILE_J 128
/* TRYING OUT LOOP TILING
unsigned int jj;

// 1st Loop Block
for (i = 0; i < N; i ++) {
	__m256 temp_u1 = _mm256_set1_ps(u1[i]); // u1
	__m256 temp_u2 = _mm256_set1_ps(u2[i]); // u2
	for (jj = 0; jj < N; jj += TILE_J) {
		for (j = jj; j < min(N, jj + TILE_J); j+= 8) {
			//printf("(ii %d, jj %d ,i %d, j %d)", ii, jj, i, j);

			// A[i][j] = A[i][j] + temp_u1 * v1[j] + temp_u2 * v2[j];
			__m256 temp_A = _mm256_load_ps(&A[i][j]); // A
			__m256 temp_v1 = _mm256_load_ps(&v1[j]); // v1
			__m256 temp_v2 = _mm256_load_ps(&v2[j]); // v2

			__m256 temp_uvA = _mm256_fmadd_ps(temp_u1, temp_v1, temp_A); // u1 * v1[j] + A[i][j]
			__m256 temp_uvAuv = _mm256_fmadd_ps(temp_u2, temp_v2, temp_uvA); //  u2 * v2[j] + temp_uvA

			_mm256_store_ps(&A[i][j], temp_uvAuv);
		}

	}
}
*/




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



