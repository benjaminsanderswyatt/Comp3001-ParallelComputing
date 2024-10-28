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
void optimized_q2(float alpha, float beta);//you will optimize this routine
void vector_optimized_q2(float alpha, float beta);//you will optimize this routine
void reworked_q2(float alpha, float beta);//you will optimize this routine
unsigned short int Compare(float alpha, float beta);
unsigned short int equal(float const a, float const b) ;

#define N 8192 //input size
__declspec(align(64)) float A[N][N], u1[N], u2[N], v1[N], v2[N], x[N], y[N], w[N], z[N], test[N];

#define TIMES_TO_RUN 100 //how many times the function will run
#define EPSILON 0.0001
#define TILE 8

int main() {

float alpha=0.23f, beta=0.45f;

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now

	initialize();

	start_1 = clock(); //start the timer 

for (int i = 0; i < TIMES_TO_RUN; i++)//this loop is needed to get an accurate ex.time value
	// slow_routine(alpha,beta);
	reworked_q2(alpha, beta);
	// optimized_q2(alpha,beta);
	// vector_optimized_q2(alpha, beta);


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

	unsigned int i, j, jj, ii;

	for (i = 0; i < N; i++) {
	//for (i = 0; i < N; i+=TILE) {
	//	for (ii = 0; ii < (i + TILE < N ? i + TILE : N); ii++) {
		x[i] += z[i];

		float temp_u1 = u1[i];
		float temp_u2 = u2[i];

		float temp_beta_y = beta * y[i];
		/*
		for (j = 0; j < N; j += TILE) {
			for (jj = j; jj < (j + TILE < N ? j + TILE : N); jj++) {
				A[i][jj] += temp_u1 * v1[jj] + temp_u2 * v2[jj];

				x[jj] += A[i][jj] * temp_beta_y;
			}

		}
		*/

		
		for (j = 0; j < N; j++) {

			A[i][j] += temp_u1 * v1[j] + temp_u2 * v2[j];

			x[j] += A[i][j] * temp_beta_y;
		}
		
	//}
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

	/*
	for (i = 0; i < N; i++) {

		float temp_u1 = u1[i];
		float temp_u2 = u2[i];

		float temp_y = y[i] * beta;

		for (j = 0; j < N; j++) {

			A[i][j] = A[i][j] + temp_u1 * v1[j] + temp_u2 * v2[j];

			x[j] = x[j] + A[i][j] * temp_y; // fmadd

		}
	}
	*/

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
			x[j] = x[j] + A[i][j] * temp_y; // fmadd
		}
	}


	for (i = 0; i < N; i++) {
		x[i] = x[i] + z[i];
	}

	/*
	for (i = 0; i < N; i++)
		for (j = 0; j < N; j++)
			w[i] = w[i] + alpha * A[i][j] * x[j];
	*/
	for (i = 0; i < N; i++) {
		float temp_w = 0.0f;
		for (j = 0; j < N; j++) {
			temp_w += alpha * A[i][j] * x[j];
		}
		w[i] = temp_w;
	}
}


void vector_optimized_q2(float alpha, float beta) {

	unsigned int i, j;

	for (i = 0; i < N; i++) {
		x[i] += z[i];

		__m256 vec_u1 = _mm256_set1_ps(u1[i]);
		__m256 vec_u2 = _mm256_set1_ps(u2[i]);

		__m256 vec_beta_y = _mm256_set1_ps(beta * y[i]);

		for (j = 0; j < N; j += 8) {
			// A[i][j] += temp_u1 * v1[j] + temp_u2 * v2[j];
			__m256 vec_v1 = _mm256_load_ps(&v1[j]); // v1
			__m256 vec_v2 = _mm256_load_ps(&v2[j]); // v2

			__m256 vec_A = _mm256_load_ps(&A[i][j]); // A

			__m256 temp_uvA = _mm256_fmadd_ps(vec_u1, vec_v1, vec_A); // u1 * v1[j] + A[i][j]
			__m256 temp_uvAuv = _mm256_fmadd_ps(vec_u2, vec_v2, temp_uvA); //  u2 * v2[j] + temp_uvA

			_mm256_storeu_ps(&A[i][j], temp_uvAuv);

			// x[j] += A[i][j] * temp_beta_y;
			__m256 vec_x = _mm256_load_ps(&x[j]); // x

			__m256 temp_xAbetay = _mm256_fmadd_ps(temp_uvAuv, vec_beta_y, vec_x);  // A[i][j] * temp_beta_y + x[j]

			_mm256_store_ps(&x[j], temp_xAbetay);
		}
	}


	for (i = 0; i < N; i++) {

		__m256 vec_temp_w = _mm256_setzero_ps();

		__m256 vec_alpha = _mm256_set1_ps(alpha); // alpha

		for (j = 0; j < N; j += 8) {
			// temp_w += alpha * A[i][j] * x[j];
			__m256 vec_A = _mm256_load_ps(&A[i][j]); // A
			__m256 vec_x = _mm256_load_ps(&x[j]); // x

			__m256 vec_Ax = _mm256_mul_ps(vec_A, vec_x);// A[i][j] * x[j]

			vec_temp_w = _mm256_fmadd_ps(vec_alpha, vec_Ax, vec_temp_w); //  alpha * Ax + temp_w
		}

		//w[i] = temp_w;
		__m128 low = _mm256_extractf128_ps(vec_temp_w, 0); // low temp_w
		__m128 high = _mm256_extractf128_ps(vec_temp_w, 1); // high temp_w

		__m128 sum_lh = _mm_add_ps(low, high); // low + high

		sum_lh = _mm_hadd_ps(sum_lh, sum_lh); // (a0 + a1 , a2 + a3 , b0 + b1 , b2 + b3)
		sum_lh = _mm_hadd_ps(sum_lh, sum_lh); // (a0 + a1 + a2 + a3 , ...)

		w[i] = _mm_cvtss_f32(sum_lh);
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



