#include <omp.h>

#include "canny.h"

unsigned char filt[N][M], gradient[N][M], grad2[N][M], edgeDir[N][M];
unsigned char gaussianMask[5][5];

void GaussianBlur() {

	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int newPixel;

	unsigned char temp;


	/* Declare Gaussian mask */
	gaussianMask[0][0] = 2;

	gaussianMask[0][1] = 4;
	gaussianMask[0][2] = 5;
	gaussianMask[0][3] = 4;
	gaussianMask[0][4] = 2;

	gaussianMask[1][0] = 4;
	gaussianMask[1][1] = 9;
	gaussianMask[1][2] = 12;
	gaussianMask[1][3] = 9;
	gaussianMask[1][4] = 4;

	gaussianMask[2][0] = 5;
	gaussianMask[2][1] = 12;
	gaussianMask[2][2] = 15;
	gaussianMask[2][3] = 12;
	gaussianMask[2][4] = 5;

	gaussianMask[3][0] = 4;
	gaussianMask[3][1] = 9;
	gaussianMask[3][2] = 12;
	gaussianMask[3][3] = 9;
	gaussianMask[3][4] = 4;

	gaussianMask[4][0] = 2;
	gaussianMask[4][1] = 4;
	gaussianMask[4][2] = 5;
	gaussianMask[4][3] = 4;
	gaussianMask[4][4] = 2;

	/*---------------------- Gaussian Blur ---------------------------------*/
	for (row = 2; row < N - 2; row++) {
		for (col = 2; col < M - 2; col++) {
			newPixel = 0;
			for (rowOffset = -2; rowOffset <= 2; rowOffset++) {
				for (colOffset = -2; colOffset <= 2; colOffset++) {

					newPixel += frame1[row + rowOffset][col + colOffset] * gaussianMask[2 + rowOffset][2 + colOffset];
				}
			}
			filt[row][col] = (unsigned char)(newPixel / 159);
		}
	}


}



#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define ROWTILE 32

#define RADTODEGREE 180 / 3.14159
#define RECIPROCAL 1.0 / 45.0
#define FOURPIRAD 4.0 / 3.14159


void Sobel() {
	int r, c, row, col;

	// Gx
	// [-1, 0, 1]
	// [-2, 0, 2]
	// [-1, 0, 1]
	const __m256i GxMask13 = _mm256_set_epi8(0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1); // 1 & 3
	const __m256i GxMask2 = _mm256_set_epi8(0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2);

	// Gy
	// [-1,-2,-1]
	// [ 0, 0, 0] zero isnt needed
	// [ 1, 2, 1]
	const __m256i GyMask1 = _mm256_set_epi8(0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1);
	const __m256i GyMask3 = _mm256_set_epi8(0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1);

	const __m256i zero = _mm256_setzero_si256();
	const __m256 pi_4 = _mm256_set1_ps(FOURPIRAD);  // 4/pi radians = 45 degrees
	const __m256 pi = _mm256_set1_ps(3.14159f);
	
	const __m256 zero_float = _mm256_setzero_ps();
	const __m256 degrees_45 = _mm256_set1_ps(45.0f);
	const __m256 degrees_180 = _mm256_set1_ps(180.0f);
	const __m256 degrees_360 = _mm256_set1_ps(360.0f);

	// Arctan approx constants
	__m256 a1 = _mm256_set1_ps(0.995354f);
	__m256 a3 = _mm256_set1_ps(-0.288679f);
	__m256 a5 = _mm256_set1_ps(0.079331f);

	__m256 half_pi = _mm256_set1_ps(1.57079632679f);

	__m256 zerof = _mm256_setzero_ps();
	__m256 negative = _mm256_set1_ps(-0.0f);

	
	#pragma omp parallel shared(gradient, edgeDir, filt)
	{

		#pragma omp for private(r, c, row, col) schedule(static) nowait
		for (int r = 1; r < N - 1; r += 32) {
			for (int c = 1; c < M - 1; c += 32) {
				for (int row = r; row < MIN(N - 4, r + 32); row += 4) { // Unroll by 4
					for (int col = c; col < MIN(c + 4, M - 32); col++) {
			
						// ------------ Load Rows ------------

						// For 0
						// Some rows overlap so dont reload
						__m256i row1_0 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col - 1]);
						__m256i row2_0___row1_1 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1]);
						__m256i row3_0___row2_1___row1_2 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col - 1]);

						// For 1 
						__m256i row3_1___row2_2___row1_3 = _mm256_loadu_si256((__m256i*) & filt[row + 2][col - 1]);

						// For 2
						__m256i row3_2___row2_3 = _mm256_loadu_si256((__m256i*) & filt[row + 3][col - 1]);

						// For 3
						__m256i row3_3 = _mm256_loadu_si256((__m256i*) & filt[row + 4][col - 1]);



						// ------------ Calc Gx ------------

						// For 0
						__m256i A_gx_0 = _mm256_maddubs_epi16(row1_0, GxMask13);
						__m256i B_gx_0 = _mm256_maddubs_epi16(row2_0___row1_1, GxMask2);
						__m256i C_gx_0___A_gx_2 = _mm256_maddubs_epi16(row3_0___row2_1___row1_2, GxMask13);

						__m256i hadd1_gx_0 = _mm256_hadd_epi16(A_gx_0, A_gx_0);
						__m256i hadd2_gx_0 = _mm256_hadd_epi16(B_gx_0, B_gx_0);
						__m256i hadd3_gx_0___hadd1_gx_2 = _mm256_hadd_epi16(C_gx_0___A_gx_2, C_gx_0___A_gx_2);

						__m256i Gx_0 = _mm256_add_epi16(hadd1_gx_0, hadd2_gx_0);
						Gx_0 = _mm256_add_epi16(Gx_0, hadd3_gx_0___hadd1_gx_2);


						// For 1
						__m256i A_gx_1 = _mm256_maddubs_epi16(row2_0___row1_1, GxMask13);
						__m256i B_gx_1 = _mm256_maddubs_epi16(row3_0___row2_1___row1_2, GxMask2);
						__m256i C_gx_1___A_gx_3 = _mm256_maddubs_epi16(row3_1___row2_2___row1_3, GxMask13);

						__m256i hadd1_gx_1 = _mm256_hadd_epi16(A_gx_1, A_gx_1);
						__m256i hadd2_gx_1 = _mm256_hadd_epi16(B_gx_1, B_gx_1);
						__m256i hadd3_gx_1___hadd1_gx_3 = _mm256_hadd_epi16(C_gx_1___A_gx_3, C_gx_1___A_gx_3);

						__m256i Gx_1 = _mm256_add_epi16(hadd1_gx_1, hadd2_gx_1);
						Gx_1 = _mm256_add_epi16(Gx_1, hadd3_gx_1___hadd1_gx_3);


						// For 2
						// A_gx_2 The same as C_gx_0 So dont recalculate
						__m256i B_gx_2 = _mm256_maddubs_epi16(row3_1___row2_2___row1_3, GxMask2);
						__m256i C_gx_2 = _mm256_maddubs_epi16(row3_2___row2_3, GxMask13);

						// hadd1_gx_2
						__m256i hadd2_gx_2 = _mm256_hadd_epi16(B_gx_2, B_gx_2);
						__m256i hadd3_gx_2 = _mm256_hadd_epi16(C_gx_2, C_gx_2);

						__m256i Gx_2 = _mm256_add_epi16(hadd3_gx_0___hadd1_gx_2, hadd2_gx_2);
						Gx_2 = _mm256_add_epi16(Gx_2, hadd3_gx_2);


						// For 3
						// A_gx_3
						__m256i B_gx_3 = _mm256_maddubs_epi16(row3_2___row2_3, GxMask2);
						__m256i C_gx_3 = _mm256_maddubs_epi16(row3_3, GxMask13);

						// hadd1_gx_3
						__m256i hadd2_gx_3 = _mm256_hadd_epi16(B_gx_3, B_gx_3);
						__m256i hadd3_gx_3 = _mm256_hadd_epi16(C_gx_3, C_gx_3);

						__m256i Gx_3 = _mm256_add_epi16(hadd3_gx_1___hadd1_gx_3, hadd2_gx_3);
						Gx_3 = _mm256_add_epi16(Gx_3, hadd3_gx_3);


						// ------------ Calc Gy ------------

						// For 0
						__m256i A_gy_0 = _mm256_maddubs_epi16(row1_0, GyMask1);
						__m256i C_gy_0 = _mm256_maddubs_epi16(row3_0___row2_1___row1_2, GyMask3);

						__m256i hadd1_gy_0 = _mm256_hadd_epi16(A_gy_0, A_gy_0);
						__m256i hadd3_gy_0 = _mm256_hadd_epi16(C_gy_0, C_gy_0);

						__m256i Gy_0 = _mm256_add_epi16(hadd1_gy_0, hadd3_gy_0);


						// For 1
						__m256i A_gy_1 = _mm256_maddubs_epi16(row2_0___row1_1, GyMask1);
						__m256i C_gy_1 = _mm256_maddubs_epi16(row3_1___row2_2___row1_3, GyMask3);

						__m256i hadd1_gy_1 = _mm256_hadd_epi16(A_gy_1, A_gy_1);
						__m256i hadd3_gy_1 = _mm256_hadd_epi16(C_gy_1, C_gy_1);

						__m256i Gy_1 = _mm256_add_epi16(hadd1_gy_1, hadd3_gy_1);


						// For 2
						__m256i A_gy_2 = _mm256_maddubs_epi16(row3_0___row2_1___row1_2, GyMask1);
						__m256i C_gy_2 = _mm256_maddubs_epi16(row3_2___row2_3, GyMask3);

						__m256i hadd1_gy_2 = _mm256_hadd_epi16(A_gy_2, A_gy_2);
						__m256i hadd3_gy_2 = _mm256_hadd_epi16(C_gy_2, C_gy_2);

						__m256i Gy_2 = _mm256_add_epi16(hadd1_gy_2, hadd3_gy_2);


						// For 3
						__m256i A_gy_3 = _mm256_maddubs_epi16(row3_1___row2_2___row1_3, GyMask1);
						__m256i C_gy_3 = _mm256_maddubs_epi16(row3_3, GyMask3);

						__m256i hadd1_gy_3 = _mm256_hadd_epi16(A_gy_3, A_gy_3);
						__m256i hadd3_gy_3 = _mm256_hadd_epi16(C_gy_3, C_gy_3);

						__m256i Gy_3 = _mm256_add_epi16(hadd1_gy_3, hadd3_gy_3);



						// ------------ Convert to float ------------
						// For sqrt and atan2

						__m256i sign_mask_0, sign_mask_1, sign_mask_2, sign_mask_3;

						// For 0
						// Gx
						__m256i unpacked_Gx_0 = _mm256_unpacklo_epi16(Gx_0, zero); // Put elements in order seperated by zeros
						sign_mask_0 = _mm256_cmpgt_epi16(zero, unpacked_Gx_0); // Mask for which are -ve (to handle 16bit int extension to 32bit float)
						__m256i extended_Gx_0 = _mm256_or_si256(unpacked_Gx_0, _mm256_slli_epi32(sign_mask_0, 16)); // Extend integers and handle signed elements
						__m256 float_Gx_0 = _mm256_cvtepi32_ps(extended_Gx_0); // Convert to float

						// Gy
						__m256i unpacked_Gy_0 = _mm256_unpacklo_epi16(Gy_0, zero);
						sign_mask_0 = _mm256_cmpgt_epi16(zero, unpacked_Gy_0);
						__m256i extended_Gy_0 = _mm256_or_si256(unpacked_Gy_0, _mm256_slli_epi32(sign_mask_0, 16));
						__m256 float_Gy_0 = _mm256_cvtepi32_ps(extended_Gy_0);


						// For 1
						// Gx
						__m256i unpacked_Gx_1 = _mm256_unpacklo_epi16(Gx_1, zero);
						sign_mask_1 = _mm256_cmpgt_epi16(zero, unpacked_Gx_1);
						__m256i extended_Gx_1 = _mm256_or_si256(unpacked_Gx_1, _mm256_slli_epi32(sign_mask_1, 16));
						__m256 float_Gx_1 = _mm256_cvtepi32_ps(extended_Gx_1);

						// Gy
						__m256i unpacked_Gy_1 = _mm256_unpacklo_epi16(Gy_1, zero);
						sign_mask_1 = _mm256_cmpgt_epi16(zero, unpacked_Gy_1);
						__m256i extended_Gy_1 = _mm256_or_si256(unpacked_Gy_1, _mm256_slli_epi32(sign_mask_1, 16));
						__m256 float_Gy_1 = _mm256_cvtepi32_ps(extended_Gy_1);


						// For 2
						// Gx
						__m256i unpacked_Gx_2 = _mm256_unpacklo_epi16(Gx_2, zero);
						sign_mask_2 = _mm256_cmpgt_epi16(zero, unpacked_Gx_2);
						__m256i extended_Gx_2 = _mm256_or_si256(unpacked_Gx_2, _mm256_slli_epi32(sign_mask_2, 16));
						__m256 float_Gx_2 = _mm256_cvtepi32_ps(extended_Gx_2);

						// Gy
						__m256i unpacked_Gy_2 = _mm256_unpacklo_epi16(Gy_2, zero);
						sign_mask_2 = _mm256_cmpgt_epi16(zero, unpacked_Gy_2);
						__m256i extended_Gy_2 = _mm256_or_si256(unpacked_Gy_2, _mm256_slli_epi32(sign_mask_2, 16));
						__m256 float_Gy_2 = _mm256_cvtepi32_ps(extended_Gy_2);


						// For 3
						// Gx
						__m256i unpacked_Gx_3 = _mm256_unpacklo_epi16(Gx_3, zero);
						sign_mask_3 = _mm256_cmpgt_epi16(zero, unpacked_Gx_3);
						__m256i extended_Gx_3 = _mm256_or_si256(unpacked_Gx_3, _mm256_slli_epi32(sign_mask_3, 16));
						__m256 float_Gx_3 = _mm256_cvtepi32_ps(extended_Gx_3);

						// Gy
						__m256i unpacked_Gy_3 = _mm256_unpacklo_epi16(Gy_3, zero);
						sign_mask_3 = _mm256_cmpgt_epi16(zero, unpacked_Gy_3);
						__m256i extended_Gy_3 = _mm256_or_si256(unpacked_Gy_3, _mm256_slli_epi32(sign_mask_3, 16));
						__m256 float_Gy_3 = _mm256_cvtepi32_ps(extended_Gy_3);




						// ------------ Sqrt Mult ------------ 
						// sqrt(Gx * Gx + Gy * Gy) Square root to maintain precision

						// For 0
						__m256 A_squared_0 = _mm256_mul_ps(float_Gx_0, float_Gx_0);

						__m256 sum_of_squares_0 = _mm256_fmadd_ps(float_Gy_0, float_Gy_0, A_squared_0);

						__m256 result_sqrt_0 = _mm256_sqrt_ps(sum_of_squares_0);


						// For 1
						__m256 A_squared_1 = _mm256_mul_ps(float_Gx_1, float_Gx_1);

						__m256 sum_of_squares_1 = _mm256_fmadd_ps(float_Gy_1, float_Gy_1, A_squared_1);

						__m256 result_sqrt_1 = _mm256_sqrt_ps(sum_of_squares_1);


						// For 2
						__m256 A_squared_2 = _mm256_mul_ps(float_Gx_2, float_Gx_2);

						__m256 sum_of_squares_2 = _mm256_fmadd_ps(float_Gy_2, float_Gy_2, A_squared_2);

						__m256 result_sqrt_2 = _mm256_sqrt_ps(sum_of_squares_2);


						// For 3
						__m256 A_squared_3 = _mm256_mul_ps(float_Gx_3, float_Gx_3);

						__m256 sum_of_squares_3 = _mm256_fmadd_ps(float_Gy_3, float_Gy_3, A_squared_3);

						__m256 result_sqrt_3 = _mm256_sqrt_ps(sum_of_squares_3);


						// ------------ Store gradient ------------

						// For 0
						float result_array_0[8];
						_mm256_storeu_ps(result_array_0, result_sqrt_0);

						gradient[row][col] = (unsigned char)result_array_0[0];
						gradient[row][col + 4] = (unsigned char)result_array_0[1];
						gradient[row][col + 8] = (unsigned char)result_array_0[2];
						gradient[row][col + 12] = (unsigned char)result_array_0[3];
						gradient[row][col + 16] = (unsigned char)result_array_0[4];
						gradient[row][col + 20] = (unsigned char)result_array_0[5];
						gradient[row][col + 24] = (unsigned char)result_array_0[6];
						gradient[row][col + 28] = (unsigned char)result_array_0[7];


						// For 1
						float result_array_1[8];
						_mm256_storeu_ps(result_array_1, result_sqrt_1);

						gradient[row + 1][col] = (unsigned char)result_array_1[0];
						gradient[row + 1][col + 4] = (unsigned char)result_array_1[1];
						gradient[row + 1][col + 8] = (unsigned char)result_array_1[2];
						gradient[row + 1][col + 12] = (unsigned char)result_array_1[3];
						gradient[row + 1][col + 16] = (unsigned char)result_array_1[4];
						gradient[row + 1][col + 20] = (unsigned char)result_array_1[5];
						gradient[row + 1][col + 24] = (unsigned char)result_array_1[6];
						gradient[row + 1][col + 28] = (unsigned char)result_array_1[7];


						// For 2
						float result_array_2[8];
						_mm256_storeu_ps(result_array_2, result_sqrt_2);

						gradient[row + 2][col] = (unsigned char)result_array_2[0];
						gradient[row + 2][col + 4] = (unsigned char)result_array_2[1];
						gradient[row + 2][col + 8] = (unsigned char)result_array_2[2];
						gradient[row + 2][col + 12] = (unsigned char)result_array_2[3];
						gradient[row + 2][col + 16] = (unsigned char)result_array_2[4];
						gradient[row + 2][col + 20] = (unsigned char)result_array_2[5];
						gradient[row + 2][col + 24] = (unsigned char)result_array_2[6];
						gradient[row + 2][col + 28] = (unsigned char)result_array_2[7];


						// For 3
						float result_array_3[8];
						_mm256_storeu_ps(result_array_3, result_sqrt_3);

						gradient[row + 3][col] = (unsigned char)result_array_3[0];
						gradient[row + 3][col + 4] = (unsigned char)result_array_3[1];
						gradient[row + 3][col + 8] = (unsigned char)result_array_3[2];
						gradient[row + 3][col + 12] = (unsigned char)result_array_3[3];
						gradient[row + 3][col + 16] = (unsigned char)result_array_3[4];
						gradient[row + 3][col + 20] = (unsigned char)result_array_3[5];
						gradient[row + 3][col + 24] = (unsigned char)result_array_3[6];
						gradient[row + 3][col + 28] = (unsigned char)result_array_3[7];



						// ------------ Arctan approx ------------

						// For 0
						// Absolute value
						__m256 abs_x_0 = _mm256_andnot_ps(negative, float_Gx_0);
						__m256 abs_y_0 = _mm256_andnot_ps(negative, float_Gy_0);

						// Make it fall between 0-1 for approximation ( |y| / (|x| + |y|) )
						__m256 sum_0 = _mm256_add_ps(abs_x_0, abs_y_0);
						__m256 t_0 = _mm256_div_ps(abs_y_0, sum_0);

						// Polynomial approximation for atan(t) ( poly = a1*t + a3*t^3 + a5*t^5 )
						__m256 t_sq_0 = _mm256_mul_ps(t_0, t_0);
						__m256 poly_0 = _mm256_fmadd_ps(t_sq_0, a5, a3);
						poly_0 = _mm256_fmadd_ps(t_sq_0, poly_0, a1);
						poly_0 = _mm256_mul_ps(t_0, poly_0);

						// Get the quadrant
						__m256 is_neg_x_0 = _mm256_cmp_ps(float_Gx_0, zerof, _CMP_LT_OQ); // x < 0
						__m256 is_neg_y_0 = _mm256_cmp_ps(float_Gy_0, zerof, _CMP_LT_OQ); // y < 0

						// Adjust based on the quadrant
						__m256 angle_0 = _mm256_blendv_ps(poly_0, _mm256_sub_ps(pi, poly_0), is_neg_x_0); // x < 0
						angle_0 = _mm256_blendv_ps(angle_0, _mm256_sub_ps(zerof, angle_0), is_neg_y_0); // y < 0


						// For 1
						// Absolute value
						__m256 abs_x_1 = _mm256_andnot_ps(negative, float_Gx_1);
						__m256 abs_y_1 = _mm256_andnot_ps(negative, float_Gy_1);

						// Make it fall between 0-1 for approximation ( |y| / (|x| + |y|) )
						__m256 sum_1 = _mm256_add_ps(abs_x_1, abs_y_1);
						__m256 t_1 = _mm256_div_ps(abs_y_1, sum_1);

						// Polynomial approximation for atan(t) ( poly = a1*t + a3*t^3 + a5*t^5 )
						__m256 t_sq_1 = _mm256_mul_ps(t_1, t_1);
						__m256 poly_1 = _mm256_fmadd_ps(t_sq_1, a5, a3);
						poly_1 = _mm256_fmadd_ps(t_sq_1, poly_1, a1);
						poly_1 = _mm256_mul_ps(t_1, poly_1);

						// Get the quadrant
						__m256 is_neg_x_1 = _mm256_cmp_ps(float_Gx_1, zerof, _CMP_LT_OQ); // x < 0
						__m256 is_neg_y_1 = _mm256_cmp_ps(float_Gy_1, zerof, _CMP_LT_OQ); // y < 0

						// Adjust based on the quadrant
						__m256 angle_1 = _mm256_blendv_ps(poly_1, _mm256_sub_ps(pi, poly_1), is_neg_x_1); // x < 0
						angle_1 = _mm256_blendv_ps(angle_1, _mm256_sub_ps(zerof, angle_1), is_neg_y_1); // y < 0


						// For 2
						// Absolute value
						__m256 abs_x_2 = _mm256_andnot_ps(negative, float_Gx_2);
						__m256 abs_y_2 = _mm256_andnot_ps(negative, float_Gy_2);

						// Make it fall between 0-1 for approximation ( |y| / (|x| + |y|) )
						__m256 sum_2 = _mm256_add_ps(abs_x_2, abs_y_2);
						__m256 t_2 = _mm256_div_ps(abs_y_2, sum_2);

						// Polynomial approximation for atan(t) ( poly = a1*t + a3*t^3 + a5*t^5 )
						__m256 t_sq_2 = _mm256_mul_ps(t_2, t_2);
						__m256 poly_2 = _mm256_fmadd_ps(t_sq_2, a5, a3);
						poly_2 = _mm256_fmadd_ps(t_sq_2, poly_2, a1);
						poly_2 = _mm256_mul_ps(t_2, poly_2);

						// Get the quadrant
						__m256 is_neg_x_2 = _mm256_cmp_ps(float_Gx_2, zerof, _CMP_LT_OQ); // x < 0
						__m256 is_neg_y_2 = _mm256_cmp_ps(float_Gy_2, zerof, _CMP_LT_OQ); // y < 0

						// Adjust based on the quadrant
						__m256 angle_2 = _mm256_blendv_ps(poly_2, _mm256_sub_ps(pi, poly_2), is_neg_x_2); // x < 0
						angle_2 = _mm256_blendv_ps(angle_2, _mm256_sub_ps(zerof, angle_2), is_neg_y_2); // y < 0


						// For 3
						// Absolute value
						__m256 abs_x_3 = _mm256_andnot_ps(negative, float_Gx_3);
						__m256 abs_y_3 = _mm256_andnot_ps(negative, float_Gy_3);

						// Make it fall between 0-1 for approximation ( |y| / (|x| + |y|) )
						__m256 sum_3 = _mm256_add_ps(abs_x_3, abs_y_3);
						__m256 t_3 = _mm256_div_ps(abs_y_3, sum_3);

						// Polynomial approximation for atan(t) ( poly = a1*t + a3*t^3 + a5*t^5 )
						__m256 t_sq_3 = _mm256_mul_ps(t_3, t_3);
						__m256 poly_3 = _mm256_fmadd_ps(t_sq_3, a5, a3);
						poly_3 = _mm256_fmadd_ps(t_sq_3, poly_3, a1);
						poly_3 = _mm256_mul_ps(t_3, poly_3);

						// Get the quadrant
						__m256 is_neg_x_3 = _mm256_cmp_ps(float_Gx_3, zerof, _CMP_LT_OQ); // x < 0
						__m256 is_neg_y_3 = _mm256_cmp_ps(float_Gy_3, zerof, _CMP_LT_OQ); // y < 0

						// Adjust based on the quadrant
						__m256 angle_3 = _mm256_blendv_ps(poly_3, _mm256_sub_ps(pi, poly_3), is_neg_x_3); // x < 0
						angle_3 = _mm256_blendv_ps(angle_3, _mm256_sub_ps(zerof, angle_3), is_neg_y_3); // y < 0



						// ------------ Edge direction ------------
						// ((atan2(Gx, Gy)) / 3.14159) * 180.0
						// Then round to nearest 45 degrees (shifted until angle within 0-180)

						// For 0
						// Round to nearest 4/pi rad (45 degrees)
						__m256 scaled_angle_0 = _mm256_mul_ps(angle_0, pi_4); // scale by 4/pi rad (45 degrees)

						__m256 rounded_angle_0 = _mm256_round_ps(scaled_angle_0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

						// Convert radians to degrees
						__m256 newAngle_0 = _mm256_mul_ps(rounded_angle_0, degrees_45);

						// Ensure angles are positive
						__m256 is_negative_0 = _mm256_cmp_ps(newAngle_0, zero_float, _CMP_LT_OS); // Mask for negative values
						newAngle_0 = _mm256_add_ps(newAngle_0, _mm256_and_ps(is_negative_0, degrees_360));  // Add 360 degrees to negative angles

						// Angles greater than 180 are pushed back -180 degrees (This is the case when an angle was -ve or is 180)
						__m256 is_greater_than_180_0 = _mm256_cmp_ps(newAngle_0, degrees_180, _CMP_GE_OS);
						newAngle_0 = _mm256_sub_ps(newAngle_0, _mm256_and_ps(is_greater_than_180_0, degrees_180));


						// For 1
						// Round to nearest 4/pi rad (45 degrees)
						__m256 scaled_angle_1 = _mm256_mul_ps(angle_1, pi_4); // scale by 4/pi rad (45 degrees)

						__m256 rounded_angle_1 = _mm256_round_ps(scaled_angle_1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

						// Convert radians to degrees
						__m256 newAngle_1 = _mm256_mul_ps(rounded_angle_1, degrees_45);

						// Ensure angles are positive
						__m256 is_negative_1 = _mm256_cmp_ps(newAngle_1, zero_float, _CMP_LT_OS); // Mask for negative values
						newAngle_1 = _mm256_add_ps(newAngle_1, _mm256_and_ps(is_negative_1, degrees_360));  // Add 360 degrees to negative angles

						// Angles greater than 180 are pushed back -180 degrees (This is the case when an angle was -ve or is 180)
						__m256 is_greater_than_180_1 = _mm256_cmp_ps(newAngle_1, degrees_180, _CMP_GE_OS);
						newAngle_1 = _mm256_sub_ps(newAngle_1, _mm256_and_ps(is_greater_than_180_1, degrees_180));


						// For 2
						// Round to nearest 4/pi rad (45 degrees)
						__m256 scaled_angle_2 = _mm256_mul_ps(angle_2, pi_4); // scale by 4/pi rad (45 degrees)

						__m256 rounded_angle_2 = _mm256_round_ps(scaled_angle_2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

						// Convert radians to degrees
						__m256 newAngle_2 = _mm256_mul_ps(rounded_angle_2, degrees_45);

						// Ensure angles are positive
						__m256 is_negative_2 = _mm256_cmp_ps(newAngle_2, zero_float, _CMP_LT_OS); // Mask for negative values
						newAngle_2 = _mm256_add_ps(newAngle_2, _mm256_and_ps(is_negative_2, degrees_360));  // Add 360 degrees to negative angles

						// Angles greater than 180 are pushed back -180 degrees (This is the case when an angle was -ve or is 180)
						__m256 is_greater_than_180_2 = _mm256_cmp_ps(newAngle_2, degrees_180, _CMP_GE_OS);
						newAngle_2 = _mm256_sub_ps(newAngle_2, _mm256_and_ps(is_greater_than_180_2, degrees_180));


						// For 3
						// Round to nearest 4/pi rad (45 degrees)
						__m256 scaled_angle_3 = _mm256_mul_ps(angle_3, pi_4); // scale by 4/pi rad (45 degrees)

						__m256 rounded_angle_3 = _mm256_round_ps(scaled_angle_3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

						// Convert radians to degrees
						__m256 newAngle_3 = _mm256_mul_ps(rounded_angle_3, degrees_45);

						// Ensure angles are positive
						__m256 is_negative_3 = _mm256_cmp_ps(newAngle_3, zero_float, _CMP_LT_OS); // Mask for negative values
						newAngle_3 = _mm256_add_ps(newAngle_3, _mm256_and_ps(is_negative_3, degrees_360));  // Add 360 degrees to negative angles

						// Angles greater than 180 are pushed back -180 degrees (This is the case when an angle was -ve or is 180)
						__m256 is_greater_than_180_3 = _mm256_cmp_ps(newAngle_3, degrees_180, _CMP_GE_OS);
						newAngle_3 = _mm256_sub_ps(newAngle_3, _mm256_and_ps(is_greater_than_180_3, degrees_180));



						// ------------ Store edgeDir ------------

						// For 0
						float AngleArray_0[8];
						_mm256_storeu_ps(AngleArray_0, newAngle_0);

						edgeDir[row][col] = AngleArray_0[0];
						edgeDir[row][col + 4] = AngleArray_0[1];
						edgeDir[row][col + 8] = AngleArray_0[2];
						edgeDir[row][col + 12] = AngleArray_0[3];
						edgeDir[row][col + 16] = AngleArray_0[4];
						edgeDir[row][col + 20] = AngleArray_0[5];
						edgeDir[row][col + 24] = AngleArray_0[6];
						edgeDir[row][col + 28] = AngleArray_0[7];


						// For 1
						float AngleArray_1[8];
						_mm256_storeu_ps(AngleArray_1, newAngle_1);

						edgeDir[row + 1][col] = AngleArray_1[0];
						edgeDir[row + 1][col + 4] = AngleArray_1[1];
						edgeDir[row + 1][col + 8] = AngleArray_1[2];
						edgeDir[row + 1][col + 12] = AngleArray_1[3];
						edgeDir[row + 1][col + 16] = AngleArray_1[4];
						edgeDir[row + 1][col + 20] = AngleArray_1[5];
						edgeDir[row + 1][col + 24] = AngleArray_1[6];
						edgeDir[row + 1][col + 28] = AngleArray_1[7];


						// For 2
						float AngleArray_2[8];
						_mm256_storeu_ps(AngleArray_2, newAngle_2);

						edgeDir[row + 2][col] = AngleArray_2[0];
						edgeDir[row + 2][col + 4] = AngleArray_2[1];
						edgeDir[row + 2][col + 8] = AngleArray_2[2];
						edgeDir[row + 2][col + 12] = AngleArray_2[3];
						edgeDir[row + 2][col + 16] = AngleArray_2[4];
						edgeDir[row + 2][col + 20] = AngleArray_2[5];
						edgeDir[row + 2][col + 24] = AngleArray_2[6];
						edgeDir[row + 2][col + 28] = AngleArray_2[7];


						// For 3
						float AngleArray_3[8];
						_mm256_storeu_ps(AngleArray_3, newAngle_3);

						edgeDir[row + 3][col] = AngleArray_3[0];
						edgeDir[row + 3][col + 4] = AngleArray_3[1];
						edgeDir[row + 3][col + 8] = AngleArray_3[2];
						edgeDir[row + 3][col + 12] = AngleArray_3[3];
						edgeDir[row + 3][col + 16] = AngleArray_3[4];
						edgeDir[row + 3][col + 20] = AngleArray_3[5];
						edgeDir[row + 3][col + 24] = AngleArray_3[6];
						edgeDir[row + 3][col + 28] = AngleArray_3[7];


					}



				}



			}




		}


		// ------------------------------------ Row Leftovers ------------------------------------

		#pragma omp for private(row, c, col) schedule(static) nowait
		for (int row = ((N - 2) / 4) * 4 + 1; row < N - 1; row++) {
			for (int c = 1; c < M - 1; c += 32) {
				for (int col = c; col < MIN(c + 4, M - 32); col++) {

					// ------------ Load Rows ------------
					__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col - 1]);
					__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1]);
					__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col - 1]);

					// ------------ Calc Gx ------------
					__m256i A_gx = _mm256_maddubs_epi16(row1, GxMask13);
					__m256i B_gx = _mm256_maddubs_epi16(row2, GxMask2);
					__m256i C_gx = _mm256_maddubs_epi16(row3, GxMask13);

					__m256i hadd1_gx = _mm256_hadd_epi16(A_gx, A_gx);
					__m256i hadd2_gx = _mm256_hadd_epi16(B_gx, B_gx);
					__m256i hadd3_gx = _mm256_hadd_epi16(C_gx, C_gx);

					__m256i Gx = _mm256_add_epi16(hadd1_gx, hadd2_gx);
					Gx = _mm256_add_epi16(Gx, hadd3_gx);

					// ------------ Calc Gx ------------
					__m256i A_gy = _mm256_maddubs_epi16(row1, GyMask1);
					__m256i C_gy = _mm256_maddubs_epi16(row3, GyMask3);

					__m256i hadd1_gy = _mm256_hadd_epi16(A_gy, A_gy);
					__m256i hadd3_gy = _mm256_hadd_epi16(C_gy, C_gy);

					__m256i Gy = _mm256_add_epi16(hadd1_gy, hadd3_gy);

					// ------------ Convert to float ------------
					__m256i sign_mask;
					// Gx
					__m256i unpacked_Gx = _mm256_unpacklo_epi16(Gx, zero); // Put elements in order seperated by zeros
					sign_mask = _mm256_cmpgt_epi16(zero, unpacked_Gx); // Mask for which are -ve (to handle 16bit int extension to 32bit float)
					__m256i extended_Gx = _mm256_or_si256(unpacked_Gx, _mm256_slli_epi32(sign_mask, 16)); // Extend integers and handle signed elements
					__m256 float_Gx = _mm256_cvtepi32_ps(extended_Gx); // Convert to float

					// Gy
					__m256i unpacked_Gy = _mm256_unpacklo_epi16(Gy, zero);
					sign_mask = _mm256_cmpgt_epi16(zero, unpacked_Gy);
					__m256i extended_Gy = _mm256_or_si256(unpacked_Gy, _mm256_slli_epi32(sign_mask, 16));
					__m256 float_Gy = _mm256_cvtepi32_ps(extended_Gy);

					// ------------ Sqrt Mult ------------
					__m256 A_squared = _mm256_mul_ps(float_Gx, float_Gx);

					__m256 sum_of_squares = _mm256_fmadd_ps(float_Gy, float_Gy, A_squared);

					__m256 result_sqrt = _mm256_sqrt_ps(sum_of_squares);

					// ------------ Store gradient ------------
					float result_array[8];
					_mm256_storeu_ps(result_array, result_sqrt);


					gradient[row][col] = (unsigned char)result_array[0];
					gradient[row][col + 4] = (unsigned char)result_array[1];
					gradient[row][col + 8] = (unsigned char)result_array[2];
					gradient[row][col + 12] = (unsigned char)result_array[3];
					gradient[row][col + 16] = (unsigned char)result_array[4];
					gradient[row][col + 20] = (unsigned char)result_array[5];
					gradient[row][col + 24] = (unsigned char)result_array[6];
					gradient[row][col + 28] = (unsigned char)result_array[7];

					// ------------ Arctan Approximation ------------
					// Absolute value
					__m256 abs_x = _mm256_andnot_ps(negative, float_Gx);
					__m256 abs_y = _mm256_andnot_ps(negative, float_Gy);

					// Make it fall between 0-1 for approximation ( |y| / (|x| + |y|) )
					__m256 sum = _mm256_add_ps(abs_x, abs_y);
					__m256 t = _mm256_div_ps(abs_y, sum);

					// Polynomial approximation for atan(t) ( poly = a1*t + a3*t^3 + a5*t^5 )
					__m256 t_sq = _mm256_mul_ps(t, t);
					__m256 poly = _mm256_fmadd_ps(t_sq, a5, a3);
					poly = _mm256_fmadd_ps(t_sq, poly, a1);
					poly = _mm256_mul_ps(t, poly);

					// Get the quadrant
					__m256 is_neg_x = _mm256_cmp_ps(float_Gx, zerof, _CMP_LT_OQ); // x < 0
					__m256 is_neg_y = _mm256_cmp_ps(float_Gy, zerof, _CMP_LT_OQ); // y < 0

					// Adjust based on the quadrant
					__m256 angle = _mm256_blendv_ps(poly, _mm256_sub_ps(pi, poly), is_neg_x); // x < 0
					angle = _mm256_blendv_ps(angle, _mm256_sub_ps(zerof, angle), is_neg_y); // y < 0


					// ------------ Edge direction ------------

					// Round to nearest 4/pi rad (45 degrees)
					__m256 scaled_angle = _mm256_mul_ps(angle, pi_4); // scale by 4/pi rad (45 degrees)

					__m256 rounded_angle = _mm256_round_ps(scaled_angle, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

					// Convert radians to degrees
					__m256 newAngle = _mm256_mul_ps(rounded_angle, degrees_45);

					// Ensure angles are positive
					__m256 is_negative = _mm256_cmp_ps(newAngle, zero_float, _CMP_LT_OS); // Mask for negative values
					newAngle = _mm256_add_ps(newAngle, _mm256_and_ps(is_negative, degrees_360));  // Add 360 degrees to negative angles

					// Angles greater than 180 are pushed back -180 degrees (This is the case when an angle was -ve or is 180)
					__m256 is_greater_than_180 = _mm256_cmp_ps(newAngle, degrees_180, _CMP_GE_OS);
					newAngle = _mm256_sub_ps(newAngle, _mm256_and_ps(is_greater_than_180, degrees_180));

					// ------------ Store edgeDir ------------
					float AngleArray[8];
					_mm256_storeu_ps(AngleArray, newAngle);

					edgeDir[row][col] = AngleArray[0];
					edgeDir[row][col + 4] = AngleArray[1];
					edgeDir[row][col + 8] = AngleArray[2];
					edgeDir[row][col + 12] = AngleArray[3];
					edgeDir[row][col + 16] = AngleArray[4];
					edgeDir[row][col + 20] = AngleArray[5];
					edgeDir[row][col + 24] = AngleArray[6];
					edgeDir[row][col + 28] = AngleArray[7];
				}
			}
		}



		// ------------------------------------ Col Leftovers ------------------------------------

		#pragma omp for private(row, col) schedule(static) nowait
		for (int row = 1; row < M - 1; row++) {
			for (int col = ((N - 2) / 32) * 32 + 1; col < N - 1; col++) {

				int Gx = 0;
				int Gy = 0;

				// Gx
				Gx += filt[row - 1][col + 1] - filt[row - 1][col - 1];
				Gx += (filt[row][col + 1] - filt[row][col - 1]) << 1; // Mult by 2 using bit shift
				Gx += filt[row + 1][col + 1] - filt[row + 1][col - 1];

				// Gy
				Gy += (filt[row + 1][col] - filt[row - 1][col]) << 1;
				Gy += filt[row + 1][col + 1] + filt[row + 1][col - 1];
				Gy -= filt[row - 1][col + 1] + filt[row - 1][col - 1];


				gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));


				float thisAngle = atan2(Gx, Gy) * RADTODEGREE;

				int newAngle;

				if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
					newAngle = 0;
				else if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
					newAngle = 45;
				else if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
					newAngle = 90;
				else if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
					newAngle = 135;


				edgeDir[row][col] = newAngle;


			}

		}
	}
}



int image_detection() {


	int i, j;
	unsigned int    row, col;
	int rowOffset;
	int colOffset;
	int Gx;
	int Gy;
	float thisAngle;
	int newAngle;
	int newPixel;

	unsigned char temp;



	/*---------------------- create the image  -----------------------------------*/
	frame1 = (unsigned char**)malloc(N * sizeof(unsigned char *));
	if (frame1 == NULL) { printf("\nerror with malloc fr"); return -1; }
	for (i = 0; i < N; i++) {
		frame1[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
		if (frame1[i] == NULL) { printf("\nerror with malloc fr"); return -1; }
	}


	//create the image
	print = (unsigned char**)malloc(N * sizeof(unsigned char *));
	if (print == NULL) { printf("\nerror with malloc fr"); return -1; }
	for (i = 0; i < N; i++) {
		print[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
		if (print[i] == NULL) { printf("\nerror with malloc fr"); return -1; }
	}

	//initialize the image
	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = 0;

	read_image(IN, frame1);


	GaussianBlur();
	   	  	

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = filt[i][j];

	write_image(OUT_NAME1, print);


	Sobel();



	/* write gradient to image*/

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = gradient[i][j];

	write_image(OUT_NAME2, print);



	for (i = 0; i < N; i++)
		free(frame1[i]);
	free(frame1);



	for (i = 0; i < N; i++)
		free(print[i]);
	free(print);


	return 0;

}




void read_image(char filename[], unsigned char **image)
{
	int inint = -1;
	int c;
	FILE *finput;
	int i, j;

	printf("  Reading image from disk (%s)...\n", filename);
	//finput = NULL;
	openfile(filename, &finput);


	for (j = 0; j < N; j++)
		for (i = 0; i < M; i++) {
			c = getc(finput);


			image[j][i] = (unsigned char)c;
		}



	/* for (j=0; j<N; ++j)
	   for (i=0; i<M; ++i) {
		 if (fscanf(finput, "%i", &inint)==EOF) {
		   fprintf(stderr,"Premature EOF\n");
		   exit(-1);
		 } else {
		   image[j][i]= (unsigned char) inint; //printf("\n%d",inint);
		 }
	   }*/

	fclose(finput);

}





void write_image(char* filename, unsigned char **image)
{
	FILE* foutput;
	errno_t err;
	int i, j;


	printf("  Writing result to disk (%s)...\n", filename);
	if ((err = fopen_s(&foutput, filename, "wb")) != NULL) {
		printf("Unable to open file %s for writing\n", filename);
		exit(-1);
	}

	fprintf(foutput, "P2\n");
	fprintf(foutput, "%d %d\n", M, N);
	fprintf(foutput, "%d\n", 255);

	for (j = 0; j < N; ++j) {
		for (i = 0; i < M; ++i) {
			fprintf(foutput, "%3d ", image[j][i]);
			if (i % 32 == 31) fprintf(foutput, "\n");
		}
		if (M % 32 != 0) fprintf(foutput, "\n");
	}
	fclose(foutput);


}










void openfile(char *filename, FILE** finput)
{
	int x0, y0;
	errno_t err;
	char header[255];
	int aa;

	if ((err = fopen_s(finput, filename, "rb")) != NULL) {
		printf("Unable to open file %s for reading\n");
		exit(-1);
	}

	aa = fscanf_s(*finput, "%s", header, 20);

	/*if (strcmp(header,"P2")!=0) {
	   fprintf(stderr,"\nFile %s is not a valid ascii .pgm file (type P2)\n",
			   filename);
	   exit(-1);
	 }*/

	x0 = getint(*finput);
	y0 = getint(*finput);





	if ((x0 != M) || (y0 != N)) {
		printf("Image dimensions do not match: %ix%i expected\n", N, M);
		exit(-1);
	}

	x0 = getint(*finput); /* read and throw away the range info */


}


int getint(FILE *fp) /* adapted from "xv" source code */
{
	int c, i, firstchar, garbage;

	/* note:  if it sees a '#' character, all characters from there to end of
	   line are appended to the comment string */

	   /* skip forward to start of next number */
	c = getc(fp);
	while (1) {
		/* eat comments */
		if (c == '#') {
			/* if we're at a comment, read to end of line */
			char cmt[256], *sp;

			sp = cmt;  firstchar = 1;
			while (1) {
				c = getc(fp);
				if (firstchar && c == ' ') firstchar = 0;  /* lop off 1 sp after # */
				else {
					if (c == '\n' || c == EOF) break;
					if ((sp - cmt) < 250) *sp++ = c;
				}
			}
			*sp++ = '\n';
			*sp = '\0';
		}

		if (c == EOF) return 0;
		if (c >= '0' && c <= '9') break;   /* we've found what we were looking for */

		/* see if we are getting garbage (non-whitespace) */
		if (c != ' ' && c != '\t' && c != '\r' && c != '\n' && c != ',') garbage = 1;

		c = getc(fp);
	}

	/* we're at the start of a number, continue until we hit a non-number */
	i = 0;
	while (1) {
		i = (i * 10) + (c - '0');
		c = getc(fp);
		if (c == EOF) return i;
		if (c<'0' || c>'9') break;
	}
	return i;
}



