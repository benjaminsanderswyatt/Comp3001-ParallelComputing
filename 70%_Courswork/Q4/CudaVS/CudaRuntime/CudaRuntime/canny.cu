#include <omp.h>
#include <cuda_runtime.h>
#include "canny.h"

unsigned char filt[N][M], gradient[N][M], grad2[N][M], edgeDir[N][M];
unsigned char gaussianMask[5][5];
signed char GxMask[3][3], GyMask[3][3];


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


void Sobel_Original() {


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


	/* Declare Sobel masks */
	GxMask[0][0] = -1; GxMask[0][1] = 0; GxMask[0][2] = 1;
	GxMask[1][0] = -2; GxMask[1][1] = 0; GxMask[1][2] = 2;
	GxMask[2][0] = -1; GxMask[2][1] = 0; GxMask[2][2] = 1;

	GyMask[0][0] = -1; GyMask[0][1] = -2; GyMask[0][2] = -1;
	GyMask[1][0] = 0; GyMask[1][1] = 0; GyMask[1][2] = 0;
	GyMask[2][0] = 1; GyMask[2][1] = 2; GyMask[2][2] = 1;

	/*---------------------------- Determine edge directions and gradient strengths -------------------------------------------*/
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			/* Calculate the sum of the Sobel mask times the nine surrounding pixels in the x and y direction */
			for (rowOffset = -1; rowOffset <= 1; rowOffset++) {
				for (colOffset = -1; colOffset <= 1; colOffset++) {

					Gx += filt[row + rowOffset][col + colOffset] * GxMask[rowOffset + 1][colOffset + 1];
					Gy += filt[row + rowOffset][col + colOffset] * GyMask[rowOffset + 1][colOffset + 1];
				}
			}

			gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));

			thisAngle = (((atan2(Gx, Gy)) / 3.14159) * 180.0);

			/* Convert actual edge direction to approximate value */
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

/*
GX
	[0] [1] [2]
[0] -1  0   1
[1] -2  0   2
[2] -1  0   1

GY
	[0] [1] [2]
[0] -1  -2  -1
[1] 0   0   0
[2] 1   2   1
*/

// Use separable filters - the (3x3) can be split into a horizonal (1x3) and a vertical (3x1) - Do horizontal then vertical
// Dont recompute angles - precomute the atan2 values (computing for every pixel adds up)
// Tile the image - memory access patterns
// Approximations - Square root? ( |Gx| + |Gy| ) ~= ( Gx^2 + Gy^2 )^0.5
// Parallelize outer loops row and col - each pixel is independant so go for it (OpenMP?)
// AVX - Multiple pixels at a time
// Declare masks - The masks are static never changing so make them before hand

//#define sqr(Gx, Gy) ((abs(Gx) + abs(Gy)) / 2)
//#define sqrted(Gx, Gy) ((Gx*Gx + Gy*Gy)^0.5)

// pixels are 8 bits ACX gives 32 per instruction




void Shifter() {
	int i, j;
	unsigned int row, col;


	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col += 30) { // += num of elements in vector
			/*
			* vec index:		0  1  2  3  4  5  6  7  8  9 ... 31
			* load row(col: x):	a, b, c, d, e, f, g, h, i, j ...
			*
			* convolute:	|   b, c, d, e, f, g, h, i, j, ? ...  -> ( This row is load(col: x+1) )
			* [-1, 0, 1]	V   -  -  -  -  -  -  -  -  -  -
			*					?  a  b  c  d  e  f  g  h  i	  -> ( This row is -load(col: x-1) )
			*
			* index 0 and index 31 are missed because of shift (x+1) & (x-1) respectivly -> 30 elements
			*/
			
			// Load Row 1
			__m256i row1_shift1 = _mm256_loadu_si256((__m256i*) & filt[row-1][col + 1]);
			__m256i row1_main = _mm256_loadu_si256((__m256i*) & filt[row-1][col]);
			__m256i row1_shiftneg1 = _mm256_loadu_si256((__m256i*) & filt[row-1][col - 1]);

			// Load Row 2
			__m256i row2_shift1 = _mm256_loadu_si256((__m256i*) & filt[row][col + 1]);
			__m256i row2_main = _mm256_loadu_si256((__m256i*) & filt[row][col]);
			__m256i row2_shiftneg1 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1]);

			// Load Row 3
			__m256i row3_shift1 = _mm256_loadu_si256((__m256i*) & filt[row+1][col + 1]);
			__m256i row3_main = _mm256_loadu_si256((__m256i*) & filt[row+1][col]);
			__m256i row3_shiftneg1 = _mm256_loadu_si256((__m256i*) & filt[row+1][col - 1]);

			/*
			// --- Gx ---
			 
			// Row 1 [-1, 0, -1] A
			__m256i Row1_shift1_x_1 = row1_shift1;
			__m256i Row1_shiftneg1_x_neg1 = row1_shiftneg1 * -1;

			__m256i Gx_A = Row1_shift1_x_1 + Row1_shiftneg1_x_neg1;


			// Row 2 [-2, 0, 2] B
			__m256i Row2_shift1_x_2 = row2_shift1 * 2;
			__m256i Row2_shiftneg1_x_ = row2_shiftneg1 * -2;

			__m256i Gx_B = Row2_shift1_x_2 + Row2_shiftneg1_x_;


			// Row 3 [-1, 0, -1] C
			__m256i Row3_shift1_x_1 = row3_shift1;
			__m256i Row3_shiftneg1_x_neg1 = row3_shiftneg1 * -1;

			__m256i Gx_C = Row3_shift1_x_1 + Row3_shiftneg1_x_neg1;


			__m256i Gx = Gx_A + Gx_B + Gx_C;


			// --- Gy ---

			// Row 1 [-1, -2, -1] A
			__m256i Row1_shift1_x_neg1 = row1_shift1 * -1;
			__m256i Row1_main_x_neg2 = row1_main * -2;
			__m256i Row1_shiftneg1_x_neg1 = row1_shiftneg1 * -1;

			__m256i Gy_A = Row1_shift1_x_neg1 + Row1_main_x_neg2 + Row1_shiftneg1_x_neg1;


			// Row 3 [1, 2, 1] C
			__m256i Row3_shift1_x_1 = row3_shift1;
			__m256i Row3_main_x_2 = row3_main * 2;
			__m256i Row3_shiftneg1_x_1 = row3_shiftneg1 * 1;

			__m256i Gy_C = Row3_shift1_x_1 + Row3_main_x_2 + Row3_shiftneg1_x_1;


			__m256i Gy = Gy_A + Gy_C;
			*/



		}
	}
}




void print_m256i_8(__m256i vec) {
	// Create an array to hold the elements of the vector
	int8_t elements[32]; // Adjust size depending on data type
	_mm256_storeu_si256((__m256i*)elements, vec); // Store the vector into the array

	printf("Vector elements: ");
	for (int i = 0; i < 32; i++) { // Adjust loop limit depending on the data type
		printf("%d ", elements[i]); // Use %d for signed integers
	}
	printf("\n");
}

void print_m128i_8(__m128i vec) {
	// Create an array to hold the elements of the vector
	int8_t elements[16]; // Adjust size depending on data type
	_mm_storeu_si128((__m128i*)elements, vec); // Store the vector into the array

	printf("Vector elements: ");
	for (int i = 0; i < 16; i++) { // Adjust loop limit depending on the data type
		printf("%d ", elements[i]); // Use %d for signed integers
	}
	printf("\n");
}


void print_m128i_16(__m128i vec) {
	// Create an array to hold the elements of the vector
	int16_t elements[8]; // Adjust size depending on data type
	_mm_storeu_si128((__m128i*)elements, vec); // Store the vector into the array

	printf("Vector elements: ");
	for (int i = 0; i < 8; i++) { // Adjust loop limit depending on the data type
		printf("%d ", elements[i]); // Use %d for signed integers
	}
	printf("\n");
}

void print_m256i_16(__m256i vec) {
	// Create an array to hold the elements of the vector
	int16_t elements[16]; // Adjust size depending on data type
	_mm256_storeu_si256((__m256i*)elements, vec); // Store the vector into the array

	printf("Vector elements: ");
	for (int i = 0; i < 16; i++) { // Adjust loop limit depending on the data type
		printf("%d ", elements[i]); // Use %d for signed integers
	}
	printf("\n");
}

void print_m256i_32(__m256i vec) {
	// Create an array to hold the elements of the vector
	int32_t elements[8]; // Adjust size depending on data type
	_mm256_storeu_si256((__m256i*)elements, vec); // Store the vector into the array

	printf("Vector elements: ");
	for (int i = 0; i < 8; i++) { // Adjust loop limit depending on the data type
		printf("%d ", elements[i]); // Use %d for signed integers
	}
	printf("\n");
}


int checkRowAt = 1;
int checkColAt = 1;

void print_loop_8(__m256i vec, int row, int col, char* hi) {
	if (row == checkRowAt && col == checkColAt) {
		printf(hi);
		printf("\n");
		print_m256i_8(vec);
		printf("\n");
	}
}

void print_loop_8_128(__m128i vec, int row, int col, char* hi) {
	if (row == checkRowAt && col == checkColAt) {
		printf(hi);
		printf("\n");
		print_m128i_8(vec);
		printf("\n");
	}
}

void print_loop_16_128(__m128i vec, int row, int col, char* hi) {
	if (row == checkRowAt && col == checkColAt) {
		printf(hi);
		printf("\n");
		print_m128i_16(vec);
		printf("\n");
	}
}

void print_loop_16(__m256i vec, int row, int col, char* hi) {
	if (row == checkRowAt && col == checkColAt) {
		printf(hi);
		printf("\n");
		print_m256i_16(vec);
		printf("\n");
	}
}

void print_single(int num, int row, int col, char* hi) {
	if (row == checkRowAt && col == checkColAt) {
		printf(hi);
		printf("\n");
		printf("num: %d",num);
		printf("\n");
	}
}



void Working() {
	int i, j;
	unsigned int row, col;

	int newAngle;

	// [-1, 0, 1]
	// [-2, 0, 2]
	// [-1, 0, 1]
	//printf("\n--- GxMask ---\n");

	__m256i GxMask1 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1);
	__m256i GxMask2 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2);
	__m256i GxMask3 = GxMask1;

	//print_m256i_8(GxMask1);
	//print_m256i_8(GxMask2);
	//print_m256i_8(GxMask3);
	



	// [-1,-2,-1]
	// [ 0, 0, 0]
	// [ 1, 2, 1]
	//printf("\n--- GyMask ---\n");
	
	__m256i GyMask1 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -1);
	__m256i GyMask3 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1);

	//print_m256i_8(GyMask1);
	//print_m256i_8(GyMask3);


	//printf("\n--- Loop ---\n");
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			
			// Load Row 1
			__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col-1]);
			// Load Row 2
			__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col-1]);
			// Load Row 3
			__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col-1]);

			//print_loop_8(row1, row, col, "row1");
			//print_loop_8(row2, row, col, "row2");
			//print_loop_8(row3, row, col, "row3");

			

			// --- Gx ---
			
			// Row 1
			__m256i A_gx = _mm256_maddubs_epi16(row1, GxMask1);
			//print_loop_16(A_gx, row, col, "A_gx");

			__m256i hadd1_gx = _mm256_hadd_epi16(A_gx, A_gx);
			//print_loop_16(hadd1_gx, row, col, "hadd1_gx");

			// Row 2
			__m256i B_gx = _mm256_maddubs_epi16(row2, GxMask2);
			//print_loop_16(B_gx, row, col, "B_gx");

			__m256i hadd2_gx = _mm256_hadd_epi16(B_gx, B_gx);
			//print_loop_16(hadd2_gx, row, col, "hadd2_gx");

			// Row 3
			__m256i C_gx = _mm256_maddubs_epi16(row3, GxMask3);
			//print_loop_16(C_gx, row, col, "C_gx");

			__m256i hadd3_gx = _mm256_hadd_epi16(C_gx, C_gx);
			//print_loop_16(hadd3_gx, row, col, "hadd3_gx");


			// Final
			__m256i Gx = _mm256_add_epi16(hadd1_gx, hadd2_gx);
			//print_loop_16(Gx, row, col, "Gx");

			Gx = _mm256_add_epi16(Gx, hadd3_gx);
			//print_loop_16(Gx, row, col, "Gx");


			// --- Gy ---
			
			// Row 1
			__m256i A_gy = _mm256_maddubs_epi16(row1, GyMask1);
			//print_loop_16(A_gy, row, col, "A_gy");

			__m256i hadd1_gy = _mm256_hadd_epi16(A_gy, A_gy);
			//print_loop_16(hadd1_gy, row, col, "hadd1_gy");

			// Row 3
			__m256i C_gy = _mm256_maddubs_epi16(row3, GyMask3);
			//print_loop_16(C_gy, row, col, "C_gy");

			__m256i hadd3_gy = _mm256_hadd_epi16(C_gy, C_gy);
			//print_loop_16(hadd3_gy, row, col, "hadd3_gy");


			// Final
			__m256i Gy = _mm256_add_epi16(hadd1_gy, hadd3_gy);
			//print_loop_16(Gy, row, col, "Gy");




			int singleGx = (signed short) _mm256_extract_epi16(Gx, 0);
			int singleGy = (signed short) _mm256_extract_epi16(Gy, 0);

			//print_single(singleGx, row, col, "singleGx");
			//print_single(singleGy, row, col, "singleGy");

			
			// Calculate gradient magnitude
			gradient[row][col] = (unsigned char)(sqrt(singleGx * singleGx + singleGy * singleGy));

			//print_single(gradient[row][col], row, col, "gradient[row][col]");


			
			// Calculate edge direction
			float thisAngle = (((atan2(singleGx, singleGy)) / 3.14159) * 180.0);


			// Convert angle to closest direction
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

float atan2_approx(float y, float x) {
	float abs_y = fabs(y) + 1e-10;
	float r, angle;
	if (x >= 0) {
		r = (x - abs_y) / (x + abs_y);
		angle = 0.78539816f;
	}
	else {
		r = (x + abs_y) / (abs_y - x);
		angle = 2.35619449f;
	}
	angle += (0.1963f * r * r - 0.9817f) * r;
	return (y < 0) ? -angle : angle;
}


void Sobel() {
	int i, j;
	int row, col;

	int newAngle;

	// [-1, 0, 1]
	// [-2, 0, 2]
	// [-1, 0, 1]
	printf("\n--- GxMask ---\n");

	__m256i GxMask13 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1);
	__m256i GxMask2 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, -2);

	print_m256i_8(GxMask13);
	print_m256i_8(GxMask2);




	// [-1,-2,-1]
	// [ 0, 0, 0]
	// [ 1, 2, 1]
	printf("\n--- GyMask ---\n");

	__m256i GyMask1 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -2, -1);
	__m256i GyMask3 = _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1);

	print_m256i_8(GyMask1);
	print_m256i_8(GyMask3);


	printf("\n--- Loop ---\n");
	//#pragma omp parallel for
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {


			// Load Row 1
			__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col - 1]);
			// Load Row 2
			__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1]);
			// Load Row 3
			__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col - 1]);

			print_loop_8(row1, row, col, "row1");
			print_loop_8(row2, row, col, "row2");
			print_loop_8(row3, row, col, "row3");



			// --- Gx ---

			// Row 1
			__m256i A_gx = _mm256_maddubs_epi16(row1, GxMask13);
			print_loop_16(A_gx, row, col, "A_gx");

			__m256i hadd1_gx = _mm256_hadd_epi16(A_gx, A_gx);
			print_loop_16(hadd1_gx, row, col, "hadd1_gx");

			// Row 2
			__m256i B_gx = _mm256_maddubs_epi16(row2, GxMask2);
			print_loop_16(B_gx, row, col, "B_gx");

			__m256i hadd2_gx = _mm256_hadd_epi16(B_gx, B_gx);
			print_loop_16(hadd2_gx, row, col, "hadd2_gx");

			// Row 3
			__m256i C_gx = _mm256_maddubs_epi16(row3, GxMask13);
			print_loop_16(C_gx, row, col, "C_gx");

			__m256i hadd3_gx = _mm256_hadd_epi16(C_gx, C_gx);
			print_loop_16(hadd3_gx, row, col, "hadd3_gx");


			// Final
			__m256i Gx = _mm256_add_epi16(hadd1_gx, hadd2_gx);
			print_loop_16(Gx, row, col, "Gx");

			Gx = _mm256_add_epi16(Gx, hadd3_gx);
			print_loop_16(Gx, row, col, "Gx");


			// --- Gy ---

			// Row 1
			__m256i A_gy = _mm256_maddubs_epi16(row1, GyMask1);
			print_loop_16(A_gy, row, col, "A_gy");

			__m256i hadd1_gy = _mm256_hadd_epi16(A_gy, A_gy);
			print_loop_16(hadd1_gy, row, col, "hadd1_gy");

			// Row 3
			__m256i C_gy = _mm256_maddubs_epi16(row3, GyMask3);
			print_loop_16(C_gy, row, col, "C_gy");

			__m256i hadd3_gy = _mm256_hadd_epi16(C_gy, C_gy);
			print_loop_16(hadd3_gy, row, col, "hadd3_gy");


			// Final
			__m256i Gy = _mm256_add_epi16(hadd1_gy, hadd3_gy);
			print_loop_16(Gy, row, col, "Gy");




			int singleGx = (signed short)_mm256_extract_epi16(Gx, 0);
			int singleGy = (signed short)_mm256_extract_epi16(Gy, 0);

			print_single(singleGx, row, col, "singleGx");
			print_single(singleGy, row, col, "singleGy");


			// Calculate gradient magnitude
			gradient[row][col] = (unsigned char)(sqrt(singleGx * singleGx + singleGy * singleGy));

			print_single(gradient[row][col], row, col, "gradient[row][col]");



			// Calculate edge direction
			float thisAngle = (((atan2(singleGx, singleGy)) / 3.14159) * 180.0);


			// Convert angle to closest direction
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




#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define TILE 32


void NewWorking() {
	int i, j;
	int row, col;
	__m128i low, high;

	int newAngle;

	// [-1, 0, 1]
	// [-2, 0, 2]
	// [-1, 0, 1]
	printf("\n--- GxMask ---\n");

	__m256i GxMask13 = _mm256_set_epi8(0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1);
	__m256i GxMask2 = _mm256_set_epi8(0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2);

	
	// [-1,-2,-1]
	// [ 0, 0, 0]
	// [ 1, 2, 1]
	printf("\n--- GyMask ---\n");

	__m256i GyMask1 = _mm256_set_epi8(0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1);
	__m256i GyMask3 = _mm256_set_epi8(0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1);

	int Z = 1024;
	int Y = 1024;
	for (int r = 1; r < Z - 1; r += TILE) {
		for (col = 1; col < Y - 1; col+=32) {
			for (int row = r; row < MIN(Z - 1, r + TILE); row++) {
				for (int inner = 0; inner < 4; inner++) {

					// Each operation calculates for many columns
					// e.g. when inner = 0 & col = 1. columns: 1, 5, 9, 13, 17, 21, 25, 29 are calculated
					// Meaning for each complete inner loop a 32 element row is calculated

					//printf("r: %d, col: %d, row: %d, inner: %d \n", r, col, row, inner);

					// Load Row 1
					__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col - 1 + inner]);
					// Load Row 2
					__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1 + inner]);
					// Load Row 3
					__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col - 1 + inner]);


					// --- Gx ---

					// Row 1
					__m256i A_gx = _mm256_maddubs_epi16(row1, GxMask13);
					//__m256i hadd1_gx = _mm256_hadd_epi16(A_gx, A_gx);
					low = _mm256_extractf128_si256(A_gx, 0);
					high = _mm256_extractf128_si256(A_gx, 1);
					__m128i hadd1_gx = _mm_hadd_epi16(low, high);

					// Row 2
					__m256i B_gx = _mm256_maddubs_epi16(row2, GxMask2);
					//__m256i hadd2_gx = _mm256_hadd_epi16(B_gx, B_gx);
					low = _mm256_extractf128_si256(B_gx, 0);
					high = _mm256_extractf128_si256(B_gx, 1);
					__m128i hadd2_gx = _mm_hadd_epi16(low, high);

					// Row 3
					__m256i C_gx = _mm256_maddubs_epi16(row3, GxMask13);
					//__m256i hadd3_gx = _mm256_hadd_epi16(C_gx, C_gx);
					low = _mm256_extractf128_si256(C_gx, 0);
					high = _mm256_extractf128_si256(C_gx, 1);
					__m128i hadd3_gx = _mm_hadd_epi16(low, high);


					// Final
					__m128i Gx = _mm_add_epi16(hadd1_gx, hadd2_gx);
					Gx = _mm_add_epi16(Gx, hadd3_gx);


					// --- Gy ---

					// Row 1
					__m256i A_gy = _mm256_maddubs_epi16(row1, GyMask1);
					//__m256i hadd1_gy = _mm256_hadd_epi16(A_gy, A_gy);
					low = _mm256_extractf128_si256(A_gy, 0);
					high = _mm256_extractf128_si256(A_gy, 1);
					__m128i hadd1_gy = _mm_hadd_epi16(low, high);

					// Row 3
					__m256i C_gy = _mm256_maddubs_epi16(row3, GyMask3);
					//__m256i hadd3_gy = _mm256_hadd_epi16(C_gy, C_gy);
					low = _mm256_extractf128_si256(C_gy, 0);
					high = _mm256_extractf128_si256(C_gy, 1);
					__m128i hadd3_gy = _mm_hadd_epi16(low, high);
					

					// Final
					__m128i Gy = _mm_add_epi16(hadd1_gy, hadd3_gy);


					signed short GxArray[8];
					_mm_storeu_si128((__m128i*)GxArray, Gx);
					
					signed short GyArray[8];
					_mm_storeu_si128((__m128i*)GyArray, Gy);


					// Calculate gradient magnitude
					gradient[row][col + inner] = (unsigned char)(sqrt(GxArray[0] * GxArray[0] + GyArray[0] * GyArray[0]));
					gradient[row][col + 4 + inner] = (unsigned char)(sqrt(GxArray[1] * GxArray[1] + GyArray[1] * GyArray[1]));
					gradient[row][col + 8 + inner] = (unsigned char)(sqrt(GxArray[2] * GxArray[2] + GyArray[2] * GyArray[2]));
					gradient[row][col + 12 + inner] = (unsigned char)(sqrt(GxArray[3] * GxArray[3] + GyArray[3] * GyArray[3]));
					gradient[row][col + 16 + inner] = (unsigned char)(sqrt(GxArray[4] * GxArray[4] + GyArray[4] * GyArray[4]));
					gradient[row][col + 20 + inner] = (unsigned char)(sqrt(GxArray[5] * GxArray[5] + GyArray[5] * GyArray[5]));
					gradient[row][col + 24 + inner] = (unsigned char)(sqrt(GxArray[6] * GxArray[6] + GyArray[6] * GyArray[6]));
					gradient[row][col + 28 + inner] = (unsigned char)(sqrt(GxArray[7] * GxArray[7] + GyArray[7] * GyArray[7]));
					


					for (int k = 0; k < 8; k++) {
						print_single(k, row, col, "k");

						// Calculate edge direction
						float thisAngle = (((atan2(GxArray[k], GyArray[k])) / 3.14159) * 180.0);


						// Convert angle to closest direction
						if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
							newAngle = 0;
						else if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
							newAngle = 45;
						else if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
							newAngle = 90;
						else if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
							newAngle = 135;

						edgeDir[row][col + inner + k*4] = newAngle;
					}
					

					
				}
			}
		}
	}
}








int roundToNearestMultiple(int number, int multiple) {
	// Calculate the remainder
	int remainder = number % multiple;

	// If the remainder is less than half of the multiple, round down, otherwise round up
	if (remainder < multiple / 2) {
		return number - remainder; // Round down
	}
	else {
		return number + (multiple - remainder); // Round up
	}
}

int roundToClosestDirection(double angle) {
	// Normalize the angle to be between -180 and 180
	if (angle > 180) {
		angle -= 360;
	}
	else if (angle < -180) {
		angle += 360;
	}

	// Round to the closest multiple of 45
	int roundedAngle = static_cast<int>((angle + 22.5) / 45) * 45;

	// Handle edge case where rounding might produce 180, which should be mapped to 0
	if (roundedAngle == 180) {
		roundedAngle = 0;
	}

	return roundedAngle;
}


void Testing() {
	int i, j;
	int row, col;
	__m128i low, high;

	int newAngle;

	// [-1, 0, 1]
	// [-2, 0, 2]
	// [-1, 0, 1]
	printf("\n--- GxMask ---\n");

	__m256i GxMask13 = _mm256_set_epi8(0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1);
	__m256i GxMask2 = _mm256_set_epi8(0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2, 0, 2, 0, -2);


	// [-1,-2,-1]
	// [ 0, 0, 0]
	// [ 1, 2, 1]
	printf("\n--- GyMask ---\n");

	__m256i GyMask1 = _mm256_set_epi8(0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1, 0, -1, -2, -1);
	__m256i GyMask3 = _mm256_set_epi8(0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1);

	int Z = 1024;
	int Y = 1024;
	for (int r = 1; r < Z - 1; r += TILE) {
		for (col = 1; col < Y - 1; col += 32) {
			for (int row = r; row < MIN(Z - 1, r + TILE); row++) {
				for (int inner = 0; inner < 4; inner++) {

					// Each operation calculates for many columns
					// e.g. when inner = 0 & col = 1. columns: 1, 5, 9, 13, 17, 21, 25, 29 are calculated
					// Meaning for each complete inner loop a 32 element row is calculated

					//printf("r: %d, col: %d, row: %d, inner: %d \n", r, col, row, inner);

					// Load Row 1
					__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col - 1 + inner]);
					// Load Row 2
					__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1 + inner]);
					// Load Row 3
					__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col - 1 + inner]);


					// --- Gx ---

					// Row 1
					__m256i A_gx = _mm256_maddubs_epi16(row1, GxMask13);
					//__m256i hadd1_gx = _mm256_hadd_epi16(A_gx, A_gx);
					low = _mm256_extractf128_si256(A_gx, 0);
					high = _mm256_extractf128_si256(A_gx, 1);
					__m128i hadd1_gx = _mm_hadd_epi16(low, high);

					// Row 2
					__m256i B_gx = _mm256_maddubs_epi16(row2, GxMask2);
					//__m256i hadd2_gx = _mm256_hadd_epi16(B_gx, B_gx);
					low = _mm256_extractf128_si256(B_gx, 0);
					high = _mm256_extractf128_si256(B_gx, 1);
					__m128i hadd2_gx = _mm_hadd_epi16(low, high);

					// Row 3
					__m256i C_gx = _mm256_maddubs_epi16(row3, GxMask13);
					//__m256i hadd3_gx = _mm256_hadd_epi16(C_gx, C_gx);
					low = _mm256_extractf128_si256(C_gx, 0);
					high = _mm256_extractf128_si256(C_gx, 1);
					__m128i hadd3_gx = _mm_hadd_epi16(low, high);


					// Final
					__m128i Gx = _mm_add_epi16(hadd1_gx, hadd2_gx);
					Gx = _mm_add_epi16(Gx, hadd3_gx);


					// --- Gy ---

					// Row 1
					__m256i A_gy = _mm256_maddubs_epi16(row1, GyMask1);
					//__m256i hadd1_gy = _mm256_hadd_epi16(A_gy, A_gy);
					low = _mm256_extractf128_si256(A_gy, 0);
					high = _mm256_extractf128_si256(A_gy, 1);
					__m128i hadd1_gy = _mm_hadd_epi16(low, high);

					// Row 3
					__m256i C_gy = _mm256_maddubs_epi16(row3, GyMask3);
					//__m256i hadd3_gy = _mm256_hadd_epi16(C_gy, C_gy);
					low = _mm256_extractf128_si256(C_gy, 0);
					high = _mm256_extractf128_si256(C_gy, 1);
					__m128i hadd3_gy = _mm_hadd_epi16(low, high);


					// Final
					__m128i Gy = _mm_add_epi16(hadd1_gy, hadd3_gy);


					signed short GxArray[8];
					_mm_storeu_si128((__m128i*)GxArray, Gx);

					signed short GyArray[8];
					_mm_storeu_si128((__m128i*)GyArray, Gy);


					// Calculate gradient magnitude
					gradient[row][col + inner] = (unsigned char)(sqrt(GxArray[0] * GxArray[0] + GyArray[0] * GyArray[0]));
					gradient[row][col + 4 + inner] = (unsigned char)(sqrt(GxArray[1] * GxArray[1] + GyArray[1] * GyArray[1]));
					gradient[row][col + 8 + inner] = (unsigned char)(sqrt(GxArray[2] * GxArray[2] + GyArray[2] * GyArray[2]));
					gradient[row][col + 12 + inner] = (unsigned char)(sqrt(GxArray[3] * GxArray[3] + GyArray[3] * GyArray[3]));
					gradient[row][col + 16 + inner] = (unsigned char)(sqrt(GxArray[4] * GxArray[4] + GyArray[4] * GyArray[4]));
					gradient[row][col + 20 + inner] = (unsigned char)(sqrt(GxArray[5] * GxArray[5] + GyArray[5] * GyArray[5]));
					gradient[row][col + 24 + inner] = (unsigned char)(sqrt(GxArray[6] * GxArray[6] + GyArray[6] * GyArray[6]));
					gradient[row][col + 28 + inner] = (unsigned char)(sqrt(GxArray[7] * GxArray[7] + GyArray[7] * GyArray[7]));



					for (int k = 0; k < 8; k++) {
						
						// Calculate edge direction
						float thisAngle = (((atan2(GxArray[k], GyArray[k])) / 3.14159) * 180.0);

						/*
						// Convert angle to closest direction
						if (((thisAngle >= -22.5) && (thisAngle <= 22.5)) || (thisAngle >= 157.5) || (thisAngle <= -157.5))
							newAngle = 0;
						else if (((thisAngle > 22.5) && (thisAngle < 67.5)) || ((thisAngle > -157.5) && (thisAngle < -112.5)))
							newAngle = 45;
						else if (((thisAngle >= 67.5) && (thisAngle <= 112.5)) || ((thisAngle >= -112.5) && (thisAngle <= -67.5)))
							newAngle = 90;
						else if (((thisAngle > 112.5) && (thisAngle < 157.5)) || ((thisAngle > -67.5) && (thisAngle < -22.5)))
							newAngle = 135;
						*/

						
						
						// Convert angle to closest direction
						//newAngle = roundToNearestMultiple(thisAngle, 45);



						int newAngle = roundToClosestDirection(thisAngle);







						edgeDir[row][col + inner + k * 4] = newAngle;
						
					}
				}
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


	//Sobel();
	//NewWorking();
	Testing();

	//Sobel_Original();

	for (i = 0; i < N; i++)
		for (j = 0; j < M; j++)
			print[i][j] = edgeDir[i][j];

	write_image("EdgeDir", print);


	

	//Shifter();

	//New();


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



