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

void Sobel_seperable() {


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

	/*
	static const int GxMask[3][3] = {
	{-1, 0, 1},
	{-2, 0, 2},
	{-1, 0, 1}
	};

	static const int GyMask[3][3] = {
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1}
	};
	*/

	/*
	Gx += filt[row - 1][col - 1] * GxMask[0][0];
	Gx += filt[row - 1][col    ] * GxMask[0][1];
	Gx += filt[row - 1][col + 1] * GxMask[0][2];

	Gx += filt[row][col - 1] * GxMask[1][0];
	Gx += filt[row][col] * GxMask[1][1];
	Gx += filt[row][col + 1] * GxMask[1][2];

	Gx += filt[row + 1][col - 1] * GxMask[2][0];
	Gx += filt[row + 1][col] * GxMask[2][1];
	Gx += filt[row + 1][col + 1] * GxMask[2][2];

	Gy += filt[row - 1][col - 1] * GyMask[0][0];
	Gy += filt[row - 1][col] * GyMask[0][1];
	Gy += filt[row - 1][col + 1] * GyMask[0][2];

	Gy += filt[row][col - 1] * GyMask[1][0];
	Gy += filt[row][col] * GyMask[1][1];
	Gy += filt[row][col + 1] * GyMask[1][2];

	Gy += filt[row + 1][col - 1] * GyMask[2][0];
	Gy += filt[row + 1][col] * GyMask[2][1];
	Gy += filt[row + 1][col + 1] * GyMask[2][2];
	*/

	//__m256i GxMask = _mm256_set_epi8(-1, 0, 1, -2, 0, 2, -1, 0, 1, -1, 0, 1, -2, 0, 2, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	//__m256i GyMask = _mm256_set_epi8(-1, -2, -1, 0, 0, 0, 1, 2, 1, -1, -2, -1, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);







	// ARE M AND N ROW / COLUMNS THE CORRECT WAY AROUND???????????????????????







	// Declare separable Sobel masks
	static const int GxRow[3] = { -1, 0, 1 };  // Horizontal filter for Gx
	static const int GxCol[3] = { 1, 2, 1 }; // Vertical filter for Gx

	static const int GyRow[3] = { 1, 2, 1 }; // Horizontal filter for Gy
	static const int GyCol[3] = { -1, 0, 1 };  // Vertical filter for Gy

	// Dynamically allocate memory for GxTemp and GyTemp
	int* GxTemp = (int*)malloc(N * M * sizeof(int)); // [row][col] -> [row * M + col]
	int* GyTemp = (int*)malloc(N * M * sizeof(int));

	// Rows init 0
	for (int i = 0; i < M; i++) {
		GxTemp[i] = 0;
		GyTemp[i] = 0;

		GxTemp[M * (N - 1) + i] = 0;
		GyTemp[M * (N - 1) + i] = 0;
	}

	// Columns init 0
	for (int i = 1; i < N - 1; i++) { // Corners have already been done
		GxTemp[i] = 0;
		GyTemp[i] = 0;

		GxTemp[M * i + M - 1] = 0;
		GyTemp[M * i + M - 1] = 0;
	}



	//---------------------------- Row convolution -------------------------------------------
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			// Gx { -1, 0, 1 }
			// Gy { 1, 2, 1 }

			Gx = 0;
			Gy = 0;
			/*
			// Horizontal filter
			Gx += filt[row - 1][col] * GxRow[0];
			Gy += filt[row - 1][col] * GyRow[0];

			// Gx += filt[row][col] * GxRow[1]; // Multiplying by 0
			Gy += filt[row][col] * GyRow[1];

			Gx += filt[row + 1][col] * GxRow[2];
			Gy += filt[row + 1][col] * GyRow[2];
			*/


			// -------------- TODO * -1 and * 2 can be bitwise operations AVX?
			Gx += filt[row - 1][col] * -1;
			Gy += filt[row - 1][col];

			// Gx += filt[row][col] * GxRow[1]; // Multiplying by 0
			Gy += filt[row][col] * 2;

			Gx += filt[row + 1][col];
			Gy += filt[row + 1][col];



			GxTemp[row * M + col] = Gx;
			GyTemp[row * M + col] = Gy;

		}
	}

	//---------------------------- Column convolution + Angles -------------------------------------------
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			// Gx { 1, 2, 1 }
			// Gy { -1, 0, 1 }

			Gx = 0;
			Gy = 0;
			/*
			// Vertical filter
			Gx += GxTemp[row][col - 1] * GxCol[0];
			Gy += GyTemp[row][col - 1] * GyCol[0];

			Gx += GxTemp[row][col] * GxCol[1];
			// Gy += GyTemp[row][col] * GyCol[1]; // Multiplying by 0

			Gx += GxTemp[row][col + 1] * GxCol[2];
			Gy += GyTemp[row][col + 1] * GyCol[2];
			*/

			Gx += GxTemp[row * M + col - 1];
			Gy += GyTemp[row * M + col - 1] * -1;

			Gx += GxTemp[row * M + col] * 2;
			// Gy += GyTemp[row][col] * GyCol[1]; // Multiplying by 0

			Gx += GxTemp[row * M + col + 1];
			Gy += GyTemp[row * M + col + 1];



			// Calculate gradient magnitude
			gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));

			// Calculate edge direction
			thisAngle = (((atan2(Gx, Gy)) / 3.14159) * 180.0);


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


	free(GxTemp);
	free(GyTemp);



	/*
	//---------------------------- Determine edge directions and gradient strengths -------------------------------------------
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col++) {

			Gx = 0;
			Gy = 0;

			// Apply horizontal Sobel filter (row-wise convolution)
			for (int offset = -1; offset <= 1; offset++) {
				Gx += filt[row][col + offset] * GxMask[offset + 1];
				Gy += filt[row + offset][col] * GyMask[offset + 1];
			}

			gradient[row][col] = (unsigned char)(sqrt(Gx * Gx + Gy * Gy));

			thisAngle = (((atan2(Gx, Gy)) / 3.14159) * 180.0);

			// Convert actual edge direction to approximate value
			if (thisAngle < 0)
				thisAngle += 180.0;

			if (thisAngle <= 22.5 || thisAngle > 157.5)
				newAngle = 0;
			else if (thisAngle <= 67.5)
				newAngle = 45;
			else if (thisAngle <= 112.5)
				newAngle = 90;
			else
				newAngle = 135;



			edgeDir[row][col] = newAngle;
		}
	}
	*/
}


void Sobel() {
	unsigned int    row, col;




	
	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col+=16) { // += num of elements in vector

			//load 3 rows from image
			__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col]);
			__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col]);
			__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col]);


			//GX
			
			// Horizontal kernal [-1,0,1]
			__m256i hk_neg1 = _mm256_set1_epi8(-1); // can be static (move outside)
			__m256i hk_0 = _mm256_set1_epi8(0);
			__m256i hk_1 = _mm256_set1_epi8(1);


			__m256i result_row1 = _mm256_maddubs_epi16(row1, hk_neg1); // returns 16 bit signed int
			__m256i result_row2 = _mm256_maddubs_epi16(row2, hk_0);
			__m256i result_row3 = _mm256_maddubs_epi16(row3, hk_1);

			__m256i horizontal_sum_x = _mm256_add_epi16(result_row1, result_row2);
			horizontal_sum_x = _mm256_add_epi16(horizontal_sum_x, result_row3); // 16 bit signed

			// Vertical kernal [1,2,1]'
			__m256i vk_1 = _mm256_set1_epi8(1); // can be static (move outside)
			__m256i vk_2 = _mm256_set1_epi8(2);

			__m256i result_row1_x = _mm256_maddubs_epi16(result_row1, vk_1);
			__m256i result_row2_x = _mm256_maddubs_epi16(result_row2, vk_2);
			__m256i result_row3_x = _mm256_maddubs_epi16(result_row3, vk_1);

			__m256i vertical_sum_x = _mm256_add_epi16(result_row1_x, result_row2_x);
			vertical_sum_x = _mm256_add_epi16(vertical_sum_x, result_row3_x);



			//GY
			
			// Horizontal kernal [1,2,1]
			__m256i y_hk_1 = _mm256_set1_epi8(1);
			__m256i y_hk_2 = _mm256_set1_epi8(2);

			__m256i result_row1_y = _mm256_maddubs_epi16(row1, y_hk_1);
			__m256i result_row2_y = _mm256_maddubs_epi16(row2, y_hk_2);
			__m256i result_row3_y = _mm256_maddubs_epi16(row3, y_hk_1);

			__m256i horizontal_sum_y = _mm256_add_epi16(result_row1_y, result_row2_y);
			horizontal_sum_y = _mm256_add_epi16(horizontal_sum_y, result_row3_y);

			// Vertical kernal [-1,0,1]'
			__m256i y_vk_neg1 = _mm256_set1_epi16(-1);
			__m256i y_vk_0 = _mm256_set1_epi16(0);
			__m256i y_vk_1 = _mm256_set1_epi16(1);

			__m256i vertical_sum_y = _mm256_add_epi16(_mm256_maddubs_epi16(result_row1_y, y_vk_neg1), _mm256_maddubs_epi16(result_row2_y, y_vk_0));

			vertical_sum_y = _mm256_add_epi16(vertical_sum_y, _mm256_maddubs_epi16(result_row3_y, y_vk_1));




			
			// Compute Approximate Gradient Magnitude (L1 Norm)
			// Square Gx and Gy
			__m256i gx_squared = _mm256_mullo_epi16(vertical_sum_x, vertical_sum_x);
			__m256i gy_squared = _mm256_mullo_epi16(vertical_sum_y, vertical_sum_y);

			// Sum Gx^2 + Gy^2
			__m256i sum_squares = _mm256_add_epi16(gx_squared, gy_squared);

			// Convert to float for sqrt calculation
			__m256 ps_squares = _mm256_cvtepi32_ps(sum_squares);

			// Compute sqrt
			__m256 magnitude = _mm256_sqrt_ps(ps_squares);

			// Convert to unsigned char and store
			__m256i gradient_result = _mm256_cvtps_epi32(magnitude);
			_mm256_storeu_si256((__m256i*)&gradient[row][col], gradient_result);
			
		}
	}





}

void Sobel_Optimized() {
	int i, j;
	unsigned int row, col;
	int Gx, Gy;

	/*
	// AVX registers for masks
	__m256i GxMask = _mm256_set_epi8(
		-1, 0, 1, -2, 0, 2, -1, 0, 1,
		-1, 0, 1, -2, 0, 2, -1, 0, 1,
		-1, 0, 1, -2, 0, 2, -1, 0, 1,
		-1, 0, 1, -2, 0, 2);

	__m256i GyMask = _mm256_set_epi8(
		-1, -2, -1, 0, 0, 0, 1, 2, 1,
		-1, -2, -1, 0, 0, 0, 1, 2, 1,
		-1, -2, -1, 0, 0, 0, 1, 2, 1,
		-1, -2, -1, 0, 0, 0, 1, 2);


	for (row = 1; row < N - 1; row++) {
		for (col = 1; col < M - 1; col += 16) { // Process 16 pixels at a time

			// Load 3 rows of 16 pixels each
			__m256i row1 = _mm256_loadu_si256((__m256i*) & filt[row - 1][col - 1]);
			__m256i row2 = _mm256_loadu_si256((__m256i*) & filt[row][col - 1]);
			__m256i row3 = _mm256_loadu_si256((__m256i*) & filt[row + 1][col - 1]);

			// Compute Gx and Gy
			__m256i Gx = _mm256_setzero_si256();
			__m256i Gy = _mm256_setzero_si256();

			Gx = _mm256_add_epi16(Gx, _mm256_maddubs_epi16(row1, GxMask));
			Gx = _mm256_add_epi16(Gx, _mm256_maddubs_epi16(row2, GxMask));
			Gx = _mm256_add_epi16(Gx, _mm256_maddubs_epi16(row3, GxMask));

			Gy = _mm256_add_epi16(Gy, _mm256_maddubs_epi16(row1, GyMask));
			Gy = _mm256_add_epi16(Gy, _mm256_maddubs_epi16(row2, GyMask));
			Gy = _mm256_add_epi16(Gy, _mm256_maddubs_epi16(row3, GyMask));

			// Compute gradient magnitude
			__m256i gradient = _mm256_hadd_epi16(_mm256_mullo_epi16(Gx, Gx), _mm256_mullo_epi16(Gy, Gy));

			// Store results back
			_mm256_storeu_si256((__m256i*) & gradient[row][col], gradient);

			// Optional: approximate direction calculations for edgeDir
			// Skip if only gradient magnitude is needed.
		}
	}

	*/
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
	//Sobel_seperable();
	//Sobel_Original();
	//Sobel_Optimized();


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



