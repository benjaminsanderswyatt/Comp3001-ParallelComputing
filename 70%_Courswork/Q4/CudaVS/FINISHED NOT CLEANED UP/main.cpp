
#include "canny.h"

void Sobel();
void Sobel_Original();
void Sobel_Test();
void benchmark();   // Remove
void compare();   // Remove


int main() {

	int out, i, j;
	   

	image_detection();


    
    //compare();   // Remove
    benchmark(); // Remove

	system("pause");
	return 0;
}




#define TIMES 100 // Remove

void benchmark() {  // Remove
    printf("\nComparing speed to the orignal sobel\n");

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TIMES; i++) {
        Sobel_Original();
    }
    
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "\nOriginal Elapsed time: " << elapsed.count() << " s\n";





    auto start_1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < TIMES; i++) {
        Sobel();
    }
    

    auto finish_1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_1 = finish_1 - start_1;
    std::cout << "\nOptimised Elapsed time: " << elapsed_1.count() << " s\n\n";
}









bool different(unsigned char **first_image, unsigned char **second_image) {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            if (first_image[i][j] != second_image[i][j]) {
                printf("\n Different at i: %d, j: %d\n", i, j);
                return true;
            }
        }
    }
    return false;
}




void compare() {
    printf("\nComparing result to the correct image\n");

    int i, j;
    unsigned char **out2_image, **correct_image;

    // Allocate memory for images
    out2_image = (unsigned char**)malloc(N * sizeof(unsigned char*));
    correct_image = (unsigned char**)malloc(N * sizeof(unsigned char*));
    for (i = 0; i < N; i++) {
        out2_image[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
        correct_image[i] = (unsigned char*)malloc(M * sizeof(unsigned char));
    }


    // Read the two edgeDir
    read_image(OUT_NAME2, out2_image);
    read_image(CORRECT, correct_image);


    // Are the images different
    if (!different(out2_image, correct_image)) {
        printf("\nCORRECT Sobel\n\n");
    }
    else {
        printf("\nINCORRECT Soble\n\n");
    }



    // Read the two images
    read_image(EDGE_NAME_TEST, out2_image);
    read_image(EDGE_NAME_CORRECT, correct_image); // Correct file


    // Are the images different
    if (!different(out2_image, correct_image)) {
        printf("\nCORRECT Sobel edgeDir\n\n");
    }
    else {
        printf("\nINCORRECT Soble edgeDir\n\n");
    }






    // Free allocated memory
    for (i = 0; i < N; i++) {
        free(out2_image[i]);
        free(correct_image[i]);
    }
    free(out2_image);
    free(correct_image);
}
