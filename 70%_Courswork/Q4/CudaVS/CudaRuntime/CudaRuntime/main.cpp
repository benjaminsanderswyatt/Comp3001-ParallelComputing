
#include "canny.h"
void Sobel_Original();  // Remove
void benchmark();   // Remove


int main() {

    benchmark();

	int out, i, j;
	   
	image_detection();

    

	system("pause");
	return 0;
}




#define TIMES 100 // Remove

void benchmark() {  // Remove

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
