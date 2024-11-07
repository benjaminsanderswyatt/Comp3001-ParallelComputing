/*
------------------DR VASILIOS KELEFOURAS-----------------------------------------------------
------------------COMP3001 ------------------------------------------------------------------
------------------PARALLEL PROGAMMING MODULE-------------------------------------------------
------------------UNIVERSITY OF PLYMOUTH, SCHOOL OF ENGINEERING, COMPUTING AND MATHEMATICS---
*/

#include <Windows.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <stdbool.h> 
#include <stdio.h>
#include <time.h>
//#include <chrono>

#define NUM_THREADS 4
#define N 100

int A[N], B[N], Acopy1[N];

void init();
void un_opt();
void un_opt2();
void un_opt3();
bool compare();

int main() {


	init();

	//define the timers measuring execution time
	clock_t start_1, end_1; //ignore this for  now
	start_1 = clock(); //start the timer (THIS IS NOT A VERY ACCURATE TIMER) - ignore this for now

	//auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only
	//auto start = std::chrono::high_resolution_clock::now(); //ACCURATE timer provided in C++ only

	for (int i = 0; i < 1; i++) {
		// un_opt();
		// un_opt2();
		un_opt3();

	}

	//auto finish = std::chrono::high_resolution_clock::now(); 
	end_1 = clock(); //end the timer - ignore this for now


	printf(" clock() method: %ldms\n", (end_1 - start_1) / (CLOCKS_PER_SEC / 1000));
	//std::chrono::duration<double> elapsed = finish - start;
	//std::cout << "Elapsed time: " << elapsed.count() << " s\n";

	system("pause"); //this command does not let the output window to close
	return 0;


}


void init() {

	int i;

	for (i = 0; i < N; i++) {
		A[i] = rand() % 50;
		B[i] = rand() % 1000;
		Acopy1[i] = A[i];
	}

}

//this is the serial version of the program you need to parallelize
void un_opt() {

	int sum = 0;
#pragma omp parallel 
	{
		int ave = 0;
		int i;

		#pragma omp for
		for (i = 0; i < N; i++) {
			ave += A[i];
		}


		#pragma omp critical
		{
			sum += ave;
		}



	}

	printf("\nave=%d\n", sum);

	
}

void un_opt2() {

	int ave = 0;

#pragma omp parallel
	{
		
		int i;

		#pragma omp for reduction(+:ave)
		for (i = 0; i < N; i++) {
			ave += A[i];
		}



	}
	

	printf("\nave=%d\n", ave);

}

void un_opt3() {

	int ave = 0;
	int i;

	#pragma omp parallel for reduction(+:ave)
	for (i = 0; i < N; i++) {
		ave += A[i];
	}

	printf("\nave=%d\n", ave);

}