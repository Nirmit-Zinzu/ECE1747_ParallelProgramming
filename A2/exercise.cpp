#include <stdio.h> // for printf
#include <stdlib.h> // for exit
#include <errno.h> // for perror
#include <math.h> // for sqrt
#include "hrtime.h"


#define STRIDE 1024
#define ITER 1000000
#define NUM_ELEM ((STRIDE)*(ITER))
#define CACHELINE_SIZE 64
#define WORDS_PER_CACHELINE  ((CACHELINE_SIZE)/sizeof(int))


int error(const char * msg)
{
    perror(msg);
    exit(1);
}

// Function that is applied to an element of the array to transform it
// After that, the transformed alement is copied into another element of the array by the caller
// In the simplest form this function just returns the element unmodified.
// Use some math function to give it more weight
int transform(int a)
{
    int n = (int)round(sqrt(a));
    return n*n;
}

void sequentialSolution(int *A)
{
    for (int n = STRIDE; n < NUM_ELEM; ++n) {
        A[n] = transform(A[n - STRIDE]);
    }
}

void wrongSolution(int *A)
{
    // Wrong solution that does not take care of dependencies
    #pragma omp parallel for
    for (int n = STRIDE; n < NUM_ELEM; ++n)
        A[n] = transform(A[n - STRIDE]);
}

void parallelSolution_1(int *A)
{
    for (int i = 1; i < ITER; ++i)
        //  #pragma omp_set_num_threads(8);
        #pragma omp parallel for
        for (int n = i * STRIDE; n < (i+1) * STRIDE; ++n)
            A[n] = transform(A[n - STRIDE]);
}

// Cache locality issue -- worse than sequential
void parallelSolution_2(int *A)
{
    #pragma omp parallel for
    for (int i = 0; i < STRIDE; i++) {
        for (int n = 0; n < ITER-1; n++) {
            A[i + (n+1)*STRIDE] = transform(A[i + n*STRIDE]);
        }
    }
}

// Transformation 
void parallelSolution_3(int *A)
{
    #pragma omp parallel for 
    for (int i = 1; i < ITER; ++i) {
        int idx = i * STRIDE;
        for (int n = 0; n < STRIDE; ++n) {
            A[idx + n] = transform(A[n]);
        }
    }
}

void parallelSolution_4(int *A)
{
    int idx;
    #pragma omp parallel for schedule(dynamic,16) private(idx) num_threads(4) 
    for (int i = 1; i < ITER; ++i) {
        idx = i * STRIDE;
        for (int n = 0; n < STRIDE; ++n) {
            A[idx + n] = transform(A[n]);
        }
    }
}

int main()
{
    double start_time, end_time;
  
    int *A = (int*)malloc(sizeof(int) * NUM_ELEM);
    if (!A)
        error("cannot allocate array");
    // Initiallize array
    #pragma omp parallel for
    for (int n = 0; n < NUM_ELEM; ++n)
        A[n] = n;
    // print values for the last 1024 block of elements
    for (int n = 0; n < STRIDE; ++n)
        printf("%d ", A[n + (ITER-1)*STRIDE]);
    printf("\n");

    start_time = getElapsedTime();

    // sequentialSolution(A);
    
    // parallelSolution_1(A);  // Shwetha's solution
    // parallelSolution_2(A);  // Cache locality issue -- worse than sequential
    // parallelSolution_3(A);  // Good Cache locality  
    parallelSolution_4(A);  // Good Cache locality with dynamic scheduling 

    // call your parallel solution here
    end_time = getElapsedTime();
    
    // verify results for the last 1024 block of elements
    for (int n = 0; n < STRIDE; ++n)
        printf("%d ", A[n + (ITER-1)*STRIDE]);
    printf("\n");

    printf("Code took %lf seconds \n", end_time - start_time);
    
    free(A);
    
    return 0;
}

