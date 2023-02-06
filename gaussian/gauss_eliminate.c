/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: February 25, 2022
 *
 * Student names(s): FIXME
 * Date: FIXME
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -std=c99 -O3 -Wall -lpthread -lm
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c gauss_pthread.c -std=c99 -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"
#include "thread.h" 

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
extern void *worker_function(void *args);

Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix*, int num_threads);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);


int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s matrix-size num-threads \n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        fprintf(stderr, "num-threads: number of threads\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    
    int status = compute_gold(U_reference.elements, A.num_rows);
  
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
    
    //print_matrix(U_reference);

    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");

    //print_matrix(U_mt);
    //printf("\n");
    gettimeofday(&start, NULL);
    gauss_eliminate_using_pthreads(&U_mt, num_threads);
    gettimeofday(&stop, NULL);
    
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    //print_matrix(U_mt);
    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix *U, int num_threads)
{
    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));    /* Data structure to store thread IDs */
    pthread_attr_t attributes;                                                      /* Thread attributes */
    pthread_attr_init(&attributes);  

    //Allocate mmeory for the thread data
    int i;
    thread_data_t *thread_data = (thread_data_t *)malloc (sizeof(thread_data_t) * num_threads);
    int chunk_size = (int)floor((float)U->num_columns/(float)num_threads); 

    /* Initialize the barrier data structures */
    barrier1.counter = 0;
    sem_init(&barrier1.counter_sem, 0, 1); 
    sem_init(&barrier1.barrier_sem, 0, 0); 

    barrier2.counter = 0;
    sem_init(&barrier2.counter_sem, 0, 1); 
    sem_init(&barrier2.barrier_sem, 0, 0); 

    barrier3.counter = 0;
    sem_init(&barrier3.counter_sem, 0, 1); 
    sem_init(&barrier3.barrier_sem, 0, 0); 

    barrier4.counter = 0;
    sem_init(&barrier4.counter_sem, 0, 1); 
    sem_init(&barrier4.barrier_sem, 0, 0); 

    barrier5.counter = 0;
    sem_init(&barrier5.counter_sem, 0, 1); 
    sem_init(&barrier5.barrier_sem, 0, 0); 

    barrier6.counter = 0;
    sem_init(&barrier6.counter_sem, 0, 1); 
    sem_init(&barrier6.barrier_sem, 0, 0); 

    //Initialize Thread Structure
    for (i = 0; i < num_threads; i++) {
        thread_data[i].tid = i; 
        thread_data[i].num_threads = num_threads;
        thread_data[i].offset = i * chunk_size;
        thread_data[i].chunk_size = chunk_size;
        thread_data[i].A = U;
    }

    //Create Threads
    for (i = 0; i < num_threads; i++)
        pthread_create(&thread_id[i], &attributes, worker_function, (void *)&thread_data[i]);
					 
    /* Join point: wait for the workers to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

    /* Free dynamically allocated data structures */
    free((void *)thread_data);
}


/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}

void print_matrix(const Matrix M){

    int i, j;
    for (i = 0; i < M.num_rows; i++){
        for (j = 0; j < M.num_columns; j++){
            printf("%10f ", M.elements[M.num_columns*i +j]);
        }
        printf("\n");
    }
}
