#include "gauss_eliminate.h" 
#include <pthread.h>
#include <semaphore.h>

typedef struct thread_data_s {
    int tid;                       
    int num_threads;                
    int offset;               
    int chunk_size;
    Matrix *A; //Pointer to the source matrix                                  
} thread_data_t;


/* Structure that defines the barrier */
typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;

barrier_t barrier1;
barrier_t barrier2;
barrier_t barrier3;
barrier_t barrier4;

barrier_t barrier5;
barrier_t barrier6;

void barrier_sync(barrier_t *, int, int);