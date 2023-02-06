#include "grid.h" 
#include <pthread.h>
#include <semaphore.h>

typedef struct thread_data_s {
    int tid;                       
    int num_threads;
    double num_elements;                
    grid_t *dest_grid; //Store pointer to a source grid             
    grid_t *src_grid;  //Store pointer to a destination grid              
    int offset;                    
    int chunk_size;                 
    double *diff;
    double *num_iter;                       
    pthread_mutex_t *mutex_for_diff;
    pthread_mutex_t *mutex_for_num_iter;
} thread_data_t;


/* Structure that defines the barrier */
typedef struct barrier_s {
    sem_t counter_sem;          /* Protects access to the counter */
    sem_t barrier_sem;          /* Signals that barrier is safe to cross */
    int counter;                /* The value itself */
} barrier_t;

barrier_t barrier1;
barrier_t barrier2;
void barrier_sync(barrier_t *, int, int);