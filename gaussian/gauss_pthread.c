#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gauss_eliminate.h"
#include "thread.h"



/* This function solves the Gauss-Seidel method on the CPU using a single thread. */
void worker_function (void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;

    int i,j,k;
    // printf("THREAD DATA 1, num_rows, TID %f, %d\n", thread_data->A->elements[0], thread_data->tid);

    for(k = 0; k<thread_data->A->num_rows; k++){    //thread_data->A->num_rows


        barrier_sync(&barrier2, thread_data->tid, thread_data->num_threads);

        //Chunk up the row and reduce it
        if(thread_data->tid < (thread_data->num_threads - 1)){
            for (j = (thread_data->offset); j < (thread_data->chunk_size + thread_data->offset); j++) {   /* Reduce the current row. */

                if (thread_data->A->elements[thread_data->A->num_columns * k + k] == 0) {
                    fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                }      

                if( (thread_data->A->num_columns * k + j) != (thread_data->A->num_columns * k + k)){
                    thread_data->A->elements[thread_data->A->num_columns * k + j] = 
                    (float)(thread_data->A->elements[thread_data->A->num_columns * k + j]/ thread_data->A->elements[thread_data->A->num_columns * k + k]);
                }	/* Division step */

                //printf("TID 0 modifies element = %d, %f\n", thread_data->A->num_columns * k + j, thread_data->A->elements[thread_data->A->num_columns * k + k]);
            }
        }else{
            for (j = (thread_data->offset); j < thread_data->A->num_columns; j++) {   /* Reduce the current row. */

                if (thread_data->A->elements[thread_data->A->num_columns * k + k] == 0) {
                    fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                }            
                
                if( (thread_data->A->num_columns * k + j) != (thread_data->A->num_columns * k + k)){
                    thread_data->A->elements[thread_data->A->num_columns * k + j] = 
                    (float)(thread_data->A->elements[thread_data->A->num_columns * k + j]/ thread_data->A->elements[thread_data->A->num_columns * k + k]);
                }	/* Division step */

                //printf("TID 1 modifies element = %d, %f \n", thread_data->A->num_columns * k + j, thread_data->A->elements[thread_data->A->num_columns * k + k]);
            }
        }


        //Barrier
        barrier_sync(&barrier1, thread_data->tid, thread_data->num_threads);
        if(thread_data->tid == 0)
            thread_data->A->elements[thread_data->A->num_columns * k + k] = 1;

        
        // if(thread_data->tid == 0){
        //     printf("Middle Way 1: \n");
        //     for (i = 0; i < thread_data->A->num_rows; i++){
        //         for (j = 0; j < thread_data->A->num_columns; j++){
        //             printf("%10f ", thread_data->A->elements[thread_data->A->num_columns*i +j]);
        //         }
        //         printf("\n");
        //     }
        //     printf("\n");
        // }
        // barrier_sync(&barrier5, thread_data->tid, thread_data->num_threads);
        /* Elimination Step */
        
        if(thread_data->tid < (thread_data->num_threads - 1)){
            //rows in chunk
            for (i = (thread_data->offset + k + 1 ); i < (thread_data->chunk_size+thread_data->offset + k + 1); i++) {

                //Cols in chunk
                    for (j = (k + 1); j < thread_data->A->num_columns; j++){
                        if((thread_data->A->num_columns * i + j) < thread_data->A->num_rows*thread_data->A->num_rows){
                        thread_data->A->elements[thread_data->A->num_columns * i + j] = 
                            thread_data->A->elements[thread_data->A->num_columns * i + j] - 
                            (thread_data->A->elements[thread_data->A->num_columns * i + k] * thread_data->A->elements[thread_data->A->num_columns * k + j]);
                        //printf("CHANGE = %d, %d, %d\n",  thread_data->A->num_columns * i + j, k, thread_data->tid);
                        // printf("A = %d, %d \n",  thread_data->A->num_columns * i + k, thread_data->tid);
                        // printf("B = %d, %d \n",  thread_data->A->num_columns * k + j, thread_data->tid);
                        }
                    }
                if((thread_data->A->num_columns * i + k)< thread_data->A->num_rows*thread_data->A->num_rows){
                    thread_data->A->elements[thread_data->A->num_columns * i + k] = 0;
                }
                //thread_data->A->elements[thread_data->A->num_columns * i + k] = 0;
            }
        }else{

            for (i = (thread_data->offset + k + 1); i < (thread_data->A->num_rows); i++) {

                //Cols in chunk
                for (j = (k + 1); j < thread_data->A->num_columns; j++){
                    thread_data->A->elements[thread_data->A->num_columns * i + j] = 
                        thread_data->A->elements[thread_data->A->num_columns * i + j] - 
                        (thread_data->A->elements[thread_data->A->num_columns * i + k] * thread_data->A->elements[thread_data->A->num_columns * k + j]);
                    //printf("CHANGE = %d, %d, %d\n",  thread_data->A->num_columns * i + j, k, thread_data->tid);
                    // printf("A = %d, %d \n",  thread_data->A->num_columns * i + k, thread_data->tid);
                    // printf("B = %d, %d \n",  thread_data->A->num_columns * k + j, thread_data->tid);
                }
                thread_data->A->elements[thread_data->A->num_columns * i + k] = 0;
            }
        }
        
        //barrier_sync(&barrier6, thread_data->tid, thread_data->num_threads);
    }

    // printf("THREAD DATA 2, num_rows %f, %d \n", thread_data->A->elements[0], thread_data->tid);


    pthread_exit(NULL);
}


/* Barrier synchronization implementation */
void barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
    int i;

    sem_wait(&(barrier->counter_sem));
    /* Check if all threads before us, that is num_threads - 1 threads have reached this point. */	  
    if (barrier->counter == (num_threads - 1)) {

        barrier->counter = 0; /* Reset counter value */
        sem_post(&(barrier->counter_sem)); 	 
        /* Signal blocked threads that it is now safe to cross the barrier */
        //printf("Thread number %d is signalling other threads to proceed\n", tid); 
        for (i = 0; i < (num_threads - 1); i++){
            sem_post(&(barrier->barrier_sem));
        }
    } 
    else { /* There are threads behind us */
        barrier->counter++;
        sem_post(&(barrier->counter_sem));
        sem_wait(&(barrier->barrier_sem)); /* Block on the barrier semaphore */
    }

    return;
}



        // for (i = (k+1); i < thread_data->A->num_rows; i++) {
        //     //barrier_sync(&barrier3, thread_data->tid, thread_data->num_threads);
        //     if(thread_data->tid < (thread_data->num_threads - 1)){

        //         //Go over all cols
        //         for (j = (thread_data->offset + k+1); j < (thread_data->chunk_size+thread_data->offset+k+1); j++){
        //             thread_data->A->elements[thread_data->A->num_columns * i + j] = 
        //                 thread_data->A->elements[thread_data->A->num_columns * i + j] - 
        //                 (thread_data->A->elements[thread_data->A->num_columns * i + k] * thread_data->A->elements[thread_data->A->num_columns * k + j]);
        //             // printf("CHANGE = %d, %d\n",  thread_data->A->num_columns * i + j, thread_data->tid);
        //             // printf("A = %d, %d \n",  thread_data->A->num_columns * i + k, thread_data->tid);
        //             // printf("B = %d, %d \n",  thread_data->A->num_columns * k + j, thread_data->tid);
        //         }	/* Elimination step */
            
        //     }else{
                
        //         //Go over all cols
        //         for (j = (thread_data->offset + k+1); j < thread_data->A->num_columns; j++){
        //             thread_data->A->elements[thread_data->A->num_columns * i + j] = 
        //                 thread_data->A->elements[thread_data->A->num_columns * i + j] - 
        //                 (thread_data->A->elements[thread_data->A->num_columns * i + k] * thread_data->A->elements[thread_data->A->num_columns * k + j]);
        //             // printf("CHANGE = %d, %d \n",  thread_data->A->num_columns * i + j, thread_data->tid);
        //             // printf("A = %d , %d\n",  thread_data->A->num_columns * i + k, thread_data->tid);
        //             // printf("B = %d , %d\n",  thread_data->A->num_columns * k + j, thread_data->tid);
        //         }	/* Elimination step */
                
                
        //     }

        //     //Set the element under the pivot to 0
        //     //barrier_sync(&barrier4, thread_data->tid, thread_data->num_threads);
        //     if(thread_data->tid == 0)
        //         thread_data->A->elements[thread_data->A->num_columns * i + k] = 0;
        // }

