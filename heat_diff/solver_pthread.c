#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "grid.h"
#include "thread.h"



/* This function solves the Gauss-Seidel method on the CPU using a single thread. */
void worker_function (void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;

    int done = 0;
    int i,j;
    double p_diff;
    float old, new;
    double eps = 1e-6;


    int num_elements;


    while(!done){
        num_elements = 0;
        p_diff = 0;


        barrier_sync(&barrier1, thread_data->tid, thread_data->num_threads);

        if(thread_data->tid == 0){
            pthread_mutex_lock(thread_data->mutex_for_diff);
            *(thread_data->diff) = 0;
            pthread_mutex_unlock(thread_data->mutex_for_diff);
        }

        if(thread_data->tid < (thread_data->num_threads - 1)){

            for (i = thread_data->offset; i < (thread_data->offset + thread_data->chunk_size); i++) { // rows
                
                for (j = 1; j < (thread_data->src_grid->dim - 1); j++) { //columns

                    old = thread_data->src_grid->element[i * thread_data->src_grid->dim + j];
                    new = 0.25 * (thread_data->src_grid->element[(i - 1) * thread_data->src_grid->dim + j] +\
                              thread_data->src_grid->element[(i + 1) * thread_data->src_grid->dim + j] +\
                              thread_data->src_grid->element[i * thread_data->src_grid->dim + (j + 1)] +\
                              thread_data->src_grid->element[i * thread_data->src_grid->dim + (j - 1)]);
                     
                    thread_data->dest_grid->element[i * thread_data->src_grid->dim + j] = new;
                    p_diff = p_diff + fabs(new - old);
                    num_elements++;
                }   
            }
        }else{

            for (i = thread_data->offset; i < (thread_data->src_grid->dim - 1); i++) { // rows
                
                for (j = 1; j < (thread_data->offset + thread_data->chunk_size - 2); j++) { //columns
                    old = thread_data->src_grid->element[i * thread_data->src_grid->dim + j];
                    new = 0.25 * (thread_data->src_grid->element[(i - 1) * thread_data->src_grid->dim + j] +\
                              thread_data->src_grid->element[(i + 1) * thread_data->src_grid->dim + j] +\
                              thread_data->src_grid->element[i * thread_data->src_grid->dim + (j + 1)] +\
                              thread_data->src_grid->element[i * thread_data->src_grid->dim + (j - 1)]);
                     
                    thread_data->dest_grid->element[i * thread_data->src_grid->dim + j] = new;
                    p_diff = p_diff + fabs(new - old);
                    num_elements++;
                }
            }
        }


        //printf("p_diff, TID  = %lf, %d \n", p_diff, thread_data->tid);
        pthread_mutex_lock(thread_data->mutex_for_diff);
        *(thread_data->diff) += p_diff/num_elements;
        pthread_mutex_unlock(thread_data->mutex_for_diff);

        

        barrier_sync(&barrier2, thread_data->tid, thread_data->num_threads);

        pthread_mutex_lock(thread_data->mutex_for_num_iter);
        *(thread_data->num_iter) = *(thread_data->num_iter) + 1;
        pthread_mutex_unlock(thread_data->mutex_for_num_iter);

        if(*(thread_data->diff)/thread_data->num_threads < eps){
            //printf("Exiting on DIFF = %lf, %f \n", *(thread_data->diff)/thread_data->num_threads, eps);
            done = 1;
        } 

        //Flip Pointers
        grid_t *temp = thread_data->src_grid;
        thread_data->src_grid = thread_data->dest_grid;
        thread_data->dest_grid = temp;
    }
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
        //lsprintf("Thread number %d is signalling other threads to proceed\n", tid); 
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



