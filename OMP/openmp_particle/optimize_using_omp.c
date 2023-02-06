/* Implementation of PSO using OpenMP.
 *
 * Author: Naga Kandasamy
 * Date: February 9, 2022
 *
 */

#define _POSIX_C_SOURCE 2

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "pso.h"
#include <sys/time.h>

/* Solve PSO */
int pso_solve_omp(char *function, swarm_t *swarm, 
                    float xmax, float xmin, int max_iter, int num_threads)
{
    int i, j, iter = 0, g = -1, private_g;
    float w = 0.79, c1 = 1.49, c2 = 1.49;
    float best_fitness, partial_fitness;

    float r1, r2;
    particle_t *particle, *gbest;
    float curr_fitness;

    unsigned int seedp = 2021;

    /*Start parallel region
    * Variables partial_fitness and private_g are private for every thread and once the threads are done
    * the global g and best fitness are updated and shared
    * every thread has a private iteration -> every thread has to go through all iterations
    */
    #pragma omp parallel private(j,i,r1,r2,curr_fitness,particle,gbest,partial_fitness, private_g) firstprivate(iter) shared(g,best_fitness,w,c1,c2,xmax,xmin,max_iter,function) num_threads(num_threads)
    {
        //int tid = omp_get_thread_num();

        while (iter < max_iter) {

            private_g = -1;
            partial_fitness = INFINITY;

            //one thread updates the best_fitness
            #pragma omp single
                best_fitness = INFINITY;

            //#pragma omp barrier

            #pragma omp for schedule(static)
                for (i = 0; i < swarm->num_particles; i++) {

                    particle = &swarm->particle[i];
                    gbest = &swarm->particle[particle->g];  /* Best performing particle from last iteration */ 
                    
                    //printf("TID: %d operates on: %d \n",tid,i);

                    for (j = 0; j < particle->dim; j++) {   /* Update this particle's state */
                        r1 = (float)rand_r(&seedp)/(float)RAND_MAX;
                        r2 = (float)rand_r(&seedp)/(float)RAND_MAX;

                        /* Update particle velocity */
                        particle->v[j] = w * particle->v[j]\
                                        + c1 * r1 * (particle->pbest[j] - particle->x[j])\
                                        + c2 * r2 * (gbest->x[j] - particle->x[j]);
                        /* Clamp velocity */
                        if ((particle->v[j] < -fabsf(xmax - xmin)) || (particle->v[j] > fabsf(xmax - xmin))) 
                            particle->v[j] = uniform(-fabsf(xmax - xmin), fabsf(xmax - xmin));

                        /* Update particle position */
                        particle->x[j] = particle->x[j] + particle->v[j];
                        if (particle->x[j] > xmax)
                            particle->x[j] = xmax;
                        if (particle->x[j] < xmin)
                            particle->x[j] = xmin;
                    } /* State update */
                    
                    /* Evaluate current fitness */
                    pso_eval_fitness(function, particle, &curr_fitness);

                    /* Update pbest */
                    if (curr_fitness < particle->fitness) {
                        particle->fitness = curr_fitness;
                        for (j = 0; j < particle->dim; j++)
                            particle->pbest[j] = particle->x[j];
                    }
                } /* Particle loop */

            /* Identify best performing particle */

            
            //g = pso_get_best_fitness(swarm);
            //Every thread gets its partial best fitness
            #pragma omp for schedule(static)
                for (i = 0; i < swarm->num_particles; i++) {
                    particle = &swarm->particle[i];
                    
                    if (particle->fitness < partial_fitness) {
                        partial_fitness = particle->fitness;
                        private_g = i;
                    }
                }   
            //Every thread compairs its partial fitness to the global one and updates
            #pragma omp critical
            { 
                if(partial_fitness < best_fitness){
                    best_fitness = partial_fitness;
                    g = private_g;
                }
            }

            #pragma omp barrier

            #pragma omp for schedule(static)
                for (i = 0; i < swarm->num_particles; i++) {
                    particle = &swarm->particle[i];
                    particle->g = g;
                }


            iter++;
            
            //Implied barrier by single
            
        } /* End of iteration */
    }
    return g;
}

int optimize_using_omp(char *function, int dim, int swarm_size, 
                       float xmin, float xmax, int num_iter, int num_threads)
{
    /* Initialize PSO */
    swarm_t *swarm;
    srand(time(NULL));

    //struct timeval start, stop;	

    //gettimeofday(&start, NULL);
    swarm = pso_init_omp(function, dim, swarm_size, xmin, xmax, num_threads);
    // gettimeofday(&stop, NULL);

    // fprintf(stderr, "Execution time omp init = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    if (swarm == NULL) {
        fprintf(stderr, "Unable to initialize PSO\n");
        exit(EXIT_FAILURE);
    }

    //Solve PSO
    int g = -1;
    g = pso_solve_omp(function, swarm, xmax, xmin, num_iter, num_threads);
    if (g >= 0) {
        fprintf(stderr, "Solution:\n");
        pso_print_particle(&swarm->particle[g]);
    }

    pso_free(swarm);
    return g;
}
