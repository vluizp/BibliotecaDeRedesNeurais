// nn_parallel.h
#ifndef NN_PARALLEL_H
#define NN_PARALLEL_H

#include "nn_types.h"
#include "nn_graph.h"
#include "nn_math.h"
#include "queue.h" // Dependencia externa

// --- Variaveis Globais (Definidas em nn_parallel.c) ---
extern pthread_mutex_t mutex, mutex2, mutex3, mutex4;
extern pthread_cond_t cond, cond2, cond3, cond4;
extern queue_t queue, queue_neurons; // Filas para Produtor/Consumidor

// --- Funcoes de Gerenciamento ---
void init_global_thread_drive(struct network *net);

// --- Funcoes Produtor/Consumidor ---
void *producer(void *args);
void *consumer(void *arg);

// --- Treinamento Paralelo ---
void treina_network2(struct network *net, float **exemplos, float ** saidas_esperadas,  
                    int numero_de_exemplos, float taxa_de_aprendizagem , int epocas);

#endif // NN_PARALLEL_H
