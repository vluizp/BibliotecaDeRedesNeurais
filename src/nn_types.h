// nn_types.h
#ifndef NN_TYPES_H
#define NN_TYPES_H

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <pthread.h>

// --- Estruturas de Dados Básicas ---

struct adjacencia
{
    int vertice;
    float peso;
    float media_movel;      // Usado para RMSProp/Adam (E[g^2])
    float media_movel_adam; // Usado para Adam (m)
    struct adjacencia * prox;
};


struct vertices
{
    struct adjacencia * cabeca; 
    int numero_do_vertice;
    float saida;
    float saida_esperada;
    float delta; // Erro retropropagado (gradiente)
};


struct grafo
{
    int vertices;
    int arestas;
    struct vertices *adj;       
};


struct camadas
{   
    struct vertices **neurons;  // Ponteiros para os vertices no grafo
    int num_neurons;
};


struct network
{
    int camadas;
    struct grafo * neuronios;
    struct camadas * layers;
    int num_neuronios;
};

// --- Estruturas para Paralelismo (nn_parallel.h/c) ---

struct thread_drive_t
{
    int *num_neuron_por_camada; 
    struct network *net;
    int camada_anterior;
    int camada_posterior;
};

// Variaveis globais de controle de threads (serao definidas/externas em nn_parallel.h)
extern struct thread_drive_t global_thread_drive;
// extern queue_t queue, queue_neurons;
extern int pode_comecar, pode_continuar;

#endif // NN_TYPES_H