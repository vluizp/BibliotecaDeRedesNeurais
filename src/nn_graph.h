// nn_graph.h
#ifndef NN_GRAPH_H
#define NN_GRAPH_H

#include "nn_types.h"

// Funcoes de Gerenciamento de Grafo/Aresta
struct adjacencia * cria_adj(int vertice, float peso);
void cria_aresta(struct grafo * grafo, int vertice_inicial, int vertice_final, float peso);
struct grafo * cria_grafo(int num_vertices);
float busca_na_lista(int vertice, struct vertices *neuron);

// Funcoes de Gerenciamento da Rede
struct network * cria_network(int *camadas);
void destroi_net(struct network *net);
int set_entradas_e_saidas(struct network *net, float * exemplos, float * saidas);

// Utilitario
void imprime_grafo(struct grafo * grafo);

#endif // NN_GRAPH_H