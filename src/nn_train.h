// nn_train.h
#ifndef NN_TRAIN_H
#define NN_TRAIN_H

#include "nn_types.h"
#include "nn_graph.h"
#include "nn_math.h"

// Estrutura auxiliar para saidas (nao essencial, mas util para retornar)
struct saidas
{
    float *saidas_camada_oculta;
    float *saidas_camada_saida;     
    int num_saidas_oculta;
    int num_saidas_saida;
};

// Funcoes Principais de Propagacao
struct saidas *feedfoward(struct network * net, float * exemplos, float * saidas, int print_saida);
void back_propagation(struct network *net);

// Otimizadores (Atualizacao de Pesos)
void atualiza_pesos(struct network *net, float taxa_de_aprendizagem); // GD Básico
void atualiza_pesos_rmsprop(struct network *net, float taxa_de_aprendizagem, float beta);
void atualiza_pesos_adam(struct network *net, float taxa_de_aprendizagem, float beta, float beta2);
void atualiza_pesos_regularizacao_l2(struct network *net, float taxa_de_aprendizagem, float lambda);

// Funcoes de Treinamento
void treina_network(struct network *net, float **exemplos, float ** saidas_esperadas,  
                    int numero_de_exemplos, float taxa_de_aprendizagem , int epocas);
void treina_network_rmsprop(struct network *net, float **exemplos, float **saidas_esperadas,
                            int numero_de_exemplos, float taxa_de_aprendizagem, int epocas, float beta);
void treina_network_adam(struct network *net, float **exemplos, float **saidas_esperadas,
                            int numero_de_exemplos, float taxa_de_aprendizagem, int epocas, float beta, float beta2);

#endif // NN_TRAIN_H