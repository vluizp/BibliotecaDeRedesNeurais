// main.c
#include "nn_graph.h"
#include "nn_train.h"
#include "nn_parallel.h" // Inclui treina_network2
#include <stdio.h>

void exemplo_uso_xor(int num_epocas, float taxa_aprendizagem)
{
    // Conjunto de dados XOR: 2 entradas, 1 saida
    float *data_train[4];
    float *labels[4];
    
    // Entradas
    data_train[0] = (float[]){0.0f, 0.0f};
    data_train[1] = (float[]){0.0f, 1.0f};
    data_train[2] = (float[]){1.0f, 0.0f};
    data_train[3] = (float[]){1.0f, 1.0f};

    // Saidas Esperadas
    labels[0] = (float[]){0.0f};
    labels[1] = (float[]){1.0f};
    labels[2] = (float[]){1.0f};
    labels[3] = (float[]){0.0f};
    
    // Arquitetura: 2 (Entrada) -> 2 (Oculta) -> 1 (Saida)
    int camadas[] = {2, 2, 1, -1};
    
    struct network *net = cria_network(camadas);

    if (net == NULL) {
        printf("Falha ao criar a rede.\n");
        return;
    }
    


    printf("\n--- Teste 1: Treinamento Sequencial (GD Padrao) ---\n");
    // Otimizador Padrao (GD)
    treina_network(net, data_train, labels, 4, taxa_aprendizagem, num_epocas);

    printf("\n--- Teste 2: Treinamento com Otimizador ADAM ---\n");
    // Otimizador Adam
    treina_network_adam(net, data_train, labels, 4, 0.01f, 100, 0.9f, 0.999f);

    printf("\n--- Teste 0: Treinamento Paralelo (Feedforward com Threads) ---\n");
    
    // Re-cria a rede para resetar os pesos, ou destrói e aloca novamente
    destroi_net(net);
    net = cria_network(camadas); // Recria com novos pesos aleatorios
    treina_network2(net, data_train, labels, 4, taxa_aprendizagem, num_epocas);

    destroi_net(net);
}

void prever_xor(struct network *net, float x1, float x2)
{
    float entradas[] = {x1, x2};
    float saidas_dummy[] = {0.0f}; // Apenas para satisfazer a assinatura

    // Apenas Feedforward
    feedfoward(net, entradas, saidas_dummy, 0);

    // Pega a saida final
    float saida_predita = net->layers[net->camadas - 1].neurons[0]->saida;
    printf("Entrada: (%.0f, %.0f) -> Saida Predita: %.4f\n", x1, x2, saida_predita);
}


int main(void)
{
   
    
    exemplo_uso_xor(1000, 0.5f); // 1000 épocas, taxa de 0.5 para GD.

    return 0;
}
