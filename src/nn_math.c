// nn_math.c
#include "nn_math.h"

// Funcao de ativacao sigmoide
float sigmoide(float x)
{
    return 1.0f / (1.0f + expf(-x)); // Usando expf para float
}

// Derivada da sigmoide (simplificada para o output da sigmoide)
float derivada_sigmoide(float x)
{
    return x * (1.0f - x);
}

// Calcula o erro quadratico medio (MSE)
float calcula_erro(struct network *net)
{
    if (net == NULL) {
        // puts("parametro nao pode ser NULL em calcula_erro");
        return 0.0f;
    }       

    float erro = 0.0f;
    int camada_saida_idx = net->camadas - 1;

    for (int i = 0; i < net->layers[camada_saida_idx].num_neurons; i++)
    {   
        struct vertices *v = net->layers[camada_saida_idx].neurons[i];
        float err = (v->saida - v->saida_esperada);     
        erro += err * err;
    }
    // Retorna a soma dos erros quadraticos, idealmente deveria ser dividido por 2
    return erro;
}