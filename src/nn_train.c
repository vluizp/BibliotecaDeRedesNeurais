#include "nn_train.h"


void treina_network(struct network *net, float **exemplos, float ** saidas_esperadas, 
					int numero_de_exemplos, float taxa_de_aprendizagem , int epocas)
{
	if (net == NULL || exemplos == NULL || saidas_esperadas == NULL)
	{	puts("parametros em treina_network nao pode ser NULL");
		return;
	}
	
	float erro_total = 0.0;
	

	for (int i = 0; i < epocas; i++)
	{	erro_total = 0.0;

		for (int j = 0; j < numero_de_exemplos; j++)
		{	feedfoward(net,exemplos[j], saidas_esperadas[j], 0);			

			float erro = calcula_erro(net);
			erro_total+= erro;			
			back_propagation(net);							
			atualiza_pesos(net, taxa_de_aprendizagem);
		}
		
	}	
	printf("epoca =  %i, erro_medio = %f\n",epocas, erro_total / numero_de_exemplos);	
	return;	
}


void atualiza_pesos_adam(struct network *net, float taxa_de_aprendizagem, float beta,float beta2)
{
    if (net == NULL)
    {
        puts("parametro -> 'net' nao pode ser NULL em atualiza_pesos_rmsprop");
        return;
    }

    for (int i = net->camadas - 2; i >= 0; i--)
    {
        for (int j = 0; j < net->layers[i].num_neurons; j++)
        {
            struct vertices *v = net->layers[i].neurons[j];
            struct adjacencia *adj = v->cabeca;

            while (adj != NULL)
            {
                float gradiente = net->neuronios->adj[adj->vertice].delta;
                float square_gradiente = gradiente * gradiente;

                // Atualização do E[g^2]
				adj->media_movel_adam = adj->media_movel_adam * beta + (1- beta)* gradiente;

                adj->media_movel = beta2 * adj->media_movel + (1 - beta2) * square_gradiente;

                float delta = (taxa_de_aprendizagem /sqrt(adj->media_movel)) * adj->media_movel_adam;
                adj->peso += delta;

                adj = adj->prox;
            }
        }
    }
    return;
}



void atualiza_pesos_rmsprop(struct network *net, float taxa_de_aprendizagem, float beta)
{
    if (net == NULL)
    {
        puts("parametro -> 'net' nao pode ser NULL em atualiza_pesos_rmsprop");
        return;
    }

    for (int i = net->camadas - 2; i >= 0; i--)
    {
        for (int j = 0; j < net->layers[i].num_neurons; j++)
        {
            struct vertices *v = net->layers[i].neurons[j];
            struct adjacencia *adj = v->cabeca;

            while (adj != NULL)
            {
                float gradiente = net->neuronios->adj[adj->vertice].delta;
                float square_gradiente = gradiente * gradiente;

                // Atualização do E[g^2]
                adj->media_movel = beta * adj->media_movel + (1 - beta) * square_gradiente;

                // Atualização do peso usando RMSprop
                float delta = taxa_de_aprendizagem * gradiente /sqrt(adj->media_movel);
                adj->peso += delta;
                adj = adj->prox;
            }
        }
    }
    return;
}

void atualiza_pesos_regularizacao_l2(struct network *net, float taxa_de_aprendizagem, float lambda)
{
    if (net == NULL)
    {
        puts("parametro -> 'net' nao pode ser NULL em atualiza_pesos_regularizacao_l2");
        return;
    }

    for (int i = net->camadas - 2; i >= 0; i--)
    {
        for (int j = 0; j < net->layers[i].num_neurons; j++)
        {
            struct vertices *v = net->layers[i].neurons[j];
            struct adjacencia *adj = v->cabeca;

            while (adj != NULL)
            {
                float gradiente = net->neuronios->adj[adj->vertice].delta;
                float peso_anterior = adj->peso;

                // Atualização do peso com regularização L2
                float novo_peso = peso_anterior - taxa_de_aprendizagem * (gradiente + 2 * lambda * peso_anterior);
                adj->peso = novo_peso;

                adj = adj->prox;
            }
        }
    }
    return;
}



void treina_network_adam(struct network *net, float **exemplos, float **saidas_esperadas,
                            int numero_de_exemplos, float taxa_de_aprendizagem, int epocas, float beta, float beta2)
{
    if (net == NULL || exemplos == NULL || saidas_esperadas == NULL)
    {
        return;
    }

    float erro_total = 0.0;

    for (int i = 0; i < epocas; i++)
    {
        erro_total = 0.0;
        for (int j = 0; j < numero_de_exemplos; j++)
        {
            feedfoward(net, exemplos[j], saidas_esperadas[j],0);
            back_propagation(net);
            float erro = calcula_erro(net);
            erro_total += erro;
            atualiza_pesos_adam(net, taxa_de_aprendizagem, beta,beta2);
        }
		printf("epoca =  %i, erro_medio = %f\n", i, erro_total / numero_de_exemplos);            
	}

    return;
}


void treina_network_rmsprop(struct network *net, float **exemplos, float **saidas_esperadas,
                            int numero_de_exemplos, float taxa_de_aprendizagem, int epocas, float beta)
{
    if (net == NULL || exemplos == NULL || saidas_esperadas == NULL)
    {
        puts("parametros em treina_network_rmsprop nao pode ser NULL");
        return;
    }

    float erro_total = 0.0;

    for (int i = 0; i < epocas; i++)
    {
        erro_total = 0.0;
        for (int j = 0; j < numero_de_exemplos; j++)
        {
            feedfoward(net, exemplos[j], saidas_esperadas[j],0);
            back_propagation(net);
            float erro = calcula_erro(net);
            erro_total += erro;
            atualiza_pesos_rmsprop(net, taxa_de_aprendizagem, beta);
        }
	printf("epoca =  %i, erro_medio = %f\n", i, erro_total / numero_de_exemplos);            
    }

    return;
}


// executa a propagacao adiante
struct saidas *feedfoward(struct network * net, float * exemplos, float * saidas, int print_saida)
{
	if (net == NULL || exemplos == NULL || saidas == NULL)
	{	puts("parametros de feedfoward nao pode ser NULL");
		return NULL;
	}

	int ret = set_entradas_e_saidas(net, exemplos, saidas);
	
	if (ret == 0)
	{	puts("parametros de feedfoward nao pode ser NULL");
		return NULL;
	}
	
	float somatorio = 0.0;	
	int pos = 0;
	int camada_posterior = 1,camada_anterior = 0;
	
		for( camada_posterior = 1; camada_posterior <= net->camadas-1; camada_posterior++ )
		{
				for (int j = 0; j < net->layers[camada_posterior].num_neurons; j++)
				{	struct vertices *v_inicio = net->layers[camada_posterior].neurons[j];
					for (int k = 0; k < net->layers[camada_anterior].num_neurons; k++)
					{	
						struct vertices *v_final  = net->layers[camada_anterior].neurons[k];
						float peso = busca_na_lista(v_inicio->numero_do_vertice, v_final);
						somatorio += v_final->saida * peso;
						//printf("v inicio:%i peso %f * saida %f vertice de entrada %i\n",v_inicio->numero_do_vertice,peso, v_final->saida, v_final->numero_do_vertice);
					}
					//printf("somatorio %f\n", somatorio);
					v_inicio->saida = sigmoide(somatorio);
				
					if (print_saida)
					{	printf("vinicio:%i saida = %f\n",v_inicio->numero_do_vertice);			
					}			
					somatorio = 0.0;
				}
			camada_anterior++;
		}

	
	return NULL;
}



void back_propagation(struct network *net)
{
	if (net == NULL)
	{
		puts("parametro de back_propagation nao pode ser NULL");
		return;		
	}

	float somatorio = 0.0;
	
 
	for (int i = net->camadas-1; i > 0; i--)
	{	somatorio = 0.0;


		for (int j = 0; j < net->layers[i].num_neurons; j++)		
		{	struct vertices *v = net->layers[i].neurons[j];
			// camada saida
			if (i == net->camadas-1)
			{	v->delta = (v->saida_esperada - v->saida)*derivada_sigmoide(v->saida);
				//printf("somatorio %f, neuronio %i  = delta %f\n",somatorio,  v->numero_do_vertice, v->delta);				
				continue;
			}
			//camada escondida
			struct adjacencia *adj = v->cabeca;
			while (adj != NULL)
			{	
				somatorio += adj->peso * net->neuronios->adj[adj->vertice].delta;				
				adj = adj->prox;
			}
			v->delta = somatorio * derivada_sigmoide(v->saida); 
			//printf("somatorio %f, neuronio %i  = delta %f\n",somatorio,  v->numero_do_vertice, v->delta);
		}
	}

	return;
}


void atualiza_pesos(struct network *net, float taxa_de_aprendizagem)
{
	if (net == NULL)
	{	puts("parametro -> 'net' nao poder ser NULL em atualiza_pesos");
		return;
	}
	
	for (int i = net->camadas-2; i >= 0; i--)
	{	
 		for (int j = 0; j < net->layers[i].num_neurons; j++)
		{	struct vertices *v     = net->layers[i].neurons[j];
			struct adjacencia *adj = v->cabeca;
			while (adj != NULL)
			{	float delta = net->neuronios->adj[adj->vertice].delta * taxa_de_aprendizagem * v->saida;
				//printf("vertice %i ", v->numero_do_vertice);
				//printf("peso antes %f ", adj->peso);
				adj->peso   = adj->peso + delta;
				//printf("peso depois %f\n", adj->peso);	 
				adj = adj->prox;
			}
		}
	}
	return;
}


