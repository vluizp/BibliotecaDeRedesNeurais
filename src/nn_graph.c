#include "nn_graph.h"
#include <stdio.h>
#include <stdlib.h>


void destroi_lista(struct adjacencia *adj)
{
	if (adj == NULL)
		return;

	struct adjacencia *aux = NULL;

	while (adj != NULL)
	{
		aux = adj;		
		adj = adj->prox;
		free(aux);		
	}

	return;
}

void destroi_grafo(struct grafo * grafo)
{
	if (grafo  == NULL)
	{	puts("parametro nao pode ser NULL em destroi_grafo");
		return;
	}

	for (int i = 0; i < grafo->vertices; i++)
		destroi_lista(grafo->adj[i].cabeca);
		
	free(grafo->adj);
}

void destroi_net(struct network *net)
{
	if (net == NULL)
	{	puts("parametro nao pode ser NULL em destroi_net");
		return ;
	}

	for (int i = 0; i < net->camadas; i++)	
	{
		free(net->layers[i].neurons);

	}	

	free(net->layers);
	destroi_grafo(net->neuronios);
	free(net->neuronios);
	free(net);
	return;
}



void imprime_grafo(struct grafo * grafo)
{
	
	if (grafo == NULL)
	{	puts("grafo nao pode ser NULL em imprime_grafo");
		return;
	}

	printf("vertices %i Aretas %i\n", grafo->vertices, grafo->arestas);

	for (int i = 0; i < grafo->vertices; i++)
	{	printf("v%i: ", i);
		struct adjacencia * adj = grafo->adj[i].cabeca;
		while (adj != NULL)
		{	printf("v%i(%f) ", adj->vertice, adj->peso);
			adj = adj->prox;			
		}
		puts("");	
	}
	return;
}

struct adjacencia * cria_adj(int vertice, float peso)
{
	struct adjacencia *new = malloc(sizeof(struct adjacencia));
	
	if (new == NULL)	
	{	puts("falha na alocacao de memoria em cria_adj");
		return NULL;			
	}

	new->media_movel = 0.0;
	new->media_movel_adam = 0.0;
	new->vertice = vertice;
	new->peso    = peso;
	new->prox    = NULL;
	return new;
} 

void cria_aresta(struct grafo * grafo, int vertice_inicial, int vertice_final, float peso)
{
	if (grafo == NULL)	
	{	puts("grafo nao pode ser NULL em cria_aresta");
		return;
	}

	struct adjacencia * new_adj  = cria_adj(vertice_final, peso);

	if (new_adj == NULL) 
	{	puts("falha na alocacao de memoria em cria_aresta");
		return;
	}

	new_adj->prox = grafo->adj[vertice_inicial].cabeca;
	grafo->adj[vertice_inicial].cabeca = new_adj;
	grafo->arestas++;
	return;
}	


struct grafo * cria_grafo(int num_vertices)
{

	struct grafo *new = malloc(sizeof(struct grafo));
		
	if (new == NULL)
	{	puts("falha na alocacao de memoria em cria_grafo");
		return NULL;
	}

	new->vertices = num_vertices;
	new->arestas  = 0;
	new->adj      = malloc(num_vertices  * sizeof(struct vertices)); 

	if (new->adj == NULL)
	{	free(new);
		puts("falha na alocacao de memoria em cria_grafo");
		return NULL;
	}

	for (int i = 0; i < num_vertices; i++)
	{
		new->adj[i].cabeca = NULL;
		new->adj[i].numero_do_vertice = i;		
	}	

	return new; 
}

// cria_network
// camadas -> vetor com valores que indicam quantos neuronios tem em cada camada, 
// ultima posicao do vetor  deve ser -1
struct network * cria_network(int *camadas)
{
	if (camadas == NULL)
	{	puts("parametro camadas nao pode ser NULL");
		return NULL;
	}
	
	struct network *new = malloc(sizeof(struct network));
	
	if (new == NULL)
	{	puts("falha na alocacao de memoria em cria_network");
		return NULL;
	}	

	int num_neuronios = 0;
	int layers = 0;

	for (int i = 0; camadas[i] != -1; i++)	
	{	num_neuronios += camadas[i];
		layers++;
	}

	new->camadas       = layers;
	new->num_neuronios = num_neuronios;
	new->neuronios     = cria_grafo(num_neuronios); 
	new->layers        = malloc(layers * sizeof(struct camadas));

	if (new->layers == NULL)	
	{	free(new);
		puts("falha na alocacao de memoria em cria_network");
		return NULL;
	}

		
	int indice_vertice = 0;
	for (int i = 0; i < layers; i++)	
	{	new->layers[i].num_neurons = camadas[i];
		new->layers[i].neurons = malloc(camadas[i] * sizeof(struct vertices *));
		if (new->layers[i].neurons == NULL)
		{	free(new);
			puts("falha na alocacao de memoria em cria_network");
			return NULL;
		}

		for (int l = 0; l < camadas[i]; l++)
			new->layers[i].neurons[l] = &new->neuronios->adj[indice_vertice++];	
	}


	//ligacoes dos vertices
	for (int i = 0; i < layers-1; i++)	
	{	for (int j = 0; j < new->layers[i].num_neurons; j++)
		{	struct vertices * v_inicio = new->layers[i].neurons[j]; 
			for (int k = 0; k < new->layers[i+1].num_neurons; k++)
			{	struct vertices * v_final = new->layers[i+1].neurons[k];
				cria_aresta(new->neuronios, v_inicio->numero_do_vertice, v_final->numero_do_vertice, (float)rand() / RAND_MAX);
			}	
		}
	}
	
	imprime_grafo(new->neuronios);

	return new;
}

int set_entradas_e_saidas(struct network *net, float * exemplos, float * saidas)
{
	if (net == NULL || exemplos == NULL || saidas == NULL)
	{	puts("parametros de set_entradas_e_saidas nao pode ser NULL");
		return 0;
	}

	for (int i = 0; i < net->layers[0].num_neurons; i++)
	{	struct vertices * v = net->layers[0].neurons[i]; 
		v->saida = exemplos[i];
	}

	for (int i = 0; i < net->layers[net->camadas-1].num_neurons; i++)
	{	struct vertices * v = net->layers[net->camadas-1].neurons[i]; 
		v->saida_esperada = saidas[i];
	}

	return 1;

}

// pode ser melhorado usando um array para representar a lista de adjacencias e fazendo busca binaria 
float busca_na_lista(int vertice, struct vertices *neuron)
{

	if (neuron == NULL)
		return 0.0;	
	
	struct adjacencia *adj = neuron->cabeca;

	while (adj != NULL)
	{	if (adj->vertice == vertice)
			return adj->peso;
		adj = adj->prox;
	}

	return 0.0;
}

