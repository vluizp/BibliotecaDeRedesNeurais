#include "nn_parallel.h"
#include "queue.h"
#include "nn_train.h"

// --- Variaveis Globais (Definidas em nn_parallel.c) ---
pthread_mutex_t mutex, mutex2, mutex3, mutex4;
pthread_cond_t cond, cond2, cond3, cond4;
queue_t queue, queue_neurons; // Filas para Produtor/Consumidor
struct thread_drive_t global_thread_drive;

int pode_comecar, pode_continuar;

void init_global_thread_drive(struct network *net)
{
	global_thread_drive.net = net;
	global_thread_drive.camada_anterior = 0;
	global_thread_drive.camada_posterior = 1;

	if ((global_thread_drive.num_neuron_por_camada = malloc(sizeof(int) * net->camadas)) == NULL)
	{	puts("erro na alocacao de memoria em init_global_thread_drive()");
		return;
	}
	
	for (int i = 1; i < net->camadas; i++)
	{
		global_thread_drive.num_neuron_por_camada[i] = net->layers[i].num_neurons;
	}	

	return;
}

//thread produtora constroi a mfila com inices dos neuronios da camada a ser ativada
void *producer(void *args)
{
    struct network *net = global_thread_drive.net;
    int i;

    while (1)
    {	pthread_mutex_lock(&mutex3);
		    if (global_thread_drive.camada_posterior >= net->camadas)
		    {
		        pode_comecar = 0;
		        pode_continuar = 1;
		        pthread_cond_signal(&cond3); // Sinaliza que pode prosseguir com a próxima fase
		    }
        pthread_mutex_unlock(&mutex3);
        // Espera até poder começar a nova camada
        pthread_mutex_lock(&mutex4);
		    while (!pode_comecar)
		        pthread_cond_wait(&cond4, &mutex4);
        pthread_mutex_unlock(&mutex4);
		
        // Adiciona neurônios da camada posterior na fila para serem processados
        pthread_mutex_lock(&mutex);
		    i = 0;
			
		    while (i < net->layers[global_thread_drive.camada_posterior].num_neurons)
		    {	
		        append(&queue, &i);
		        i++;
		    }
		    //queue_print(&queue, print_int);
		    pthread_cond_broadcast(&cond);  // Acorda todas as threads consumidoras
        pthread_mutex_unlock(&mutex);

        pthread_mutex_lock(&mutex2);
			while (queue_len(&queue_neurons) < global_thread_drive.num_neuron_por_camada[global_thread_drive.camada_posterior])
		        pthread_cond_wait(&cond2, &mutex2);
		    queue_reinit(&queue_neurons);
			global_thread_drive.camada_anterior++;
		    global_thread_drive.camada_posterior++;
        pthread_mutex_unlock(&mutex2);
    }
}

// thread consumidora , faz uso a fila criada pela thread produtora .
void *consumer(void *arg)
{
    int neuron_idx;
    int thread_id = *(int *)arg;

    struct network *net = global_thread_drive.net;
    while (1)
    {
        // Espera por trabalho na fila
        pthread_mutex_lock(&mutex);
		    while (queue_len(&queue) == 0)
		        pthread_cond_wait(&cond, &mutex);
		    dequeue(&queue, &neuron_idx);
        pthread_mutex_unlock(&mutex);

        // Processa o neurônio da camada atual
        struct vertices *v_final = net->layers[global_thread_drive.camada_posterior].neurons[neuron_idx];
        float somatorio = 0.0;

        for (int k = 0; k < net->layers[global_thread_drive.camada_anterior].num_neurons; k++)
        {
           // printf("Thread %i processando neurônio  %i na camada pos %i usando camada ant %i\n", thread_id, neuron_idx, global_thread_drive.camada_posterior,global_thread_drive.camada_anterior);
            struct vertices *v_inicio = net->layers[global_thread_drive.camada_anterior].neurons[k];
            float peso = busca_na_lista(v_final->numero_do_vertice, v_inicio);
            somatorio += v_inicio->saida * peso;
        }

        v_final->saida = sigmoide(somatorio);

        pthread_mutex_lock(&mutex2);
		    append(&queue_neurons, &neuron_idx);
		    if (queue_len(&queue_neurons) == global_thread_drive.num_neuron_por_camada[global_thread_drive.camada_posterior])
		    {	
		        pthread_cond_signal(&cond2); // Sinaliza que a camada foi completamente processada
		    }
        pthread_mutex_unlock(&mutex2);
    }
}


void treina_network2(struct network *net, float **exemplos, float ** saidas_esperadas, 
					int numero_de_exemplos, float taxa_de_aprendizagem , int epocas)
{
	if (net == NULL || exemplos == NULL || saidas_esperadas == NULL)
	{	puts("parametros em treina_network nao pode ser NULL");
		return;
	}
	
	float erro_total = 0.0;

	int num_threads = 2; // valor ad-hoc a ideia ter o valor maximo de neuronios em alguma camada   
	pthread_t thread[num_threads], prod_thread;
	init_queue(&queue, sizeof(int), NULL);
	init_queue(&queue_neurons, sizeof(int), NULL);
	init_global_thread_drive(net);
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);
	pthread_mutex_init(&mutex2, NULL);
	pthread_cond_init(&cond2, NULL);
	pthread_mutex_init(&mutex3, NULL);
	pthread_cond_init(&cond3, NULL);
	pthread_mutex_init(&mutex4, NULL);
	pthread_cond_init(&cond4, NULL);

	pode_continuar = 0;	
	pthread_create(&prod_thread, NULL, producer, NULL);
		
	for (int i = 0; i < num_threads; i++)
		pthread_create(&thread[i], NULL, consumer, (void*)&i);
	
	for (int i = 0; i < epocas; i++)
	{	erro_total = 0.0;
		
		for (int j = 0; j < numero_de_exemplos; j++)
		{	
			set_entradas_e_saidas(net, exemplos[j], saidas_esperadas[j]);
			
			// Inicia o processamento da primeira camada
			pthread_mutex_lock(&mutex4);
				pode_comecar = 1;
				pthread_cond_signal(&cond4);
			pthread_mutex_unlock(&mutex4);

			// Espera até que todas as camadas tenham sido processadas
			pthread_mutex_lock(&mutex3);
				while (!pode_continuar)			
					pthread_cond_wait(&cond3, &mutex3);
			pthread_mutex_unlock(&mutex3);

			float erro = calcula_erro(net);
			printf("epoca= %i, erro %f\n", epocas, erro);
			erro_total += erro;

			// Backpropagation e atualização de pesos
			back_propagation(net);
			atualiza_pesos(net, taxa_de_aprendizagem);

			pthread_mutex_lock(&mutex2);
				pode_continuar = 0;	
				global_thread_drive.camada_anterior = 0;
		    	global_thread_drive.camada_posterior = 1;		
			pthread_mutex_unlock(&mutex2);
		}
		
	}


	
	queue_destroy(&queue);
	queue_destroy(&queue_neurons);
	
	printf("epoca =  %i, erro_medio = %f\n",epocas, erro_total / numero_de_exemplos);	
	return;	
}

