#include "queue.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

void init_queue (queue_t * q, int sizeoftype, void (*freeFunc)(void *))
{
	if (sizeoftype == 0 || !q)	
		return; 
	
	if (!freeFunc)
		q->freeFunc = NULL;
	else
		q->freeFunc = freeFunc;
		
	q->queue = NULL;
	q->last  = NULL;
	q->nelem = 0;
	q->ltype = sizeoftype;
	
	return;
}


int queue_len (queue_t *q)
{
	if (!q)	
	{	puts("queue_len() :queue nao pode ser NULL");	
		return 0;
	}
	
	//printf("lista len %i\n", q->nelem);		
	return q->nelem;
}

void append (queue_t * q, void * obj)
{
	if (!q || !obj)	
		return;
		 
	void   * new  = malloc(q->ltype);
	elem_t * node = malloc(sizeof(elem_t));	
	
	if (!new || !node)
	{	puts("append() :sem memoria");
		return;
	}
	
	node->obj  = new;
	node->next = NULL;
	memcpy(node->obj, obj, q->ltype);
	
	q->nelem++;
	
	if (!q->queue) 
	{	q->queue = node;
		q->last  = node; 
		return;
	}
 
	q->last->next = node;
	q->last       = node;
	return;
}

void dequeue (queue_t *q, void *obj)
{
	if (!q || !obj)
	{	puts("dequeue() :fila ou obj nao pode ser NULL");
		return;	
	}
	
	if (!q->queue) 
		return;
		
	elem_t *aux = q->queue;
	q->queue    = q->queue->next;
	memcpy(obj,aux->obj, q->ltype);
	
	if (aux == q->last) 
	{	q->last  = NULL;
		q->queue = NULL;
	}
	 
	free(aux->obj);
	free(aux);
	q->nelem--;
	return; 
}

void queue_print (queue_t *q, void (*print)(void *))
{
	if (!q || !print)
	{	puts("queue_print() :fila ou print nao pode ser NULL");
		return;	
	}
	
	elem_t * aux = q->queue;
	
	while (aux != NULL)
	{	print(aux->obj);
		aux = aux->next; 
	}
	
	return;
}

void queue_reinit(queue_t * q)
{

	if (!q)
		return;
			
	elem_t *aux, *aux2;	
	
	aux = q->queue;
	 	
	while (aux != NULL)
	{	aux2 = aux->next;
		if (q->freeFunc)
			q->freeFunc(aux->obj);
		else
			free(aux->obj);
		free(aux);
		aux = aux2; 
	}
	q->queue = NULL;
	q->nelem = 0;
	return;		

}
void queue_destroy(queue_t *q)
{
	if (!q)
		return;
			
	elem_t *aux, *aux2;	
	
	aux = q->queue;
	 	
	while (aux != NULL)
	{	aux2 = aux->next;
		if (q->freeFunc)
			q->freeFunc(aux->obj);
		else
			free(aux->obj);
		free(aux);
		aux = aux2; 
	}
	//printf("pointer queue %p ", q->queue);	
	q->queue    = NULL;
	q->nelem    = 0;
	q->freeFunc = NULL;
	q->ltype    = 0;
	return;	
}
