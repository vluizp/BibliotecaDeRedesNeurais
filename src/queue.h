#ifndef __QUEUE__
#define __QUEUE__



typedef struct elem 
{	
	void * obj;	
	struct elem * next;
		
}elem_t;

typedef struct Queue_t
{	
	int nelem;
	int ltype;
	elem_t *queue;
	elem_t *last;
	void (*freeFunc)(void *);	 
}queue_t;


void init_queue (queue_t * q, int sizeoftype, void (*freeFunc)(void *));
int  queue_len (queue_t *q);
void append (queue_t * q, void * obj);
void dequeue (queue_t *q, void *obj);
void queue_print (queue_t *q, void (*print)(void *));
void queue_destroy(queue_t *q);
void queue_reinit(queue_t * q);
#endif