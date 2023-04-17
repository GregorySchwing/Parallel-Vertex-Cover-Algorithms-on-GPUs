#ifndef BWDWORKLIST_H
#define BWDWORKLIST_H

typedef unsigned int Ticket;
typedef unsigned long long int HT;

typedef union {
	struct {int numWaiting; int numEnqueued;};
	unsigned long long int combined;
} Counter;

struct WorkList{
	unsigned int size;
	unsigned int threshold;
    volatile int* list;
    volatile unsigned int* listNumDeletedVertices;
    volatile Ticket *tickets;
    HT *head_tail;
	int* count;
	Counter * counter;
};
#endif
