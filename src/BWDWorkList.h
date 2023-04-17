#ifndef BWDWORKLIST_H
#define BWDWORKLIST_H
#include <stdint.h>
#include <stdbool.h>
typedef unsigned int Ticket;
typedef unsigned long long int HT;

typedef union {
	struct {int numWaiting; int numEnqueued;};
	unsigned long long int combined;
} Counter;

struct WorkList{
	unsigned int size;
	unsigned int threshold;
    volatile bool* mutex;
    volatile int* list;
    volatile unsigned int* listNumDeletedVertices;
	// For augmenting/contracting blossoms
	volatile uint64_t * listBTypeEdges;

    volatile Ticket *tickets;
    HT *head_tail;
	int* count;
	Counter * counter;
};
#endif
