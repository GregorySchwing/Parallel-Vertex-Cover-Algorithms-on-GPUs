#ifndef BWDWORKLIST_CUH
#define BWDWORKLIST_CUH

#include "config.h"
#include "CSRGraphRep.h"
#include "BWDWorkList.h"

WorkList allocateWorkList(CSRGraph graph, pvc::Config config, unsigned int numBlocks){
	WorkList workList;
	workList.size = config.globalListSize;
	workList.threshold = config.globalListThreshold * workList.size;

	volatile int* list_d;
	volatile unsigned int * listNumDeletedVertices_d;
	volatile Ticket *tickets_d;
	HT *head_tail_d;
	int* count_d;
	Counter * counter_d;
	cudaMalloc((void**) &list_d, (graph.vertexNum) * sizeof(int) * workList.size);
	cudaMalloc((void**) &listNumDeletedVertices_d, sizeof(unsigned int) * workList.size);
	cudaMalloc((void**) &tickets_d, sizeof(Ticket) * workList.size);
	cudaMalloc((void**) &head_tail_d, sizeof(HT));
	cudaMalloc((void**) &count_d, sizeof(int));
	cudaMalloc((void**) &counter_d, sizeof(Counter));
	
	workList.list = list_d;
	workList.listNumDeletedVertices = listNumDeletedVertices_d;
	workList.tickets = tickets_d;
	workList.head_tail = head_tail_d;
	workList.count=count_d;
	workList.counter = counter_d;

	HT head_tail = 0x0ULL;
	Counter counter;
	counter.combined = 0;
	cudaMemcpy(head_tail_d,&head_tail,sizeof(HT),cudaMemcpyHostToDevice);
	cudaMemcpy((void*)list_d, graph.degree, (graph.vertexNum) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemset((void*)&listNumDeletedVertices_d[0], 0, sizeof(unsigned int));
	cudaMemset((void*)&tickets_d[0], 0, workList.size * sizeof(Ticket));
	cudaMemset(count_d, 0, sizeof(int));
	cudaMemcpy(counter_d, &counter ,sizeof(Counter),cudaMemcpyHostToDevice);

	return workList;
}


WorkList allocateWorkListEdges(int vertexNum, pvc::Config config, unsigned int numBlocks){
	WorkList workList;
	workList.size = config.globalListSize;
	workList.threshold = config.globalListThreshold * workList.size;

	volatile int* list_d;
	volatile unsigned int * listNumDeletedVertices_d;
	volatile Ticket *tickets_d;
	HT *head_tail_d;
	int* count_d;
	Counter * counter_d;
	cudaMalloc((void**) &list_d, (vertexNum) * sizeof(int) * workList.size);
	cudaMalloc((void**) &listNumDeletedVertices_d, sizeof(unsigned int) * workList.size);
	cudaMalloc((void**) &tickets_d, sizeof(Ticket) * workList.size);
	cudaMalloc((void**) &head_tail_d, sizeof(HT));
	cudaMalloc((void**) &count_d, sizeof(int));
	cudaMalloc((void**) &counter_d, sizeof(Counter));
	
	workList.list = list_d;
	workList.listNumDeletedVertices = listNumDeletedVertices_d;
	workList.tickets = tickets_d;
	workList.head_tail = head_tail_d;
	workList.count=count_d;
	workList.counter = counter_d;

	HT head_tail = 0x0ULL;
	Counter counter;
	counter.combined = 0;
	cudaMemcpy(head_tail_d,&head_tail,sizeof(HT),cudaMemcpyHostToDevice);
	cudaMemset((void*)&listNumDeletedVertices_d[0], 0, sizeof(unsigned int));
	cudaMemset((void*)&tickets_d[0], 0, workList.size * sizeof(Ticket));
	cudaMemset(count_d, 0, sizeof(int));
	cudaMemcpy(counter_d, &counter ,sizeof(Counter),cudaMemcpyHostToDevice);

	return workList;
}



WorkList allocateWorkListEdmonds(CSRGraph graph, pvc::Config config, unsigned int numBlocks){

	WorkList workList;
	workList.size = config.globalListSize;
	workList.threshold = config.globalListThreshold * workList.size;

	volatile int* list_d;
	volatile unsigned int * listNumDeletedVertices_d;
	volatile Ticket *tickets_d;
	HT *head_tail_d;
	int* count_d;
	Counter * counter_d;
	cudaMalloc((void**) &list_d, 1 * sizeof(int) * workList.size);
	cudaMalloc((void**) &listNumDeletedVertices_d, sizeof(unsigned int) * workList.size);
	cudaMalloc((void**) &tickets_d, sizeof(Ticket) * workList.size);
	cudaMalloc((void**) &head_tail_d, sizeof(HT));
	cudaMalloc((void**) &count_d, sizeof(int));
	cudaMalloc((void**) &counter_d, sizeof(Counter));
	
	workList.list = list_d;
	workList.listNumDeletedVertices = listNumDeletedVertices_d;
	workList.tickets = tickets_d;
	workList.head_tail = head_tail_d;
	workList.count=count_d;
	workList.counter = counter_d;

	HT head_tail = 0x0ULL;
	Counter counter;
	counter.combined = 0;
	cudaMemcpy(head_tail_d,&head_tail,sizeof(HT),cudaMemcpyHostToDevice);
	//cudaMemcpy((void*)list_d, graph.degree, 1 * sizeof(int), cudaMemcpyHostToDevice);
	//cudaMemset((void*)&listNumDeletedVertices_d[0], 0, sizeof(unsigned int));
	cudaMemset((void*)list_d, 0, workList.size * sizeof(int));
	cudaMemset((void*)listNumDeletedVertices_d, 0, workList.size * sizeof(unsigned int));
	cudaMemset((void*)&tickets_d[0], 0, workList.size * sizeof(Ticket));
	cudaMemset(count_d, 0, sizeof(int));
	cudaMemcpy(counter_d, &counter ,sizeof(Counter),cudaMemcpyHostToDevice);

	return workList;
}

void cudaFreeWorkList(WorkList workList){
	cudaFree((void*)workList.list);
	cudaFree(workList.head_tail);
	cudaFree((void*)workList.listNumDeletedVertices);
	cudaFree(workList.counter);
	cudaFree(workList.count);
}

#endif
