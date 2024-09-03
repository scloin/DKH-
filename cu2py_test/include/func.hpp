#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <time.h>
#include <stdlib.h>
#include <vector>

static cudaGraph_t graph;
static cudaGraphExec_t graphExec;
static bool graph_created;

//host func
void adding(int* d_data0, cudaStream_t stream0, int size);
//void adding_re(int* d_data0, cudaStream_t stream0, int size);

//util func
void Aprint(int* A, size_t n);

//device func
__global__ void kernel_mult(int* d_data0);
__global__ void relaunchSelf();
__global__ void launchTailGraph(cudaGraphExec_t graph);
__global__ void loopKernel(cudaGraphConditionalHandle handle);