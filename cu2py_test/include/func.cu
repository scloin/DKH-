#include <func.hpp>

#define BLOCK_SIZE 512
#define BLOCK_QUEUE_SIZE 512

void Aprint(int* A, size_t n){
    for (int i=0; i<n; i++){
        printf("%d ", A[i]);
        if (i%64==63){
            printf("\n");
        }
    }
}

__global__ void kernel_mult(int* d_data0){
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    *(d_data0+tid)+=1;
}
void adding(int* d_data0, cudaStream_t stream0, int size){
    
    int num_blocks;
    /*kernel*/
    num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);

    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);

    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
    kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);

    // CUDA_CHECK(cudaStreamSynchronize(stream0));
    // cudaDeviceSynchronize();
}

// 나중에 test
__device__ int relaunchCount = 1;

__global__ void relaunchSelf() {
    int relaunchMax = 100000;

    if (threadIdx.x == 0) {
        if (relaunchCount < relaunchMax) {
            cudaGraphLaunch(cudaGetCurrentGraphExec(), cudaStreamGraphTailLaunch);
        }

        relaunchCount++;
    }
}

// void adding_re(int* d_data0, cudaStream_t stream0, int size){
//     int num_blocks;
//     /*kernel*/
//     num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
//     kernel_mult<<<num_blocks, BLOCK_SIZE, 0, stream0>>>(d_data0);
//     cudaDeviceSynchronize();
//     relaunchSelf<<<1,1,0,stream0>>>();

// }

// __global__ void kernel_mult_r(int* d_data0){
//     const int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     *(d_data0+tid)+=1;
// }

__global__ void launchTailGraph(cudaGraphExec_t graph) {
    cudaGraphLaunch(graph, cudaStreamGraphTailLaunch);
}

__global__ void loopKernel(cudaGraphConditionalHandle handle)
{
    static int count = 100000;
    cudaGraphSetConditional(handle, --count ? 1 : 0);
}