#include <func.hpp>
std::vector<cudaGraphNode_t> _node_list;
#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

using namespace std;


int main(){
    clock_t g_start, g_end;
    graph_created=false;
    int size =1<<10;
    printf("%d\n",size);
    cudaStream_t stream0;
    cudaStream_t stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));
    /*init host memory*/
    int* h_data0;
    int* h_data1;
    int* h_result;
    h_data0 = (int*)calloc(size*3, sizeof(int));
    h_data1 = &h_data0[size];
    h_result = &h_data0[size*2];

    srand(990720);
    for (int i=0;i<size;i++){
        h_data0[i]= rand()%10;
    }
    // Aprint(h_data0, size);
    // Aprint(h_data1, size);

    /*init device memory*/
    int* d_data0;
    int* d_data1;
    int* d_result;
    CUDA_CHECK(cudaMalloc((void**)&d_data0, (size*3)*sizeof(int)));
    d_data1=&d_data0[size];
    d_result=&d_data0[size*2];

    /*memcpyH2D*/
    CUDA_CHECK(cudaMemcpyAsync(d_data0, h_data0, size*3*sizeof(int), cudaMemcpyHostToDevice, stream0));

    cudaGraphExec_t gExec1, gExec2, gExec3;
    cudaGraphNode_t a, b,c;
    cudaGraph_t g1, g2, g3;
    cudaGraphCreate(&g2,0);

    CUDA_CHECK(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));
    adding(d_data0, stream0, size);
    CUDA_CHECK(cudaStreamEndCapture(stream0, &g3));

    cudaGraphAddChildGraphNode(&b, g2, NULL, 0, g3);
    struct cudaKernelNodeParams kr={0};
    kr.func=(void*)relaunchSelf;
    kr.gridDim=1;
    kr.blockDim=1;
    kr.sharedMemBytes=0;
    kr.kernelParams=NULL;
    kr.extra=NULL;

    cudaGraphAddKernelNode(&a, g2, &b,1,&kr);
    // cudaGraphAddKernelNode(&c, g1, &a,2,&kr);

    cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(gExec2, stream0);

        //printf("%3d %lf\n",i, (double)((end - start)));
    cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal);
    launchTailGraph<<<1, 1, 0, stream0>>>(gExec2);
    cudaStreamEndCapture(stream0, &g1);
    cudaGraphInstantiate(&gExec1, g1);
    g_start = clock();
    cudaGraphLaunch(gExec1, stream0);
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    g_end = clock();
    printf("total %lf s\n", ((float)(g_end - g_start)/1000000));
    
    /*memcpyD2H*/
    CUDA_CHECK(cudaMemcpyAsync(h_data0, d_data0, size*3*sizeof(int), cudaMemcpyDeviceToHost, stream0));
    Aprint(h_data0, size);
    // Aprint(h_data1, size);
    // Aprint(h_result, size);

    /*free*/
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    CUDA_CHECK(cudaFree(d_data0));
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    free(h_data0);
    return 0;
} 

