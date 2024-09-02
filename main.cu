#include <func.hpp>

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

using namespace std;

int main(){
    clock_t g_start, g_end;
    
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
    g_start = clock();
    for (int i=0; i<100000;i++){
    adding(d_data0, stream0, size);
    }
    g_end = clock();
    printf("total %lf s\n", ((float)(g_end - g_start)/1000000));

    /*memcpyD2H*/
    CUDA_CHECK(cudaMemcpyAsync(h_data0, d_data0, size*3*sizeof(int), cudaMemcpyDeviceToHost, stream0));
    Aprint(h_data0, size);
    // Aprint(h_data1, size);
    // Aprint(h_result, size);

    /*free*/
    CUDA_CHECK(cudaFree(d_data0));
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    free(h_data0);
    return 0;
} 

