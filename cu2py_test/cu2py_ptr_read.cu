#include <torch/extension.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cstring>
#include <ATen/cuda/CUDAGraph.h>
#define DSIZE 28*28

void* buffer;
int* head;
int* tail;
int* out;

#define cudaCheckErrors(msg) \
  do { \
  cudaError_t __err = cudaGetLastError(); \
  if (__err != cudaSuccess) { \
  fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
    msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
  fprintf(stderr, "*** FAILED - ABORTING\n"); \
  exit(1); \
  } \
  } while (0)

__global__ void polling(int* tail){
  int check=1;
  while(check!=0){
    //wait for head==0
    check=atomicExch(&tail[0],1);
  }
}

__global__ void _set_tail(int* head){
  atomicExch(&head[0],0);
}

void set_tail(){
    _set_tail<<<1,1>>>(head);
    cudaDeviceSynchronize();
}
void polling_input(){
    polling<<<1,1>>>(tail);
    cudaDeviceSynchronize();
}
torch::Tensor read_and_return_tensor_ptr(const std::string &fifo_path) {
    
    cudaIpcMemHandle_t my_handle;
    unsigned char handle_buffer[sizeof(my_handle) + 1] = {0};

    FILE *fp = fopen(fifo_path.c_str(), "r");
    if (fp == NULL) {
        throw std::runtime_error("fifo open fail");
    }

    for (int i = 0; i < sizeof(my_handle); i++) {
        int ret = fscanf(fp, "%c", handle_buffer + i);
        if (ret == EOF) {
            throw std::runtime_error("received EOF");
        } else if (ret != 1) {
            throw std::runtime_error("fscanf returned an error");
        }
    }
    cudaMalloc(&buffer, DSIZE*sizeof(float)+3*sizeof(int));
    memcpy((unsigned char *)(&my_handle), handle_buffer, sizeof(my_handle));

    cudaIpcOpenMemHandle((void **)&buffer, my_handle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("IPC handle fail");
    out=(int*)(buffer+sizeof(float)*DSIZE);
    head=(int*)(buffer+sizeof(float)*DSIZE+sizeof(int));
    tail=(int*)(buffer+sizeof(float)*DSIZE+2*sizeof(int));
    int result;
    int h_head;
    int h_tail;
    cudaMemcpy((void **)&result, out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void **)&h_head, head, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void **)&h_tail, tail, sizeof(int), cudaMemcpyDeviceToHost);
    printf("%d %d %d\n", result, h_head, h_tail);
    // GPU 메모리를 직접 가리키는 Tensor 생성
    torch::Tensor output_tensor = torch::from_blob(buffer, {DSIZE}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_and_return_tensor_ptr", &read_and_return_tensor_ptr, "Read data and return tensor pointer");
    m.def("polling_input", &polling_input, "Read data and return tensor pointer");
    m.def("set_tail", &set_tail, "Read data and return tensor pointer");
}

