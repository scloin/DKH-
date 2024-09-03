#include <torch/extension.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cstring>
#include <ATen/cuda/CUDAGraph.h>


#define DSIZE 3

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

torch::Tensor read_and_return_tensor_ptr(const std::string &fifo_path) {
    float *data;
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

    memcpy((unsigned char *)(&my_handle), handle_buffer, sizeof(my_handle));

    cudaIpcOpenMemHandle((void **)&data, my_handle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("IPC handle fail");

    // GPU 메모리를 직접 가리키는 Tensor 생성
    torch::Tensor output_tensor = torch::from_blob(data, {DSIZE}, torch::device(torch::kCUDA).dtype(torch::kFloat32));

    return output_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_and_return_tensor_ptr", &read_and_return_tensor_ptr, "Read data and return tensor pointer");
}
