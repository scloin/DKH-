#include <torch/extension.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cstring>

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



void read_ipc_handle(torch::Tensor &output_tensor, const std::string &fifo_path) {
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

    int flag = 0;
    while (flag == 0) {
        cudaMemcpy(output_tensor.data_ptr<float>(), data, DSIZE * sizeof(float), cudaMemcpyDeviceToHost);
        cudaCheckErrors("memcopy fail");

        printf("values read from GPU memory: %f %f %f\n", output_tensor[0].item<float>(), output_tensor[1].item<float>(), output_tensor[2].item<float>());

        sleep(1);

        if (output_tensor[0].item<float>() < 0) {
            flag = -1;
        }
    }

    cudaIpcCloseMemHandle(data);
    cudaFree(data);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("read_ipc_handle", &read_ipc_handle, "Read IPC handle from FIFO and copy GPU memory to Tensor");
}
