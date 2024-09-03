#include <torch/extension.h>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAGraph.h>
#include <func.hpp>

#define CUDA_CHECK(val) { \
	if (val != cudaSuccess) { \
		fprintf(stderr, "Error %s at line %d in file %s\n", cudaGetErrorString(val), __LINE__, __FILE__); \
		exit(val); \
	} \
}

using namespace std;



torch::Tensor cuda_graph_example(int size) {
    // GPU 메모리를 PyTorch 텐서로 관리하기 위해 torch::Tensor 사용
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor h_data_tensor = torch::zeros({size * 3}, options);
    int* h_data0 = h_data_tensor.data_ptr<int>();
    int* h_data1 = h_data0 + size;
    int* h_result = h_data0 + (size * 2);

    // 데이터 초기화
    srand(990720);
    for (int i = 0; i < size; i++) {
        h_data0[i] = rand() % 10;
    }

    // 장치 메모리 할당 및 복사
    torch::Tensor d_data_tensor = torch::empty_like(h_data_tensor);
    int* d_data0 = d_data_tensor.data_ptr<int>();
    int* d_data1 = d_data0 + size;
    int* d_result = d_data0 + (size * 2);
    for (int i = 0; i < size; i++) {
        printf("%d ",h_data0[i]);
    }
    printf("\n");
    CUDA_CHECK(cudaMemcpyAsync(d_data0, h_data0, size * 3 * sizeof(int), cudaMemcpyHostToDevice));

    // CUDA 그래프 생성 및 실행
    cudaGraph_t g1, g2, g3;
    cudaGraphExec_t gExec1, gExec2;
    cudaGraphNode_t a, b;
    printf("1\n");
    CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    // 가정: `adding`은 d_data0를 사용하여 CUDA 커널을 실행
    adding(d_data0, 0, size);
    CUDA_CHECK(cudaStreamEndCapture(0, &g3));

    CUDA_CHECK(cudaGraphCreate(&g2, 0));
    CUDA_CHECK(cudaGraphAddChildGraphNode(&b, g2, nullptr, 0, g3));

    struct cudaKernelNodeParams kr = {0};
    kr.func = (void*)relaunchSelf;
    kr.gridDim = 1;
    kr.blockDim = 1;
    kr.sharedMemBytes = 0;
    kr.kernelParams = nullptr;
    kr.extra = nullptr;

    CUDA_CHECK(cudaGraphAddKernelNode(&a, g2, &b, 1, &kr));

    CUDA_CHECK(cudaGraphInstantiate(&gExec2, g2, cudaGraphInstantiateFlagDeviceLaunch));
    CUDA_CHECK(cudaGraphUpload(gExec2, 0));

    CUDA_CHECK(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal));
    launchTailGraph<<<1, 1, 0>>>(gExec2);
    CUDA_CHECK(cudaStreamEndCapture(0, &g1));

    CUDA_CHECK(cudaGraphInstantiate(&gExec1, g1, cudaGraphInstantiateFlagDeviceLaunch));

    clock_t g_start, g_end;
    g_start = clock();
    CUDA_CHECK(cudaGraphLaunch(gExec1, 0));
    CUDA_CHECK(cudaStreamSynchronize(0));
    g_end = clock();
    printf("Total %lf s\n", ((float)(g_end - g_start) / CLOCKS_PER_SEC));

    // 결과를 CPU로 복사
    CUDA_CHECK(cudaMemcpyAsync(h_data0, d_data0, size * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    printf("1\n");
    // 그래프 및 스트림 메모리 해제
    cudaGraphDestroy(g1);
    cudaGraphExecDestroy(gExec1);
    cudaGraphDestroy(g2);
    cudaGraphExecDestroy(gExec2);
    
    // 결과 텐서를 반환
    return h_data_tensor;
}

// PyTorch 모듈 정의
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_graph_example", &cuda_graph_example, "Execute CUDA graph and return results as tensor", py::arg("size"));
}
