#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int M, N, O;
	char* S;

	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	Layer(int M, int N, int O, char* S);

	~Layer();

	void setOutput(float *data);
	void setOutputwithPadding(float *data, float * d_data);
	void clear();
	void bp_clear();
	void save_to_json();
	void fetchmodel();
};


// Utility CUDA kernel functions
__device__ float step_function(float v);
__global__ void apply_step_function(float *input, float *output, const int N);
__global__ void makeError(float *err, float *output, unsigned int Y, const int N);
__global__ void apply_grad(float *output, float *grad, const int N);

// Forward propagation kernels
__global__ void fp_preact_c1(float input[28][28], float preact[6][28][28], float weight[6][5][5]);
__global__ void fp_bias_c1(float preact[6][28][28], float bias[6]);
__global__ void fp_preact_s1(float input[6][28][28], float preact[6][14][14]);
__global__ void fp_bias_s1(float preact[6][14][14], float bias[1]);
__global__ void fp_preact_c2(float input[6][14][14], float preact[16][10][10], float weight[16][6][5][5]);
__global__ void fp_bias_c2(float preact[16][10][10], float bias[16]);
__global__ void fp_preact_s2(float input[16][10][10], float preact[16][5][5]);
__global__ void fp_bias_s2(float preact[16][5][5], float bias[1]);
__global__ void fp_preact_c3(float input[16][5][5], float preact[120], float weight[120][16][5][5]);
__global__ void fp_bias_c3(float preact[120], float bias[120]);
__global__ void fp_preact_f1(float input[400], float preact[120], float weight[120][400]);
__global__ void fp_bias_f1(float preact[120], float bias[120]);
__global__ void fp_preact_f2(float input[120], float preact[10], float weight[10][120]);
__global__ void fp_bias_f2(float preact[10], float bias[10]);

__global__ void apply_padding2(float input[28][28], float output[32][32]);