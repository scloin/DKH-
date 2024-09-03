import torch
import cuda_graph_extension

size = 1024
result_tensor = cuda_graph_extension.cuda_graph_example(size)
print(result_tensor)
