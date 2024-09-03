import torch
import ipc_extension

fifo_path = "testfifo"

# GPU 메모리를 가리키는 Tensor를 반환하는 함수를 호출
tensor = ipc_extension.read_and_return_tensor_ptr(fifo_path)
print(tensor.device)
# Tensor를 출력
print("Initial tensor values:", tensor)

# 예를 들어, 이후에 Tensor의 값을 계속 확인 가능
import time
for _ in range(10):
    print("Updated tensor values:", tensor)
    time.sleep(1)
