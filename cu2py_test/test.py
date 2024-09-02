import torch
import ipc_extension

# PyTorch tensor를 생성
output_tensor = torch.zeros(3, device='cuda')

# FIFO 파일 경로 지정
fifo_path = "testfifo"

# C++ 확장 모듈 호출
ipc_extension.read_ipc_handle(output_tensor, fifo_path)

# 결과 출력
print("Final tensor values:", output_tensor.cpu().numpy())
