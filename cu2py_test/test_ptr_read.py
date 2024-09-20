import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ipc_extension
import os
torch.backends.cudnn.benchmark = True
fifo_path = "testfifo"
# GPU 메모리를 가리키는 Tensor를 반환하는 함수를 호출

#print(tensor.device)
# Tensor를 출력
#print("Initial tensor values:", tensor)

# 예를 들어, 이후에 Tensor의 값을 계속 확인 가능
import time
# os.environ['CUDA_LAUNCH_BLOCKING']="1"
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

# time.sleep(1)


transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)), #mnist
])
trainset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_train)

trainloader = DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=2)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.classes =  num_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device=torch.device("cuda") 
net=SimpleCNN().to(device)
optimizer=nn.CrossEntropyLoss()
criterion = optim.SGD(net.parameters(),lr=0.01,momentum=0.9, weight_decay=5e-4)
def train_and_save(net, trainloader, device):
    net.train()

    for i in range(20):
        train_loss = 0
        correct = 0
        total = 0
        batch_idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            try:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            
                torch.autograd.set_detect_anomaly(True) # type: ignore

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            except Exception as E:
                print('ERROR! : ',E)
            
        print("train loss : %0.4f  train acc : %0.2f" %(train_loss/(batch_idx+1),100.*correct/total))

    net.eval()

    torch.save({
    'model':net.state_dict()
    }, "./model/cnn.pt")

#train_and_save()

# net=SimpleCNN()

checkpoint= torch.load("./model/cnn.pt", weights_only=True)
net.load_state_dict(checkpoint['model'])
net.eval()
torch.cuda.empty_cache()
import torch
import ctypes
from torch.cuda import current_device, Event
net.to(device)

tensor = ipc_extension.read_and_return_tensor_ptr(fifo_path)

# net.to(device)
t=transforms.Normalize((0.5), (0.5))
stop = torch.ones(1)*-1
stop = stop.to(device="cuda:0")


ipc_extension.polling_input()
#torch.cuda.synchronize(device)
_, outputs = net(torch.reshape((tensor-0.5)/0.5, (1,1,28,28))).max(1)
#torch.cuda.synchronize(device)
ipc_extension.set_tail()
torch.cuda.synchronize(device)
time.sleep(0.01)
# print(outputs)
# time.sleep(0.01)

ipc_extension.polling_input()
#torch.cuda.synchronize(device)
_, outputs = net(torch.reshape((tensor-0.5)/0.5, (1,1,28,28))).max(1)
#print(outputs)
#torch.cuda.synchronize(device)
ipc_extension.set_tail()
torch.cuda.synchronize(device)
time.sleep(0.01)
# print(outputs)
# time.sleep(0.01)

ipc_extension.polling_input()
#torch.cuda.synchronize(device)
_, outputs = net(torch.reshape((tensor-0.5)/0.5, (1,1,28,28))).max(1)
#print(outputs)
#torch.cuda.synchronize(device)
ipc_extension.set_tail()
torch.cuda.synchronize(device)
time.sleep(0.01)
# print(outputs)
# time.sleep(0.01)

ipc_extension.polling_input()
#torch.cuda.synchronize(device)
_, outputs = net(torch.reshape((tensor-0.5)/0.5, (1,1,28,28))).max(1)
#print(outputs)
#torch.cuda.synchronize(device)
ipc_extension.set_tail()
torch.cuda.synchronize(device)
time.sleep(0.01)
# print(outputs)
# time.sleep(0.01)

ipc_extension.polling_input()
#torch.cuda.synchronize(device)
_, outputs = net(torch.reshape((tensor-0.5)/0.5, (1,1,28,28))).max(1)
#print(outputs)
#torch.cuda.synchronize(device)
ipc_extension.set_tail()
torch.cuda.synchronize(device)
time.sleep(0.01)
# print(outputs)
# time.sleep(0.01)

ipc_extension.polling_input()
#torch.cuda.synchronize(device)
_, outputs = net(torch.reshape((tensor-0.5)/0.5, (1,1,28,28))).max(1)
#print(outputs)
#torch.cuda.synchronize(device)
ipc_extension.set_tail()
torch.cuda.synchronize(device)
time.sleep(0.01)
# print(outputs)
# time.sleep(0.01)
