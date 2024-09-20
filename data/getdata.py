# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torchvision
# import torchvision.transforms as transforms
# import torch.backends.cudnn as cudnn
# import torch.utils.data as data
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])
# testset=torchvision.datasets.Imagenette(root='./',split='val', size='320px', download=False, transform=transform_test)
# testloader = data.DataLoader(
#         testset, batch_size=1, shuffle=False)

# for i, (x, y) in enumerate(testloader):
#     x, y = x.to('cuda:0'), y.to('cuda:0')
#     print(y[0].size(), x[0].size())
import torch
from PIL import Image
import torchvision
from torchvision import transforms
import torch.utils.data as data
import numpy as np
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
# or any of these variants
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

# input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# _testset=torchvision.datasets.Imagenette(root='./',split='val', size='320px', download=False)
# _testloader = data.DataLoader(
#         _testset, batch_size=1, shuffle=False)

# for i, (input_batch, y) in enumerate(_testset):
#         img_np = np.array(input_batch)
#         print(img_np, img_np.shape)
#         break
testset=torchvision.datasets.Imagenette(root='./',split='val', size='320px', download=False, transform=preprocess)
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
testloader = data.DataLoader(
        testset, batch_size=100, shuffle=False)
# move the input and model to GPU for speed if available
for i, (input_batch, y) in enumerate(testloader):
    # print(input_batch[0], input_batch[0].shape)
    # break
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
    # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
    #print(output[0])
    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    #print(probabilities)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 1)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item(), top5_catid[i])