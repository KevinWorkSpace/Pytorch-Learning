import torch
import torch.nn as nn
import numpy as np
import torchvision
import torchvision.transforms as transforms

# x = torch.tensor(1., requires_grad = True)
# w = torch.tensor(2., requires_grad=True)
# b = torch.tensor(3., requires_grad=True)
#
# y = w * x + b
# y.backward()
# print(x.grad)
# print(w.grad)
# print(b.grad)

# x = torch.randn(5, 2)
# y = torch.randn(5, 3)
# linear = nn.Linear(2, 3)
# print(linear.weight)
# print(linear.bias)
#
# criterion = nn.MSELoss()
# optim = torch.optim.SGD(linear.parameters(), lr=0.01)
# pred = linear(x)
# loss = criterion(pred, y)
# print(loss.item())
# loss.backward()
# print("dL/dw: ", linear.weight.grad)
# print("dL/db: ", linear.bias.grad)
# optim.step()
#
# pred = linear(x)
# loss = criterion(y, pred)
# print("loss after one iteration: ", loss.item())

# x = np.array([[1, 2], [2, 3]])
# y = torch.from_numpy(x)
# print(y)
# z = y.numpy()
# print(z)

# train_dataset = torchvision.datasets.CIFAR10(root="../data", train=True, transform=transforms.ToTensor(), download=True)
# image, label = train_dataset[0]
# print(image.size())
# print(label)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=64,
#                                            shuffle=True)
# data_iter = iter(train_loader)
# images, labels = data_iter.next()
# for images, labels in train_loader:
#     pass

# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self):
#         # TODO
#         # 1. Initialize file paths or a list of file names.
#         pass
#
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         pass
#
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         return 0
# customDataset = CustomDataset()
# train_loader2 = torch.utils.data.DataLoader(dataset=customDataset, batch_size=64, shuffle=True)

resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False
resnet.fc = nn.Linear(resnet.fc.in_features, 10)
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print(outputs.size())