import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import numpy as np

batch_size = 100
epochs = 5
input_size = 28 * 28
num_class = 10
learning_rate = 0.01

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../data',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
criterion = nn.CrossEntropyLoss()
model = nn.Linear(input_size, num_class)
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)
        pred = model(images)
        loss = criterion(pred, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, epochs, i + 1, total_step, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        preds_prob = model(images)
        _, preds = torch.max(preds_prob.data, 1)
        correct += (preds == labels).sum()
        total += batch_size
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))




