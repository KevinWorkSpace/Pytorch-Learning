import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

batch_size = 100
input_size = 28 * 28
hidden_size = 500
num_class = 10
epochs = 1
learning_rate = 0.01


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


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


neuralNet = NeuralNet(input_size, hidden_size, num_class)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(neuralNet.parameters(), lr=learning_rate)
total_step = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size)
        preds = neuralNet(images)
        loss = criterion(preds, labels)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (i+1) % 100 == 0:
            print("[{}/{}]epochs, [{}/{}]step, loss: {:.4f}".format(epoch+1, epochs, i+1, total_step, loss.item()))

correct = 0
with torch.no_grad:
    for images, labels in test_loader:
        images = images.reshape(-1, input_size)
        preds = neuralNet(images)
        _, predicted = torch.max(preds, 1)
        print(_)
        correct += (predicted == labels).sum()
    print("Accuracy: {:.4f}".format(correct/len(test_loader)))