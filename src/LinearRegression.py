import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

input_size = 1
output_size = 1
learning_rate = 0.01
epochs = 60

X_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
Y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)
model = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    x = torch.from_numpy(X_train)
    y = torch.from_numpy(Y_train)
    output = model(x)
    loss = criterion(y, output)
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (epoch + 1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(X_train)).detach().numpy()
plt.plot(X_train, Y_train, 'ro', label='Original data')
plt.plot(X_train, predicted, label='Fitted line')
plt.legend()
plt.show()

# # Save the model checkpoint
# torch.save(model.state_dict(), 'model.ckpt')