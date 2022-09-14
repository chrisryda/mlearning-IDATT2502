import torch
import matplotlib.pyplot as plt
import numpy as np
from CSVLoader import load_csv
from tqdm import tqdm
        
training_data = load_csv("./day_length_weight.csv")
length__weigth_data = training_data[:, 1:]

day_data = torch.reshape(training_data[:, 0], (-1, 1))
length_data = torch.reshape(training_data[:, 1], (-1, 1))
weight_data = torch.reshape(training_data[:, 2], (-1, 1))

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in tqdm(range(10_000)):
    model.loss(length__weigth_data, day_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(length__weigth_data, day_data)))

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Length')
ax.set_ylabel('Weight')
ax.set_zlabel('Age (days)')

ax.scatter3D(length_data, weight_data, day_data)

x_axis = torch.linspace(torch.min(length_data), torch.max(length_data), 2)
y_axis = torch.linspace(torch.min(weight_data), torch.max(weight_data), 2)

x,y = torch.meshgrid(x_axis, y_axis)

inputs = torch.stack((x.reshape(-1, 1), y.reshape(-1, 1)), 1).reshape(-1, 2)

z_axis = model.f(inputs).detach()

ax_f = ax.plot_trisurf(
    x.reshape(1, -1)[0],
    y.reshape(1, -1)[0],
    z_axis.reshape(1, -1)[0],
    alpha=0.5,
    color='green'
)

plt.show()
