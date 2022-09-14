import torch
import matplotlib.pyplot as plt
from CSVLoader import load_csv
from tqdm import tqdm

training_data = load_csv("./length_weight.csv")
length_data = torch.reshape(training_data[:, 0], (-1, 1))
weight_data = torch.reshape(training_data[:, 1], (-1, 1))


class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b 

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in tqdm(range(5000)):
    model.loss(length_data, weight_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(length_data, weight_data)))

# Visualize result
plt.plot(length_data, weight_data, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('length')
plt.ylabel('weight')

x = torch.tensor([[torch.min(length_data)], [torch.max(length_data)]])
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = xW+b$')
plt.legend()
plt.show()
