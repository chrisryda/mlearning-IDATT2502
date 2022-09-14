import torch
import matplotlib.pyplot as plt
from CSVLoader import load_csv
from tqdm import tqdm

training_data = load_csv("./day_head_circumference.csv")

day_data = torch.reshape(training_data[:, 0], (-1, 1))
head_circumference_data = torch.reshape(training_data[:, 1], (-1, 1))

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return 20 * torch.sigmoid((x @ self.W + self.b)) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD([model.W, model.b], 0.000001)
for epoch in tqdm(range(10_000)):
    model.loss(day_data, head_circumference_data).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(day_data, head_circumference_data)))

# Visualize result
plt.plot(day_data, head_circumference_data, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Age (days)')
plt.ylabel('head circumference')

x = torch.tensor(torch.linspace(torch.min(day_data), torch.max(day_data), 100)).reshape(-1, 1)
plt.plot(x.detach(), model.f(x).detach(), label='$\\hat y = f(x) = 20\\sigma(xW+b) + 31$')

plt.legend()
plt.show()
