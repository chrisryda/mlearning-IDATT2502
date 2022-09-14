import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([1.0, 1.0, 1.0, 0.0]).reshape(-1, 1)

class SigmoidModel:

    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
    
    def logits(self, x):
            return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = SigmoidModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent (SGD)
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in tqdm(range(10_000)):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('NAND-operator')
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')

x1_train = x_train[:, 0]
x2_train = x_train[:, 1]

ax.scatter3D(x1_train, x2_train, y_train)

x1_axis = torch.linspace(torch.min(x1_train), torch.max(x1_train), 2)
x2_axis = torch.linspace(torch.min(x2_train), torch.max(x2_train), 2)

meshgrid = torch.meshgrid(x1_axis, x2_axis)
x = meshgrid[0]
y = meshgrid[1]

inputs = torch.stack((x.reshape(-1, 1), y.reshape(-1, 1)), 1).reshape(-1, 2)
z_axis = model.f(inputs).detach()
# print(z_axis)

ax_f = ax.plot_trisurf(
    x.reshape(1, -1)[0],
    y.reshape(1, -1)[0],
    z_axis.reshape(1, -1)[0],
    alpha=0.5,
    color='green'
)

plt.show()