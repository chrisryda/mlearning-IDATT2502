import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

EPOCHS = 100_000
LEARNING_RATE = 0.1

x_train = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).reshape(-1, 2)
y_train = torch.tensor([0.0, 1.0, 1.0, 0.0]).reshape(-1, 1)

class NeuralXOR:
    def __init__(self):
        self.W_1 = torch.tensor([[0.37, -0.37], [0.37, -0.37]], requires_grad=True)
        self.W_2 = torch.tensor([[0.37], [0.37]], requires_grad=True)
        self.b_1 = torch.tensor([[-0.17]], requires_grad=True)
        self.b_2 = torch.tensor([[-0.17]], requires_grad=True)

    def _logits_1(self, x):
        return x @ self.W_1 + self.b_1

    def _logits_2(self, x):
        return self._h(x) @ self.W_2 + self.b_2

    def _h(self, x):
        return torch.sigmoid(self._logits_1(x))

    def f(self, x):
        return torch.sigmoid(self._logits_2(x))

    def loss_f(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self._logits_2(x), y)

model = NeuralXOR()

optimizer = torch.optim.SGD([model.W_1, model.b_1, model.W_2, model.b_2], LEARNING_RATE)
for epoch in tqdm(range(EPOCHS)):
    model.loss_f(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

# print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss_f(train_x, train_y)))

print(model.f(torch.tensor([0.0, 0.0])))
print(model.f(torch.tensor([1.0, 0.0])))
print(model.f(torch.tensor([0.0, 1.0])))
print(model.f(torch.tensor([1.0, 1.0])))

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('XOR-operator')
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

ax_f = ax.plot_trisurf(
    x.reshape(1, -1)[0],
    y.reshape(1, -1)[0],
    z_axis.reshape(1, -1)[0],
    alpha=0.5,
    color='green'
)

plt.show()