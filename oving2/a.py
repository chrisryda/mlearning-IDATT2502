import torch
import matplotlib.pyplot as plt
from tqdm import tqdm 

x_train = torch.tensor([0.0, 1.0]).reshape(-1,1)
y_train = torch.tensor([1.0, 0.0]).reshape(-1,1)

class SigmoidModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
            return x @ self.W + self.b
    
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

model = SigmoidModel()

optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in tqdm(range(10_000)):
    model.loss(x_train, y_train).backward()
    optimizer.step()

    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.title('NOT-operator')
plt.xlabel('x')
plt.ylabel('y')

x = torch.tensor([torch.min(x_train), torch.max(x_train)]).reshape(-1, 1)
plt.plot(x.detach(), model.f(x).detach(), label='$\\hat y = f(x) = \\sigma(xW+b)$')
plt.legend()
plt.show()