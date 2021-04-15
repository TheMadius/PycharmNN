import torch
import matplotlib.pyplot as plt

devise = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SinNet(torch.nn.Module):
    def __init__(self, n_hidder_neurons):
        super(SinNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, n_hidder_neurons)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(n_hidder_neurons, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x


def loss(pred, target):
    return ((pred - target)**2).mean()


def predict(net, x, y):
    plt.plot(x.numpy(), y.numpy(), 'o', label='Res')
    plt.plot(x.numpy(), (net.forward(x)).data.numpy(), 'o', label='Pres')
    plt.title('$y= sin(x)$')
    plt.show()


x_train = torch.rand(100) * 20 - 10
y_train = torch.sin(x_train)

y_train = y_train + torch.randn(y_train.shape) / 5
print(x_train.shape)
x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_train = x_train.to(devise)
y_train = y_train.to(devise)

print(x_train.shape)
x_val = torch.linspace(-10, 10, 100)
y_val = torch.sin(x_val)

x_val.unsqueeze_(1)
y_val.unsqueeze_(1)

x_val = x_train.to(devise)
y_val = y_train.to(devise)

sine_net = SinNet(50)
sine_net = sine_net.to(devise)

optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.001)

for _ in range(20000):
    optimizer.zero_grad()
    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)
    loss_val.backward()
    optimizer.step()

y_pred = sine_net.forward(x_train)
print(loss(y_pred, y_train))
predict(sine_net, x_val, y_val)

