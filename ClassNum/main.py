import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as numers

devise = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MNISY_train = numers.MNIST("./", download=True, train=True)
MNISY_test = numers.MNIST("./", download=True, train=False)

X_train = MNISY_train.train_data
X_test = MNISY_test.train_data
Y_train = MNISY_train.train_labels
Y_test = MNISY_test.train_labels

X_train = X_train.float()
X_test = X_test.float()

X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()


class NeironNet(torch.nn.Module):
    def __init__(self):
        super(NeironNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.ac1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.ac2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(400, 120)
        self.ac3 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(120, 84)
        self.ac4 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.ac2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 400)

        x = self.fc1(x)
        x = self.ac3(x)
        x = self.fc2(x)
        x = self.ac4(x)
        x = self.fc3(x)
        return x


net = NeironNet()
net = net.to(devise)

loss = torch.nn.CrossEntropyLoss()
optimizator = torch.optim.Adam(net.parameters(), lr=1.0e-4)

batch_size = 50

X_test = X_test.to(devise)
Y_test = Y_test.to(devise)

for epoch in range(10000):
    order = np.random.permutation(len(X_train))
    for index in range(0, len(X_train), batch_size):
        optimizator.zero_grad()

        batch_index = order[index:index + batch_size]
        X_batch = X_train[batch_index].to(devise)
        Y_batch = Y_train[batch_index].to(devise)

        preds = net.forward(X_batch)

        loss_value = loss(preds, Y_batch)
        loss_value.backward()

        optimizator.step()

    test_preds = net.forward(X_test)
    accuracy = (test_preds.argmax(dim=1) == Y_test).float().mean()
    print(accuracy)
