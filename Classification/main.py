import torch
import numpy as np
import sklearn.datasets
from sklearn.model_selection import train_test_split

wine = sklearn.datasets.load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, :2],
    wine.target,
    test_size=0.3,
    shuffle=True)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


class WineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(WineNet, self).__init__()

        self.fc1 = torch.nn.Linear(2, n_hidden_neurons)
        self.activ1 = torch.nn.Sigmoid()
       # self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
       #self.activ2 = torch.nn.Sigmoid()
        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
       # x = self.fc2(x)
       # x = self.activ2(x)
        x = self.fc3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


wine_net = WineNet(5)

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(wine_net.parameters(),
                             lr=1.0e-3)

batch_size = 10

for epoch in range(10000):
    order = np.random.permutation(len(X_train))
    for index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()
        batch_index = order[index:index + batch_size]

        x_batch = X_train[batch_index]
        y_batch = y_train[batch_index]

        preds = wine_net.forward(x_batch)

        loss_val = loss(preds, y_batch)
        loss_val.backward()

        optimizer.step()

    if epoch % 100 == 0:
        test_preds = wine_net.forward(X_test)
        test_preds = test_preds.argmax(dim=1)
        print((test_preds == y_test).float().mean())
