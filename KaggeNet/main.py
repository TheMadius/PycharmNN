import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms, models
from tqdm import tqdm

def printIm(train_dataset):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    for im, _ in train_dataset:
        plt.imshow(im.permute(1, 2, 0) * std + mean)
        plt.show()


train_dir = "train"
val_dir = "val"
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

batch_size = 8
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)

class NeironNet(torch.nn.Module):
    def __init__(self):
        super(NeironNet, self).__init__()
        self.conv1_1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv1_2 = torch.nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1)
        self.ac1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=6)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=0)
        self.conv2_2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.ac2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=16)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(46656, 120)
        self.ac3 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(120, 84)
        self.ac4 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(84, 2)

    def forward(self, x):
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.ac1(x)
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.ac2(x)
        x = self.bn2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), 46656)

        x = self.fc1(x)
        x = self.ac3(x)
        x = self.fc2(x)
        x = self.ac4(x)
        x = self.fc3(x)
        return x

net = NeironNet()
loss = torch.nn.CrossEntropyLoss()
optimizator = torch.optim.Adam(net.parameters(), lr=1.0e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizator, step_size = 7, gamma=0.1)


for epoch in range(100):
    for X_batch, Y_batch in tqdm(train_dataset):
        X_batch = X_batch.unsqueeze(0).float()
        optimizator.zero_grad()
        net.train()

        preds = net.forward(X_batch)

        loss_value = loss(preds, Y_batch)
        loss_value.backward()

        optimizator.step()
    net.eval()
    accuracy = 0.0
    count = 0
    for X_test, Y_test in tqdm(val_dataset):
        X_batch = X_batch.unsqueeze(0).float()
        test_preds = net.forward(X_test)
        accuracy += (test_preds.argmax(dim=1) == Y_test).float().mean()
        ++count
    print(accuracy)
