import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import relu, max_pool2d
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from tqdm import tqdm


class MLPModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        out1 = nn.functional.relu(self.fc1(x))
        out2 = self.fc2(out1)
        return out2


class CNNModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, padding=1)
        self.drop1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(8 * 3 * 3, 36)
        self.drop2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(36, 10)

    def forward(self, x):
        out = max_pool2d(relu(self.conv1(x)), kernel_size=2)
        out = max_pool2d(relu(self.conv2(out)), kernel_size=2)
        out = max_pool2d(relu(self.conv3(out)), kernel_size=2)
        out = out.view(out.shape[0], -1)
        out = relu(self.fc1(self.drop1(out)))
        out = self.fc2(self.drop2(out))
        return out


class Lenet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.LazyLinear(84),
            nn.ReLU(),
            nn.LazyLinear(10)
        )

    def forward(self, x):
        return self.net(x)


def train_epoch(train_dataloader, model, optimizer):
    total = 0
    total_correct = 0
    pbar = tqdm(train_dataloader)
    for bx, by in pbar:
        bx, by = bx.to(device), by.to(device)
        # print("bx.shape", bx.shape, "by.shape", by.shape)
        out = model(bx)
        # print("out.shape", out.shape)
        loss = nn.functional.cross_entropy(out, by)
        pred = torch.argmax(out, dim=1)
        # print("loss", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # break

        total_correct += (pred == by).sum().item()
        total += len(by)
        pbar.set_description(f"acc {total_correct / total:.4f}")


def test_epoch(test_dataloader, model):
    total = 0
    total_correct = 0
    pbar = tqdm(test_dataloader)
    for bx, by in pbar:
        bx, by = bx.to(device), by.to(device)
        out = model(bx)
        pred = torch.argmax(out, dim=1)

        total_correct += (pred == by).sum().item()
        total += len(by)
        pbar.set_description(f"test acc {total_correct / total:.4f}")


if __name__ == "__main__":
    transform = T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_dataloader = DataLoader(mnist_train, batch_size=16, shuffle=True)

    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_dataloader = DataLoader(mnist_test, batch_size=16, shuffle=False)

    n_epoch = 10
    device = 1
    # model = MLPModule().to(device)
    # model = CNNModule().to(device)
    model = Lenet5().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(n_epoch):
        train_epoch(train_dataloader, model, optimizer)
        test_epoch(test_dataloader, model)
