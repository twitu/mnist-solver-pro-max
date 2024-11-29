import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3 * 3, 40),
            nn.ReLU(),
            nn.Linear(40, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return nn.LogSoftmax(dim=1)(x)


use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
model = Net()
summary(model, input_size=(1, 28, 28))
model = Net().to(device)

# torch.manual_seed(1)
batch_size = 32

# kwargs = {"num_workers": 1, "pin_memory": True} if use_mps else {}
kwargs = {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.RandomRotation(5),
                transforms.RandomAffine(5, shear=5),
                # transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
            ]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
    **kwargs,
)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=batch_size,
    shuffle=True,
    **kwargs,
)


def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_description(desc=f"loss={loss.item()} batch_id={batch_idx}")


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


epochs = 20
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    steps_per_epoch=2 * len(train_loader),
    epochs=epochs,
    pct_start=0.3,
    div_factor=25,
    final_div_factor=1000,
    anneal_strategy="cos",
)

for epoch in range(1, epochs):
    train(model, device, train_loader, optimizer, scheduler, epoch)
    test(model, device, test_loader)
