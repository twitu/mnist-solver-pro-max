import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from torchsummary import summary
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # First block
        self.conv1 = nn.Conv2d(1, 10, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv1x1_1 = nn.Conv2d(20, 10, 1)

        # Second block
        self.conv3 = nn.Conv2d(10, 20, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(20)
        self.conv4 = nn.Conv2d(20, 20, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(20)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Final block
        self.conv5 = nn.Conv2d(20, 40, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(40)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(40, 10, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout(x)
        x = F.relu(self.conv1x1_1(x))

        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        # Final block
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.gap(x)
        x = self.fc(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=1)

use_mps = torch.backends.mps.is_available()
device = torch.device("mps" if use_mps else "cpu")
model = Net()
summary(model, input_size=(1, 28, 28))
model = Net().to(device)

torch.manual_seed(1)
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

def visualize_misclassification(mode, test_loader, device, num_to_show=10):
    model.eval()
    mis_images = []
    mis_labels = []
    mis_predis = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            pred = output.argmax(dim=1, keepdim=True)
            wrong_mask = pred.squeeze() != target

            if wrong_mask.sum() > 0:
                incorrect_images = data[wrong_mask]
                incorrect_targets = target[wrong_mask]
                incorrect_preds = pred[wrong_mask]

                for img, true_label, pred_label in zip(incorrect_images, incorrect_targets, incorrect_preds):
                    mis_images.append(img.cpu())
                    mis_labels.append(true_label.cpu())
                    mis_predis.append(pred_label.cpu())

                if len(mis_images) >= num_to_show:
                    break

    plt.figure(figsize=(15, 3))
    for i in range(min(num_to_show, len(mis_images))):
        plt.subplot(1, num_to_show, i + 1)
        img = mis_images[i].squeeze().numpy()
        plt.imshow(img, cmap="gray")
        plt.title(
            f"True: {mis_labels[i].item()}\nPred: {mis_predis[i].item()}\n",
            fontsize=8
        )
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def test(model, device, test_loader, visualize=True):
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

    if visualize:
        visualize_misclassification(model, test_loader, device)


epochs = 20
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,              # Maximum learning rate
    epochs=20,                # Total number of epochs
    steps_per_epoch=len(train_loader),  # Number of steps in each epoch
    pct_start=0.2,           # Percentage of training to increase lr
    div_factor=10.0,         # Initial lr = max_lr/div_factor
    three_phase=False,       # Use three phase learning
    final_div_factor=100,    # Min lr = initial_lr/final_div_factor
)

for epoch in range(epochs):
    train(model, device, train_loader, optimizer, scheduler, epoch)
    test(model, device, test_loader, False)
