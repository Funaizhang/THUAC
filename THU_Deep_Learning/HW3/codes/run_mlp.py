import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from layer import BatchNorm1d
from utils import LOG_INFO, plot_loss_acc
import time

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 100
LOG_INTERVAL = 100
EPOCHS = 50

# TODO: adjust these hyperparameters
LR = 0.01
MM = 0.9
WD = 0.0

device = 'cpu'
kwargs = {'num_workers': 1, 'pin_memory': True}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
       transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=TRAIN_BATCH_SIZE, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False,
        transform=transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
       ])),
    batch_size=TEST_BATCH_SIZE, shuffle=False, **kwargs)

# TODO: implement your network architecture
# MLP (Linear-BN-ReLU-Linear-BN-ReLU-Linear)
model = nn.Sequential(
        nn.Linear(784, 512),
#        BatchNorm1d(512), # c = 512
        nn.ReLU(),
        nn.Linear(512, 128),
#        BatchNorm1d(128), # c = 128
        nn.ReLU(),
        nn.Linear(128, 10) # output shape: N x 10
).to(device)

optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MM, weight_decay=WD)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_list = []
    acc_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)  # flatten
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        acc = pred.eq(target.view_as(pred)).float().mean()
        acc_list.append(acc.item())
        if batch_idx % LOG_INTERVAL == 0:
            train_loss = np.mean(loss_list)
            train_acc = np.mean(acc_list)
            msg = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tAvg Loss: {:.4f}\tAvg Acc: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), np.mean(loss_list), np.mean(acc_list))
            LOG_INFO(msg)
            loss_list.clear()
            acc_list.clear()
    
    return train_loss, train_acc

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = model(data)
            test_loss += F.cross_entropy(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_acc))

    return test_loss, test_acc


# actual execution
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader)

    # save train & test loss & accuracy 
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    test_loss_list.append(test_loss)
    test_acc_list.append(test_acc)

end_time = time.time()

plot_loss_acc(EPOCHS, train_loss_list, test_loss_list, 'MLP', 'without BN', loss=True)
plot_loss_acc(EPOCHS, train_acc_list, test_acc_list, 'MLP', 'without BN', loss=False)
print('Time per epoch = {:.2f}s'.format((end_time-start_time)/EPOCHS))
# with BN: 12.83s per epoch
# without BN: 13.20s per epoch