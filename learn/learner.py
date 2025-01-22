import torch as torch
import torch.utils.data as D
from learn.neural_network import NeuralNetwork
import torch.nn as nn


BATCH_SIZE = 64
NUM_EPOCHS = 5


def train(dataloader, model, loss_fn, optimizer, device):
    model.train()

    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, model, loss_fn, device):
    model.eval()

    dataset_size, num_batches = len(dataloader.dataset), len(dataloader)
    test_loss, num_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    num_correct /= dataset_size

    print(f'accuracy: {(100 * num_correct):0.1f}%, avg loss: {test_loss:>8f} \n')


def learn(dataset):
    train_dataset, test_dataset = D.random_split(dataset, [0.80, 0.20])

    train_dataloader, test_dataloader = D.DataLoader(train_dataset, batch_size=BATCH_SIZE), \
                                        D.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    device = (
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else
        'cpu'
    )
    model = NeuralNetwork(len(dataset.vocab),
                          dataset.encoding_size).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for e in range(NUM_EPOCHS):
        print(f'epoch {e + 1} -')
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, device)
