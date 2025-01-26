import torch as torch
import torch.utils.data as D
from model.neural_network import NeuralNetwork
import torch.nn as nn
from model.model import Model


DEVICE = Model.get_device()
BATCH_SIZE = 64
NUM_EPOCHS = 5
MODEL_FILE_NAME = 'model.pt'


def train(dataloader, net, loss_fn, optimizer):
    net.train()

    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        pred = net(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test(dataloader, net, loss_fn):
    net.eval()

    dataset_size, num_batches = len(dataloader.dataset), len(dataloader)
    test_loss, num_correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)

            pred = net(X)
            test_loss += loss_fn(pred, y).item()
            num_correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    num_correct /= dataset_size

    print(f'accuracy: {(100 * num_correct):0.1f}%, avg loss: {test_loss:>8f} \n')


def learn(dataset, res_path):
    train_dataset, test_dataset = D.random_split(dataset, [0.80, 0.20])
    train_dataloader, test_dataloader = D.DataLoader(train_dataset, batch_size=BATCH_SIZE), \
                                        D.DataLoader(test_dataset, batch_size=BATCH_SIZE)

    net = NeuralNetwork(
        len(dataset.vectorizer.vocab),
        dataset.vectorizer.encoding_size
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    for e in range(NUM_EPOCHS):
        print(f'epoch {e + 1} -')
        train(train_dataloader, net, loss_fn, optimizer)
        test(test_dataloader, net, loss_fn)

    Model(net, dataset.vectorizer).save(res_path)
