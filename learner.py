import torch as torch
import torch.utils.data as D
from model.nets.fnn import FNN
import torch.nn as nn
from model.model import Model


DEVICE = Model.get_device()


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


def learn(dataset, train_ratio, batch_size, num_epochs, model_path):
    train_dataset, test_dataset = D.random_split(dataset, [train_ratio, 1.0 - train_ratio])
    train_dataloader, test_dataloader = D.DataLoader(train_dataset, batch_size=batch_size), \
                                        D.DataLoader(test_dataset, batch_size=batch_size)

    net = FNN(
        len(dataset.vectorizer.vocab),
        dataset.vectorizer.encoding_size,
        dataset.count_classifiers()
    ).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

    for e in range(num_epochs):
        print(f'epoch {e + 1} -')
        train(train_dataloader, net, loss_fn, optimizer)
        test(test_dataloader, net, loss_fn)

    Model(net, dataset.vectorizer).save(model_path)
