from torch import nn


class NeuralNetwork(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super().__init__()

        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(vocab_size * embedding_size, 3),
        )


    def forward(self, x):
        return self.stack(self.flatten(x))
