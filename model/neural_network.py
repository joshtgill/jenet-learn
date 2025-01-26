from torch import nn


class NeuralNetwork(nn.Module):

    EMBEDDING_SIZE = 50

    def __init__(self, vocab_size, encoding_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, self.EMBEDDING_SIZE)
        self.flatten = nn.Flatten()
        self.stack = nn.Sequential(
            nn.Linear(encoding_size * self.EMBEDDING_SIZE, 3),
        )


    def forward(self, x):
        return self.stack(self.flatten(self.embedding(x)))
