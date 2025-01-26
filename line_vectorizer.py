import torch
import torch.nn.functional as F


class LineVectorizer():

    def __init__(self, vocab, encoding_size):
        self.vocab = vocab
        self.encoding_size = encoding_size


    def __call__(self, x):
        x = x.iloc[0]

        # Make a one-hot vector for each character in a string
        one_hot = F.one_hot(
            torch.tensor([self.vocab.get(c) for c in x]),
            num_classes=len(self.vocab)
        )

        # Pad one-hot to largest string length
        return F.pad(
            one_hot,
            (0, 0, 0, self.encoding_size - len(one_hot)),
            mode='constant',
            value=0
        ).to(torch.float32)
