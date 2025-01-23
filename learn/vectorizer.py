import torch
import torch.nn.functional as F


class Vectorizer(object):

    def __call__(self, line, vocab, encoding_size):
        # Make a one-hot vector for each character in a string
        one_hot = F.one_hot(
            torch.tensor([vocab.get(c) for c in line]),
            num_classes=len(vocab)
        )

        # Pad one-hot to largest string length
        return F.pad(
            one_hot,
            (0, 0, 0, encoding_size - len(one_hot)),
            mode='constant',
            value=0
        ).to(torch.float32)
