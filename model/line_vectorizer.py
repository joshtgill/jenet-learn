import torch
import torch.nn.functional as F


class LineVectorizer():

    def __init__(self, vocab, encoding_size):
        self.vocab = vocab
        self.encoding_size = encoding_size


    def __call__(self, x):
        # only 1 x column in dataframe
        x = x.iloc[0]

        tokens = torch.tensor([self.vocab.get(c) for c in x])
        return F.pad(tokens, (0, self.encoding_size - len(tokens)), value=0)
