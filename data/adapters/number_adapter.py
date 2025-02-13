from data.adapters.base_adapter import BaseAdapter
import string
import random


class NumberAdapter(BaseAdapter):

    def __init__(self, min_length=0, max_length=9, contain_digits=True, contain_letters=False, prefix=''):
        self.min_length = min_length
        self.max_length = max_length
        self.prefix = prefix

        self.srcs = ''
        if contain_digits:
            self.srcs += string.digits
        if contain_letters:
            self.srcs += string.ascii_letters


    def sample(self):
        number = ''.join([random.choice(self.srcs)])
        for _ in range(random.randint(self.min_length, self.max_length)):
            number += random.choice(self.srcs)

        return f'{self.prefix}{number}'
