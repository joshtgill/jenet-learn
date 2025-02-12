from data.adapters.base_adapter import BaseAdapter
import random


class NumberAdapter(BaseAdapter):

    def __init__(self, min, max):
        self.min = min
        self.max = max


    def sample(self):
        return random.randint(self.min, self.max)
