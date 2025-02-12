from data.adapters.base_adapter import BaseAdapter
import random


class TextAdapter(BaseAdapter):

    def __init__(self, *src_paths):
        self.srcs = [[line.strip() for line in open(src_path)] for src_path in src_paths]


    def sample(self):
        # Randomly select an item in 2D list
        return random.choice(random.choice(self.srcs))
