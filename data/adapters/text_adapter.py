from data.adapters.base_adapter import BaseAdapter
import random


class TextAdapter(BaseAdapter):

    def __init__(self, *src_paths):
        self.srcs = [[line.strip() for line in open(src_path)] for src_path in src_paths]


    def sample(self, k):
        # Randomly select an item in 2D list
        samples = []
        for i in range(len(self.srcs)):
            samples.extend(
                random.sample(
                    self.srcs[i],
                    min(
                        k // len(self.srcs) + (1 if i < k % len(self.srcs) else 0),
                        len(self.srcs[i])
                    )
                )
            )

        return samples
