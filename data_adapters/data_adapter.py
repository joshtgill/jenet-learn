from abc import abstractmethod


class DataAdapter:

    def make(self, num_samples):
        samples = []
        for _ in range(num_samples):
            while True:
                sample = self.sample()
                if sample not in samples:
                    break
            samples.append(sample)

        return samples


    @abstractmethod
    def sample(self):
        pass
