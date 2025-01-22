from abc import abstractmethod


class DataAdapter:

    def __init__(self, res_path):
        self.res_path = res_path


    def make(self, num_samples):
        samples = []
        for _ in range(num_samples):
            while True:
                sample = self.sample_line()
                if sample not in samples:
                    break
            samples.append(sample)

        return samples


    @abstractmethod
    def sample_line(self):
        pass
