from abc import abstractmethod


class BaseAdapter:

    def make(self, num_samples):
        MAX_ATTEMPS = 10
        samples = []
        for _ in range(num_samples):
            num_attempts = 0
            while True:
                sample, num_attempts = self.sample(), num_attempts + 1
                if sample not in samples or num_attempts == MAX_ATTEMPS:
                    # Only add sample if repeating
                    break
            samples.append(sample)

        return samples


    @abstractmethod
    def sample(self):
        pass
