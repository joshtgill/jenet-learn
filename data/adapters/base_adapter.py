from abc import abstractmethod


class BaseAdapter:

    @abstractmethod
    def sample(self, k):
        pass
