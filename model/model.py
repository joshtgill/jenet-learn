import torch
import pandas as pd


class Model:

    MODEL_FILE_NAME = 'model.pt'

    def __init__(self, net=None, vectorizer=None):
        self.net = net
        self.vectorizer = vectorizer


    @staticmethod
    def get_device():
        return 'cuda' if torch.cuda.is_available() else \
               'mps' if torch.backends.mps.is_available() else \
               'cpu'


    def save(self, res_path):
        torch.save(self, res_path + self.MODEL_FILE_NAME)


    @staticmethod
    def load(res_path):
        return torch.load(res_path + Model.MODEL_FILE_NAME,
                          map_location=Model.get_device(),
                          weights_only=False)


    def query(self, line):
        self.net.eval()

        return torch.argmax(
            self.net(self.vectorizer(pd.DataFrame([line])[0],).unsqueeze(0).to(Model.get_device()))
        ).item()
