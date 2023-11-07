import io
import pickle
from typing import IO

import torch


class Unpickler(pickle.Unpickler):
    def __init__(self, file: IO[bytes], auto_cuda: bool = True):
        super().__init__(file)
        self.__auto_cuda: bool = auto_cuda

    @property
    def auto_cuda(self):
        return self.__auto_cuda

    @property
    def device(self) -> str:
        return torch.device('cuda' if self.auto_cuda and torch.cuda.is_available() else 'cpu')

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        else:
            return super().find_class(module, name)
