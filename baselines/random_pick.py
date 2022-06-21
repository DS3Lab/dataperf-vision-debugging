import numpy as np
from sklearn.model_selection import train_test_split
from appraiser import Appraiser

class RandomAppraiser(Appraiser):
    def __init__(self) -> None:
        super().__init__()
        self.name='random'
        
    def fit(self, train_X, train_y, val_X, val_y):
        self.shape = train_X.shape[0]
        self.sample = np.random.choice(self.shape, self.shape, replace=False)
        

    def propose(self, budget):
        return self.sample[:budget]