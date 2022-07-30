import numpy as np
from sklearn.model_selection import train_test_split
from appraiser import Appraiser

class CustomAppraiser(Appraiser):
    def __init__(self) -> None:
        super().__init__()
        self.name='my_debug'
        
    def fit(self, train_X, train_y, val_X, val_y):
        """
        Fit your appraiser with the noisy training and validation data.
        """
        self.shape = train_X.shape[0]
        self.sample = np.random.choice(self.shape, self.shape, replace=False)
        

    def propose(self, budget):
        """
        Return the indices to the first :budget samples, which you want the benchmark to fix. 
        """
        return self.sample[:budget]