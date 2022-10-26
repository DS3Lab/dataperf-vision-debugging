import torch
from appraiser import Appraiser
# we use LogisticClassifier here as a proxy
from classifier import LogisticClassifier as Classifier
from baselines.ptif.calc_influence_function import calc_img_wise
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class InfluenceFunctionAppraiser(Appraiser):
    def __init__(self) -> None:
        super().__init__()
        self.name='influence_function'

    def create_dataloader(self, X, y):
        X_pt = torch.tensor(X)
        y_pt = torch.tensor(y)
        dataset = TensorDataset(X_pt, y_pt)
        return DataLoader(dataset, batch_size=32, shuffle=False)

    def fit(self, train_X, train_y, val_X, val_y):
        self.clf = Classifier()
        self.clf.fit(train_X, train_y)
        self.model = self.clf.model
        self.train_loader = self.create_dataloader(train_X, train_y)
        self.test_loader = self.create_dataloader(val_X, val_y)
        config = {
            'outdir': 'outdir',
            'seed': 42,
            'gpu': -1,
            'num_classes': 2,
            'test_sample_num': 40,
            'test_start_index': 0,
            'recursion_depth': 200,
            'r_averaging': 50,
            'scale': None,
            'damp': None,
            'calc_method': 'img_wise',
            'log_filename': None,
        }
        
        influences, harmful, helpful = calc_img_wise(config, self.model, self.train_loader, self.test_loader)
        print("Harmfulness: {}".format(harmful))
        print("Helpfulness: {}".format(helpful))
        self.harmfulness = helpful
    
    def propose(self, budget):
        return self.harmfulness[:budget]
