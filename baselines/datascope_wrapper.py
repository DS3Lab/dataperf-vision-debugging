import numpy as np
from sklearn import pipeline
from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance, ImportanceMethod
from appraiser import Appraiser
from classifier import LogisticClassifier as Classifier

utility_pipeline = pipeline.make_pipeline(
    Classifier()
)

utility = SklearnModelAccuracy(utility_pipeline)

class ShapleyAppraiser(Appraiser):
    def __init__(self, importance_method) -> None:
        self.importance_method = importance_method
        self.name=""
        if self.importance_method == ImportanceMethod.MONTECARLO:
            self.name='mc_shapley'
        elif self.importance_method == ImportanceMethod.BRUTEFORCE:
            self.name='bruteforce_shapley'
        elif self.importance_method == ImportanceMethod.NEIGHBOR:
            self.name='neighbor_shapley (datascope)'
        else:
            raise ValueError(f"Unknown algorithm {self.importance_method}")
        super().__init__()
        
    def fit(self, train_X, train_y, val_X, val_y):
        if self.importance_method == ImportanceMethod.MONTECARLO:
            importance = ShapleyImportance(
                    method=self.importance_method,
                    utility=utility,
                    mc_iterations=100,
                )
        else:
            importance = ShapleyImportance(method=self.importance_method, utility=utility)
        train_y = np.squeeze(train_y)
        val_y = np.squeeze(val_y)
        importances = importance.fit(train_X, train_y).score(val_X, val_y)
        self.importances = (importances).argsort()
        
    def propose(self, budget):
        return self.importances[:budget]