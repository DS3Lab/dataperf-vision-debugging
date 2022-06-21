import numpy as np
from sklearn import pipeline
from datascope.importance.common import SklearnModelAccuracy
from datascope.importance.shapley import ShapleyImportance
from appraiser import Appraiser
from classifier import XGBClassifier as Classifier

utility_pipeline = pipeline.make_pipeline(
    Classifier()
)

utility = SklearnModelAccuracy(utility_pipeline)

class ShapleyAppraiser(Appraiser):
    def __init__(self, importance_method) -> None:
        self.importance_method = importance_method
        self.name=f'{self.importance_method.replace(" ", "")}_shapley'
        super().__init__()
        
    def fit(self, train_X, train_y, val_X, val_y):
        importance = ShapleyImportance(method=self.importance_method, utility=utility)
        train_y = np.squeeze(train_y)
        val_y = np.squeeze(val_y)
        importances = importance.fit(train_X, train_y).score(val_X, val_y)
        self.importances = (importances).argsort()
        
    def propose(self, budget):
        return self.importances[:budget]