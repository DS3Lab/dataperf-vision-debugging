import numpy as np
from pyarrow import parquet as pq
import torch
import xgboost
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class XGBClassifier():
    def __init__(self):
        self.xgb = xgboost.XGBClassifier()

    def fit(self, features, labels):
        self.xgb.fit(features, labels)
    
    def predict(self, feature):
        return self.xgb.predict(feature)

    def evaluate(self, test_features, test_labels):
        test_labels = test_labels.ravel()
        pred = self.predict(test_features)
        accuracy = (pred == test_labels).mean() 
        return round(accuracy, 2)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        
        # First fully connected layer
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class LinearClassifier():
    def __init__(self):
        torch.manual_seed(42)
        self.model = LinearModel()
    
    def fit(self, features, labels, epochs=50, lr=0.01, batch_size=8):
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=0.9)
        self.model.train(True)
        for epoch in range(epochs):
            for x, y in dataloader:
                y_pred = self.model(x.float())
                loss = criterion(y_pred, y.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

    def predict(self, feature):
        self.model.eval()
        feature = torch.tensor(feature)
        with torch.no_grad():
            output = self.model(feature)
        return output

    def evaluate(self, test_features, test_labels):
        test_labels = test_labels.ravel()
        pred = self.predict(test_features)
        pred = torch.sigmoid(pred)
        pred = np.where(pred <= 0.5, 0, pred)
        pred = np.where(pred > 0.5, 1, pred)
        accuracy = (pred == test_labels).mean()
        return round(accuracy, 2)

if __name__=="__main__":
    train = pq.read_table("embeddings/09j2d_train_0.3_200.parquet")
    test = pq.read_table("embeddings/09j2d_test.parquet")
    train_X, train_y = np.vstack(train.column("encoding").to_numpy()), np.vstack(train.column("label").to_numpy())
    test_X, test_y = np.vstack(test.column("encoding").to_numpy()), np.vstack(test.column("label").to_numpy())
    clf = XGBClassifier()
    clf.fit(train_X, train_y)
    linear_clf = LinearClassifier()
    linear_clf.fit(train_X, train_y)
    linear_acc = linear_clf.evaluate(test_X, test_y)
    xgboost_acc = clf.evaluate(test_X, test_y)
    print("Linear Classifier Accuracy: {}".format(linear_acc))
    print("XGBoost Classifier Accuracy: {}".format(xgboost_acc))