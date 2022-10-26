import numpy as np
from pyarrow import parquet as pq
import sklearn
import torch
import xgboost
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class XGBClassifier():
    def __init__(self):
        self.xgb = xgboost.XGBClassifier(
            eval_metric='logloss',
            use_label_encoder=False,
        )

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

class LogisticClassifier():
    def __init__(self):
        torch.manual_seed(42)
        self.model = LinearModel()
        self.threshold = 0.5

    def fit(self, features, labels, epochs=20, lr=0.01, batch_size=64, weight_decay=0.9):
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model.train(True)
        for epoch in range(epochs):
            for x, y in dataloader:
                optimizer.zero_grad()
                y_pred = self.model(x.float())
                y_pred = y_pred.reshape(y.shape)
                loss = criterion(y_pred, y.float())
                loss.backward()
                optimizer.step()
            # print(list(self.model.parameters()))
        # adjust the threshold, on the training set
        cand_thres = list(np.arange(0.1, 1, 0.05))
        # find highest accuracy
        best_acc = 0
        best_thres = 0
        for thres in cand_thres:
            self.threshold = thres
            accuracy = self.evaluate(features, labels)
            if accuracy > best_acc:
                best_acc = accuracy
                best_thres = thres
        self.threshold = best_thres

    def predict(self, feature):
        self.model.eval()
        if not isinstance(feature, torch.Tensor):
            feature = torch.tensor(feature).float()
        with torch.no_grad():
            output = self.model(feature)
            output = torch.sigmoid(output)
        output = np.where(output <= self.threshold, 0, output)
        output = np.where(output > self.threshold, 1, output)
        output = output.ravel()
        return output

    def evaluate(self, test_features, test_labels):
        test_labels = test_labels.ravel()
        if isinstance(test_labels, torch.Tensor):
            test_labels = test_labels.numpy()
        pred = self.predict(test_features)
        accuracy = (pred == test_labels).mean()
        return round(accuracy, 2)


if __name__ == "__main__":
    train = pq.read_table("embeddings/09j2d_test_500.parquet")
    test = pq.read_table("embeddings/09j2d_val_100.parquet")
    train_X, train_y = np.vstack(train.column("encoding").to_numpy()), np.vstack(
        train.column("label").to_numpy())
    test_X, test_y = np.vstack(test.column("encoding").to_numpy()), np.vstack(
        test.column("label").to_numpy())
    # clf = XGBClassifier()
    # clf.fit(train_X, train_y)
    # xgboost_acc = clf.evaluate(test_X, test_y)
    # print("XGBoost Classifier Accuracy: {}".format(xgboost_acc))
    test_X = torch.tensor(test_X)
    test_y = torch.tensor(test_y)
    linear_clf = LogisticClassifier()
    linear_clf.fit(train_X, train_y)
    linear_acc = linear_clf.evaluate(test_X, test_y)
    print("Logistic Classifier Accuracy: {}".format(linear_acc))
    # sklearn_clf = LogisticRegressionCV()
    # sklearn_clf.fit(train_X, train_y)
    # sklearn_acc = sklearn_clf.score(test_X, test_y)
    # print("Sklearn Classifier Accuracy: {}".format(sklearn_acc))
