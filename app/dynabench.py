import os
import sys
from pyarrow import parquet as pq
import numpy as np
from classifier import LogisticClassifier as Classifier
from utils import fix,calc_auc_from_submission
import pandas as pd
from my_debug.selection import CustomAppraiser
import json

submission_path = "../submissions/"
train_size = 300
results_path = "../results/"

if __name__=="__main__":
    # in dynabench submissions, we take all files under submission folder
    submissions = [x for x in os.listdir(submission_path) if x.endswith(".json")]
    tasks = set([x.split("_")[0] for x in submissions])
    for task in tasks:
        submitted_evaluations = []
        train_filepath = os.path.join(
            "embeddings/", f"{task}_train_0.3_300.parquet")
        test_filepath = os.path.join(
            "embeddings/", f"{task}_test_500.parquet")
        groud_truth = os.path.join("data", f"dataset_{task}_train.csv")
        gt_df = pd.read_csv(groud_truth)
        train_file = pq.read_table(train_filepath)
        test_file = pq.read_table(test_filepath)
        test_X, test_y = np.vstack(test_file.column("encoding").to_numpy()), np.vstack(
            test_file.column("label").to_numpy())
        train_X, train_y = np.vstack(train_file.column(
            "encoding").to_numpy()), np.vstack(train_file.column("label").to_numpy())
        before_clf = Classifier()
        before_clf.fit(train_X, train_y)
        before_acc = before_clf.evaluate(test_X, test_y)
        print(f"On task [{task}]: Accuracy before cleaning: {before_acc}")
        # find submissions related to this task
        task_submissions = [x for x in submissions if x.split("_")[0]== task ]
        for submission in task_submissions:
            submission_file = os.path.join(submission_path, submission)
            method_name = submission.replace(".txt","").replace(f"{task}_","")
            with open(submission_file, "r") as f:
                lines = f.readlines()
            proposed_fixes = [int(x.rstrip()) for x in lines]
            for i in range(1, train_size+1):
                print(f"Evaluating {method_name} {i}/{train_size}", flush=True)
                new_train_set, len_fixes = fix(
                    proposed_fixes[:i], train_file, i, gt_df)
                new_train_X, new_train_y = np.vstack(new_train_set.column(
                    "encoding").to_numpy()), np.vstack(new_train_set.column("label").to_numpy())
                new_train_X = new_train_X.astype(np.float32)
                new_train_y = new_train_y.astype(np.float32)
                # re-initialise and re-fit the classifier
                new_clf = Classifier()
                new_clf.fit(new_train_X, new_train_y)
                after_acc = new_clf.evaluate(test_X, test_y)
                submitted_evaluations.append({
                        "submission": method_name,
                        "accuracy": after_acc,
                        "fixes": len_fixes,
                    })
        result = {
            "before_acc": before_acc,
            "submitted_evaluations": submitted_evaluations,
            "auc_score": calc_auc_from_submission(submitted_evaluations)
        }
        with open(os.path.join(results_path, f"{task}_evaluation.json"), "w") as f:
            json.dump(result, f)