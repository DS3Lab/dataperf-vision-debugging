import json
import os
from utils import fix
from pyarrow import parquet as pq
from classifier import XGBClassifier as Classifier
import numpy as np
from tqdm import tqdm
import pandas as pd
import yaml
from yaml import Loader
import sys
from utils import calc_auc_from_submission

def run_correction_eval(
):
    is_docker = False 
    if len(sys.argv)>1 and sys.argv[1] == "docker":
        is_docker=True
    if is_docker:
        with open("task_setup_docker.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    else:
        with open("task_setup.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)

    submission_path = setup['paths']['submission_folder']
    results_path = setup['paths']['results_folder']

    for task in setup["tasks"]:
        data_id = task["data_id"]
        noise_level = task["noise_level"]
        train_size = task["train_size"]
        test_size = task["test_size"]
        groud_truth = os.path.join("data", f"dataset_{data_id}_train.csv")
        gt_df = pd.read_csv(groud_truth)

        # Loading training and testing sets
        train_filepath = os.path.join(setup['paths']['embedding_folder'], f"{task['data_id']}_train_{task['noise_level']}_{task['train_size']}.parquet")
        test_filepath = os.path.join(setup['paths']['embedding_folder'], f"{task['data_id']}_test_{task['test_size']}.parquet")
        
        train_file = pq.read_table(train_filepath)
        test_file = pq.read_table(test_filepath)

        test_X, test_y = np.vstack(test_file.column("encoding").to_numpy()), np.vstack(test_file.column("label").to_numpy())
        train_X, train_y = np.vstack(train_file.column("encoding").to_numpy()), np.vstack(train_file.column("label").to_numpy())

        # Fit the classifier on the training set (before cleaning)
        before_clf = Classifier()
        before_clf.fit(train_X, train_y)
        before_acc = before_clf.evaluate(test_X, test_y)
        print(f"On task [{data_id}]: Accuracy before cleaning: {before_acc}")

        # find submissions
        submissions = [x for x in os.listdir(submission_path) if x.endswith(".txt") and x.split("_")[0] == data_id]
        
        submitted_evaluations = []
        for submission in submissions:
            submission_file = os.path.join(submission_path, submission)
            method = submission.replace(".txt", "").replace(f"{data_id}_", "")
            with open(submission_file, "r") as f:
                lines = f.readlines()
            proposed_fixes = [int(x.rstrip()) for x in lines]
            progress_bar = tqdm(range(1, train_size+1))
            for i in range(1, train_size+1):
                progress_bar.set_description(f"Budget {i}")
                # perform the fix, reload the train_X, train_y
                new_train_set, len_fixes = fix(proposed_fixes[:i], train_file, i, gt_df)
                new_train_X, new_train_y = np.vstack(new_train_set.column("encoding").to_numpy()), np.vstack(new_train_set.column("label").to_numpy())

                # re-initialise and re-fit the classifier
                new_clf = Classifier()
                new_clf.fit(new_train_X, new_train_y)
                after_acc = new_clf.evaluate(test_X, test_y)
                submitted_evaluations.append({
                    "submission": submission.replace(".txt", "").replace(f"{data_id}_", ""),
                    "accuracy":after_acc,
                    "fixes": len_fixes,
                    }
                )
                progress_bar.update(1)
                progress_bar.set_postfix(acc=after_acc, method=method)
        """
        Here we calculate the auc score with the submitted_evaluations
        """
        result = {
            "before_acc": before_acc,
            "submitted_evaluations": submitted_evaluations,
            "auc": calc_auc_from_submission(submitted_evaluations)
        }
        with open(os.path.join(results_path, f"{data_id}_evaluation.json"), "w") as f:
            json.dump(result, f)

if __name__=="__main__":
    run_correction_eval()