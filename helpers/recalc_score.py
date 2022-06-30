import os
from loguru import logger
import numpy as np
from sklearn import metrics


def my_calc_auc_from_submission(submissions, ratio_cleaned=0.95):
    methods = set([x['submission'] for x in submissions])
    scores = {
        k: {
            'auc': 0,
            'fraction_fixes': 0
        } for k in methods
    }
    for method in methods:
        submission = [x for x in submissions if x['submission'] == method]
        logger.info(f"Scoring for submission {method}...")
        x = np.array([x['fixes'] for x in submission])
        y = np.array([x['accuracy'] for x in submission])
        auc_score = metrics.auc(x, y)
        scores[method]['auc'] = auc_score/len(x)
        """
        Here calculate the number of fixes that could achieve ratio_cleaned * accuracy (achieved on cleaned dataset)
        """
        target_y = ratio_cleaned * y[len(x)-1]
        fraction_fixes = -1
        for x_idx, x_val in enumerate(y):
            if x_val > target_y:
                fraction_fixes = x_idx
                break
        # the name should be changed later...
        scores[method]['fraction_fixes'] = fraction_fixes/len(x)
    return scores

def recalc_score(fpath):
    import json
    with open(fpath, 'r') as fp:
        data = json.load(fp)
    submissions = data['submitted_evaluations']
    score = my_calc_auc_from_submission(
        submissions, ratio_cleaned=0.95
    )
    data['score'] = score
    with open(fpath, 'w') as fp:
        json.dump(data, fp)

if __name__=="__main__":
    all_evaluations = [x for x in os.listdir("results") if x.endswith(".json")]
    for each in all_evaluations:
        recalc_score(os.path.join("results", each))
    