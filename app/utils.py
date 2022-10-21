import numpy as np
import pyarrow as pa
from sklearn import metrics
from loguru import logger
def fix(proposed_fixes, train, budget, gt_df):
    if len(proposed_fixes) > budget:
        raise ValueError("Submission takes more budget than expected, {}>{}".format(len(proposed_fixes), budget))
    fixed_points = train.take(proposed_fixes)
    fixed_points = fixed_points.to_pydict()
    train = train.to_pydict()
    # mutate labels
    # find the ground truth
    gt_labels = [gt_df[gt_df['ImageID'] == filename]['hv_label'].values[0] for filename in fixed_points['filename']]
    for row_id, each in enumerate(proposed_fixes):
        train['label'][each] = gt_labels[row_id]
    d = pa.Table.from_pydict(train)
    return d, len(proposed_fixes)

def calc_auc_from_submission(submissions, ratio_cleaned=0.95):
    methods = set([x['submission'] for x in submissions])
    scores = {
        k: {
            'auc': -1,
            'fraction_fixes': -1
        } for k in methods
    }
    for method in methods:
        submission = [x for x in submissions if x['submission'] == method]
        logger.info(f"Scoring for submission {method}...")
        x = np.array([x['fixes'] for x in submission])
        y = np.array([x['accuracy'] for x in submission])
        """
        Here calculate the number of fixes that could achieve ratio_cleaned * accuracy (achieved on cleaned dataset)
        """
        target_y = ratio_cleaned * y[len(x)-1]
        for i in range(len(x)):
            if y[i] >=target_y:
                scores[method]['fraction_fixes'] = i/len(x)
                break
    return scores