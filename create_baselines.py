import os
import yaml
import numpy as np
from yaml import Loader
from pyarrow import parquet as pq
from baselines.shapley import ShapleyAppraiser
from baselines.random_pick import RandomAppraiser
from datascope.importance.shapley import ImportanceMethod
from baselines.influence_function import InfluenceFunctionAppraiser
from datascope.importance.shapley import ImportanceMethod
from timeit import default_timer as timer
import sys

if __name__ == "__main__":
    is_docker = False 
    if len(sys.argv)>1 and sys.argv[1] == "docker":
        is_docker=True
    if is_docker:
        with open("task_setup_docker.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    else:
        with open("task_setup.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    baselines = []
    
    
    
    for task in setup['tasks']:

        # we re-initialise the appraisers for each task

        for appraiser in setup['baselines']:
            if appraiser['name'] == 'mc_shapley':
                importance_appraiser = ShapleyAppraiser(importance_method=ImportanceMethod.MONTECARLO)
            elif appraiser['name'] == 'bruteforce_shapley':
                importance_appraiser = ShapleyAppraiser(importance_method=ImportanceMethod.BRUTEFORCE)
            elif appraiser['name'] == 'neighbor_shapley':
                importance_appraiser = ShapleyAppraiser(importance_method=ImportanceMethod.NEIGHBOR)
            elif appraiser['name'] == 'random':
                importance_appraiser = RandomAppraiser()
            elif appraiser['name'] == 'influence_function':
                importance_appraiser = InfluenceFunctionAppraiser()
            else:
                raise ValueError(f"Unknown algorithm {appraiser['name']}")
        
        baselines.append(importance_appraiser)

        train_filepath = os.path.join(setup['paths']['embedding_folder'], f"{task['data_id']}_train_{task['noise_level']}_{task['train_size']}.parquet")
        test_filepath = os.path.join(setup['paths']['embedding_folder'], f"{task['data_id']}_test_{task['test_size']}.parquet")
        val_filepath = os.path.join(setup['paths']['embedding_folder'], f"{task['data_id']}_val_{task['val_size']}.parquet")

        train_file = pq.read_table(train_filepath)
        test_file = pq.read_table(test_filepath)
        val_file = pq.read_table(val_filepath)
        
        train_X, train_y = np.vstack(train_file.column("encoding").to_numpy()), np.vstack(train_file.column("label").to_numpy())
        test_X, test_y = np.vstack(test_file.column("encoding").to_numpy()), np.vstack(test_file.column("label").to_numpy())
        val_X, val_y = np.vstack(val_file.column("encoding").to_numpy()), np.vstack(val_file.column("label").to_numpy())

        for appraiser in baselines:
            print(f"Fitting {appraiser.name} on {task['data_id']} ")
            start = timer()
            appraiser.fit(train_X, train_y, val_X, val_y)
            # we write a single file, where budget is the length of the dataset - and we fix from 1 to the length of the dataset.
            proposed_fixes = appraiser.propose(budget=train_X.shape[0]+1)
            end = timer()
            print(f" - took {end-start} seconds")
            with open(os.path.join(setup['paths']['submission_folder'], f"{task['data_id']}_{appraiser.name}.txt"), 'w+') as f:
                for item in proposed_fixes:
                    f.write(f"{item}\n")
            with open(os.path.join(setup['paths']['submission_folder'], f"time_{task['data_id']}_{appraiser.name}.txt"), 'w+') as f:
                f.write(f"{end-start}")