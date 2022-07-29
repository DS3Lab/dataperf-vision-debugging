import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import sys
import yaml
from yaml import Loader


def get_cost_time(row):
    row = row.to_dict()
    method = row['method']
    data_id = row['data_id']
    with open(os.path.join(submission_path, f"time_{data_id}_{method}.txt"), 'r') as fp:
        time = fp.read()
    return float(time)


def aggregate_data(evaluation_file, result_folder):
    data_id = evaluation_file.split("_")[0]
    evaluation_folder = result_folder

    with open(os.path.join(evaluation_folder, evaluation_file), "r") as f:
        results = json.load(f)
    data = results["auc_score"]
    data = pd.DataFrame.from_dict(data, orient='index')
    data['method'] = data.index
    data['data_id'] = data_id
    data['time'] = data.apply(get_cost_time, axis=1)
    return data

def plot(data, result_folder, score_metric='auc'):
    sns.set(
        font="DejaVu Sans",
        context="paper",
        style="whitegrid",
        font_scale=2,
        rc={'figure.figsize': (10, 9)},
    )
    sns.color_palette("colorblind", as_cmap=True)
    ax = sns.scatterplot(
        x=score_metric,
        y="time",
        hue="method",
        style="data_id",
        s=200,
        data=data,
    )
    ax.set_title(f"Valuation time vs. {score_metric}")
    ax.set_yscale("log")
    """
    Add score to the legend
    """
    handles, labels = ax.get_legend_handles_labels()
    # find methods
    methods = set(data['method'].tolist())
    handles, labels = handles[1:1+len(methods)], labels[1:1+len(methods)]
    for h in handles:
        h.set_sizes([200])
    
    for idx, label in enumerate(labels):
        if label=='mc_shapley':
            labels[idx] = 'TMC Shapley x100'
        elif label == 'neighbor_shapley (datascope)':
            labels[idx] = 'DataScope Shapley'
        elif label == 'random':
            labels[idx] = 'Random'
        elif label == 'influence_function':
            labels[idx] = 'Influence Function'
    print(labels)
    plt.legend(markerscale=1.3)
    leg = ax.legend(handles=handles, labels=labels)
    figure_path = os.path.join(result_folder, f"{score_metric}_speed.png")
    plt.savefig(figure_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def run_plotting(evaluation_folder="results/"):

    results = os.listdir(evaluation_folder)
    results = [x for x in results if x.endswith(".json")]
    frames = []
    for result in results:
        df = aggregate_data(result, result_folder=evaluation_folder)
        frames.append(df)
    data = pd.concat(frames)
    data.sort_values(by='method', inplace=True)
    plot(data, result_folder=evaluation_folder, score_metric='auc')
    plot(data, result_folder=evaluation_folder, score_metric='fraction_fixes')


if __name__ == "__main__":
    is_docker = False
    if len(sys.argv) > 1 and sys.argv[1] == "docker":
        is_docker = True
    if is_docker:
        with open("task_setup_docker.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    else:
        with open("task_setup.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    results_path = setup['paths']['results_folder']
    submission_path = setup['paths']['submission_folder']
    run_plotting(results_path)