import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import sys
import yaml
from yaml import Loader

def extract_method(row):
    method = row['submission']
    return method

def plot(evaluation_file, result_folder):
    data_id = evaluation_file.split("_")[0]
    evaluation_folder = "results/"
    sns.set(
        font="DejaVu Sans",
        context="paper",
        style="whitegrid",
        font_scale=2,
        rc={'figure.figsize':(10,9)},
    )
    sns.color_palette("colorblind", as_cmap=True)
    with open(os.path.join(evaluation_folder, evaluation_file), "r") as f:
        results = json.load(f)
    before_acc = results['before_acc']
    data = results["submitted_evaluations"]
    data = pd.DataFrame(data)
    data['method'] = data.apply(extract_method, axis=1)
    data.sort_values(by='method', inplace=True)
    ax = sns.lineplot(
        x="fixes",
        y="accuracy",
        lw=2,
        hue="method",
        data=data
    )
    ax.set_title(f"Cleaning on {data_id}")
    plt.axhline(before_acc, linestyle='--', label='Before Correction', lw=2, color='black')
    
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles = handles, labels = labels)
    leg = ax.legend()
    for line in leg.get_lines():
        line.set_linewidth(4.0)
    # Slice list to remove first handle
    figure_path = os.path.join(result_folder, )
    plt.savefig(f"results/{data_id}_evaluation.png", bbox_inches='tight', pad_inches=0)
    plt.close()

def run_plotting(evaluation_folder="results/"):
    
    results = os.listdir(evaluation_folder)
    results = [x for x in results if x.endswith(".json")]
    for result in results:
        plot(result, result_folder=evaluation_folder)

if __name__=="__main__":
    is_docker = True if sys.argv[1] == "docker" else False
    if is_docker:
        with open("task_setup_docker.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    else:
        with open("task_setup.yml", "r") as f:
            setup = yaml.load(f, Loader=Loader)
    submission_path = setup['paths']['submission_folder']
    results_path = setup['paths']['results_folder']
    run_plotting(results_path)