import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score
import os
import pandas as pd

def create_f1_array(dataframe):
    tests = dataframe['TEST'].unique()
    f1_results = []
    for test in tests:
        df = dataframe[dataframe['TEST']==test]
        pred = np.array(df['PREDICTED'])
        actual = np.array(df['ACTUALS'])
        p=0.5
        f1 = f1_score(actual, [1 if i >= p else 0 for i in pred])
        f1_results .append(f1)

    return f1_results

def plot_cv_results(entity_results, observation_results,title):
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10

    red_line = (206 / 255, 61 / 255, 48 / 255, 0.9)  # rgba
    red_envelope = (242 / 255, 71 / 255, 56 / 255, 0.6)
    red_marker_fill = (242 / 255, 143 / 255, 107 / 255, 1.0)

    blue_line = (29 / 255, 66 / 255, 115 / 255, 0.9)
    blue_envelope = (4 / 255, 196 / 255, 217 / 255, 0.6)
    blue_marker_fill = (4 / 255, 196 / 255, 217 / 255, 1.0)
    purple_line = (114 / 255, 62 / 255, 152 / 255, 1.0)

    median1 = np.median(entity_results)
    median2 = np.median(observation_results)

    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(script_path))
    df_results = pd.DataFrame({'ENTITY_F1':entity_results, 'OBSERVATION_F1':observation_results})
    df_results.to_csv(os.path.join(parent_directory, 'Outputs', 'F1_crossvalidation_results.csv'))

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(13 / 2.54, 6.5 / 2.54))

    ## List 1: Histogram + KDE
    sns.histplot(entity_results, kde=False, ax=axes, stat='count', color=red_envelope, edgecolor='black',alpha=0.5)
    axes.axvline(median1, color='red', linestyle='--', label=f'ES median = {median1:.2f}')
    ax_2 = axes.twinx()
    sns.kdeplot(entity_results, ax=ax_2, color=red_line, linewidth=2)
    #axes.set_title('Entity-level splitting')
    axes.legend()
    #axes.set_xlabel('Macro F1')

    # List 2: Histogram + KDE
    sns.histplot(observation_results, kde=False, ax=axes, stat='count', color=blue_envelope, edgecolor=blue_line)
    axes.axvline(median2, color='blue', linestyle='--', label=f'OS median = {median2:.2f}')
    #ax1_2 = axes[1].twinx()
    sns.kdeplot(observation_results, ax=ax_2, color=blue_line, linewidth=2)
    axes.set_title('Splitting strategy comparison')
    axes.legend()
    axes.set_xlabel ('Macro F1')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(parent_directory, 'Outputs',f'{title}.svg'))
    plt.show()
    plt.close(fig)

