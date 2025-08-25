from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress, stats, shapiro
import os
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix, permanova

def select_data(data, index_array,sample_id_column,target_column):
    selected_data = data[data[sample_id_column].isin(index_array.values)]
    selected_data_no_label = selected_data.loc[:, ~selected_data.columns.isin([target_column])]
    label = selected_data[target_column]

    return selected_data_no_label,label

def custom_stratified_train_test_split(dataframe, sample_id_column, stratify_column, fraction):
    unique_samples = dataframe[[sample_id_column, stratify_column]].drop_duplicates()
    #TODO I removed the random_state so that multiple iterations will sample differently
    train_sample_ids, test_sample_ids = train_test_split(
        unique_samples,
        stratify=unique_samples[stratify_column],
        test_size=fraction,
        shuffle=True
        )#random_state=19
    #sanity check to ensure train and test entities are distinct
    mutually_exclusive = set(train_sample_ids['GSWA_sample_id'].values).isdisjoint(test_sample_ids['GSWA_sample_id'].values)
    print(f'data subsets are mutually exclusive: {mutually_exclusive}')
    train = dataframe[dataframe[sample_id_column].isin(train_sample_ids['GSWA_sample_id'].values)]
    test = dataframe[dataframe[sample_id_column].isin(test_sample_ids['GSWA_sample_id'].values)]
    return train, test

def custom_stratified_group_kfold(dataframe, entity_identifier_column, class_column, n_splits=5, seed=None):
    """
       Custom function to split data into stratified k-folds.

       Parameters:
       dataframe: data to split into folds
       column: grouping column in dataframe
       n_splits: number of splits to create

       Returns:
       splits : list of tuples
           A list of (train_indices, test_indices) tuples for each fold.
       """

    unique_samples = dataframe[[entity_identifier_column, class_column]].drop_duplicates()

    # Initialize dictionaries to hold the indices for each class
    class_labels_and_sample_indices = defaultdict(list)

    # Group the indices by their corresponding class in y
    class_labels = unique_samples[class_column].unique()
    class_labels = np.sort(class_labels)
    for label in class_labels:
        # This creates a dictionary with key that is class label. Each key contains the indices of samples in that class
        class_labels_and_sample_indices[label].extend(unique_samples[unique_samples[class_column] == label].index.to_list())

    # Shuffle each class' indices
    for index in class_labels_and_sample_indices:
        np.random.seed(seed)
        np.random.shuffle(class_labels_and_sample_indices[index])

    # Split each class into approximately equal-sized folds
    class_folds = {label: np.array_split(indices, n_splits) for label, indices in class_labels_and_sample_indices.items()}

    # Create the stratified folds by collecting one fold from each class
    folds = [[] for _ in range(n_splits)]
    for label, fold_indices in class_folds.items():
        for i in range(n_splits):
            folds[i].extend(fold_indices[i])

    # Create a list of (train_indices, test_indices) for each fold
    splits = []
    for i in range(n_splits):
        #Collect the sample id's associated with the train and test split
        test_samples = unique_samples.loc[folds[i], entity_identifier_column]
        train_samples = unique_samples[~unique_samples.index.isin(folds[i])][entity_identifier_column]
        no_overlap = not test_samples.isin(train_samples).any()
        if not no_overlap:
            duplicates = list(test_samples[test_samples.isin(train_samples)].unique())
            print('samples are duplicated between train and test splits')
            print(duplicates)
            break
        else:

            #Using those sample id's, find the indices of observations related to the sample id's in the original dataframe
            test_indices = dataframe[dataframe[entity_identifier_column].isin(test_samples)].index #np.array(folds[i])
            train_indices = dataframe[dataframe[entity_identifier_column].isin(train_samples)].index#np.array([idx for j in range(n_splits) if j != i for idx in folds[j]])
            yield(train_indices, test_indices)

def plot_cv_results(entity_results, observation_results,title, output_location):
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10

    red_line = (206 / 255, 61 / 255, 48 / 255, 0.9)  # rgba
    red_envelope = (242 / 255, 71 / 255, 56 / 255, 0.6)

    blue_line = (29 / 255, 66 / 255, 115 / 255, 0.9)
    blue_envelope = (4 / 255, 196 / 255, 217 / 255, 0.6)

    median1 = np.median(entity_results)
    median2 = np.median(observation_results)

    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(script_path))
    df_results = pd.DataFrame({'ENTITY_F1':entity_results, 'OBSERVATION_F1':observation_results})
    df_results.to_csv(os.path.join(parent_directory, 'Outputs',output_location, 'F1_crossvalidation_results.csv'))

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18 / 2.54, 8 / 2.54))

    ## List 1: Histogram + KDE
    sns.histplot(entity_results, kde=False, ax=axes, stat='count', color=red_envelope, edgecolor='black',alpha=0.5, binwidth=0.5)
    axes.axvline(median1, color='red', linestyle='--', label=f'ES median: {median1:.2f}')
    ax_2 = axes.twinx()
    sns.kdeplot(entity_results, ax=ax_2, color=red_line, linewidth=2)
    axes.legend()

    # List 2: Histogram + KDE
    sns.histplot(observation_results, kde=False, ax=axes, stat='count', color=blue_envelope, edgecolor=blue_line, binwidth = 0.5)
    axes.axvline(median2, color='blue', linestyle='--', label=f'OS median: {median2:.2f}')
    sns.kdeplot(observation_results, ax=ax_2, color=blue_line, linewidth=2)
    axes.legend()
    axes.set_xlabel ('Mean squared error')

    #Add the reference line for Scharf et al. (2004)
    axes.axvline(x = 3.12**2, color='black', linestyle=':', label=f'Scharf et al. (2024): {3.12**2:.2f}')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(parent_directory, 'Outputs',output_location,f'{title}.svg'))
    plt.show()
    plt.close(fig)

def return_regression_results(dataframe):
    #scharf et al. (2024) report median prediction for a sample vs sample actual

    #Thus sort dataframe by actual (same sample will have the same SiO2 value for all zircon) per kfold iteration (fold per repeat)
    dataframe.sort_values(by=['repeat', 'fold', 'actual'])

    #Then calculate median predicted.
    median_df = dataframe.groupby(['repeat', 'fold', 'actual'])['predicted'].median().reset_index()

    x = median_df['actual']
    y = median_df['predicted']
    slope, intercept, r_value, p_value, std_err = linregress(median_df['actual'], median_df['predicted'])
    y_model = slope * x + intercept
    r_squared = r_value ** 2

    #to help the hexplot display clearly in the silica range of interested, I'm going to clip values to 45-70
    data_clipped = median_df[(median_df['predicted']<=70) & (median_df['predicted']>=45)]
    x_clipped = median_df['actual']
    y_clipped = median_df['predicted']
    y_model_clipped = slope * x + intercept

    return x, y, y_model, r_squared, x_clipped, y_clipped, y_model_clipped

def linear_regression_of_prediction_results(entity_df, observation_df, output_location):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10

    red_line = (206 / 255, 61 / 255, 48 / 255, 0.9)  # rgba
    blue_line = (29 / 255, 66 / 255, 115 / 255, 0.9)

    entity_x, entity_y, entity_model_y, entity_r_squared, ent_x_clipped, ent_y_clipped, ent_y_model_clipped = return_regression_results(entity_df)
    observation_x, observation_y, observation_model_y, observation_r_squared, obs_x_clipped, obs_y_clipped, obs_y_model_clipped = return_regression_results(observation_df)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12 / 2.54, 12 / 2.54))
    #hxb_entity = axes[0].hexbin(ent_x_clipped, ent_y_clipped, gridsize=50, cmap='Reds')
    #axes[0].set_aspect('equal', adjustable='box')
    #axes[0].set_xlim(45, 70)  # Example x-axis limits
    #axes[0].set_ylim(45, 70)
    axes[0].scatter(entity_x, entity_y, color=red_line, s=16)
    axes[0].plot(entity_x, entity_model_y, color =red_line, label=f'Entity-split R2:{entity_r_squared:.1f} ' )
    axes[0].set_title('Entity-level splitting')
    axes[0].legend()
    axes[0].set_xlabel('Actual whole-rock silica (%)')
    axes[0].set_ylabel('Predicted whole-rock silica (%)')
    axes[0].set_xlim(40, 80)
    axes[0].set_ylim(40, 80)
    #divider_entity = make_axes_locatable(axes[0])
    #cax_ent = divider_entity.append_axes("right", size="5%", pad=0.05)
    #cbar_entity = fig.colorbar(hxb_entity, cax=cax_ent)
    #cbar_entity.set_label("Counts")

    #hxb_observation = axes[1].hexbin(obs_x_clipped, obs_y_clipped, gridsize=(50), cmap='Blues')
    #axes[1].set_aspect('equal', adjustable='box')
    #axes[1].set_xlim(45, 70)  # Example x-axis limits
    #axes[1].set_ylim(45, 70)
    axes[1].scatter(observation_x, observation_y, color=blue_line, s=16)
    axes[1].plot(observation_x, observation_model_y, color=blue_line, label=f'Observation-split R2:{observation_r_squared:.1f} ')
    axes[1].set_title('Observation-level splitting')
    axes[1].legend()
    axes[1].set_xlabel('Actual whole-rock silica (%)')
    axes[1].set_ylabel('Predicted whole-rock silica (%)')
    axes[1].set_xlim(40, 80)
    axes[1].set_ylim(40, 80)
    #divider_obs = make_axes_locatable(axes[1])
    #cax_obs = divider_obs.append_axes("right", size="5%", pad=0.05)
    #cbar_obs = fig.colorbar(hxb_observation, cax=cax_obs)
    #cbar_obs.set_label("Counts")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(script_path))

    fig.savefig(os.path.join(parent_directory, 'Outputs',output_location, 'LinearRegressionResults.svg'))
    plt.show()
    plt.close(fig)

def test_independence_of_entities(dataframe, entity_id, target_column,class_column, output_location):

    #details for results output
    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(script_path))
    output_file = os.path.join(parent_directory, 'Outputs',output_location, 'PERMANOVA_results.xlsx')

    # Firstly, we'd expect the different classes to look different geochemically
    # (otherwise why else would be design a predictive system?)
    # So let's look at each class independently.
    #In this study, classes are the 'bins' assigned to zircon. I.e. we will consider each silica bin separately as the study expects zircon from high silica rocks to look
    #different to those from lower silica rocks

    # Putting all features on the same scale
    scaler = StandardScaler()
    features = [col for col in dataframe.columns if col not in (entity_id, class_column, target_column)]
    dataframe.loc[:,features] = scaler.fit_transform(dataframe.loc[:,features])

    all_classes_permanova_results=[]
    all_pairwise_permanova_results = []
    for silica_class in dataframe[class_column].unique():
        df = dataframe[dataframe[class_column] == silica_class]

        #Let's keep only those groups with 10 or more data points.
        filtered_df = df[df.groupby(entity_id)[entity_id].transform('count') >= 10]
        feature_cols = [col for col in filtered_df.columns if col not in (entity_id, class_column, target_column)]

        # Compute distance matrix
        data_matrix = filtered_df[feature_cols].values
        groups = filtered_df[entity_id].values
        distance_array = pdist(data_matrix, metric='euclidean')
        distance_matrix = DistanceMatrix(squareform(distance_array))

        # PERMANOVA
        global_result = permanova(distance_matrix, groups, permutations=999)
        all_classes_permanova_results.append(
            {'Class': silica_class,
             'pseudo-F': global_result['test statistic'],
             'p-value': global_result['p-value'],
             }
        )

        if global_result['p-value'] <0.05:
            pairwise_pvalues = []
            group1_ids = []
            group2_ids = []
            unique_groups = np.unique(groups)
            pairwise_combinations = list(combinations(unique_groups, 2))
            for group1, group2 in pairwise_combinations:
                subset_data_indices = np.where((groups == group1) | (groups == group2))[0]
                subset_ids = [distance_matrix.ids[i] for i in subset_data_indices]
                subset_dm = distance_matrix.filter(subset_ids)
                subset_groups = groups[subset_data_indices]
                pairwise_result = permanova(subset_dm, subset_groups)
                pairwise_pvalues.append(pairwise_result['p-value'])
                group1_ids.append(group1)
                group2_ids.append(group2)
            reject, pvals_corrected, _, _  = multipletests(pairwise_pvalues, method='fdr_bh')
            all_pairwise_permanova_results.append(
                {'Class': [silica_class]*len(pairwise_combinations),
                 'Group1': group1_ids,
                 'Group2': group2_ids,
                 'Raw pvalues': pairwise_pvalues,
                 'Adjusted p-values': pvals_corrected,
                 'Reject': reject
                }
            )

    all_classes_permanova_df = pd.DataFrame(all_classes_permanova_results)
    permanova_dfs = [pd.DataFrame(d) for d in all_pairwise_permanova_results]
    all_pairwise_permanova_df =  pd.concat(permanova_dfs, ignore_index=True)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        all_classes_permanova_df.to_excel(writer, sheet_name='Global_PERMANOVA', index=False)
        all_pairwise_permanova_df.to_excel(writer, sheet_name='PERMANOVA_pairwise', index=False)
