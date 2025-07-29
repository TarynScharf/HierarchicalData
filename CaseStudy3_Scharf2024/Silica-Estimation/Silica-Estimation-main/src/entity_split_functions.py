from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress

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


    train = dataframe[dataframe[sample_id_column].isin(train_sample_ids['GSWA_sample_id'].values)]
    test = dataframe[dataframe[sample_id_column].isin(test_sample_ids['GSWA_sample_id'].values)]
    return train, test

def custom_stratified_kfold(dataframe, entity_identifier_column, class_column, n_splits=5):
    """
       Custom function to split data into stratified k-folds.

       Parameters:
       X : array-like, shape (n_samples,)
           The data to be split into folds.
       y : array-like, shape (n_samples,)
           The target labels to stratify by.
       n_splits : int, default=5
           Number of folds.

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

def plot_cv_results(entity_results, observation_results,title):
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10

    red_line = (206 / 255, 61 / 255, 48 / 255, 0.9)  # rgba
    red_envelope = (242 / 255, 71 / 255, 56 / 255, 0.6)

    blue_line = (29 / 255, 66 / 255, 115 / 255, 0.9)
    blue_envelope = (4 / 255, 196 / 255, 217 / 255, 0.6)

    median1 = np.median(entity_results)
    median2 = np.median(observation_results)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13 / 2.54, 6.5 / 2.54))

    ## List 1: Histogram + KDE
    sns.histplot(entity_results, kde=False, ax=axes[0], stat='count', color=red_envelope, edgecolor='black',alpha=0.5)
    axes[0].axvline(median1, color='red', linestyle='--', label=f'Median = {median1:.2f}')
    ax0_2 = axes[0].twinx()
    sns.kdeplot(entity_results, ax=ax0_2, color=red_line, linewidth=2)
    axes[0].set_title('Entity-level splitting')
    axes[0].legend()
    axes[0].set_xlabel('MSE')

    # List 2: Histogram + KDE
    sns.histplot(observation_results, kde=False, ax=axes[1], stat='count', color=blue_envelope, edgecolor=blue_line)
    axes[1].axvline(median2, color='red', linestyle='--', label=f'Median = {median2:.2f}')
    ax1_2 = axes[1].twinx()
    sns.kdeplot(observation_results, ax=ax1_2, color=blue_line, linewidth=2)
    axes[1].set_title('Observation-level splitting')
    axes[1].legend()
    axes[1].set_xlabel ('MSE')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{title}.svg')
    plt.show()
    plt.close(fig)

def return_regression_results(dataframe):
    x = dataframe['actual']
    y = dataframe['predicted']
    slope, intercept, r_value, p_value, std_err = linregress(dataframe['actual'], dataframe['predicted'])
    y_model = slope * x + intercept
    r_squared = r_value ** 2

    #to help the hexplot display clearly in the silica range of interested, I'm going to clip values to 45-70
    data_clipped = dataframe[(dataframe['predicted']<=70) & (dataframe['predicted']>=45)]
    x_clipped = data_clipped['actual']
    y_clipped = data_clipped['predicted']
    y_model_clipped = slope * x + intercept

    return x, y, y_model, r_squared, x_clipped, y_clipped, y_model_clipped

def hexbin_of_prediction_results(entity_df, observation_df):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 10

    red_line = (206 / 255, 61 / 255, 48 / 255, 0.9)  # rgba
    red_envelope = (242 / 255, 71 / 255, 56 / 255, 0.6)

    blue_line = (29 / 255, 66 / 255, 115 / 255, 0.9)
    blue_envelope = (4 / 255, 196 / 255, 217 / 255, 0.6)

    entity_x, entity_y, entity_model_y, entity_r_squared, ent_x_clipped, ent_y_clipped, ent_y_model_clipped = return_regression_results(entity_df)
    observation_x, observation_y, observation_model_y, observation_r_squared, obs_x_clipped, obs_y_clipped, obs_y_model_clipped = return_regression_results(observation_df)

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13 / 2.54, 6.5 / 2.54))
    hxb_entity = axes[0].hexbin(ent_x_clipped, ent_y_clipped, gridsize=50, cmap='Reds')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(45, 70)  # Example x-axis limits
    axes[0].set_ylim(45, 70)
    #axes[0].scatter(entity_x, entity_y, cmap='Reds')
    axes[0].plot(entity_x, entity_model_y, color =red_line, label=f'Entity-split R2:{entity_r_squared:.1f} ' )
    axes[0].set_title('Entity-level splitting')
    axes[0].legend()
    axes[0].set_xlabel('Actual whole-rock silica (%)')
    axes[0].set_ylabel('Predicted whole-rock silica (%)')
    divider_entity = make_axes_locatable(axes[0])
    cax_ent = divider_entity.append_axes("right", size="5%", pad=0.05)
    cbar_entity = fig.colorbar(hxb_entity, cax=cax_ent)
    cbar_entity.set_label("Counts")

    hxb_observation = axes[1].hexbin(obs_x_clipped, obs_y_clipped, gridsize=(50), cmap='Blues')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(45, 70)  # Example x-axis limits
    axes[1].set_ylim(45, 70)
    #axes[1].scatter(observation_x, observation_y, cmap='Blues')
    axes[1].plot(observation_x, observation_model_y, color=blue_line, label=f'Observation-split R2:{observation_r_squared:.1f} ')
    axes[1].set_title('Observation-level splitting')
    axes[1].legend()
    axes[1].set_xlabel('Actual whole-rock silica (%)')
    axes[1].set_ylabel('Predicted whole-rock silica (%)')
    divider_obs = make_axes_locatable(axes[1])
    cax_obs = divider_obs.append_axes("right", size="5%", pad=0.05)
    cbar_obs = fig.colorbar(hxb_observation, cax=cax_obs)
    cbar_obs.set_label("Counts")


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'LinearRegressionResults_hexbin.svg')
    plt.show()
    plt.close(fig)


