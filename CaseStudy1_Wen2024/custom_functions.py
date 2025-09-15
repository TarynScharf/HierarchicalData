from collections import defaultdict
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from skbio.stats.composition import ilr, closure
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from itertools import combinations

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

    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(script_path)
    df_results = pd.DataFrame({'ENTITY_F1':entity_results, 'OBSERVATION_F1':observation_results})
    df_results.to_csv(os.path.join(parent_directory, 'Outputs', 'F1_crossvalidation_results.csv'))

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(18 / 2.54, 8 / 2.54))

    ## List 1: Histogram + KDE
    sns.histplot(entity_results, kde=False, ax=axes, stat='count', color=red_envelope, edgecolor='black',alpha=0.5, binwidth=0.01)
    axes.axvline(median1, color='red', linestyle='--', label=f'ES median = {median1:.2f}')
    axes.axvline(0.91, color='black', linestyle=':', label=f'Wen et al. (2024) = 0.91')
    ax_2 = axes.twinx()
    sns.kdeplot(entity_results, ax=ax_2, color=red_line, linewidth=2)
    axes.legend()

    # List 2: Histogram + KDE
    sns.histplot(observation_results, kde=False, ax=axes, stat='count', color=blue_envelope, edgecolor=blue_line, binwidth = 0.01)
    axes.axvline(median2, color='blue', linestyle='--', label=f'OS median = {median2:.2f}')
    sns.kdeplot(observation_results, ax=ax_2, color=blue_line, linewidth=2)
    axes.legend()
    axes.set_xlabel ('Macro F1')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(parent_directory, 'Outputs',f'{title}.svg'))
    plt.show()
    plt.close(fig)

def plot_confusion_matrix(y_true, y_pred,data_splitting_type,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    else:
        pass

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Normalised predictions per class')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes) - 0.5, -0.5)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(script_path)
    plt.savefig(f"{parent_directory}/Outputs/{data_splitting_type}_confusion_matrix.svg", dpi=300, bbox_inches='tight')
    plt.close()

def custom_stratified_group_kfold(dataframe, column, n_splits=5, seed=None):
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

    unique_samples = dataframe[[column, 'label']].drop_duplicates()

    # Initialize dictionaries to hold the indices for each class
    class_labels_and_sample_indices = defaultdict(list)

    # Group the indices by their corresponding class in y
    class_labels = unique_samples['label'].unique()
    class_labels = np.sort(class_labels)
    for label in class_labels:
        # This creates a dictionary with key that is class label. Each key contains the indices of samples in that class
        class_labels_and_sample_indices[label].extend(unique_samples[unique_samples['label'] == label].index.to_list())

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
        test_samples = unique_samples.loc[folds[i], column]
        train_samples = unique_samples[~unique_samples.index.isin(folds[i])][column]
        no_overlap = not test_samples.isin(train_samples).any()
        if not no_overlap:
            duplicates = list(test_samples[test_samples.isin(train_samples)].unique())
            print('samples are duplicated between train and test splits')
            print(duplicates)
            break
        else:

            #Using those sample id's, find the indices of observations related to the sample id's in the original dataframe
            test_indices = dataframe[dataframe[column].isin(test_samples)].index #np.array(folds[i])
            train_indices = dataframe[dataframe[column].isin(train_samples)].index#np.array([idx for j in range(n_splits) if j != i for idx in folds[j]])
            yield(train_indices, test_indices)

def select_data(dataframe, entity_id_column, entity_id_array):

    selected_data = dataframe[dataframe[entity_id_column].isin(entity_id_array.values)]

    # Sanity check to ensure that the data subset contains all classes:
    print(f'labels in data subset: {selected_data['label'].unique()}')

    # Split into x and y subsets and get rid of the entity_id column
    y_data = selected_data['label']
    x_data = selected_data.drop(columns = ['label',entity_id_column])

    return x_data,y_data

def custom_stratified_train_test_split(dataframe, entity_id_column, test_size_fraction):
    #Create a list of unique entity_id - label pairs
    unique_samples = dataframe[[entity_id_column, 'label']].drop_duplicates()
    #create stratified splits from this df of unique pairs
    train_entity_ids, test_entity_ids, train_labels, test_labels = train_test_split(
        unique_samples[entity_id_column],
        unique_samples['label'],
        stratify=unique_samples['label'],
        test_size=test_size_fraction,
        shuffle=True,
        random_state=42)

    # Sanity check to ensure that no entity_id's are shared between the train and test splits.
    train_groups = set(train_entity_ids)
    no_shared_groups = train_groups.isdisjoint(set(test_entity_ids))
    if not no_shared_groups:
        print(f'No groups shared between train and test entity-splits: {no_shared_groups}')
        return None, None, None, None

    # select all data points that match the entity_id's assigned to each data subset
    print(f'No groups shared between train and test entity-splits: {no_shared_groups}')
    X_train, y_train = select_data(dataframe, entity_id_column, train_entity_ids)
    X_test, y_test = select_data(dataframe, entity_id_column, test_entity_ids)

    return X_train, X_test, y_train, y_test

def safe_closure(row, pseudocount=1e-6):
    row = row.copy()
    row[row <= 0] = pseudocount
    return closure(row)

def test_independence_of_entities(dataframe, entity_id, target_column, compositional_columns):
    #This function was created with the help of ChatGpt

    #details for results output
    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(script_path)
    output_file = os.path.join(parent_directory, 'Outputs', 'Assessment_of_entity_dissimilarity.xlsx')

    # Firstly, we'd expect the different classes to look different geochemically
    # (otherwise why else would we design a predictive system?)
    # So let's look at each class independently.
    all_classes_anova_results = []
    all_classes_tukey_results = []
    all_classes_manova_results=[]
    all_classes_manova_pairwise = []
    for deposit_class in dataframe[target_column].unique():
        df = dataframe[dataframe[target_column] == deposit_class]
        df_compositional_data = df[compositional_columns].copy()

        # ILR can't handle zeros, so assign a very small value.
        df_compositional_data[df_compositional_data<=0] = 1e-6
        closed_df = df_compositional_data.apply(closure, axis=1, result_type='expand')
        closed_df.columns = df_compositional_data.columns

        ilr_data = ilr(closed_df)
        ilr_df = pd.DataFrame(ilr_data, columns=[f'ILR{i + 1}' for i in range(ilr_data.shape[1])])
        ilr_df[entity_id] = df[entity_id].values

        # --- MANOVA ---
        ilr_cols = [col for col in ilr_df.columns if col != entity_id]
        formula = f"{' + '.join(ilr_cols)} ~ {entity_id}"
        manova = MANOVA.from_formula(formula, data=ilr_df)
        manova_summary = manova.mv_test()

        # Parse MANOVA result for storage
        manova_stat_table = manova_summary.results[entity_id]['stat']
        for stat in manova_stat_table.index:
            row = manova_stat_table.loc[stat]
            all_classes_manova_results.append({
                'Class': deposit_class,
                'Test': stat,
                'Value': row['Value'],
                'F Value': row['F Value'],
                'Num DF': row['Num DF'],
                'Den DF': row['Den DF'],
                'p-value': row['Pr > F']
            })

        # --- MANOVA Pairwise ---
        unique_entities = ilr_df[entity_id].unique()
        pairwise_combinations = list(combinations(unique_entities, 2))
        for group1, group2 in pairwise_combinations:
            pair_df = ilr_df[ilr_df[entity_id].isin([group1, group2])].copy()
            formula_pair = f"{' + '.join(ilr_cols)} ~ {entity_id}"
            manova_pair = MANOVA.from_formula(formula_pair, data=pair_df)
            manova_pair_summary = manova_pair.mv_test()
            stat_table = manova_pair_summary.results[entity_id]['stat']
            for stat in stat_table.index:
                row = stat_table.loc[stat]
                all_classes_manova_pairwise.append({
                    'Class': deposit_class,
                    'Entity 1': group1,
                    'Entity 2': group2,
                    'Test': stat,
                    'Value': row['Value'],
                    'F Value': row['F Value'],
                    'Num DF': row['Num DF'],
                    'Den DF': row['Den DF'],
                    'P-value': row['Pr > F'],
                    'Reject': True if row['Pr > F'] <0.05 else False
                })

        ## --- ANOVA + Tukey ---
        pvals = []
        col_names = []
        eta_squared = []
        for col in ilr_df.columns[:-1]:  # Exclude entity_id
            model = ols(f"{col} ~ {entity_id}", data=ilr_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            pval = anova_table.loc[entity_id, 'PR(>F)']
            pvals.append(pval)
            col_names.append(col)

            sum_of_squares_between_groups = anova_table.loc[entity_id, 'sum_sq']
            total_sum_of_squares = anova_table['sum_sq'].sum()
            eta_squared.append(sum_of_squares_between_groups / total_sum_of_squares)

        # Apply FDR (Benjamini-Hochberg)
        reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

        # Print results
        all_anova_results = []
        all_tukey_results = []
        for col, raw_p,eta_sq, adj_p, rej in zip(col_names, pvals,eta_squared, pvals_corrected, reject):
            all_anova_results.append({
                'Class': deposit_class,
                'ILR Coordinate': col,
                'Raw p-value': raw_p,
                'Adjusted p-value (Benjamini-Hochberg)': adj_p,
                'Significant': rej,
                'Eta squared': eta_sq
            })

            # If significant, perform Tukey's HSD
            if rej:
                tukey = pairwise_tukeyhsd(endog=ilr_df[col], groups=ilr_df[entity_id], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                tukey_df.insert(0, 'Class', deposit_class)
                tukey_df.insert(1, 'ILR Coordinate', col)
                all_tukey_results.append(tukey_df)

        anova_df = pd.DataFrame(all_anova_results)
        all_classes_anova_results.append(anova_df)

        tukey_df = pd.concat(all_tukey_results, ignore_index=True) if all_tukey_results else pd.DataFrame()
        all_classes_tukey_results.append(tukey_df)

    anova_df_final = pd.concat(all_classes_anova_results, ignore_index=True)
    tukey_df_final = pd.concat(all_classes_tukey_results, ignore_index=True)
    manova_df_final = pd.DataFrame(all_classes_manova_results)
    manova_pairwise_df_final = pd.DataFrame(all_classes_manova_pairwise)

    for df in [anova_df_final,tukey_df_final,manova_df_final, manova_pairwise_df_final]:
        df['Class'] = df['Class'].replace({0: 'I', 1: 'II', 2: 'III', 3: 'IV', 4: 'V'})

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        anova_df_final.to_excel(writer, sheet_name='ANOVA_results', index=False)
        tukey_df_final.to_excel(writer, sheet_name='Tukey_results', index=False)
        manova_df_final.to_excel(writer, sheet_name='MANOVA_results', index=False)
        manova_pairwise_df_final.to_excel(writer, sheet_name='MANOVA_pairwise', index=False)
