import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.stats.composition import ilr, closure
from sklearn.metrics import roc_auc_score, f1_score
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests


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

def safe_closure(row, pseudocount=1e-6):
    row = row.copy()
    row[row <= 0] = pseudocount
    return closure(row)

def test_independence_of_entities(dataframe, entity_id, target_column, compositional_columns):
    #This function was created with the help of ChatGpt

    #details for results output
    script_path = os.path.abspath(__file__)
    parent_directory = os.path.dirname(os.path.dirname(script_path))
    output_file = os.path.join(parent_directory, 'Outputs', 'Anova_results.xlsx')

    # Firstly, we'd expect the different classes to look different geochemically
    # (otherwise why else would be design a predictive system?)
    # So let's look at each class independently.
    all_classes_anova_results = []
    #all_classes_tukey_results = []
    all_classes_manova_results=[]
    for deposit_class in dataframe[target_column].unique():
        df = dataframe[dataframe[target_column] == deposit_class]
        df_compositional_data = df[compositional_columns].copy()
        df_compositional_data[df_compositional_data<=0] = 1e-6

        # ILR can't handle zeros, so assign a very small value.
        closed_df = df_compositional_data.apply(closure, axis=1, result_type='expand')
        closed_df.columns = df_compositional_data.columns

        ilr_data = ilr(closed_df)
        ilr_df = pd.DataFrame(ilr_data, columns=[f'ILR{i + 1}' for i in range(ilr_data.shape[1])])
        ilr_df[entity_id] = df[entity_id].values

        #There are many groups containing only 1 zircon. let's keep only those groups with 10 or more data points.
        filtered_ilr_df = ilr_df[ilr_df.groupby('ENTITY_ID')['ENTITY_ID'].transform('count') >= 10]

        # MANOVA
        ilr_cols = [col for col in ilr_df.columns if col != entity_id]
        formula = f"{' + '.join(ilr_cols)} ~ {entity_id}"
        manova = MANOVA.from_formula(formula, data=filtered_ilr_df)
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
        #  ANOVA
        pvals = []
        col_names = []
        eta_squared = []
        for col in ilr_df.columns[:-1]:
            model = ols(f"{col} ~ {entity_id}", data=filtered_ilr_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            pval = anova_table.loc[entity_id, 'PR(>F)']
            pvals.append(pval)
            col_names.append(col)

            sum_of_squares_between_groups = anova_table.loc[entity_id, 'sum_sq']
            total_sum_of_squares = anova_table['sum_sq'].sum()
            eta_squared.append(sum_of_squares_between_groups / total_sum_of_squares)

        # Apply FDR (Benjamini-Hochberg)
        reject, pvals_corrected, _, _ = multipletests(pvals, method='fdr_bh')

        all_anova_results = []
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
            ## Note: I do not perform Tukey because the with the number of groups, it is extremely time-consuming
            '''if rej:
                filtered_ilr_df = ilr_df[ilr_df.groupby('ENTITY_ID')['ENTITY_ID'].transform('count') >= 10]
                tukey = pairwise_tukeyhsd(endog=filtered_ilr_df[col], groups=filtered_ilr_df[entity_id], alpha=0.05)
                tukey_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])
                tukey_df.insert(0, 'Class', deposit_class)
                tukey_df.insert(1, 'ILR Coordinate', col)
                all_tukey_results.append(tukey_df)'''
        anova_df = pd.DataFrame(all_anova_results)
        all_classes_anova_results.append(anova_df)

        #tukey_df = pd.concat(all_tukey_results, ignore_index=True) if all_tukey_results else pd.DataFrame()
        #all_classes_tukey_results.append(tukey_df)
    anova_df_final = pd.concat(all_classes_anova_results, ignore_index=True)
    #tukey_df_final = pd.concat(all_classes_tukey_results, ignore_index=True)
    manova_df_final = pd.DataFrame(all_classes_manova_results)

    for df in [anova_df_final,manova_df_final]:#tukey_df_final,
        df['Class'] = df['Class'].replace({0: 'Subalkaline', 1: 'Alkaline'})

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        anova_df_final.to_excel(writer, sheet_name='ANOVA_results', index=False)
        #tukey_df_final.to_excel(writer, sheet_name='Tukey_results', index=False)
        manova_df_final.to_excel(writer, sheet_name='MANOVA_results', index=False)