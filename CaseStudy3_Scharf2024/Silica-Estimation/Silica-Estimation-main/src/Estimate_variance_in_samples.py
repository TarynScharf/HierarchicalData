import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#define file locations
folder_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
input_case_study_results = 'Data_Files/Merged_case_study_results.csv'
input_case_study_path= os.path.join(folder_path, input_case_study_results)
case_study_grain_estimates = pd.read_csv(input_case_study_path)
output_file = os.path.join(folder_path,f'Outputs/stats_{os.path.splitext(os.path.basename(input_case_study_results))[0].lower()}.csv')

input_training_results ='Outputs/prediction_on_training_dataset.csv'
input_training_results_path = os.path.join(folder_path, input_training_results)
training_dataset_grain_estimates = pd.read_csv(input_training_results_path)
group_sizes = training_dataset_grain_estimates.groupby('GSWA_sample_id').transform('count')
training_dataset_grain_estimates['group_size'] =group_sizes['PC1'] #actually any column from group_sizes would work
grouped_training_results = training_dataset_grain_estimates.groupby('GSWA_sample_id').mean()
stats_for_training_results = training_dataset_grain_estimates.groupby('GSWA_sample_id').var(0)
stats_for_training_results['group_size'] = grouped_training_results['group_size']
stats_for_training_results['actual_silica'] = grouped_training_results['actual_silica']

# Define size categories based on ranges
def size_category(value):
    if value < 5:
        return 5
    elif value <= 10:
        return 10
    elif value <= 15:
        return 15
    elif value <= 20:
        return 20
    elif value <= 25:
        return 25
    elif value <= 30:
        return 30
    elif value <= 35:
        return 35
    elif value <= 40:
        return 40
    elif value <= 45:
        return 45
    else:
        return 50

size_mapping = {
    5:60,
    10:90,
    15:120,
    20:150,
    25:180,
    30:210,
    35:240,
    40:270,
    45:300,
    50:330
}

custom_purple = '#7E139C'
# Create a custom palette that fades from yellow to purple
#custom_palette = sns.blend_palette(['#7BC8F6','plum', custom_purple], as_cmap=True)
custom_palette = sns.blend_palette(['gold','orange','#7E139C', 'cornflowerblue', 'mediumturquoise'], as_cmap=True)


# Sample colors from the colormap
num_colors = 7  # Specify the number of colors to sample
#num_colors = 9  # Specify the number of colors to sample
colors = [custom_palette(i / (num_colors - 1)) for i in range(num_colors)]
#colors = [custom_palette(i / (num_colors - 1)) for i in range(num_colors)]

# Display the custom palette
sns.palplot(colors)
plt.show()

# Create a new column for size categories
stats_for_training_results['Size_Category'] = stats_for_training_results['group_size'].apply(size_category)
y_value = 'PC2' #Should note here that PC3 is similar across the silica range. PC2 shows a change in variance between mafic and felsic
sns.scatterplot(data=stats_for_training_results, x='actual_silica', y=y_value, size ='Size_Category', sizes=size_mapping, hue ='Size_Category',palette=colors)
plt.ylabel('Sample PC2 variance')
plt.yticks(range(0,round(stats_for_training_results[y_value].max())+1, 1))
plt.ylim(0, None)
plt.xticks(range(round(stats_for_training_results['actual_silica'].min()-2),round(stats_for_training_results['actual_silica'].max())+2, 2))
plt.xlim(round(stats_for_training_results['actual_silica'].min()-2), None)
plt.xlabel('Whole-rock silica (%)')
plt.show()

#iterate through samples
list_of_samples = case_study_grain_estimates['sample_id'].unique()
all_sample_results = []
np.random.seed(42)
for sampleid in list_of_samples:
    df_sample = case_study_grain_estimates[case_study_grain_estimates['sample_id'] == sampleid]

    #create lists to hold test results
    sample_variance_list = []
    list_of_tests = []
    table_headers = []

    #define the number of tests to be run on the sample
    grain_count = df_sample.shape[0]
    grain_count_tests = np.arange(1, grain_count + 1)

    #run the tests
    for test in grain_count_tests:
        test_results = []
        #each test will be run 100 times
        for y in range(100):
            samples = df_sample.sample(n=test, replace=False)
            median = samples['predicted_silica'].median()
            error = abs(df_sample['SiO2_pct'].median() - median) #All SiO2_pct values are the same, so median or mean or value for first row will all return the same value
            test_results.append(error) #this is the median value for a test (1 to 100) at a given grain count
            #test_results.append(median)#(error) #this is the median value for a test (1 to 100) at a given grain count
        list_of_tests.append(test_results)
        table_headers.append(f'GrainCount_{test}')
    transposed = np.array(list_of_tests).T.tolist()
    df_sample_results = pd.DataFrame(transposed, columns = table_headers) #name the headers according to tests 1-100
    df_sample_results.insert(loc=0,column='sample_id', value = sampleid)
    df_sample_results.insert(loc=1,column='SiO2_pct', value = median)
    all_sample_results.append(df_sample_results)
final_results_table = pd.concat(all_sample_results)
final_results_table.to_csv(output_file)

#calculate variances on tests per sample:
variance_table = final_results_table.drop('SiO2_pct', axis=1)

def transpose_table(table):
    transpose_stats_table = table.transpose()
    transpose_stats_table = transpose_stats_table.reset_index()
    transpose_stats_table['X'] = transpose_stats_table.index + 1
    transpose_stats_table.drop('index', axis=1, inplace=True)
    return transpose_stats_table

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 8
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

#plot the variance
stats_table = variance_table.groupby('sample_id').var()  # .quantile(0.95)#
variance_plot_table = transpose_table(stats_table)
for sample in list_of_samples:
    sns.lineplot(data=variance_plot_table, x='X', y=sample, label=sample, ax= axs[0])
axs[0].set_ylabel('Variance in median silica prediction error')
axs[0].set_ylim(0, None)
axs[0].set_yticks(range(0,round(variance_plot_table.iloc[:, :7].max(skipna=True).max())+2, 1))
axs[0].set_xticks(range(1,24, 1))
axs[0].set_xlim(1, None)
axs[0].set_xlabel('Grain count')
axs[0].legend(title = 'Sample')

#plot the 95% variance in error
stats_table = variance_table.groupby('sample_id').quantile(0.95)
quantile_plot_table = transpose_table(stats_table)
for sample in list_of_samples:
    sns.lineplot(data=quantile_plot_table, x='X', y=sample, label=sample, ax= axs[1])
axs[1].axhline(y=5, color='gray', linestyle='--')
axs[1].set_ylabel('95% percentile of median silica prediction error')
axs[1].set_ylim(0, None)
axs[1].set_yticks(range(0,round(quantile_plot_table.iloc[:, :7].max(skipna=True).max())+2, 1))
axs[1].set_xticks(range(1,24, 1))
axs[1].set_xlim(1, None)
axs[1].set_xlabel('Grain count')
axs[1].legend(title = 'Sample')

#plot the median of error
stats_table = variance_table.groupby('sample_id').median()
median_plot_table = transpose_table(stats_table)
for sample in list_of_samples:
    sns.lineplot(data=median_plot_table, x='X', y=sample, label=sample, ax= axs[2])
axs[2].set_ylabel('50% percentile of median silica prediction error')
axs[2].set_ylim(0, None)
axs[2].set_yticks(range(0,round(median_plot_table.iloc[:, :7].max(skipna=True).max())+2, 1))
axs[2].axhline(y=5, color='gray', linestyle='--')
axs[2].set_xticks(range(1,24, 1))
axs[2].set_xlim(1, None)
axs[2].set_xlabel('Grain count')
axs[2].legend(title = 'Sample')

plt.show()
print('Complete')





