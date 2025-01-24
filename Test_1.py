from Utils import plot_boxplots_of_subsampling_results,create_output_subfolders,execute_test,plot_subsampling_results,data_pairplot, data_boxplots

'''
Hypothesis: 
    Geoscience data is often genetically structured. Splitting by observation allows for data leakage because of
    these genetic relationships. This data leakage results in inflated performance measurements.
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    The test is repeated 100 times and the resultant MSE values are presented in box-plot format.
'''

# Set up number of test iterations
number_of_test_iterations = 5
number_of_entities_per_test = 100
entity_increment = None
number_of_observations_per_entity = 50

# Create output folder each time the script is run
results_folder,supporting_data_folder = create_output_subfolders(parent_folder='Outputs', name = 'Test1')

# Test subsampling strategies
target_variable = 'variable1'
list_sl_predictions, list_ol_predictions = execute_test(target_variable,
                                                               number_of_test_iterations,
                                                               number_of_entities_per_test,
                                                               number_of_observations_per_entity,
                                                               supporting_data_folder
                                                               )

# Plot test results
list_mse_el, list_mape_el, list_mse_ol, list_mape_ol = plot_subsampling_results(number_of_test_iterations,
                                                                                number_of_entities_per_test,
                                                                                entity_increment,
                                                                                list_sl_predictions,
                                                                                list_ol_predictions,
                                                                                supporting_data_folder)

plot_boxplots_of_subsampling_results(list_mse_el, list_mape_el, list_mse_ol, list_mape_ol, results_folder)

'''for i in range(len(list_data_for_plotting)):
    data_pairplot(list_data_for_plotting[i], i, False, supporting_data_folder)
    data_boxplots(list_data_for_plotting[i], target_variable, i, False,supporting_data_folder)

plt.close('all')'''
'''
#Set up lists to record results per test iteration
list_mse_el = []
list_mape_el = []
list_mse_ol = []
list_mape_ol = []

#Create a plot that will record the results of data splitting
fig_model_results, axs_model_results = plt.subplots(nrows=number_of_test_iterations, ncols=2, sharey=True, figsize=(12, 6 * number_of_test_iterations))
axs_model_results_flatten= axs_model_results.flatten()

list_data_for_plotting = []

#Run the test iterations
for i in range(number_of_test_iterations):
    print (f"Test iteration {i+1} proceeding...")
    sampler = FictitiousSampler(
        intraclass_variability=0,
        interclass_variability=1
    )

    entity_observation_pairs = sampler.generate_entity_observation_pairs(
        number_of_entities=number_of_entities_per_test,
        average_number_of_observations_per_entity=number_of_observations_per_entity
    )
    list_data_for_plotting.append(entity_observation_pairs)

    sl_predictions = predict(entity_observation_pairs,SplittingStrategy.ENTITY_LEVEL,target_variable,i, supporting_data_folder)
    ol_predictions = predict(entity_observation_pairs,SplittingStrategy.OBSERVATION_LEVEL,target_variable,i,supporting_data_folder)
    mse_el, mape_el, mse_ol, mape_ol = plot_prediction_results(axs_model_results_flatten, i, number_of_entities_per_test, sl_predictions, ol_predictions)
    list_mse_el.append(mse_el)
    list_mape_el.append(mape_el)
    list_mse_ol.append(mse_ol)
    list_mape_ol.append(mape_ol)

plt.tight_layout()
file_path_model_results = os.path.join(supporting_data_folder, 'subsampling_results.svg')
plt.savefig(file_path_model_results)
plt.close('all')

# Plot pairplots of datasets for each test iteration
# and the boxplots of datasets versus the target variable for each test iteration.
#This is time-consuming and should be commented out if not needed


#Create the final output boxplot that shows how mse and mape vary across all test iterations
fig = plt.figure(figsize=(8, 6))
ax_mse_boxplot = fig.add_subplot(111)

df_results = pd.DataFrame({
    'entity_mse':list_mse_el,
    'observation_mse':list_mse_ol,
    'entity_mape':list_mape_el,
    'observation_mape':list_mape_ol
})

ax_mse_boxplot = df_results.boxplot(figsize=(8, 6), showfliers=False)
ax_mse_boxplot.set_title("Test results over 100 iterations")
file_path_mse_boxplots = os.path.join(results_folder, f'mse_boxplots.svg')
with pd.ExcelWriter(os.path.join(results_folder, 'mse_results.xlsx'), engine='openpyxl') as writer:
    df_results.to_excel(writer, sheet_name='MSE_RESULTS', index=False)
ax_mse_boxplot.savefig(file_path_mse_boxplots, dpi=300, bbox_inches="tight")
plt.close('all')'''
print('complete')