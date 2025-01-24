
from Utils import create_output_subfolders,execute_test, plot_subsampling_results,plot_scatterplots

'''
Hypothesis: 
    Geoscience data is often genetically structured. Splitting by observation allows for data leakage because of
    these genetic relationships. This data leakage results in inflated performance measurements.	
    The size of this effect decreases when the number of entities is increased, not the number of observations
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    The test is repeated 100 times and the resultant MSE values are presented in box-plot format.
    In each iteration, the number of entities remains constant and the number of observations is increased.
'''

# Set up number of test iterations
number_of_test_iterations = 10
number_of_entities_per_test = 100
entity_increment = 50
number_of_observations_per_entity = 50
intraclass_variability=0
interclass_variability=1

# Create output folder each time the script is run
results_folder,supporting_data_folder = create_output_subfolders(parent_folder='Outputs', name="Test2b")

# Test subsampling strategies
target_variable = 'variable1'
list_sl_predictions, list_ol_predictions = execute_test(target_variable,
                                                               number_of_test_iterations,
                                                               number_of_entities_per_test,
                                                               number_of_observations_per_entity,
                                                               supporting_data_folder,
                                                               intraclass_variability,
                                                               interclass_variability,
                                                               entity_increment,
                                                               False)
# Plot test results
list_mse_el, list_mape_el, list_mse_ol, list_mape_ol = plot_subsampling_results(number_of_test_iterations,
                                                                                number_of_entities_per_test,
                                                                                number_of_observations_per_entity,
                                                                                entity_increment,
                                                                                list_sl_predictions,
                                                                                list_ol_predictions,
                                                                                results_folder,
                                                                                entity_increment,
                                                                                False
                                                                                )


plot_scatterplots(list_mse_el,
                  list_mape_el,
                  list_mse_ol,
                  list_mape_ol,
                  number_of_entities_per_test,
                  number_of_observations_per_entity,
                  number_of_test_iterations,
                  results_folder,
                  entity_increment,
                  False)

print('Complete')
# Plot pairplots of datasets for each test iteration
# and the boxplots of datasets versus the target variable for each test iteration.
#This is time-consuming and should be commented out if not needed
'''for i in range(len(list_data_for_plotting)):
    data_pairplot(list_data_for_plotting[i], i, False, supporting_data_folder)
    data_boxplots(list_data_for_plotting[i], target_variable, i, False,supporting_data_folder)

plt.close('all')'''

