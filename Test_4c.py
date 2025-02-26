
from Utils import create_output_subfolders, execute_test, plot_subsampling_results, lineplots_of_prediction_metrics, IncrementType, \
    list_mse_and_mape_for_all_iterations

'''
Hypothesis: 
    Geoscience data is often genetically structured. Splitting by observation allows for data leakage because of
    these genetic relationships. This data leakage results in inflated performance measurements.	
    The size of this effect decreases when the number of entities is increased, not the number of observations
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    The test is repeated 100 times and the resultant MSE values are presented in box-plot format.
    In each iteration, the number of entities remains constant and the number of observations is increased.
    This is a replicate of Test 2b but instead of predicting variable 1 we are predicting feature 2.
    I.e. 3 features of an observation are used to predict the 2nd feature. 
    This is a many-to-many relationship in the sense that the predictive value is not constant for multiple observations
'''

# Set up number of test iterations
number_of_test_iterations = 15
number_of_entities_per_test = 100
increment = 50
increment_type = IncrementType.OBSERVATION
number_of_observations_per_entity = 50
intraclass_variability=0.5
interclass_variability=1
coefficient=1
runs_per_iteration= 100
test_name = 'Test4c'
reporting = False
target_variable = 'variable2'

parameter_dict= {
    'number_of_test_iterations': [number_of_test_iterations],
    'number_of_entities_per_test': [number_of_entities_per_test],
    'increment': [increment],
    'increment_type': [increment_type.name],
    'number_of_observations_per_entity': [number_of_observations_per_entity],
    'intraclass_variability': [intraclass_variability],
    'interclass_variability': [interclass_variability],
    'coefficient': [coefficient],
    'runs_per_iteration':[runs_per_iteration],
    'test_name': [test_name],
    'reporting': [reporting],
    'target_variable': [target_variable]
}

# Create output folder each time the script is run
results_folder,supporting_data_folder = create_output_subfolders(parameter_dict, parent_folder='Outputs', name=test_name)

# Test subsampling strategies
list_mse_el,list_mse_el_5, list_mse_el_95, list_mse_ol,list_mse_ol_5, list_mse_ol_95, list_el_predictions, list_ol_predictions,list_data_for_plotting = execute_test(
    target_variable,
    number_of_test_iterations,
    number_of_entities_per_test,
    number_of_observations_per_entity,
    coefficient,
    supporting_data_folder,
    intraclass_variability,
    interclass_variability,
    increment_type,
    increment,
    reporting,
    runs_per_iteration
)

#Plot line plots of mse vs iterations
lineplots_of_prediction_metrics(
    list_mse_el,
    list_mse_el_5,
    list_mse_el_95,
    list_mse_ol,
    list_mse_ol_5,
    list_mse_ol_95,
    number_of_entities_per_test,
    number_of_observations_per_entity,
    number_of_test_iterations,
    interclass_variability,
    intraclass_variability,
    coefficient,
    results_folder,
    increment_type,
    increment
    )

# Plot scatter plots of each iteration actual vs predicted
if reporting:
    plot_subsampling_results(
        number_of_test_iterations,
        number_of_entities_per_test,
        number_of_observations_per_entity,
        interclass_variability,
        intraclass_variability,
        coefficient,
        list_el_predictions,
        list_ol_predictions,
        supporting_data_folder,
        increment_type,
        increment
    )

print('Complete')


