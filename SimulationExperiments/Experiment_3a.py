
from Utils import create_output_subfolders, execute_test, plot_subsampling_results, lineplots_of_prediction_metrics, IncrementType, \
    list_mse_and_mape_for_all_iterations, plot_differences

'''
Hypothesis: 
    Geoscience data is often genetically structured. When the variance between entities is high, data leakage has a 
    larger impact on model performance metrics. Here, ‘impact’ refers to the underestimation of prediction error and 
    thus prediction uncertainty on new entities. 
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    In each iteration, the amount of entities, observations, and intersample variance remains constant.
    The intrasample variance increases in each iteration.
'''

# Set up number of test iterations
number_of_test_iterations = 100
number_of_entities_per_test = 100
increment = 0.5
increment_type = IncrementType.INTRASAMPLE_VARIANCE
number_of_observations_per_entity = 50
intraclass_variability=0.5
interclass_variability=3
coefficient=1
runs_per_iteration= 100
test_name = 'Test3a'
reporting = False
target_variable = 'variable1'

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
results = execute_test(
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
    results.ave_mse_el,
    results.ave_mse_el_5,
    results.ave_mse_el_95,
    results.ave_mse_ol,
    results.ave_mse_ol_5,
    results.ave_mse_ol_95,
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

'''plot_differences(
    list_mse_el,
    list_mse_ol,
    number_of_entities_per_test,
    number_of_observations_per_entity,
    number_of_test_iterations,
    interclass_variability,
    intraclass_variability,
    results_folder,
    increment_type,
    increment
)'''

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


