from datetime import datetime, date

from Utils import create_output_subfolders, execute_test, plot_subsampling_results, lineplots_of_prediction_metrics, IncrementType, \
    list_mse_and_mape_for_all_iterations

'''
Hypothesis: 
    Geoscience data is often genetically structured. The impact of the data leakage is affected by the strength of the 
    predictive relationship between the input features and target variable, with weaker relationships having higher risk 
    of incorrectly appearing predictive. Conversely, when the signal between input features and output target variable 
    is strong, data leakage will be a lesser contributor to the overall predictive power of the model. 
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    The weighted contribution of latent variable A to observation features 2 and 3 is increased incrementally for 
    several iterations. As the weighting increases, the relative contribution of noise decreases and the predictive link
     between input features and output target variable strengthens. At each step the experiment is rerun 100 times.
'''

# Set up number of test iterations
number_of_test_iterations = 30
number_of_entities_per_test = 100
increment = 1
increment_type = IncrementType.COEFFICIENT
number_of_observations_per_entity = 50
intraclass_variability=0.5
interclass_variability=3
coefficient=1
test_name = 'Test4'
reporting = False
runs_per_iteration= 100
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

# Plot test results
if reporting:
    plot_subsampling_results(
        number_of_test_iterations,
        number_of_entities_per_test,
        number_of_observations_per_entity,
        interclass_variability,
        intraclass_variability,
        coefficient,
        results.list_el_predictions,
        results.list_ol_predictions,
        supporting_data_folder,
        increment_type,
        increment
    )

print(f'Complete.')




