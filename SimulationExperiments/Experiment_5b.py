
from Utils import create_output_subfolders, execute_test, plot_subsampling_results, lineplots_of_prediction_metrics, IncrementType, \
    list_mse_and_mape_for_all_iterations

'''
Hypothesis: 
    Geoscience data is often genetically structured. The phenomenon of data leakage due to hierarchical structuring can 
    occur in both many-to-one relationships (e.g. data from many minerals used to predict a value about the single source 
    rock) and one-to-one (e.g. data from a mineral used to predict another characteristic about that same mineral). 
    As observations are constructed using entity latent variables, observations from a given entity are expected to be 
    more like one another, and less like observations derived from a separate entity. Consequently, hierarchical data 
    structures should be considered when splitting data in a one-to-one predictive relationship.
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    In each iteration, the number of entities is increased.
    This is a replicate of Experiment 2.1 but instead of predicting variable 1 we are predicting feature 2.
    I.e. 3 features of an observation are used to predict the 2nd feature. 
    This is a many-to-many relationship in the sense that the predictive value is not constant for multiple observations.
'''

# Set up number of test iterations
number_of_test_iterations = 25
number_of_entities_per_test = 100
increment = 100
increment_type = IncrementType.ENTITY
number_of_observations_per_entity = 50
intraclass_variability=0.5
interclass_variability=3
coefficient=1
runs_per_iteration= 100
test_name = 'Test5b'
reporting = False
target_variable = 'feature4'


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

# Plot scatter plots of each iteration actual vs predicted
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

print('Complete')


