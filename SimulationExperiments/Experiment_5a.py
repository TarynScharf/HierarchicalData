from Utils import (plot_boxplots_of_subsampling_results,create_output_subfolders,
                   execute_test,plot_subsampling_results,data_pairplot, data_boxplots,IncrementType,
                   list_mse_and_mape_for_all_iterations)
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
    This is a replicate of Experiment 1, but instead of predicting variable 1 we are predicting feature 4.
    I.e. 3 features of an observation are used to predict the 4th feature of that same observation. 
    This is a many-to-many relationship in the sense that the predictive value is not constant for multiple observations.
'''

# Set up number of test iterations
number_of_test_iterations = 1000
number_of_entities_per_test = 100
number_of_observations_per_entity = 50
intraclass_variability=0.5
interclass_variability=3
increment = None
increment_type = IncrementType.NONE
coefficient=1
runs_per_iteration= 1
test_name = 'Test5a'
reporting = False
target_variable = 'feature4'
hold_out_data = True

parameter_dict= {
    'number_of_test_iterations': [number_of_test_iterations],
    'number_of_entities_per_test': [number_of_entities_per_test],
    'increment': [increment],
    'increment_type': [increment_type.name],
    'number_of_observations_per_entity': [number_of_observations_per_entity],
    'intraclass_variability': [intraclass_variability],
    'interclass_variability': [interclass_variability],
    'coefficient': [coefficient],
    'runs_per_iteration': [runs_per_iteration],
    'test_name': [test_name],
    'reporting': [reporting],
    'target_variable': [target_variable],
    'hold_out_data':[hold_out_data]
}

# Create output folder each time the script is run
results_folder,supporting_data_folder = create_output_subfolders(parameter_dict, parent_folder='Outputs', name = test_name)

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
    runs_per_iteration,
    hold_out_data
)

'''plot_boxplots_of_subsampling_results(
    results.ave_mse_el, 
    results.ave_mse_ol,
    number_of_test_iterations, 
    results_folder)'''

plot_boxplots_of_subsampling_results(
    results.ave_mse_el,
    results.hold_out_mse_el,
    results.ave_mse_ol,
    results.hold_out_mse_ol,
    number_of_test_iterations,
    results_folder
)

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

    for i in range(len(results.list_data_for_plotting)):
        data_pairplot(results.list_data_for_plotting[i], i, False, supporting_data_folder)
        data_boxplots(results.list_data_for_plotting[i], target_variable, i, False,supporting_data_folder)

print('complete')