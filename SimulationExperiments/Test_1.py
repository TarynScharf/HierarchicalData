from Utils import (plot_boxplots_of_subsampling_results,create_output_subfolders,
                   execute_test,plot_subsampling_results,data_pairplot, data_boxplots,IncrementType,
                   list_mse_and_mape_for_all_iterations)
'''
Hypothesis: 
    Geoscience data is often genetically structured. Splitting by observation allows for data leakage because of
    these genetic relationships. This data leakage results in inflated performance measurements.
Test:
    Create multiple entities and test model performance with observation-level and entity-level data splitting.
    The test is repeated 100 times and the resultant MSE values are presented in box-plot format.
'''

# Set up number of test iterations
number_of_test_iterations = 1000 #as each iteration is performed 10 times and the results of average, for Test 1 this amounts to 10 * number_of_test_iterations
number_of_entities_per_test = 100
number_of_observations_per_entity = 50
intraclass_variability=15
interclass_variability=3
increment = None
increment_type = IncrementType.NONE
coefficient=1
runs_per_iteration= 1
test_name = 'Test1'
reporting = True
target_variable = 'variable1'
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
    'runs_per_iteration':[runs_per_iteration],
    'test_name': [test_name],
    'reporting': [reporting],
    'target_variable': [target_variable],
    'hold_out_data':[hold_out_data]
}

# Create output folder each time the script is run
results_folder,supporting_data_folder = create_output_subfolders(parameter_dict,parent_folder='Outputs', name = test_name)

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