import os
import statistics
from datetime import datetime
from enum import Enum
from typing import Tuple, List
import numpy as np
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sympy.logic.boolalg import Boolean
from tqdm import tqdm

from fictitious_sampler_predictive import FictitiousSamplerPredictive
from testing_framework import generate_entity_observation_dataframe, SplittingStrategy, predict, PredictionData


class IncrementType(Enum):
    ENTITY = 1
    OBSERVATION = 2
    INTRASAMPLE_VARIANCE = 3
    INTERSAMPLE_VARIANCE = 4
    NONE = 5

def create_output_subfolders(parent_folder:str, name: str):
    now = datetime.now()
    formatted_datetime = now.strftime("%Y%m%d%H%M%S")
    output_folder = os.path.join(os.path.dirname(__file__), parent_folder)
    Test_run_folder = os.path.join(output_folder, f"{name}_{formatted_datetime}")
    supporting_data_folder = os.path.join(Test_run_folder,'supporting_data')
    results_folder = os.path.join(Test_run_folder,'results')
    if not os.path.exists(Test_run_folder):  # Check if it already exists
        os.makedirs(Test_run_folder)
        os.makedirs(supporting_data_folder)
        os.makedirs(results_folder)

    return results_folder,supporting_data_folder

def lineplots_of_prediction_metrics(
    list_mse_el,
    list_mse_el_5,
    list_mse_el_95,
    list_mse_ol,
    list_mse_ol_5,
    list_mse_ol_95,
    number_of_entities,
    number_of_observations,
    number_of_test_iterations,
    interclass_variability,
    intraclass_variability,
    output_folder,
    increment_type,
    increment
    ):
        # Create the final output boxplot that shows how mse and mape vary across all test iterations
        if increment is not None:
            match increment_type:
                case IncrementType.ENTITY:
                    number_per_iteration = np.arange(
                        number_of_entities,
                        number_of_entities +  number_of_test_iterations * increment,
                        increment
                    )

                case IncrementType.OBSERVATION:
                    number_per_iteration = np.arange(
                        number_of_observations,
                        number_of_observations + number_of_test_iterations * increment,
                        increment)

                case IncrementType.INTERSAMPLE_VARIANCE:
                    number_per_iteration = np.arange(
                        interclass_variability,
                        interclass_variability + number_of_test_iterations * increment,
                        increment)

                case IncrementType.INTRASAMPLE_VARIANCE:
                    number_per_iteration = np.arange(
                        intraclass_variability,
                        intraclass_variability + number_of_test_iterations * increment,
                        increment)

                case IncrementType.NONE:
                    print('No incremental data to plot')
                    exit

        df_results = pd.DataFrame({
            'entity_mse': list_mse_el,
            'entity_mse_5p':list_mse_el_5,
            'entity_mse_95p':list_mse_el_95,
            'observation_mse': list_mse_ol,
            'observation_mse_5p':list_mse_ol_5,
            'observation_mse_95p':list_mse_ol_95

        })

        plt.figure(figsize=(20, 5))
        plt.plot(number_per_iteration, list_mse_el, color='red',ls = '-',marker = 'o', alpha=0.7, label='Entity-level MSE')
        plt.plot(number_per_iteration, list_mse_el_5, color='red',ls = '--',marker = 'o', alpha=0.7, label='Entity-level MSE 5th')
        plt.plot(number_per_iteration, list_mse_el_95, color='red',linestyle = ':',marker = 'o', alpha=0.7, label='Entity-level MSE 95th')

        plt.plot(number_per_iteration, list_mse_ol, color='blue', alpha=0.7,marker = 'o', label='Observation-level MSE')
        plt.plot(number_per_iteration, list_mse_ol_5, color='blue',ls = '--',marker = 'o', alpha=0.7, label='Observation-level MSE 5th')
        plt.plot(number_per_iteration, list_mse_ol_95, color='blue',ls = ':',marker = 'o', alpha=0.7, label='Observation-level MSE 95th ')

        plt.title(f'MSE comparison')
        plt.xlabel(f'{increment_type.name.lower()} count')
        plt.legend()

        #axes[1].plot(number_per_iteration, list_mape_el, color='red', alpha=0.7,label='Entity-level MAPE')
        #axes[1].plot(number_per_iteration, list_mape_ol, color='blue', alpha=0.7,label='Observation-level MAPE')
        #axes[1].set_title(f'MAPE comparison')
        #axes[1].set_xlabel(f'{increment_type.name.lower()} count')
        #axes[1].legend()

        file_path_mse_scatter = os.path.join(output_folder, f'mse_scatterplots.svg')
        plt.savefig(file_path_mse_scatter, dpi=300, bbox_inches="tight")
        plt.close('all')

        with pd.ExcelWriter(os.path.join(output_folder, 'mse_results.xlsx'), engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='MSE_RESULTS', index=False)



def plot_differences(
    list_mse_el,
    list_mse_ol,
    number_of_entities,
    number_of_observations,
    number_of_test_iterations,
    interclass_variability,
    intraclass_variability,
    output_folder,
    increment_type,
    increment
    ):
    # Create the final output boxplot that shows how mse and mape vary across all test iterations
    if increment is not None:
        match increment_type:
            case IncrementType.ENTITY:
                number_per_iteration = np.linspace(
                    number_of_entities,
                    number_of_test_iterations * increment,
                    number_of_test_iterations
                )

            case IncrementType.OBSERVATION:
                number_per_iteration = np.linspace(
                    number_of_observations,
                    number_of_test_iterations * increment,
                    number_of_test_iterations)

            case IncrementType.INTERSAMPLE_VARIANCE:
                number_per_iteration = np.linspace(
                    interclass_variability,
                    number_of_test_iterations * increment,
                    number_of_test_iterations)

            case IncrementType.INTRASAMPLE_VARIANCE:
                number_per_iteration = np.linspace(
                    intraclass_variability,
                    number_of_test_iterations * increment,
                    number_of_test_iterations)

            case IncrementType.NONE:
                print('No incremental data to plot')
                exit

    df_results = pd.DataFrame({
        'entity_mse': list_mse_el,
        'observation_mse': list_mse_ol,
    })

    df_results['mse_diff'] = abs(df_results['entity_mse'] - df_results['observation_mse'])

    plt.figure(figsize=(20, 5))
    plt.plot(number_per_iteration, df_results['mse_diff'], color='green', alpha=0.7)
    plt.title(f'Entity MSE - Observation MSE')
    plt.xlabel(f'{increment_type.name.lower()} count')

    #axes[1].plot(number_per_iteration, df_results['mape_diff'], color='green', alpha=0.7)
    #axes[1].set_title(f'Entity MAPE - observation MAPE')
    #axes[1].set_xlabel(f'{increment_type.name.lower()} count')

    file_path_mse_scatter = os.path.join(output_folder, f'diff_with_increasing_{increment_type.name.lower()}.svg')
    plt.savefig(file_path_mse_scatter, dpi=300, bbox_inches="tight")
    plt.close('all')

def plot_boxplots_of_subsampling_results(list_mse_el,
                                         list_mse_ol,
                                         output_folder):

    # Create the final output boxplot that shows how mse and mape vary across all test iterations
    df_results = pd.DataFrame({
        'entity_mse': list_mse_el,
        'observation_mse': list_mse_ol,
    })
    plt.figure(figsize=(8, 6))

    ax_mse_boxplot = df_results.boxplot(figsize=(8, 6), showfliers=False)
    ax_mse_boxplot.set_title("Test results over iterations")
    file_path_mse_boxplots = os.path.join(output_folder, f'mse_boxplots.svg')

    with pd.ExcelWriter(os.path.join(output_folder, 'mse_results.xlsx'), engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='MSE_RESULTS', index=False)
    plt.savefig(file_path_mse_boxplots, dpi=300, bbox_inches="tight")
    plt.close('all')

def return_mse_and_mape(test, predict):
    mse = sklearn.metrics.mean_squared_error(test, predict)
    mape = sklearn.metrics.mean_absolute_percentage_error(test, predict)
    return mse, mape

def plot_actual_vs_predicted(ax, title, x_bounds, predications:PredictionData):
    ax.scatter(x=predications.y_test, y=predications.y_predict,facecolor=(39/255, 56/255, 139/255, 0.4), edgecolor=(39/255, 56/255, 139/255, 0.7), marker = 'o')
    mse, mape = return_mse_and_mape(predications.y_test, predications.y_predict)
    xmin, xmax = min(predications.y_test.values), max(predications.y_test.values)
    ymin, ymax = min(predications.y_predict), max(predications.y_predict)
    common_min = min(xmin,ymin )
    common_max = max(xmax,ymax)
    ax.set_xlim(common_min, common_max)
    ax.set_ylim(common_min, common_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title(title)
    ax.text(x=0.05, y=0.95,s=f'mse: {mse:.2f} \nmape: {mape:.2f}',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
    return mse, mape

def plot_prediction_results(
        list_of_plot_axes,
        test_number:int,
        entity_count:int,
        number_of_observations_per_entity: int,
        interclass_variability: float,
        intraclass_variability: float,
        entity_level_results:PredictionData,
        observation_level_results:PredictionData
    ):

    ax_sl = list_of_plot_axes[test_number * 2]
    ax_ol = list_of_plot_axes[test_number * 2 + 1]

    title = f'Test {test_number+1}, {entity_count} ent, {number_of_observations_per_entity} obs, {interclass_variability} inter_v, {intraclass_variability} intra_v'
    min_x = min(min(observation_level_results.y_test), min(entity_level_results.y_test))
    max_x = max(max(observation_level_results.y_test), max(entity_level_results.y_test))
    x_bounds = (0.9*min_x, max_x*1.1)

    plot_actual_vs_predicted(ax_sl, title + ", entity splitting", x_bounds, entity_level_results)
    plot_actual_vs_predicted(ax_ol, title + ", observation splitting", x_bounds, observation_level_results)


def list_mse_and_mape_for_all_iterations(
    list_el_predictions,
    list_ol_predictions,
    number_of_test_iterations
):
    list_mse_el = []
    list_mape_el = []
    list_mse_ol = []
    list_mape_ol = []
    for i  in range(number_of_test_iterations):
        mse_el, mape_el = return_mse_and_mape(list_el_predictions[i].y_test, list_el_predictions[i].y_predict)
        mse_ol, mape_ol = return_mse_and_mape(list_ol_predictions[i].y_test, list_ol_predictions[i].y_predict)
        list_mse_el.append(mse_el)
        list_mape_el.append(mape_el)
        list_mse_ol.append(mse_ol)
        list_mape_ol.append(mape_ol)

    return list_mse_el, list_mse_ol, list_mape_el, list_mape_ol


def plot_subsampling_results(
        number_of_test_iterations,
        number_of_entities,
        number_of_observations_per_entity,
        interclass_variability,
        intraclass_variability,
        list_sl_predictions,
        list_ol_predictions,
        results_folder,
        increment_type,
        increment=None
):
    fig_model_results, axs_model_results = plt.subplots(
        nrows=number_of_test_iterations,
        ncols=2,
        figsize=(12, 6 * number_of_test_iterations)
    )

    list_of_axs_for_plotting_model_results = axs_model_results.flatten()

    for i in range(number_of_test_iterations):
        plot_prediction_results(
            list_of_axs_for_plotting_model_results,
            i,
            number_of_entities,
            number_of_observations_per_entity,
            interclass_variability,
            intraclass_variability,
            list_sl_predictions[i],
            list_ol_predictions[i]
        )

        if increment is not None:
            match increment_type:
                case IncrementType.ENTITY:
                    number_of_entities += increment

                case IncrementType.OBSERVATION:
                    number_of_observations_per_entity += increment

                case IncrementType.INTERSAMPLE_VARIANCE:
                    interclass_variability += increment

                case IncrementType.INTRASAMPLE_VARIANCE:
                    intraclass_variability += increment

                case IncrementType.NONE:
                    continue

    plt.tight_layout()
    file_path_model_results = os.path.join(results_folder, 'subsampling_results.svg')
    plt.savefig(file_path_model_results)
    plt.close('all')

def data_pairplot(entity_observation_pairs:List[Tuple],
                  test:int, visualise: Boolean,
                  output_folder: str):
    #generte the data pairplot
    data = generate_entity_observation_dataframe(entity_observation_pairs)

    #plot data, display plot, and save the plot to the Outputs folder
    sns.pairplot(data)
    file_path_pairplot_results = os.path.join(output_folder, f'test{test+1}_data_pairplot.svg')
    plt.savefig(file_path_pairplot_results)

    if visualise:
        plt.show()

    plt.close()

def data_boxplots(entity_observation_pairs: List[Tuple],
                  target_variable: str,
                  test_number: int,
                  visualise: Boolean,
                  output_folder: str):
    fig_boxplot, axs_boxplot = plt.subplots(nrows=4, ncols=1, sharey=False, figsize=(30, 10), sharex=True)
    ax_boxplots = axs_boxplot.flatten()
    data_df = generate_entity_observation_dataframe(entity_observation_pairs)

    # sort dataframe in ascending order of the target variable
    # get tick formatted tick values
    sorted_data_df = data_df.sort_values(by=target_variable)
    tick_values = [f"{val:.1f}" for val in sorted_data_df[target_variable].unique()]

    for i in range(len(ax_boxplots)):
        sorted_data_df.boxplot(by=target_variable, column=[f'feature{i + 1}'], ax=ax_boxplots[i], grid=False,
                               showfliers=False)
        ax_boxplots[i].set_title("")
        #ax_boxplots[i].set_xlabel(target_variable)
        ax_boxplots[i].set_ylabel(f'Feature {i + 1}')

        # Customize x-ticks
        ax_boxplots[i].set_xticks(range(1, len(tick_values) + 1))
        ax_boxplots[i].set_xticklabels(tick_values)

    plt.suptitle(f"Dataset features versus {target_variable}")
    plt.tight_layout()
    file_path_data_boxplots = os.path.join(output_folder, f'test{test_number +1}_data_boxplots.svg')
    plt.savefig(file_path_data_boxplots)

    if visualise:
        plt.show()

    plt.close()

def execute_test(
        target_variable,
        number_of_test_iterations,
        number_of_entities,
        number_of_observations_per_entity,
        supporting_data_folder,
        intraclass_variability,
        interclass_variability,
        increment_type,
        increment = None,
        reporting = False
):

    list_sl_predictions = []
    list_ol_predictions = []
    list_data_for_plotting = []

    ave_mse_el = []
    ave_mse_el_5= []
    ave_mse_el_95 = []
    ave_mse_ol = []
    ave_mse_ol_5 = []
    ave_mse_ol_95 = []
    #ave_mape_el = []
    #ave_mape_ol = []

    for i in range(number_of_test_iterations):
        print (f"Test iteration {i+1} proceeding...")

        # Each iteration will be repeated 10 times and the mse and mape value averaged
        list_mse_el=[]
        list_mse_ol=[]
        #list_mape_el=[]
        #list_mape_ol=[]

        for j in range(10): # tqdm(range(10), desc='Executing...'):

            sampler = FictitiousSamplerPredictive(
                intraclass_variability=intraclass_variability,
                interclass_variability=interclass_variability
            )

            entity_observation_pairs = sampler.generate_entity_observation_pairs(
                number_of_entities=number_of_entities,
                average_number_of_observations_per_entity=number_of_observations_per_entity
            )

            el_predictions, mse_el, mape_el = predict(entity_observation_pairs,SplittingStrategy.ENTITY_LEVEL,target_variable,i, supporting_data_folder,reporting)
            ol_predictions, mse_ol, mape_ol = predict(entity_observation_pairs,SplittingStrategy.OBSERVATION_LEVEL,target_variable,i,supporting_data_folder,reporting)

            if reporting and j==0:
                #Save the iteration's data for plotting pairplots and dataset boxplots (optional)
                #Save only the first dataset produced, as an example to visualise in plotting, as the iteration is repeated 10 times
                list_data_for_plotting.append(entity_observation_pairs)
                list_sl_predictions.append(el_predictions)
                list_ol_predictions.append(ol_predictions)

            list_mse_el.append(mse_el)
            list_mse_ol.append(mse_ol)
            #list_mape_el.append(mape_el)
            #list_mape_ol.append(mape_ol)

        ave_mse_el.append(statistics.mean(list_mse_el))
        ave_mse_el_5.append(np.percentile(list_mse_el,5))
        ave_mse_el_95.append(np.percentile(list_mse_el,95))
        ave_mse_ol.append(statistics.mean(list_mse_ol))
        ave_mse_ol_5.append(np.percentile(list_mse_ol, 5))
        ave_mse_ol_95.append(np.percentile(list_mse_ol, 95))
        #ave_mape_el.append(statistics.mean(list_mape_el))
        #ave_mape_ol.append(statistics.mean(list_mape_ol))

        if increment is not None:
            match increment_type:
                case IncrementType.ENTITY:
                    number_of_entities += increment

                case IncrementType.OBSERVATION:
                    number_of_observations_per_entity += increment

                case IncrementType.INTERSAMPLE_VARIANCE:
                    interclass_variability += increment

                case IncrementType.INTRASAMPLE_VARIANCE:
                    intraclass_variability += increment

                case IncrementType.NONE:
                    continue

    if reporting:
        return ave_mse_el,ave_mse_el_5, ave_mse_el_95, ave_mse_ol,ave_mse_ol_5, ave_mse_ol_95, list_sl_predictions,list_ol_predictions,list_data_for_plotting
        #return ave_mse_el,ave_mse_ol, ave_mape_el, ave_mape_ol,list_sl_predictions,list_ol_predictions,list_data_for_plotting
    else:
        return ave_mse_el,ave_mse_el_5, ave_mse_el_95, ave_mse_ol,ave_mse_ol_5, ave_mse_ol_95, None, None, None
        #return ave_mse_el,ave_mse_ol, ave_mape_el, ave_mape_ol, None, None, None

def tune_hyperparameters(x_train, y_train):
    max_depth = [int(x) for x in range(2, 20)]
    min_samples_split = [int(x) for x in range(2, 5)]
    max_leaf_nodes = [int(x) for x in range(5, 25)]
    min_samples_leaf = [int(x) for x in range(2,5)]
    n_estimators = [20,50,100,150,200,250]
    max_features = [2,4]
    grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_leaf_nodes': max_leaf_nodes,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    rf = RandomForestRegressor(random_state=42)

    grid_search = HalvingGridSearchCV(estimator=rf,param_grid=grid, factor=3,resource='n_samples', scoring='neg_mean_absolute_percentage_error', cv=5, verbose=3)
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_
