import itertools
import os
import statistics
from datetime import datetime
from enum import Enum
from typing import Tuple, List
import numpy as np
import pandas as pd
import scipy.stats
import sklearn
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sympy.logic.boolalg import Boolean
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

from fictitious_sampler_predictive import FictitiousSamplerPredictive
from fictitious_sample_simple import FictitiousSamplerSimple
from testing_framework import *

class IncrementType(Enum):
    ENTITY = 1
    OBSERVATION = 2
    INTRASAMPLE_VARIANCE = 3
    INTERSAMPLE_VARIANCE = 4
    COEFFICIENT = 5
    NONE = 6

def create_output_subfolders(script_parameters,parent_folder:str, name: str):
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

    inputs_df = pd.DataFrame(script_parameters)
    inputs_df.to_excel(os.path.join(results_folder, 'readme.xlsx'))

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
    coefficient,
    output_folder,
    increment_type,
    increment
    ):
        # Create the final output boxplot that shows how mse and mape vary across all test iterations
        if increment is not None:
            match increment_type:
                case IncrementType.ENTITY:
                    number_per_iteration = number_of_entities + np.arange(number_of_test_iterations) * increment

                case IncrementType.OBSERVATION:
                    number_per_iteration = number_of_observations +np.arange(number_of_test_iterations)*increment

                case IncrementType.INTERSAMPLE_VARIANCE:
                    number_per_iteration = interclass_variability +np.arange(number_of_test_iterations)*increment

                case IncrementType.INTRASAMPLE_VARIANCE:
                    number_per_iteration = intraclass_variability +np.arange(number_of_test_iterations)*increment

                case IncrementType.COEFFICIENT:
                    number_per_iteration = coefficient + np.arange(number_of_test_iterations) * increment

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

        df_results['mse_diff'] = abs(df_results['entity_mse'] - df_results['observation_mse'])
        with pd.ExcelWriter(os.path.join(output_folder, 'mse_results.xlsx'), engine='openpyxl') as writer:
            df_results.to_excel(writer, sheet_name='MSE_RESULTS', index=False)

        width = 18.2/2.54 #converting cm to inches because matplotlib takes dimensions in inches
        height = 5/2.54 #converting cm to inches
        fig,ax1 = plt.subplots(1,1,figsize=(width,height))
        plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype':42, 'axes.linewidth':1})

        red_line = (206/255, 61/255, 48/255,0.9) #rgba
        red_envelope = (242/255, 71/255, 56/255,0.6)
        red_marker_fill = (242/255, 143/255, 107/255,1.0)

        blue_line = (29/255, 66/255, 115/255, 0.9)
        blue_envelope = (4/255, 196/255, 217/255,0.6)
        blue_marker_fill = (4/255, 196/255, 217/255,1.0)
        purple_line = (114/255, 62/255, 152/255, 1.0)

        line_graph_width = 1.0
        thin_line_width = 0.5
        marker_size = 3

        ax1.fill_between(number_per_iteration,list_mse_el_5,list_mse_el_95, color=red_envelope, linewidth = thin_line_width)
        ln_entities = ax1.plot(number_per_iteration, list_mse_el, color=red_line,ls = '-',linewidth = line_graph_width,marker = 'o',markeredgecolor=red_line,markeredgewidth = thin_line_width,markerfacecolor=red_marker_fill,markersize=marker_size,  label='Entity-split')

        ax1.fill_between(number_per_iteration, list_mse_ol_5, list_mse_ol_95, color='white', alpha=0.6, linewidth = thin_line_width)
        ax1.fill_between(number_per_iteration, list_mse_ol_5, list_mse_ol_95, color=blue_envelope,linewidth = thin_line_width)
        ln_observations = ax1.plot(number_per_iteration, list_mse_ol, color=blue_line,linewidth = line_graph_width, marker = 'o',markeredgecolor=blue_line,markeredgewidth = thin_line_width, markerfacecolor=blue_marker_fill,markersize=marker_size, label='Observation-split')

        ax2 = ax1.twinx()
        ln_differences = ax2.plot(number_per_iteration, df_results['mse_diff'],linestyle='dashed',dashes=(2,2),linewidth = line_graph_width, color=purple_line,label='MSE difference')
        ax2.set_ylabel('MSE difference', rotation=90)

        lines = ln_entities +ln_observations+ln_differences
        labels = [l.get_label() for l in lines]
        plt.legend(lines,labels)

        ax1.tick_params('both', length=3, width=1, which='major')
        ax2.tick_params('both', length=3, width=1, which='major')

        plt.title(f'')
        ax1.set_ylabel('Mean squared error')
        ax1.set_xlabel(f'{increment_type.name.capitalize()} count')

        plt.tight_layout(pad=0.5)
        file_path_mse_scatter = os.path.join(output_folder, f'mse_scatterplots.pdf')
        plt.savefig(file_path_mse_scatter, dpi=500)
        plt.close('all')

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

def plot_boxplots_of_subsampling_results(
        list_mse_el: object,
        hold_out_mse_el: object,
        list_mse_ol: object,
        hold_out_mse_ol: object,
        number_of_test_iterations: object,
        output_folder: object
) -> None:

    # Create the final output boxplot that shows how mse and mape vary across all test iterations
    df_results = pd.DataFrame({
        'entity_mse': list_mse_el,
        'hold_out_entity_mse':hold_out_mse_el,
        'observation_mse': list_mse_ol,
        'hold_out_observation_mse':hold_out_mse_ol
    })

    with pd.ExcelWriter(os.path.join(output_folder, 'mse_results.xlsx'), engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='MSE_RESULTS', index=False)

    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'pdf.fonttype': 42, 'axes.linewidth': 1})
    width = 8/2.54 #converting cm to inches because matplotlib takes dimensions in inches
    height = 6/2.54 #converting cm to inches
    positions=[-0.4, -0.2, 0.2,0.4]

    fig = plt.figure(figsize=(width, height))
    ax_mse_boxplot = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_mse_boxplot.set_position([0.1, 0.1, 0.9, 0.9])
    ymin = df_results.values.min()
    ymax = df_results.values.max()

    ax_mse_boxplot.set_ylim(ymin-0.1, ymax+0.1)
    ax_mse_boxplot.set_xlim(-1,3)
    violins = ax_mse_boxplot.violinplot(dataset = df_results, positions=positions, widths=0.3)
    colours=[(242/255, 71/255, 56/255),(242/255, 71/255, 56/255),(4/255, 196/255, 217/255),(4/255, 196/255, 217/255)]
    for i, box in enumerate(violins['bodies']):
        box.set_facecolor(colours[i])
        box.set_edgecolor('black')
        box.set_linewidth(1)

    for line in ax_mse_boxplot.lines:
        line.set_color('black')
        line.set_linewidth(1)


    ks_test_results = []
    populations_to_compare= df_results.columns.values
    population_pairs = list(itertools.combinations(range(4), 2))

    for i, j in population_pairs:
        results = scipy.stats.ks_2samp(df_results[populations_to_compare[i]], df_results[populations_to_compare[j]])
        ks_test_results.append({'Pair': f"{populations_to_compare[i]} vs {populations_to_compare[j]}", 'Statistic': results.statistic, 'P-Value': results.pvalue})

    # Adjust p-values for multiple testing
    pvals = [r['P-Value'] for r in ks_test_results]
    adjusted = multipletests(pvals, method='bonferroni')  # or 'fdr_bh' for Benjamini-Hochberg
    for i, adj_p in enumerate(adjusted[1]):
        ks_test_results[i]['Adjusted_P_Value'] = adj_p
        ks_test_results[i]['Reject_Null_(Adj)'] = adjusted[0][i]

    ks_results_df = pd.DataFrame(ks_test_results)
    file_path_ks_results = os.path.join(output_folder, f'test1_ks_results.csv')
    ks_results_df.to_csv(file_path_ks_results)

    ax_mse_boxplot.set_title(f"")
    #ax_mse_boxplot.set_yscale('log')
    ax_mse_boxplot.tick_params(axis='y', which = 'both', length=3, width=1)
    ax_mse_boxplot.tick_params(left=False, right=True, labelright=True, labelleft=False, bottom=False,
                               labelbottom=False, direction='in', pad=-20)
    ax_mse_boxplot.get_yaxis().set_ticks_position("right")
    ax_mse_boxplot.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_mse_boxplot.text(0.8, 0.5, 'Mean squared error', transform=ax_mse_boxplot.transAxes,
            rotation=90, verticalalignment='center', horizontalalignment='right', fontsize=10)

    file_path_mse_boxplots = os.path.join(output_folder, f'mse_boxplots.pdf')
    #plt.tight_layout(pad=0)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(file_path_mse_boxplots, dpi=300,bbox_inches="tight", pad_inches=0.01)
    plt.close('all')

    # Plot empirical CDFs
    plt.figure(figsize=(12, 8))
    for population in populations_to_compare:
        sorted_data = np.sort(df_results[population])
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        plt.plot(sorted_data, cdf, label=population)

    plt.title('Empirical CDFs of Populations')
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    file_path_cdf = os.path.join(output_folder, f'test1_ks_cdf.pdf')
    plt.savefig(file_path_cdf, dpi=300, bbox_inches="tight", pad_inches=0.01)
    plt.close()


def return_mse_and_mape(test, predict):
    mse = sklearn.metrics.mean_squared_error(test, predict)
    mape = sklearn.metrics.mean_absolute_percentage_error(test, predict)
    return mse, mape

def plot_actual_vs_predicted(ax, title, x_bounds, predications:EvaluationResults):
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
        coefficient: float,
        entity_level_results:EvaluationResults,
        observation_level_results:EvaluationResults
    ):

    ax_sl = list_of_plot_axes[test_number * 2]
    ax_ol = list_of_plot_axes[test_number * 2 + 1]

    title = f'Test {test_number+1}, {entity_count} ent, {number_of_observations_per_entity} obs, {interclass_variability} inter_v, {intraclass_variability} intra_v, {coefficient} coef'
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
        coefficient,
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
            coefficient,
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

                case IncrementType.INTRASAMPLE_VARIANCE:
                    coefficient += increment


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

def data_creation_and_prediction(intraclass_variability,interclass_variability,number_of_entities,
                                 number_of_observations_per_entity,target_variable,iteration,supporting_data_folder,reporting):
    sampler = FictitiousSamplerPredictive(
                intraclass_variability=intraclass_variability,
                interclass_variability=interclass_variability)

    entity_observation_pairs = sampler.generate_entity_observation_pairs(
        number_of_entities=number_of_entities,
        average_number_of_observations_per_entity=number_of_observations_per_entity
            )

    el_predictions, mse_el, mape_el = predict(entity_observation_pairs,SplittingStrategy.ENTITY_LEVEL,target_variable,iteration, supporting_data_folder,reporting)
    ol_predictions, mse_ol, mape_ol = predict(entity_observation_pairs,SplittingStrategy.OBSERVATION_LEVEL,target_variable,iteration,supporting_data_folder,reporting)

    return el_predictions,mse_el,ol_predictions,mse_ol,entity_observation_pairs,entity_observation_pairs


class TestResults:
    list_sl_predictions = []
    list_ol_predictions = []
    list_data_for_plotting = []

    ave_mse_el = []
    ave_mse_el_5 = []
    ave_mse_el_95 = []

    ave_mse_ol = []
    ave_mse_ol_5 = []
    ave_mse_ol_95 = []

    hold_out_mse_el = []
    hold_out_mse_ol = []


def execute_test(
        target_variable,
        number_of_test_iterations,
        number_of_entities,
        number_of_observations_per_entity,
        feature_coefficient,
        supporting_data_folder,
        intraclass_variability,
        interclass_variability,
        increment_type,
        increment = None,
        reporting = False,
        runs_per_iteration=10,
        hold_out_data=False
):
    results = TestResults()

    for i in range(number_of_test_iterations):
        print (f"Test iteration {i+1} proceeding...")

        # Each iteration will be repeated 10 times and the mse and mape value averaged
        list_mse_el=[]
        list_mse_ol=[]
        list_holdout_mse_el = []
        list_holdout_mse_ol = []

        for j in range(runs_per_iteration): # tqdm(range(10), desc='Executing...'):

            sampler = FictitiousSamplerPredictive(
                intraclass_variability=intraclass_variability,
                interclass_variability=interclass_variability
            )

            entity_observation_pairs = sampler.generate_entity_observation_pairs(
                number_of_entities=number_of_entities,
                average_number_of_observations_per_entity=number_of_observations_per_entity,
                coefficient = feature_coefficient
            )

            variable_names, feature_names = get_variables_and_feature_names(entity_observation_pairs)
            el_predictions, model_el = predict(entity_observation_pairs,SplittingStrategy.ENTITY_LEVEL,target_variable,i, supporting_data_folder,feature_names, reporting)
            ol_predictions, model_ol = predict(entity_observation_pairs,SplittingStrategy.OBSERVATION_LEVEL,target_variable,i,supporting_data_folder,feature_names, reporting)

            if hold_out_data:
                hold_out_observation_pairs = sampler.generate_entity_observation_pairs(
                    number_of_entities=int(number_of_entities*0.2),
                    average_number_of_observations_per_entity=number_of_observations_per_entity,
                    coefficient=feature_coefficient
                )
                hold_out_dataset = generate_entity_observation_dataframe(hold_out_observation_pairs)
                hold_out_results_el = evaluate_model(model_el, feature_names, target_variable, hold_out_dataset)
                hold_out_results_ol = evaluate_model(model_ol, feature_names, target_variable, hold_out_dataset)

                list_holdout_mse_el.append(hold_out_results_el.mse)
                list_holdout_mse_ol.append(hold_out_results_ol.mse)

            if reporting and j==0:
                #Save the iteration's data for plotting pairplots and dataset boxplots (optional)
                #Save only the first dataset produced, as an example to visualise in plotting, as the iteration is repeated 10 times
                results.list_data_for_plotting.append(entity_observation_pairs)
                results.list_sl_predictions.append(el_predictions)
                results.list_ol_predictions.append(ol_predictions)

            list_mse_el.append(el_predictions.mse)
            list_mse_ol.append(ol_predictions.mse)

        results.ave_mse_el.append(statistics.mean(list_mse_el))
        results.ave_mse_el_5.append(np.percentile(list_mse_el,5))
        results.ave_mse_el_95.append(np.percentile(list_mse_el,95))
        results.ave_mse_ol.append(statistics.mean(list_mse_ol))
        results.ave_mse_ol_5.append(np.percentile(list_mse_ol, 5))
        results.ave_mse_ol_95.append(np.percentile(list_mse_ol, 95))
        results.hold_out_mse_el.append(statistics.mean(list_holdout_mse_el))
        results.hold_out_mse_ol.append(statistics.mean(list_holdout_mse_ol))

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

                case IncrementType.COEFFICIENT:
                    feature_coefficient += increment

                case IncrementType.NONE:
                    continue

    if reporting:
        results.list_sl_predictions = None
        results.list_ol_predictions = None
        results.list_data_for_plotting = None

    return results


def execute_simple_test(
        target_variable,
        number_of_test_iterations,
        number_of_entities,
        number_of_observations_per_entity,
        feature_coefficient,
        supporting_data_folder,
        intraclass_variability,
        interclass_variability,
        increment_type,
        increment=None,
        reporting=False,
        runs_per_iteration=10
):
    list_sl_predictions = []
    list_ol_predictions = []
    list_data_for_plotting = []

    ave_mse_el = []
    ave_mse_el_5 = []
    ave_mse_el_95 = []
    ave_mse_ol = []
    ave_mse_ol_5 = []
    ave_mse_ol_95 = []

    for i in range(number_of_test_iterations):
        print(f"Test iteration {i + 1} proceeding...")

        # Each iteration will be repeated a user-specified number of times and the mse and mape value averaged
        list_mse_el = []
        list_mse_ol = []

        for j in range(runs_per_iteration):  # tqdm(range(10), desc='Executing...'):

            sampler = FictitiousSamplerSimple(
                intraclass_variability=intraclass_variability,
                interclass_variability=interclass_variability
            )

            entity_observation_pairs = sampler.generate_entity_observation_pairs(
                number_of_entities=number_of_entities,
                average_number_of_observations_per_entity=number_of_observations_per_entity,
                coefficient=feature_coefficient
            )

            el_predictions, mse_el, mape_el = predict(entity_observation_pairs, SplittingStrategy.ENTITY_LEVEL,
                                                      target_variable, i, supporting_data_folder, reporting)
            ol_predictions, mse_ol, mape_ol = predict(entity_observation_pairs, SplittingStrategy.OBSERVATION_LEVEL,
                                                      target_variable, i, supporting_data_folder, reporting)

            if reporting and j == 0:
                # Save the iteration's data for plotting pairplots and dataset boxplots (optional)
                # Save only the first dataset produced, as an example to visualise in plotting, as the iteration is repeated 10 times
                list_data_for_plotting.append(entity_observation_pairs)
                list_sl_predictions.append(el_predictions)
                list_ol_predictions.append(ol_predictions)

            list_mse_el.append(mse_el)
            list_mse_ol.append(mse_ol)

        ave_mse_el.append(statistics.mean(list_mse_el))
        ave_mse_el_5.append(np.percentile(list_mse_el, 5))
        ave_mse_el_95.append(np.percentile(list_mse_el, 95))
        ave_mse_ol.append(statistics.mean(list_mse_ol))
        ave_mse_ol_5.append(np.percentile(list_mse_ol, 5))
        ave_mse_ol_95.append(np.percentile(list_mse_ol, 95))

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

                case IncrementType.COEFFICIENT:
                    feature_coefficient += increment

                case IncrementType.NONE:
                    continue

    if reporting:
        return ave_mse_el, ave_mse_el_5, ave_mse_el_95, ave_mse_ol, ave_mse_ol_5, ave_mse_ol_95, list_sl_predictions, list_ol_predictions, list_data_for_plotting
    else:
        return ave_mse_el, ave_mse_el_5, ave_mse_el_95, ave_mse_ol, ave_mse_ol_5, ave_mse_ol_95, None, None, None


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
