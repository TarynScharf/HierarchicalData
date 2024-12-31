from dataclasses import dataclass
import os
import random
from abc import abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, List, Iterable, Tuple


import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import sklearn
from sklearn.experimental import enable_halving_search_cv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import HalvingGridSearchCV


class AbstractSampler[Sample, Analysis]:
    @abstractmethod
    def _generate_samples(self, number_of_samples:int)->Iterable[Sample]:
        '''
        :param number_of_samples: Number of sample objects to generate
        :return: list of sample objects
        '''
        pass
    @abstractmethod
    def _generate_observations(self, sample:Sample, number_of_analyses:int)->Iterable[Analysis]:
        '''
        :param sample: sample object output by the generate_samples method
        :param number_of_analyses: The size of the population to generate.
        :return: list of analyses
        '''
        pass

    def generate_sample_observation_pairs(self,number_of_samples, average_number_of_observations_per_sample):
        sample_observation_pairs =[]

        samples = self._generate_samples(number_of_samples)
        for sample in samples:
            variance = int(average_number_of_observations_per_sample/4)
            number_of_observations = max(1, random.randint(-variance,variance ) + average_number_of_observations_per_sample)
            observations = self._generate_observations(sample, number_of_observations)
            sample_observation_pairs.append((sample,observations))
        return sample_observation_pairs

def generate_sample_observation_dataframe(sample_observation_pairs):
    list_of_df_rows = []
    for sample, observations in sample_observation_pairs:
        sample_variables = list(sample.__dict__.values())
        for observation in observations:
            observation_features = list(observation.__dict__.values())
            df_row = sample_variables + observation_features
            list_of_df_rows.append(df_row)
    variable_names, feature_names = get_variables_and_feature_names(sample_observation_pairs)
    data_headers = variable_names + feature_names
    data = pd.DataFrame(list_of_df_rows, columns=data_headers)
    return data

def visualise_data(sample_observation_pairs:List[Tuple]):
    data = generate_sample_observation_dataframe(sample_observation_pairs)
    sns.pairplot(data)
    plt.show()

def get_variables_and_feature_names(sample_observation_pairs:List[Tuple]):
    variable_names = list(sample_observation_pairs[0][0].__dict__.keys())
    feature_names = list(sample_observation_pairs[0][1][0].__dict__.keys())
    return variable_names,feature_names

class SplittingStrategy(Enum):
    SAMPLE_LEVEL = 1
    OBSERVATION_LEVEL=2

def train_test_split(sample_observation_pairs:List[Tuple], splitting_strategy:SplittingStrategy):
    train = None
    test = None
    match splitting_strategy:
        case SplittingStrategy.SAMPLE_LEVEL:
            split_index = int(len(sample_observation_pairs)*0.2)
            random.shuffle(sample_observation_pairs)
            test_pairs = sample_observation_pairs[:split_index]
            test = generate_sample_observation_dataframe(test_pairs)
            train_pairs = sample_observation_pairs[split_index:]
            train = generate_sample_observation_dataframe(train_pairs)

        case SplittingStrategy.OBSERVATION_LEVEL:
            df_data = generate_sample_observation_dataframe(sample_observation_pairs)
            split_index = int(df_data.shape[0] * 0.2)
            shuffled_df_data = df_data.sample(frac=1).reset_index(drop=True)
            test = shuffled_df_data.iloc[:split_index,:]
            train = shuffled_df_data.iloc[split_index:,:]

    return train,test

def create_x_y(data_df:pd.DataFrame,feature_names:List[str],target_variable_name:str):
    x = data_df.loc[:,feature_names]
    y=data_df.loc[:,target_variable_name]
    return x,y

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

@dataclass
class PredictionData:
    x_train: Any #Dataframe
    y_train: Any
    x_test : Any
    y_test : Any
    y_predict : Any


def predict(sample_observation_pairs:List[Tuple],splitting_strategy:SplittingStrategy, target_variable_name:str):
    variable_names,feature_names = get_variables_and_feature_names(sample_observation_pairs)
    train,test = train_test_split(sample_observation_pairs,splitting_strategy)
    train_x, train_y = create_x_y(train,feature_names,target_variable_name)
    test_x, test_y = create_x_y(test,feature_names,target_variable_name)
    rf = RandomForestRegressor(max_depth=9, max_features=4, max_leaf_nodes=16,
                          min_samples_leaf=5, random_state=42)
    #tune_hyperparameters(train_x, train_y)
    number_of_samples = len(sample_observation_pairs[0][1])
    rf.fit(train_x,train_y)
    prediction_results = rf.predict(test_x)
    return PredictionData(train_x, train_y, test_x, test_y, prediction_results)


def plot_prediction_results(
        axes, 
        test_number:int, 
        sample_count:int, 
        sample_level_results:PredictionData, 
        observation_level_results:PredictionData
    ):
    ax_sl = axes[test_number*2]
    ax_ol = axes[test_number*2+1]

    title = f'Test {test_number}, {sample_count} samples ,'
    min_x = min(min(observation_level_results.y_test), min(sample_level_results.y_test))
    max_x = max(max(observation_level_results.y_test), max(sample_level_results.y_test))
    x_bounds = (0.9*min_x, max_x*1.1)

    plot_actual_vs_predicated(ax_sl, title + "sample-level splitting", x_bounds, sample_level_results)
    plot_actual_vs_predicated(ax_ol, title + "observation-level splitting", x_bounds, observation_level_results)

def plot_actual_vs_predicated(ax, title, x_bounds, predications:PredictionData):
    ax.scatter(x=predications.y_test, y=predications.y_predict)
    mse_sl = sklearn.metrics.mean_squared_error(predications.y_test, predications.y_predict)
    mape_sl = sklearn.metrics.mean_absolute_percentage_error(predications.y_test, predications.y_predict)
    ax.set_xlabel('Actual values')
    ax.set_ylabel('Predicted values')
    ax.set_title(title)
    ax.set_xlim(x_bounds)
    ax.text(x=0.05, y=0.95,s=f'mse: {mse_sl:.2f} \nmape: {mape_sl:.2f}',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
