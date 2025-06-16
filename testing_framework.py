from dataclasses import dataclass
import os
import random
from abc import abstractmethod
from enum import Enum
from typing import Any, List, Iterable, Tuple
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

class AbstractSampler[AbstractInitialConditions, AbstractObservation]:
    @abstractmethod
    def _sample_the_initial_conditions(self, number_of_samples:int)->Iterable[AbstractInitialConditions]:
        '''
        This function simulates the "inital conditions" that lead to a geological sample. E.g. The initial magmatic conditions such as temperature, pressure, oxygen fugacity that result in a geological rock sample.
        (This is not attempting to directly model these magmatic parameters but to simulate a system - in the general sense - that is dependent on multiple parameters!)
        '''
        pass
    @abstractmethod
    def _sample_the_observations(self, conditions:AbstractInitialConditions, number_of_analyses:int)->Iterable[AbstractObservation]:
        '''
        :param conditions: sample object output by the generate_samples method
        :param number_of_analyses: The size of the population to generate.
        :return: list of analyses
        '''
        pass

    def generate_entity_observation_pairs(self, number_of_entities, average_number_of_observations_per_entity,coefficient):
        sample_observation_pairs =[]

        samples = self._sample_the_initial_conditions(number_of_entities)
        for sample in samples:
            variance = int(average_number_of_observations_per_entity / 4)
            number_of_observations = max(1, random.randint(-variance,variance ) + average_number_of_observations_per_entity)
            observations = self._sample_the_observations(sample, number_of_observations, coefficient)
            sample_observation_pairs.append((sample,observations))
        return sample_observation_pairs

def generate_entity_observation_dataframe(entity_observation_pairs):
    list_of_df_rows = []
    for entity, observations in entity_observation_pairs:
        entity_variables = list(entity.__dict__.values())
        for observation in observations:
            observation_features = list(observation.__dict__.values())
            df_row = entity_variables + observation_features
            list_of_df_rows.append(df_row)
    variable_names, feature_names = get_variables_and_feature_names(entity_observation_pairs)
    data_headers = variable_names + feature_names
    data = pd.DataFrame(list_of_df_rows, columns=data_headers)
    return data

def get_variables_and_feature_names(entity_observation_pairs:List[Tuple]):
    variable_names = list(entity_observation_pairs[0][0].__dict__.keys())
    feature_names = list(entity_observation_pairs[0][1][0].__dict__.keys())
    return variable_names,feature_names

class SplittingStrategy(Enum):
    ENTITY_LEVEL = 1
    OBSERVATION_LEVEL=2

def train_test_split(entity_observation_pairs:List[Tuple], splitting_strategy:SplittingStrategy):
    train = None
    test = None
    match splitting_strategy:
        case SplittingStrategy.ENTITY_LEVEL:
            split_index = int(len(entity_observation_pairs)*0.2)
            random.shuffle(entity_observation_pairs)
            test_pairs = entity_observation_pairs[:split_index]
            test = generate_entity_observation_dataframe(test_pairs)
            train_pairs = entity_observation_pairs[split_index:]
            train = generate_entity_observation_dataframe(train_pairs)

        case SplittingStrategy.OBSERVATION_LEVEL:
            df_data = generate_entity_observation_dataframe(entity_observation_pairs)
            split_index = int(df_data.shape[0] * 0.2)
            shuffled_df_data = df_data.sample(frac=1).reset_index(drop=True)
            test = shuffled_df_data.iloc[:split_index,:]
            train = shuffled_df_data.iloc[split_index:,:]

    return train,test

def create_x_y(data_df:pd.DataFrame,feature_names:List[str],target_variable_name:str):

    y=data_df.loc[:,target_variable_name]

    if target_variable_name not in feature_names:
        x = data_df.loc[:, feature_names]
    else:
        updated_feature_names = [feature for feature in feature_names if feature is not target_variable_name]
        x = data_df.loc[:, updated_feature_names]
    return x,y

@dataclass
class EvaluationResults:
    #x_train: Any #Dataframe
    #y_train: Any
    #x_test : Any
    y_test : Any
    y_predict : Any
    mse: Any

def train_model(feature_names, training_data, target_variable_name):
    train_x, train_y = create_x_y(training_data, feature_names, target_variable_name)
    rf = RandomForestRegressor(random_state=42)  # RandomForestRegressor(max_depth=9, max_features=4, max_leaf_nodes=16,min_samples_leaf=5, random_state=42)
    rf.fit(train_x, train_y)
    return rf

def predict(entity_observation_pairs:List[Tuple],
            splitting_strategy:SplittingStrategy,
            target_variable_name:str,
            test_number:int,
            output_folder: str,
            feature_names,
            reporting = False
            ):

    train, test = train_test_split(entity_observation_pairs, splitting_strategy)

    if reporting:
        write_dataset(train,test,splitting_strategy, test_number,output_folder)

    model = train_model(feature_names,train,target_variable_name)

    return evaluate_model(model, feature_names, target_variable_name, test), model

def evaluate_model(model, feature_names, target_variable_name, dataset):
    data_x, data_y = create_x_y(dataset, feature_names, target_variable_name)
    prediction_results = model.predict(data_x)
    mse = sklearn.metrics.mean_squared_error(data_y, prediction_results)

    return EvaluationResults(y_test=data_y,
                             y_predict=prediction_results,
                             mse = mse)

def write_dataset(train_dataset, test_dataset,splitting_strategy,test_number, output_folder):
    test_description=None
    match splitting_strategy:
        case SplittingStrategy.ENTITY_LEVEL:
            test_description = 'entity_level'
        case SplittingStrategy.OBSERVATION_LEVEL:
            test_description = 'observation_level'

    file_name = f'Iteration{test_number+1}_'+'_datasets_'+ test_description +'.xlsx'
    file_path = os.path.join(output_folder, file_name)

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        train_dataset.to_excel(writer, sheet_name='train_data', index=False)
        test_dataset.to_excel(writer, sheet_name='test_data', index=False)



