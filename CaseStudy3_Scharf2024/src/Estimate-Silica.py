import copy
import csv
import datetime
import glob
import os
import random
import numpy as np
#import optuna
#import shap
#from keras.callbacks import History
#from optuna.trial import TrialState
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import operator as op
from functools import reduce
from itertools import combinations
#import tensorflow as tf
#import tensorflow_addons as tfa
#from tensorflow.keras.layers.experimental import preprocessing
#import tensorflow.keras.backend as K

def pca(df, features, ML_features=None, u_th_features=None, keep_columns = None, pca_model = None, scaling_data = None, output_location = None):
    #it's at this point that sampleid, bin, and CL textures are removed. Only shape and possibly U&Th (depending on the ML_features variable passed) form part of x, created below
    df.reset_index(drop=True, inplace=True)
    x = df.loc[:, features].values

    if scaling_data is not None:
        scaling_data.reset_index(drop=True, inplace=True)
        y = scaling_data.loc[:, features].values
        y_means = y.mean(axis=0)
        y_stds = y.std(axis=0)
        x_scaled = (x-y_means)/y_stds
    else:
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

    if pca_model == None:
        pca_model = PCA(n_components=3)
        pca_model.fit(x_scaled)
        pca_features = pca_model.transform(x_scaled)
    else:
        pca_features = pca_model.transform(x_scaled)

    #write out the variance per principal component
    print(f'pca variance:{pca_model.explained_variance_ratio_}')
    df_pca_loadings = pd.DataFrame(pca_model.components_, columns=features)
    df_pca_loadings.index = ['PC1', 'PC2', 'PC3']

    if u_th_features is not None:
        x_uth = df.loc[:, u_th_features].values
        x1 = pd.concat([pd.DataFrame(pca_features),pd.DataFrame(x_uth)], axis=1)
        column_headers = np.concatenate([['PC1', 'PC2', 'PC3'],np.array(u_th_features)])
        x1.columns = [column_headers]
    else:
        x1 = pd.DataFrame(pca_features[:, 0:3])
        x1.columns = ['PC1', 'PC2', 'PC3']

    if ML_features is not None:
        x_ml_features = df.loc[:, ML_features]
        x2 = pd.concat([x1,x_ml_features], axis = 1)
    else:
        x2 = pd.DataFrame(pca_features[:,0:3])
        x2.columns = ['PC1', 'PC2', 'PC3']

    if keep_columns is not None:
        keep = df.loc[:, keep_columns]
        x3 = pd.concat([x2,keep], axis = 1)
    else:
        x3 = x2

    return x3, pca_model

def build_classifier(train_features, n_layers, n_units,dropout):
    CLASSES =1
    normalizer =preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    model = tf.keras.Sequential()

    for i in range(n_layers):
        model.add(
            tf.keras.layers.Dense(
                n_units[i],
                kernel_regularizer=tf.keras.regularizers.l2()
            )
        )
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(CLASSES))
    model.compile(optimizer= 'Nadam', loss=tf.losses.MeanSquaredError(),metrics=[mape_metric, smape_metric, concordance_cc, pearson_cc, tfa.metrics.RSquare(dtype=tf.float32)])
    return model

def assign_silica_bins(df):
    df.loc[df['SiO2_pct'].between(0, 45, 'right'), 'bin'] = 45
    df.loc[df['SiO2_pct'].between(45, 50, 'right'), 'bin'] = 50
    df.loc[df['SiO2_pct'].between(50, 55, 'right'), 'bin'] = 55
    df.loc[df['SiO2_pct'].between(55, 60, 'right'), 'bin'] = 60
    df.loc[df['SiO2_pct'].between(60, 65, 'right'), 'bin'] = 65
    df.loc[df['SiO2_pct'].between(65, 70, 'right'), 'bin'] = 70
    df.loc[df['SiO2_pct'].between(70, 75, 'right'), 'bin'] = 75
    df.loc[df['SiO2_pct'].between(75, 80, 'right'), 'bin'] = 80
    df.loc[df['SiO2_pct'].between(80, 85, 'right'), 'bin'] = 85

def evaluate_model(x_test, y_test, model):
    y_pred = model.predict(x_test)
    y_test.reset_index(inplace=True, drop=True)
    results = pd.concat([y_test, pd.DataFrame(y_pred)], axis=1)
    results.columns=['actual', 'predicted']

    # calculate metrics on an individual grain basis
    r2, ccc, pearson, mape, smape, mse = get_performance_metrics(y_test, y_pred)
    metrics_df = pd.DataFrame([[r2, ccc, pearson, mape, smape, mse]], columns=['r2', 'ccc', 'pearson', 'mape', 'smape', 'mse'])

    return results,metrics_df

def evaluate_model_with_shap(x_test, y_test, model,xtrain,CL, shap_output_location):
    y_pred = model.predict(x_test)
    y_test.reset_index(inplace=True, drop=True)
    results = pd.concat([y_test, pd.DataFrame(y_pred)], axis=1)
    results.columns=['actual', 'predicted']

    #calculate metics on an individual grain basis
    r2, ccc, pearson, mape, smape, mse = get_performance_metrics(y_test,y_pred)
    metrics_df = pd.DataFrame([[r2, ccc, pearson, mape, smape, mse]], columns = ['r2', 'ccc', 'pearson', 'mape', 'smape', 'mse'])

    #create the shap model
    #provide the training dataset and then sample therefrom to create the explainer
    samples_1000 = shap.sample(xtrain, nsamples=1000, random_state=42)
    explainer = shap.Explainer(model.predict, samples_1000, output_names='silica', feature_names=['PC1','PC2','PC3','oscillatory_zonation', 'sector_zonation', 'homogenous_zonation'], seed=42)

    #apply the explainer to the test dataset, to get the outputs
    shap_values = explainer(x_test)
    write_shap_values_to_csv(shap_values, shap_values.data, x_test, y_test, shap_output_location)
    shap.plots.beeswarm(shap_values, max_display=15)

    if CL:
        shap_values_df = pd.DataFrame(shap_values.values, columns=['shapPC1','shapPC2','shapPC3','shaposcillatory_zonation', 'shapsector_zonation', 'shaphomogenous_zonation'])
        data_values_df = pd.DataFrame(shap_values.data, columns = ['PC1','PC2','PC3','oscillatory_zonation', 'sector_zonation', 'homogenous_zonation'])
        pc1_df = pd.concat([data_values_df, shap_values_df], axis=1)
        sns.scatterplot(data=pc1_df, x='oscillatory_zonation', y='shaposcillatory_zonation',label = 'OZ')
        sns.scatterplot(data=pc1_df, x='sector_zonation', y='shapsector_zonation', label = 'SZ')
        sns.scatterplot(data=pc1_df, x='homogenous_zonation', y='shaphomogenous_zonation', label = 'HZ')
        sns.scatterplot(data=pc1_df, x='PC1', y='shapPC1', label='PC1')
        sns.scatterplot(data=pc1_df, x='PC2', y='shapPC2', label='PC2')
        sns.scatterplot(data=pc1_df, x='PC3', y='shapPC3', label='PC3')
    else:
        shap_values_df = pd.DataFrame(shap_values.values, columns=['shapPC1', 'shapPC2', 'shapPC3'])
        data_values_df = pd.DataFrame(shap_values.data, columns=['PC1', 'PC2', 'PC3'])
        pc1_df = pd.concat([data_values_df, shap_values_df], axis=1)
        sns.scatterplot(data = pc1_df, x='PC1', y='shapPC1', label = 'PC1')
        sns.scatterplot(data = pc1_df, x='PC2', y='shapPC2', label = 'PC2')
        sns.scatterplot(data = pc1_df, x='PC3', y='shapPC3',label = 'PC3')

    clustering = shap.utils.hclust(x_test, y_test)
    shap.plots.bar(shap_values, clustering=clustering, max_display=15)

    return results,metrics_df

def write_shap_values_to_csv(shap_values, shap_data_values, original_test_data_values, silica_values, shap_output_location):
    #o indicates 'original'
    headers= ['oPC1', 'oPC2', 'oPC3', 'oOZ', 'oSZ', 'oHZ','PC1', 'PC2', 'PC3', 'ocillatory_zonation', 'sector_zonation', 'homogenous_zonation','shapPC1', 'shapPC2', 'shapPC3', 'shapOZ', 'shapSZ', 'shapHZ' 'Silica']
    date = datetime.date.today()
    date_tag = date.strftime("%b-%d-%Y")
    csv_file_name = f'shap_results_{date_tag}.csv'
    with open(os.path.join(shap_output_location, csv_file_name), 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for i in range(len(shap_values.values)):
            original_values_list = list(original_test_data_values.iloc[i])
            data_values_list = list(shap_data_values[i])
            shap_values_list = list(shap_values.values[i])

            zipped_list = list(zip(original_values_list,data_values_list,shap_values_list))
            values = [val for item in zipped_list for val in item]
            values.append(silica_values.values[i])
            shap_values_string = list(map(str, values))
            writer.writerow(shap_values_string)

def get_date_time():
    now = datetime.datetime.now()
    date = now.strftime("%d%m%Y%H%M%S")
    return date

def ncr(n, r):
    #taken from:https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python
    r = min(r, n-r)
    number = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return number // denom

def calculate_number_of_zircon_required_to_balance_each_class(df, augmentation = 1):
    # this takes the dataset (as dataframe) that need to have its classes balanced
    # returns a summary dataframe providing classes, number of zircon in each class, and number of zircon required to balance each class

    #housekeeping
    df.reset_index(inplace=True, drop=True)

    #calculate how many data points are in each class
    zircons_per_bin = df.groupby(['bin'])['area'].count().reset_index()
    zircons_per_bin.columns = ['bin', 'num_grains']

    #Identify the largest class, apply augmentation factor
    max_value = zircons_per_bin['num_grains'].max() * augmentation

    #calculate how many additional zircon are required to balance each class
    zircons_per_bin['required_zircons'] = max_value - zircons_per_bin['num_grains']

    return zircons_per_bin

def balance_classes_with_aggregates(df, zircons_required_per_class,N=3):
    #df: dataframe containing dataset to balance
    #zircons_required_per_class: summary dataset returned by calculate_number_of_zircon_required_to_balance_each_class

    #housekeeping
    df.reset_index(inplace=True, drop=True)

    #calculate how many aggregates could be made, per sample
    zircons_per_sample = df.groupby(['bin','GSWA_sample_id'])['area'].count().reset_index()
    zircons_per_sample.columns = ['bin','GSWA_sample_id', 'num_zircon_in_sample']
    list_of_integers_per_sample={}
    for index, row in zircons_per_sample.iterrows():
        list_of_integers_per_sample[row[1]]=list(range(0, zircons_per_sample.loc[index,'num_zircon_in_sample']))

    zircons_per_sample['num_combinations_per_sample'] = zircons_per_sample.apply(lambda zircons_per_sample: ncr(int(zircons_per_sample['num_zircon_in_sample']), N), axis=1)

    #calculate how many aggregates are required from each sample in a class, to balance the class
    zircons_per_sample['num_aggregates_per_sample']=0
    bin_list = list(zircons_required_per_class['bin'])
    diff_list = list(zircons_required_per_class['required_zircons'])
    combinations_per_bin =  zircons_per_sample.groupby(['bin'])['num_combinations_per_sample'].sum()
    df_combinations_per_bin = combinations_per_bin.reset_index()
    bins_allowing_duplication = []

    for i in range(len(bin_list)):
        duplicate_aggregates=False
        min_aggregates_list = []
        if diff_list[i] > 0:
            diff = diff_list[i]
            if df_combinations_per_bin.loc[df_combinations_per_bin['bin']==bin_list[i]]['num_combinations_per_sample'].values[0]<diff:
                print(f'WARNING: Insufficient zircon in bin {bin_list[i]} to create the required augmentations')
                duplicate_aggregates = True
                bins_allowing_duplication.append(bin_list[i])

            # the number of additional aggregates per sample needed to balance this bin
            df_bin = zircons_per_sample.loc[zircons_per_sample['bin'] == bin_list[i]]  # extract all samples that are in the given bin
            samples = list(df_bin['GSWA_sample_id'])  # get a list of all the samples in the given bin
            min_aggregates_list = list(df_bin['num_aggregates_per_sample'])  # get a list of the min number of aggregates that can be made for each sample
            max_aggregates_list = list(df_bin['num_combinations_per_sample'])

            while diff > 0:
                for j in range(len(samples)):
                    if duplicate_aggregates:
                        #if there aren't enough zircon to create sufficient unique combinations (ie aggregates), then we'll simply use the sample aggregates multiple times
                        if diff > 0:
                            min_aggregates_list[j] = min_aggregates_list[j] + 1
                            diff = diff - 1
                            continue

                    else:
                        #if there are sufficient zircon to create the required number of unique combinations (ie. aggregates), then don't oversample any given sample
                        if min_aggregates_list[j] + 1 <= max_aggregates_list[j] and diff > 0:
                            min_aggregates_list[j] = min_aggregates_list[j] + 1
                            diff = diff - 1
                            continue
                        else:
                            continue

            for k in range(len(samples)):
                #write the calculated required number of aggregates, to the dataframe
                zircons_per_sample.loc[(zircons_per_sample['bin'] == bin_list[i]) & (zircons_per_sample['GSWA_sample_id'] == samples[k]), 'num_aggregates_per_sample'] = min_aggregates_list[k]

    #create the aggregates
    list_of_unique_sampleids = list(pd.unique(zircons_per_sample['GSWA_sample_id']))
    aggregates=[]
    for sample_id in list_of_unique_sampleids:
        # isolate the sample in it's own dataframe
        df_sampleid = df.loc[df['GSWA_sample_id'] == sample_id]
        df_sampleid.reset_index(drop=True, inplace=True)

        # create a "pool" of integers, 0 - number of crystals in the sample.
        list_of_integers = list_of_integers_per_sample[sample_id]
        comb = list(combinations(list_of_integers, N))
        if comb == []:
            comb = [list_of_integers]
        aggregate_rows = []
        if int(zircons_per_sample[zircons_per_sample['GSWA_sample_id']==sample_id]['num_aggregates_per_sample']) > 0:
            num_aggregates = int(zircons_per_sample[zircons_per_sample['GSWA_sample_id']==sample_id]['num_aggregates_per_sample'])
            if int(zircons_per_sample[zircons_per_sample['GSWA_sample_id']==sample_id]['bin']) in bins_allowing_duplication:
                choice = random.choices(comb,k=num_aggregates)
                for aggregate in list(choice):
                    aggregate_rows.append(list(aggregate)) #turn tuple into list
            else:
                comb_list_of_indices = list(range(0,len(comb)))
                randomly_selected_indices = np.random.choice(comb_list_of_indices, size = num_aggregates, replace = False)
                choice = [comb[i] for i in randomly_selected_indices]
                for aggregate in choice:
                    aggregate_rows.append(list(aggregate)) #turn tuple into list

        for aggregate in aggregate_rows:
            combination_rows = df_sampleid.iloc[aggregate]
            df_mean = pd.DataFrame(combination_rows.mean()).transpose()
            aggregates.append(df_mean)

    aggregated_dataset = pd.concat(aggregates, ignore_index=True)
    balanced_dataset = pd.concat([aggregated_dataset,df],ignore_index=True)

    return balanced_dataset

def load_dataset(filepath, outliers, columns,UTH, all_data=True):
    df = pd.read_csv(filepath, usecols=columns)
    df.sort_values(['GSWA_sample_id'], inplace=True)

    if outliers is not None:
        df_outliers_removed = df[~df['GSWA_sample_id'].isin(outliers)]
        df_outliers_removed.sort_values(['GSWA_sample_id'], inplace=True)
        df_outliers_removed.reset_index(drop=True, inplace=True)
    else:
        df_outliers_removed = df

    if UTH:
        #drop all rows that aren't analysed for U and Th
        df_complete = df_outliers_removed.dropna(how='any')
        df_complete.reset_index(drop=True, inplace=True)
    elif UTH == False and all_data == False:
        #drop all rows that aren't analysed for U and Th, even though the scenario doesn't specifically require analysed grains only
        df_complete = df_outliers_removed.dropna(how='any')
        df_complete.reset_index(drop=True, inplace=True)
    else:
        #use all zircon in the datas set, for the scenario
        df_complete = df_outliers_removed

    return df_complete

def predict(model_path, x_data):
    # 1 - load the model
    model = tf.keras.models.load_model(model_path,{"concordance_cc": concordance_cc, "pearson_cc": pearson_cc, "r_squared": r_squared, "mape_metric": mape_metric, "smape_metric": smape_metric})
    model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[mape_metric, smape_metric, concordance_cc, pearson_cc, tfa.metrics.RSquare(dtype=tf.float32)])

    y_pred = model.predict(x_data)
    results = pd.DataFrame(y_pred, columns=['predicted_silica'])

    prediction_results = pd.concat([x_data,results], axis=1)

    return prediction_results

def train_and_evaluate(scenario,train_log_filepath, train_dataset, val_dataset, xtrain, x_test_OG, y_test_OG, batch_size, epochs,CL, model_path, shap_output_location=False):
    # instantiate the model
    if model_path is None:
        #Model path is none if we're running a Kfold cross-validation
        csvLogger = tf.keras.callbacks.CSVLogger(train_log_filepath)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0,
            patience=20,
            verbose=0,
            mode='min',
            baseline=None,
            restore_best_weights=True)

        #This builds a model using the Optuna parameters detailed in the manuscript.
        #Update as necessary, if the Optuna optimisations are rerun
        if scenario == 'scenario_1':
            model = build_classifier(xtrain, 4, [120, 80, 110, 80],0.05)

        elif scenario == 'scenario_2':
            model = build_classifier(xtrain, 2, [150,80],0.01)

        elif scenario == 'scenario_3':
            model = build_classifier(xtrain, 9, [30, 20, 30,50,20, 30, 120, 40, 30], 0.01)

        elif scenario == 'scenario_4':
            model = build_classifier(xtrain, 3, [80,90,50], 0.00)

        elif scenario == 'scenario_5':
            model = build_classifier(xtrain, 5, [120, 100, 120, 30, 40], 0.00)

        elif scenario == 'scenario_6':
            model = build_classifier(xtrain, 2, [10,50], 0.01)

        elif scenario == 'scenario_7':
            model = build_classifier(xtrain, 5, [80, 60, 30, 30, 30], 0.00)

        model.fit(train_dataset, batch_size=batch_size, epochs=epochs, validation_data=val_dataset, validation_batch_size=batch_size, validation_freq=1, callbacks=[csvLogger, early_stopping])
        test_results, test_metrics = evaluate_model(x_test_OG, y_test_OG, model)
    else:
        #If the model has been specified, we're testing an existing model.
        model = tf.keras.models.load_model(model_path,
                                           {"concordance_cc": concordance_cc, "pearson_cc": pearson_cc, "r_squared": r_squared, "mape_metric": mape_metric, "smape_metric": smape_metric})
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=[mape_metric, smape_metric, concordance_cc, pearson_cc, tfa.metrics.RSquare(dtype=tf.float32)])
        test_results, test_metrics = evaluate_model_with_shap(x_test_OG, y_test_OG, model,xtrain,CL,shap_output_location)

    results = [test_results]
    return results,test_metrics, model

def create_results_folder(path,description):
    date = get_date_time()
    folder_name = f'Results_{description}_{date}'
    output_location = os.path.join(path,folder_name)
    if not os.path.exists(output_location):
        os.makedirs(output_location)
    print(f'Created {output_location}')
    return output_location

def train_test_model(scenario,cap_data, input_data_filepath, aggregate_size, resampling_repeats, test_split_size, CL, UTH, all_data, outliers_to_remove, epochs, n_trials= None, batchsize= None, Test=None, model_filepath=None):
    #trial description for reporting
    if Test == 'Optuna':
        description = f'Optuna_{scenario}'
    elif Test == 'Kfold':
        description = f'Kfold_{scenario}'
    elif Test == 'Test':
        description = f'Test_{scenario}'
    else:
        print('Test variable can be Optuna, Kfold or Test. Incorrect test variable assignment.')
        return

    description = description+f'_{aggregate_size}AGGx{resampling_repeats}Resample_SHAPE'
    if UTH:
        description = description+"_UTH"

    if CL:
        description = description + "_CL"

    if outliers_to_remove is not None:
        for outlier in outliers_to_remove:
            description = description+f'_{outlier}'

    #create an output folder in the script's directory, with the name defined by the description variable
    output_folder_location = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Outputs')
    location = create_results_folder(output_folder_location, description)

    #Establish the list of fields to import from the data
    if CL:
        ML_features = ['oscillatory_zonation', 'sector_zonation', 'homogenous_zonation']
    else:
        ML_features = None

    if UTH:
        columns = ['GSWA_sample_id', 'area', 'equivalent_diameter', 'perimeter', 'minor_axis_length', 'major_axis_length', 'solidity', 'convex_area', 'form_factor', 'roundness', 'compactness', 'aspect_ratio', 'minimum_Feret', 'maximum_Feret', 'U238_ppm', 'Th232_ppm', 'SiO2_pct', 'oscillatory_zonation', 'sector_zonation','homogenous_zonation']
        pca_features = ['area', 'equivalent_diameter', 'perimeter', 'minor_axis_length', 'major_axis_length', 'solidity', 'convex_area', 'form_factor', 'roundness', 'compactness', 'aspect_ratio', 'minimum_Feret', 'maximum_Feret']
        u_th_features = ['U238_ppm', 'Th232_ppm']

    else:
        columns = ['GSWA_sample_id', 'area', 'equivalent_diameter', 'perimeter', 'minor_axis_length', 'major_axis_length', 'solidity', 'convex_area', 'form_factor', 'roundness', 'compactness', 'aspect_ratio', 'minimum_Feret', 'maximum_Feret', 'U238_ppm', 'Th232_ppm', 'SiO2_pct', 'oscillatory_zonation', 'sector_zonation','homogenous_zonation']
        pca_features = ['area', 'equivalent_diameter', 'perimeter', 'minor_axis_length', 'major_axis_length', 'solidity', 'convex_area', 'form_factor', 'roundness', 'compactness', 'aspect_ratio', 'minimum_Feret', 'maximum_Feret']
        u_th_features = None

    #1 - load raw data and remove outlier samples
    df1 = load_dataset(input_data_filepath,outliers_to_remove,columns,UTH,all_data)

    #2 - assign silica bins to the dataset
    assign_silica_bins(df1)
    if cap_data:
        # Here we limit the silica values to below 69%
        df=df1[df1['SiO2_pct']<=69]
    else:
        df = df1

    #3 - set aside the test subset
    x = df.loc[:, columns]
    y = df[['bin']]
    x_train_OG, x_test_OG, y_train_OG, y_test_OG = train_test_split(x, y, test_size=test_split_size, stratify=y)
    test_OG = pd.concat([x_test_OG,y_test_OG], axis=1)
    train_OG = pd.concat([x_train_OG, y_train_OG], axis=1)

    #4.1 - balance classes
    zircons_required_per_bin = calculate_number_of_zircon_required_to_balance_each_class(train_OG,resampling_repeats)
    train_balanced_classes = balance_classes_with_aggregates(train_OG, zircons_required_per_bin, N=aggregate_size)

    #4.2 - label each dataset
    train_balanced_classes['Dataset'] = 'train'
    test_OG['Dataset'] = 'test'

    #4.3 - recombine the train and test subsets for PCA
    df_train_test = pd.concat([train_balanced_classes,test_OG], ignore_index=True)

    #4.4 - PCA on train and test data
    keep_columns = ['SiO2_pct', 'Dataset', 'GSWA_sample_id']
    df_pca, pca_loadings = pca(df_train_test, pca_features, ML_features,u_th_features, keep_columns, pca_model = None, scaling_data = None, output_location=location)

    #4.5 - separate into train and test datasets again
    df_train_pca = df_pca[df_pca['Dataset']=='train']
    df_train_pca = df_train_pca.sample(frac=1).reset_index(drop=True)

    fields = [header for header in df_train_pca.columns.values if header not in keep_columns]
    x_train_pca = df_train_pca[fields]
    y_train_pca = df_train_pca[['SiO2_pct']]

    df_test_pca = df_pca[df_pca['Dataset'] == 'test']
    x_test_pca = df_test_pca[fields]
    y_test_pca = df_test_pca[['SiO2_pct']]

    #save the data sets for later review
    df.to_csv(os.path.join(location, f'OriginalDataset_OutliersRemoved_SilicaCapped.csv'))
    df_pca.to_csv(os.path.join(location,f'ModelInputs.csv'))
    df_train_test.to_csv(os.path.join(location,f'PCAInputs.csv'))

    if Test == 'Optuna':
        # Set up and run the Optuna optimisation
        #This will build and test several model architectures. For this reason, model_filepath is set to None
        model_filepath = None
        xtrain, xval, ytrain, yval = train_test_split(x_train_pca, y_train_pca, shuffle=True, test_size=test_split_size, stratify=y_train_pca)
        train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
        train_dataset = train_dataset.batch(batchsize, drop_remainder=True)
        val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval))
        val_dataset = val_dataset.batch(batchsize, drop_remainder=True)
        optimise(n_trials, epochs,train_dataset,val_dataset,xtrain, output_location =location)
        return

    elif Test == 'Kfold':
        model_filepath = None
        # If we're testing, execute the 10-fold cross-validation
        #This will build, train and test the model 10 times. For this reason, model_filepath is set to None.
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        results = []
        metrics = []
        for i, (train_index, test_index) in enumerate(kfold.split(x_train_pca, y_train_pca)):
            xtrain, xval = x_train_pca.iloc[train_index, :], x_train_pca.iloc[test_index, :]
            ytrain, yval = y_train_pca.iloc[train_index], y_train_pca.iloc[test_index]
            train_dataset = tf.data.Dataset.from_tensor_slices((xtrain, ytrain))
            train_dataset = train_dataset.batch(batchsize, drop_remainder=True)
            val_dataset = tf.data.Dataset.from_tensor_slices((xval, yval))
            val_dataset = val_dataset.batch(batchsize, drop_remainder=True)
            filepath = os.path.join(os.path.join(location, f'{description}_log_fold{i}.csv'))
            fold_results, fold_metrics, _ = train_and_evaluate(scenario,filepath, train_dataset, val_dataset, xtrain, x_test_pca, y_test_pca, batchsize, epochs,CL, model_path=None)
            results.append(fold_results)
            metrics.append(fold_metrics)

        date = get_date_time()
        with open(os.path.join(location, f'{description}_{date}.csv'), 'a', newline='') as f:
            for i in range(len(results)):
                f.write(f'FOLD {i + 1} \n')
                for df in results[i]:
                    df.to_csv(f)
                    f.write("\n")
                    metrics[i].to_csv(f)
                    f.write("\n")
        return

    elif Test == 'Test':
        if model_filepath is None:
            print('No model filepath specified.')
            return

        #We need the xtrain data, for SHAP.
        #This could be loaded from file, but as the processes is reproduceable, I'll just remake the file
        xtrain, xval, ytrain, yval = train_test_split(x_train_pca, y_train_pca, shuffle=True, test_size=test_split_size, stratify=y_train_pca)

        results = []
        metrics = []
        fold_results, fold_metrics, _ = train_and_evaluate(scenario, None, None, None, xtrain, x_test_pca, y_test_pca, batchsize, epochs,CL, model_filepath,shap_output_location=location)

        results.append(fold_results)
        metrics.append(fold_metrics)
        date = get_date_time()

        with open(os.path.join(location, f'{description}_{date}.csv'), 'a', newline='') as f:
            for i in range(len(results)):
                f.write(f'Model_test_results {i + 1} \n')
                for df in results[i]:
                    df.to_csv(f)
                    f.write("\n")
                    metrics[i].to_csv(f)
                    f.write("\n")
    else:
        print('No test selected.')
        return

class custom_callback_save_model(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None, trial = None):
        if epoch>100:
            save_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Models',f'epoch_{epoch}'))
            self.model.save(save_path)
            print(f'Model {epoch} saved')

def setup_seed(seed):
    #This exists to establish reproduceability of tensorflow runs.
    #Taken from https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
    import random
    os.environ['PYTHONHASHSEED'] = str(seed) #Set `PYTHONHASHSEED` environment variable at a fixed value
    random.seed(seed) #Set `python` built-in pseudo-random generator at a fixed value
    np.random.seed(seed) #set numpy pseudo-random generator at a fixed value
    tf.random.set_seed(seed) #Set the `tensorflow` pseudo-random generator at a fixed value. This is CPU fix.
    os.environ['TF_DETERMINISTIC_OPS'] = '1' #tensorflow gpu fix. Must have pip install tensorflow-determinism

def apply_model(use_UTH,use_CL,model_path,dataset_for_pca_loadings,data_columns,data_to_predict_on,prediction_columns,output_location,description,keep_columns=None):
    #This function is written specifically for applying manuscript models to the manuscript case study
    #It makes assumptions about field names

    #housekeeping
    if use_UTH:
        pca_features = ['area', 'equivalent_diameter', 'perimeter', 'minor_axis_length', 'major_axis_length', 'solidity', 'convex_area', 'form_factor', 'roundness', 'compactness', 'aspect_ratio', 'minimum_Feret', 'maximum_Feret']
        u_th_features = ['U238_ppm', 'Th232_ppm']
    else:
        pca_features = ['area', 'equivalent_diameter', 'perimeter', 'minor_axis_length', 'major_axis_length', 'solidity', 'convex_area', 'form_factor', 'roundness', 'compactness', 'aspect_ratio', 'minimum_Feret', 'maximum_Feret']
        u_th_features = None

    if use_CL:
        ML_features = ['oscillatory_zonation', 'sector_zonation', 'homogenous_zonation']
    else:
        ML_features = None

    #1- load the original train_test data that pca was performed on, to generate the pca_loadings
    df_for_pca = pd.read_csv(dataset_for_pca_loadings, usecols = data_columns)
    _, pca_loadings = pca(df_for_pca, pca_features,ML_features,u_th_features, keep_columns=None, pca_model=None,output_location=output_location)

    #2 load the data set to predict on:
    df_data_all = pd.read_csv(data_to_predict_on, usecols = prediction_columns)

    if use_UTH:
        df_data = df_data_all.dropna(how='any')
        df_data.reset_index(drop=True, inplace=True)
    else:
        df_data = df_data_all

    # The new data must be scale by the model's training dataset. Hence scaling_data is used in this funcition
    df_data_pca,_ = pca(df_data, pca_features,ML_features,u_th_features, keep_columns=keep_columns, pca_model=pca_loadings, scaling_data=df_for_pca,output_location=output_location)
    #3 - create x data sets for prediction
    if use_CL:
        x = df_data_pca.loc[:,['PC1', 'PC2', 'PC3', 'oscillatory_zonation', 'sector_zonation','homogenous_zonation']]
    else:
        x = df_data_pca.loc[:, ['PC1', 'PC2', 'PC3']]

    #4 - apply model to data
    prediction_results = predict(model_path, x)
    spot_info = df_data_pca.loc[:,keep_columns]
    results = pd.concat([spot_info,prediction_results],axis=1)

    #5 - save results
    date = get_date_time()
    description = description
    results.to_csv(os.path.join(output_location, f'prediction_outputs_{description}_{date}.csv'))

def get_performance_metrics(true_values, predicted_values):
    x = tf.convert_to_tensor(np.squeeze(true_values.to_numpy(dtype='float32')))
    y = tf.convert_to_tensor(np.squeeze(predicted_values), dtype='float32')

    ccc = concordance_cc(x, y)
    pearson = pearson_cc(x, y)
    mape = mape_metric(x, y)
    smape = smape_metric(x,y)
    r2 = r_squared(x, y)
    mse = mean_squared_error(x,y)
    return r2.numpy(), ccc.numpy(), pearson.numpy(), mape.numpy(), smape.numpy(), mse.numpy()

def mean_squared_error(x, y):
    mse = K.mean((x-y)**2)
    return mse

def concordance_cc(x, y):
    ''' Concordance Correlation Coefficient'''
    sxy = K.sum((x - K.mean(x))*(y - K.mean(y)))/tf.cast(K.shape(y)[0],tf.float32)
    rhoc = 2.0 * sxy / (K.var(x) + K.var(y) + (K.mean(x) - K.mean(y)) ** 2)
    return rhoc

def pearson_cc(x,y):
    #https://stackoverflow.com/questions/72710792/creating-pearson-correlation-metrics-using-tensorflow-tensor
    # https://github.com/WenYanger/Keras_Metrics

    mean_x = K.mean(x, axis = 0)
    mean_y = K.mean(y, axis = 0)
    xm, ym = x - mean_x, y - mean_y
    r_num = K.sum(xm * ym)
    x_square_sum = K.sum(xm * xm)
    y_square_sum = K.sum(ym * ym)
    r_den = K.sqrt(x_square_sum * y_square_sum)
    r = r_num /(r_den)
    pearson_tensor = K.mean(r)

    return pearson_tensor

def r_squared(x,y):
    metric = tfa.metrics.RSquare()
    metric.update_state(x,y)
    result = metric.result()
    return result

def mape_metric(x,y):
    mape = 100 * K.mean(K.abs((x - y)/x))
    return mape

def smape_metric(x,y):
    smape = 100 * K.mean(K.abs(y - x) /(K.abs(x) + K.abs(y)))
    return smape

def delete_model(pattern,path):
    all_files = glob.glob(os.path.join(path, pattern))
    for file in all_files:
        if os.path.exists(file):
            os.remove(file)

def optuna_logging_callback(study, trial, path):
    write_headers = False
    log_name = f"{path}/{study.study_name}_log.csv"
    trial_number = trial.number
    trial_value = trial.value
    trial_params = trial.params
    trial_state = trial.state.name
    trial_number_epochs = trial.user_attrs['epochs']
    if trial_state == "COMPLETE":
        loss = trial.user_attrs['loss']
        val_loss = trial.user_attrs['val_loss']
        validation_r2 =trial.user_attrs['val_r2']
        train_r2 = trial.user_attrs['train_r2']
        validation_ccc = trial.user_attrs['validation_ccc']
        train_ccc = trial.user_attrs['train_ccc']
        validation_pearson = trial.user_attrs['validation_pearson']
        train_pearson = trial.user_attrs['train_pearson']
        validation_mape = trial.user_attrs['validation_mape']
        train_mape= trial.user_attrs['train_mape']
        validation_smape=trial.user_attrs['validation_smape']
        train_smape=trial.user_attrs['train_smape']
    else:
        pattern = f'trial_{trial.number}_epoch_*.h5'
        delete_model(pattern,path)
        loss = -9999
        val_loss = -9999
        validation_r2 = -9999
        train_r2 = -9999
        validation_ccc = -9999
        train_ccc = -9999
        validation_pearson = -9999
        train_pearson = -9999
        validation_mape = -9999
        train_mape = -9999
        validation_smape = -9999
        train_smape = -9999

    row = [trial_number, trial_state,trial_number_epochs, trial_value,val_loss,loss, validation_r2, train_r2, validation_ccc, train_ccc, validation_pearson, train_pearson, validation_mape, train_mape, validation_smape, train_smape, trial_params]

    if not os.path.exists(log_name):
        write_headers = True
        headers = ["Trial","Trial State","Num_Epochs", "Val_MSE",'val_loss','train_loss', 'validation_r2', 'train_r2', 'validation_ccc', 'train_ccc', 'validation_pearson', 'train_pearson', 'validation_mape', 'train_mape', 'validation_smape', 'train_smape', "Parameters"]

    with open (log_name, "a", newline = "") as log_file:
        writer = csv.writer(log_file)
        if write_headers:
            writer.writerow(headers)
        writer.writerow(row)

def add_trial_number_to_output_files(path):
    for path, folder, files in os.walk(path):
        for file in files:
            if os.path.splitext(file)[1] == '.csv' and file.split('_')[0] == 'trial':
                trial = file.split('_')[1]
                df = pd.read_csv(os.path.join(path, file))
                try:
                    df.insert(1, 'Trial', trial)
                    df.to_csv(os.path.join(path, file))
                except:
                    continue
    print('Trial numbers added to output csv files.')

def merge_output_results(path):
    all_files = glob.glob(os.path.join(path, "trial_*.csv"))
    df_from_each_file = (pd.read_csv(file, sep=',') for file in all_files)
    df_merged = pd.concat(df_from_each_file, ignore_index=True)
    df_merged.to_csv(os.path.join(path, "merged_results.csv"))
    print('Merged_results.csv created')

def optuna_define_model(trial, train_features):
    # This is for running Optuna to select model parameters
    normalizer =preprocessing.Normalization()
    normalizer.adapt(np.array(train_features))
    print("Features mean: %.2f" % (normalizer(np.array(train_features)).numpy().mean()))
    print("Features std: %.2f" % (normalizer(np.array(train_features)).numpy().std()))

    n_layers = trial.suggest_int('n_layers', 1, 10) #7,12
    model = tf.keras.Sequential()
    dropout_options = [0,0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]
    p = trial.suggest_categorical("dropout", dropout_options)
    for i in range(n_layers):
        num_hidden = 10*trial.suggest_int("n_units_l{}".format(i), 2, 15, log=True)
        model.add(
            tf.keras.layers.Dense(
                num_hidden,
                kernel_regularizer=tf.keras.regularizers.l2()
            )
        )
        model.add(tf.keras.layers.LeakyReLU(alpha=0.01))
        model.add(tf.keras.layers.Dropout(p))

    model.add(tf.keras.layers.Dense(1,kernel_regularizer=tf.keras.regularizers.l2()))
    model.compile(optimizer='Nadam', loss=tf.losses.MeanSquaredError(), metrics=[mape_metric,smape_metric, concordance_cc, pearson_cc,tfa.metrics.RSquare(dtype=tf.float32)]) # y_shape=(1,)
    return model

def objective(trial,train_dataset,val_dataset,x_train,output_location,EPOCHS):

    model = optuna_define_model(trial, x_train)
    global MODEL_NAME
    MODEL_NAME = F'trial_{trial.number}'

    # generate the model and optimizer
    filepath = os.path.join(output_location, f'trial_{trial.number}_log.csv')
    csvLogger = tf.keras.callbacks.CSVLogger(filepath)
    save_model = custom_callback_save_model()
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        verbose=0,
        mode='min',
        baseline=None,
        restore_best_weights=True)

    # Train model
    history = History()
    history = model.fit(train_dataset, batch_size=32, epochs=EPOCHS, validation_data=val_dataset, validation_batch_size=32, validation_freq=1, callbacks=[csvLogger,early_stopping])
    model.save(os.path.join(output_location,f'last_epoch_model_trial{trial.number}'))
    val_metric = history.history['val_loss'][-1]
    trial.set_user_attr('val_loss', history.history['val_loss'][-1])
    trial.set_user_attr('loss', history.history['loss'][-1])
    trial.set_user_attr('val_mse', history.history['val_loss'][-1])
    trial.set_user_attr('train_mse', history.history['loss'][-1])
    trial.set_user_attr('val_r2', history.history['val_r_square'][-1])
    trial.set_user_attr('train_r2', history.history['r_square'][-1])
    trial.set_user_attr('validation_ccc', history.history['val_concordance_cc'][-1])
    trial.set_user_attr('train_ccc', history.history['concordance_cc'][-1])
    trial.set_user_attr('validation_pearson', history.history['val_pearson_cc'][-1])
    trial.set_user_attr('train_pearson', history.history['pearson_cc'][-1])
    trial.set_user_attr('validation_mape', history.history['val_mape_metric'][-1])
    trial.set_user_attr('train_mape', history.history['mape_metric'][-1])
    trial.set_user_attr('validation_smape', history.history['val_smape_metric'][-1])
    trial.set_user_attr('train_smape', history.history['smape_metric'][-1])
    trial.set_user_attr('epochs', len(history.epoch))

    return val_metric

def optimise(N_TRIALS,EPOCHS,train_dataset,val_dataset,x_train,output_location):
    # ensure results are reproduceable
    RANDOM_SEED = 42
    setup_seed(RANDOM_SEED)

    # set up optimisation metadata
    study_name = f'optuna_{N_TRIALS}trials_{EPOCHS}epochs'

    study_direction = 'minimize'  # Regression is optimised on MSE, as we want to minimise the MSE

    study = optuna.create_study(study_name=study_name, direction=study_direction, sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED), pruner=optuna.pruners.HyperbandPruner())
    objective_function = lambda trial: objective(trial,train_dataset,val_dataset,x_train,output_location,EPOCHS)
    optuna_logging_callback_function = lambda study, trial: optuna_logging_callback(study, trial, output_location)
    study.optimize(objective_function, n_trials=N_TRIALS, callbacks=[optuna_logging_callback_function])
    optuna.importance.get_param_importances(study)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    add_trial_number_to_output_files(output_location)
    merge_output_results(output_location)

#Ensure reproduceability of tensorflow runs.
setup_seed(42)

#Specify which model to apply to a test data subset or to new data:
#1) This is the name of the folder in which all models are located once they created. Model are created by running the train_test_model function using the parameter Test = 'Optuna'
models_folder =os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Models')

#This just makes it a bit easier to rerun scenarios. Specify the scenario option:
#If you're testing a model, you must have created a model for the appropriate scenario.
scenario = 'scenario_2'

if scenario == 'scenario_1':
    use_CL = True
    use_UTH = True
    use_all_data = True
    cap_data_at_69 = False
    model_description = 'scenario1_last_epoch_model_trial134'

elif scenario == 'scenario_2':
    use_CL=True
    use_UTH = True
    use_all_data=False
    cap_data_at_69 = True
    model_description = 'scenario2_last_epoch_model_trial57'

elif scenario == 'scenario_3':
    use_CL = False
    use_UTH = True
    use_all_data = False
    cap_data_at_69 = True
    model_description = 'scenario3_ last_epoch_model_trial31'

elif scenario == 'scenario_4':
    use_CL = True
    use_UTH = False
    use_all_data = False
    cap_data_at_69 = True
    model_description = 'scenario4_last_epoch_model_trial153'

elif scenario == 'scenario_5':
    use_CL = True
    use_UTH = False
    use_all_data = True
    cap_data_at_69 = True
    model_description = 'scenario5_last_epoch_model_trial47'

elif scenario == 'scenario_6':
    use_CL = False
    use_UTH = False
    use_all_data = False
    cap_data_at_69 = True
    model_description = 'scenario6_last_epoch_model_trial113'

elif scenario == 'scenario_7':
    use_CL = False
    use_UTH = False
    use_all_data = True
    cap_data_at_69 = True
    model_description = 'scenario7_last_epoch_model_trial32'

#Optimise the models, or run 5-fold cross-validation, or test an optimimal model

train_test_model(
    scenario = scenario,
    cap_data = cap_data_at_69,
    input_data_filepath=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Data_files','Dataset.csv'),  # update the file path to the input data file
    aggregate_size=2, #number of zircon to aggregate to create an additional data point
    resampling_repeats=2, #factor by which the largest silica bin is increased
    test_split_size=0.1,  #percentage of data set to keep aside as a test subset
    CL=use_CL, # whether or not to use cathodoluminescence texture
    UTH=use_UTH, # whether or not to use U and Th data per grain
    all_data=use_all_data, #For scenarios that don't use U and Th, choose whether to test on all the data or the analysis-constrained data
    outliers_to_remove= None,  # list any outlier sample ID to remove. Provide this input as an array e.g. [180933,168936]
    epochs=2, #How many epochs to train for (early-stopping is applied, though)
    n_trials=2, #Number of Optuna trials to run
    batchsize=32, #batchsize of 32 is the Tensorflow default
    Test = 'Optuna',#Indicates wich action to take. Options: 'Optuna', 'Kfold', 'Test'. If not specified, the function exits once data sets are created.
    model_filepath = os.path.join(models_folder, model_description) # path to the model you are applying on the test data. If not None, specify the model here using this:  os.path.join(models_folder, description)
    )

#Use this function to apply an existing model to a new data set.
#This  function is commented out. To use this function, remove the starting and triple quotation marks
'''
apply_model(
    use_UTH = True, #if you're not using UTH, then this is false (e.g. scenario 2 versus scenario 4). Use the  input data for the specified model.
    use_CL = True,# if you're not using CL, then this is false. (e.g. scenario 1). Use the  input data for the specified model.
    model_path =os.path.join(models_folder, model_description), # path to the model you are applying on new data
    #you will need to pass your new data through the same PCA that the training data went through.
    #This links to the original data on which the PCA was modelled, allowing you to recreate the PCA and apply it to new data
    dataset_for_pca_loadings =os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Outputs', 'Results_Optuna_2AGGx2Resample_SHAPE_UTH_CL_20062023130738','PCAInputs.csv'),
    data_columns = ['GSWA_sample_id','area','equivalent_diameter','perimeter','minor_axis_length','major_axis_length','solidity','convex_area','form_factor','roundness','compactness','aspect_ratio','minimum_Feret','maximum_Feret','U238_ppm','Th232_ppm','SiO2_pct','oscillatory_zonation', 'sector_zonation','homogenous_zonation'], #columns in the PCAInputs data that you will need
    data_to_predict_on=os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Data_Files','Greenland_dataset.csv'), #the data file to run the silica predictions on,
    prediction_columns = ['sample_id','analytical_spot','group','area','equivalent_diameter','perimeter','minor_axis_length','major_axis_length','solidity','convex_area','form_factor','roundness','compactness','aspect_ratio','minimum_Feret','maximum_Feret','U238_ppm','Th232_ppm', 'oscillatory_zonation', 'sector_zonation','homogenous_zonation'], #columns required in the dataset that prediction will be run on
    output_location =os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),'Outputs'),
    description = scenario,
    keep_columns =['analytical_spot', 'sample_id', 'group'] #columns in the dataset, which aren't used for prediction, but which are nice to have in the results file.
    )'''



