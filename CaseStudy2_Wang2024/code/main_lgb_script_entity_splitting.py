#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
from tqdm import tqdm
#from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from pathlib import Path
from custom_functions import create_f1_array, plot_cv_results, test_independence_of_entities

def plot_confusion_matrix(y_true, y_pred, classes,type,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        # print('Confusion matrix, without normalization')

    # print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    ax.set_ylim(len(classes) - 0.5, -0.5)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    #return ax
    parent_folder = Path(__file__).resolve().parent.parent
    plt.savefig(f"{parent_folder}/Outputs/{type}_confusion_matrix.svg", dpi=300, bbox_inches='tight')
    plt.close()

def cal(y, oof_lgb, p=0.5):
    print("AUC score: {}".format(roc_auc_score(y, oof_lgb)))
    print("F1 score: {}".format(f1_score(y, [1 if i >= p else 0 for i in oof_lgb])))
    print("Precision score: {}".format(precision_score(y, [1 if i >= p else 0 for i in oof_lgb])))
    print("Recall score: {}".format(recall_score(y, [1 if i >= p else 0 for i in oof_lgb])))

'''Cell 2'''
def ks_calc_cross(data,pred,y_label):
    crossfreq = pd.crosstab(data[pred[0]],data[y_label[0]])
    crossdens = crossfreq.cumsum(axis=0) / crossfreq.sum()
    crossdens['gap'] = abs(crossdens[0] - crossdens[1])
    ks = crossdens[crossdens['gap'] == crossdens['gap'].max()]
    return ks,crossdens

def ks_calc_auc(data,pred,y_label):
    fpr,tpr,thresholds= roc_curve(data[y_label[0]],data[pred[0]])
    ks = max(tpr-fpr)
    return ks

def ks_calc_2samp(data,pred,y_label):
    Bad = data.loc[data[y_label[0]]==1,pred[0]]
    Good = data.loc[data[y_label[0]]==0, pred[0]]
    data1 = Bad.values
    data2 = Good.values
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.concatenate([data1,data2])
    cdf1 = np.searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = (np.searchsorted(data2,data_all,side='right'))/(1.0*n2)
    ks = np.max(np.absolute(cdf1-cdf2))
    cdf1_df = pd.DataFrame(cdf1)
    cdf2_df = pd.DataFrame(cdf2)
    cdf_df = pd.concat([cdf1_df,cdf2_df],axis = 1)
    cdf_df.columns = ['cdf_Bad','cdf_Good']
    cdf_df['gap'] = cdf_df['cdf_Bad']-cdf_df['cdf_Good']
    return ks,cdf_df

'''Cell 3'''
train_data_list = ['training_raw data_1.xlsx','training_raw data_0.xlsx']
train_data = pd.DataFrame()
for file in train_data_list:
    _ = pd.read_excel('../data/{}'.format(file))
    if '1' in file:
        _['label'] = 1
    else:
        _['label'] = 0
    _.columns = ['CITATION','SAMPLE NAME','ROCK TYPE','TI', 'Y', 'NB', 'LA', 'CE', 'CE*', 'PR', 'ND', 'SM', 'EU', 'EU*', 'GD', 'TB', 'DY',
       'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'TH', 'U', 'label']
    train_data = pd.concat([train_data,_], axis=0)

# Here I establish an entity-ID for entity_splitting.
# I include citation, as sample names are not universally unique (i.e. they may be replicated in different studies).
train_data['ENTITY_ID'] = train_data['CITATION'] + train_data['SAMPLE NAME']
train_data = train_data.drop(['ROCK TYPE', 'CITATION', 'SAMPLE NAME'], axis=1)

# Sample JJD_28 of citation JARA J.J. occurs in both the alkaline and subalkaline datasets. It is here removed as a duplicate.
train_data = train_data[train_data['ENTITY_ID'] != '[24690] JARA J. J. (2021)samp. JJJD_28']

#Check how many entities there are per category
train_data = train_data.reset_index(drop=True)
print(f'Entities per class in the full dataset: \n {train_data.groupby('label')['ENTITY_ID'].nunique()}')
print(f'Total datapoints in the full dataset: {train_data.shape[0]}')

valid = pd.read_excel('../data/{}'.format('application data.xlsx'))
valid['label']=np.nan

print('#'*20)
print('miss rate_training data')
print(train_data.isnull().sum() / (valid.shape[0]))
print('#'*20)
print('miss rate_application data')
print(valid.isnull().sum() / (valid.shape[0]))

'''Cell 4'''
feas = ['TI', 'Y', 'NB', 'LA', 'CE', 'CE*', 'PR', 'ND', 'SM', 'EU', 'EU*', 'GD', 'TB', 'DY',
       'HO', 'ER', 'TM', 'YB', 'LU', 'HF', 'TA', 'TH', 'U']
for fea in feas:
    train_data['{}_isnull'.format(fea)] = 0
    valid['{}_isnull'.format(fea)] = 0
    train_data.loc[train_data[fea].isnull(), '{}_isnull'.format(fea)] =1
    valid.loc[valid[fea].isnull(), '{}_isnull'.format(fea)] =1

'''Cell 5'''
train_data = train_data.fillna(-99)
valid = valid.fillna(-99)
test_independence_of_entities(train_data, entity_id='ENTITY_ID', target_column='label', compositional_columns=feas)

'''Cell 6'''
X_train, X_test = train_test_split(train_data, test_size=0.3, random_state=42)

#Check number of entities per group, as this split does not stratify:
print('Observation splitting')
print(f'Entities per class in the train dataset: \n {X_train.groupby('label')['ENTITY_ID'].nunique()}')
print('Total data in train dataset:X_train.shape[0] ')
print(f'Entities per class in the test dataset: \n {X_test.groupby('label')['ENTITY_ID'].nunique()}')
print('Total data in test dataset:X_test.shape[0] ')

'''Cell 7'''
feas_expode = []
for i in range(len(feas)-1):
    for j in range(i,len(feas)):
        for data in [train_data,valid]:
            data['{}+{}'.format(feas[i],feas[j])] = data[feas[i]]+data[feas[j]]
            data['{}-{}'.format(feas[i],feas[j])] = data[feas[i]]-data[feas[j]]
            data['{}*{}'.format(feas[i],feas[j])] = data[feas[i]]*data[feas[j]]
            data['{}/{}'.format(feas[i],feas[j])] = data[feas[i]]/(data[feas[j]]+1e-10)
#           data['{}^{}'.format(feas[i],feas[j])] = data[feas[i]]**(train_data[feas[j]]+1e-10)

'''Cell 8'''
for i in tqdm(range(len(feas)-2)):
    for j in range(i,len(feas)-1):
        for k in range(j,len(feas)):
            for data in [train_data,valid]:
                data['{}+{}+{}'.format(feas[i],feas[j],feas[k])] = data[feas[i]]+data[feas[j]]+data[feas[k]]
                data['{}-{}-{}'.format(feas[i],feas[j],feas[k])] = data[feas[i]]-data[feas[j]]-data[feas[k]]
                data['{}*{}*{}'.format(feas[i],feas[j],feas[k])] = data[feas[i]]*data[feas[j]]*data[feas[k]]
                data['{}/{}/{}'.format(feas[i],feas[j],feas[k])] = data[feas[i]]/(data[feas[j]]+1e-10)/(data[feas[k]]+1e-10)

'''Cell 9'''
X_train, X_test = train_test_split(train_data, test_size=0.3, random_state=42)

# The previous train-test split is an observation-split (random splitting), as implemented by Wang et al. (2024).
# Here I perform the entity_splitting equivalent, for comparison.
# Note that neither of these splitting methodologies will stratify the datasets
gss = GroupShuffleSplit(n_splits = 1, train_size = 0.7, random_state = 42) #only perform this split once
le = LabelEncoder()
train_data["ENTITY_ID"] = le.fit_transform(train_data["ENTITY_ID"])
groups = train_data['ENTITY_ID'].astype(str)#use entity-ID to group data points
X_train_entity=[]
X_test_entity=[]
for i, (train_index, test_index) in enumerate(gss.split(train_data, groups=groups)):
    X_train_entity = train_data.iloc[train_index]
    X_test_entity = train_data.iloc[test_index]

#Sanity check to ensure that no entity_id's are shared between the train and test splits.
train_groups = set(X_train_entity['ENTITY_ID'])
no_shared_groups = train_groups.isdisjoint(set(X_test_entity['ENTITY_ID']))
print(f'No groups shared between train and test: {no_shared_groups}')

'''Cell 10'''
y_train = X_train['label']
y_test = X_test['label']
y_train_entity = X_train_entity['label']
y_test_entity = X_test_entity['label']

cat_cols = []

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
X_train.shape, X_test.shape
X_train_entity  = X_train_entity.reset_index(drop=True)
X_test_entity = X_test_entity.reset_index(drop=True)

#features = list(set(X_train.columns.tolist()) - set(['label','prob']))
features = list(set(X_train_entity.columns.tolist()) - set(['label', 'prob', 'SAMPLEID']))
#print(features)

'''Cell 11'''
from sklearn.metrics import f1_score,roc_auc_score
def F1_score(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.where(preds>0.5,1,0)
    f_score = f1_score(labels, preds, average = 'binary')
    return 'f1_score', f_score, True

'''Cell 12'''
def lgb_5fold(X_train,y_train, X_test, valid, train_data, features_slc, entity = False, groups = None):
    '''
    :param X_train: x of dataset for model training
    :param y_train: y of dataset for model training
    :param X_test: x of dataset for model testing
    :param valid: external validation dataset
    :param train_data: All data (includes train and test data)
    :param features_slc: features to use for training
    :param entity: whether to use entity-splitting should be applied
    :param groups: group to use for entity_splitting
    :return:
    '''

    df_feature_importance = pd.DataFrame()
    df_feature_importance['features'] = features_slc
    df_feature_importance['gain'] = 0

    params = {
        'objective': 'binary',
        'metric': 'binary_error',  # binary_error
        'learning_rate': 0.05,  # 0.052
        'subsample': 0.8,
        'subsample_freq': 3,
        'num_iterations': 2000,
        'is_unbalance': True
    }

    #I'd like to repeat the 5-fold cross-validation 10 times
    #This is to create several cross-validation F1-values so that I can plot a histogram of F1 values
    predictions_lgb_10x5_repeats = {'TEST':[],'PREDICTED':[], 'ACTUALS':[]}
    predictions_lgb = np.zeros((len(X_test), 5*10))

    for repeat in range(10):

        # This kfold will only use the training data, in accordance with the original methodology.
        if not entity:
            KF = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat)
            KF_splits = KF.split(X_train.values, y_train.values)
        else:
            KF = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=repeat)
            KF.get_n_splits(X_train.values, y_train.values)
            KF_splits = KF.split(X_train.values, y_train.values, groups)

        oof_lgb = np.zeros(len(X_train))
        valid_lgb = np.zeros((len(valid), 5))
        train_lgb = np.zeros((len(train_data), 5))

        for fold_, (trn_idx, val_idx) in enumerate(KF_splits):
            repeat_fold = 5 * repeat + fold_
            print("fold n°{}".format(fold_))
            print('trn_idx:', trn_idx)
            print('val_idx:', val_idx)
            print(f'Num train entities: {X_train.iloc[trn_idx]['ENTITY_ID'].nunique()}')
            print(f'Num test entities: {X_train.iloc[val_idx]['ENTITY_ID'].nunique()}')
            trn_data = lgb.Dataset(X_train.iloc[trn_idx][features_slc], label=y_train.iloc[trn_idx],categorical_feature=cat_cols,)
            val_data = lgb.Dataset(X_train.iloc[val_idx][features_slc], label=y_train.iloc[val_idx],categorical_feature=cat_cols,)
            num_round = 10000
            clf = lgb.train(
                params,
                trn_data,
                num_round,
                valid_sets=[trn_data, val_data],
                #verbose_eval=100,
                #early_stopping_rounds=200,
                feval=F1_score,
                callbacks=[
                    early_stopping(stopping_rounds=200),
                    log_evaluation(period=100)
                ]
            )
            oof_lgb[val_idx] = clf.predict(X_train.iloc[val_idx][features_slc], num_iteration=clf.best_iteration) #predices on the fold's validation data.
            predictions_lgb[:, repeat_fold] = clf.predict(X_test[features_slc], num_iteration=clf.best_iteration) #predicts on the hold-out test dataset
            valid_lgb[:, fold_] = clf.predict(valid[features_slc], num_iteration=clf.best_iteration) #predicts on the hold-out validation dataset
            train_lgb[:, fold_] = clf.predict(train_data[features_slc], num_iteration=clf.best_iteration) #predicts on all the train data
            df_feature_importance['gain'] += clf.feature_importance() / 5
            predictions_lgb_10x5_repeats['TEST'].extend([f"{repeat}.{fold_}"]*len(X_test['label']))
            predictions_lgb_10x5_repeats['PREDICTED'].extend(np.array(predictions_lgb[:, fold_]))
            predictions_lgb_10x5_repeats['ACTUALS'].extend(np.array(X_test['label']))

    df_predictions_lgb_10x5_repeats = pd.DataFrame(predictions_lgb_10x5_repeats)

    return oof_lgb, predictions_lgb, valid_lgb, train_lgb, df_feature_importance,df_predictions_lgb_10x5_repeats

'''Cell 13'''
#features_slc = features
#oof_lgb,predictions_lgb,valid_lgb,train_lgb,df_feature_importance = lgb_5fold(X_train,X_test,valid,train_data,12,features_slc)
#features_slc = df_feature_importance[df_feature_importance['gain']>0].sort_values(by='gain',ascending=False).head(120).features
#oof_lgb,predictions_lgb,valid_lgb,train_lgb,df_feature_importance = lgb_5fold(X_train,X_test,valid,train_data,12,features_slc)
#features_slc=list(set(df_feature_importance.sort_values(by='gain', ascending=False)['features']))
#print(features_slc)

'''Cell 14'''
#The selected 120 features are present here. The code for feature selection is present in In [13].
features_slc = ['CE**EU*EU', 'YB/LU', 'NB/LA/CE*', 'TM/YB', 'Y/LA/TB', 'ER/YB', 'TH/U', 'EU**EU**TA', 'ER/TM', 'PR/ND/EU*', 'LU/TH', 'HO-LU', 'TI*TI*CE', 'TB/DY', 'TH-U', 'DY/HO', 'TA/U', 'Y/CE*/TH', 'NB*EU**U', 'Y/CE*/EU', 'TI*TI*DY', 'Y/DY/HF', 'HF-U', 'ND-EU-EU', 'TI*EU**U', 'CE**ND*HF', 'NB/LU/TA', 'EU/GD', 'Y*HF*TA', 'Y/CE/EU*', 'Y/YB/TH', 'YB/LU/HF', 'CE-ND-TA', 'PR/ND/TA', 'SM/GD', 'TI/NB/EU*', 'GD/TB', 'CE/U', 'Y/NB', 'Y*EU**HF', 'TI*TI*U', 'HO/ER', 'LA+EU*+TA', 'CE/DY/U', 'CE/EU/TA', 'TI*TI*TA', 'TI/EU*/HF', 'TI*Y*TA', 'CE/HF/HF', 'CE/GD', 'TI*EU*HF', 'CE/HF/TH', 'CE/GD/HF', 'CE/EU*/TH', 'TI*Y*EU*', 'TI*HF*HF', 'EU**HF*TA', 'HF-TH-U', 'EU/GD/HF', 'YB/LU/TA', 'CE/EU*', 'NB/CE/CE', 'Y/CE/U', 'CE/DY/DY', 'Y/TH', 'PR/EU/TA', 'CE/EU*/EU*', 'EU**HF*HF', 'Y/TM', 'Y/HO', 'Y/NB/EU*', 'NB/HF/U', 'ER-LU-LU', 'HO-LU-TA', 'TI*NB*TA', 'Y/EU*/U', 'Y/ER', 'Y/NB/ER', 'TI/EU*/TA', 'HO/TM', 'CE/TA/TH', 'TM/YB/HF', 'TI*HF*TA', 'TB/LU/HF', 'ER/LU', 'Y/DY/TA', 'DY-LU-LU', 'CE/EU/TH', 'ER/TM/HF', 'TA*U', 'TM/LU', 'NB/HF/TH', 'CE*EU**HF', 'Y*CE*EU*', 'Y/NB/YB', 'ND-EU-EU*', 'NB/TH/U', 'CE/TA/U', 'Y/SM', 'TI/HF/HF', 'Y/YB', 'TI/HF', 'EU*/TA/TA', 'CE/EU/HF', 'DY-LU-TA', 'CE/SM/HF', 'DY/LU', 'ER/YB/HF', 'Y/DY', 'Y/NB/CE', 'SM/EU/TA', 'CE/EU/EU*', 'CE/EU*/U', 'NB/HF/TA', 'NB/TA', 'TB/DY/HF', 'NB-TA-TA', 'SM/GD/HF', 'EU/EU*/GD', 'CE/TH']
oof_lgb,predictions_lgb,valid_lgb,train_lgb,df_feature_importance,predictions_lgb_10x5_repeats = lgb_5fold(X_train, y_train, X_test, valid, train_data, features_slc)
oof_lgb_ent,predictions_lgb_ent,valid_lgb_ent,train_lgb_ent,df_feature_importance_ent,predictions_lgb_ent_10x5_repeats = lgb_5fold(X_train_entity, y_train_entity, X_test_entity, valid, train_data, features_slc, entity=True, groups = X_train_entity['ENTITY_ID'])

'''Cell 15'''
predictions_lgb = predictions_lgb.mean(axis=1)
predictions_lgb_ent = predictions_lgb_ent.mean(axis=1)

'''Cell 16'''
p = 0.5
print('#'*20)
print('train entities')
cal(y_train_entity, oof_lgb_ent, p=p)
print('#'*20)
print('test entities')
cal(y_test_entity.values,predictions_lgb_ent,p=p)
entity_f1_array = create_f1_array(predictions_lgb_ent_10x5_repeats)

p = 0.5
print('#'*20)
print('train observations')
cal(y_train, oof_lgb, p=p)
print('#'*20)
print('test observations')
cal(y_test.values,predictions_lgb,p=p)
observation_f1_array = create_f1_array(predictions_lgb_10x5_repeats)

#Create a histogram of F1 results
plot_cv_results(entity_results=entity_f1_array , observation_results=observation_f1_array ,title='f1_histograms')

print('#'*20)
print('Observation train confusion matrix')
class_names = np.array(["0","1"])
plot_confusion_matrix(y_train, [1 if i >= p else 0 for i in oof_lgb], classes=class_names, type='observation_train', normalize=False, cmap = plt.cm.Blues)
plot_confusion_matrix(y_train, [1 if i >= p else 0 for i in oof_lgb], classes=class_names, type='observation_train_normalised', normalize=True, cmap = plt.cm.Blues)

print('#'*20)
print('Observation test confusion matrix')
class_names = np.array(["0","1"])
plot_confusion_matrix(y_test.values, [1 if i >= p else 0 for i in predictions_lgb], classes=class_names, type='observation_test', normalize=False, cmap = plt.cm.Blues)
plot_confusion_matrix(y_test.values, [1 if i >= p else 0 for i in predictions_lgb], classes=class_names, type='observation_test_normalised', normalize=True,cmap = plt.cm.Blues)

print('#'*20)
print('Entity train confusion matrix')
class_names = np.array(["0","1"])
plot_confusion_matrix(y_train_entity, [1 if i >= p else 0 for i in oof_lgb_ent], classes=class_names,type='entity_train', normalize=False, cmap = plt.cm.Reds)
plot_confusion_matrix(y_train_entity, [1 if i >= p else 0 for i in oof_lgb_ent], classes=class_names,type='entity_train_normalised', normalize=True, cmap = plt.cm.Reds)

print('#'*20)
print('Entity test confusion matrix')
class_names = np.array(["0","1"])
plot_confusion_matrix(y_test_entity.values, [1 if i >= p else 0 for i in predictions_lgb_ent], classes=class_names,type='entity_test', normalize=False, cmap = plt.cm.Reds)
plot_confusion_matrix(y_test_entity.values, [1 if i >= p else 0 for i in predictions_lgb_ent], classes=class_names,type='entity_test_normalised', normalize=True, cmap = plt.cm.Reds)

print('#'*20)
print('positive number in application data:')
print(np.sum([1 if f>=p else 0 for f in valid_lgb.mean(axis=1).tolist()]))

train_data['prob'] = train_lgb.mean(axis=1)
train_data['pred'] = [1 if f>=p else 0 for f in train_lgb.mean(axis=1).tolist()]

valid['prob'] = valid_lgb.mean(axis=1)
valid['pred'] = [1 if f>=p else 0 for f in valid_lgb.mean(axis=1).tolist()]

'''Cell 17'''
cols_output = features_slc + ['label','prob','pred']
train_data[cols_output].to_excel('../res/training data_label.xlsx', index=False)
valid[cols_output].to_excel('../res/application data_label.xlsx', index=False)
df_feature_importance.sort_values(by='gain', ascending=False).to_excel('../res/feature importance.xlsx', index=False)

'''Cell 18'''
print('#'*20)
print('feature importance：')
print(df_feature_importance.sort_values(by='gain', ascending=False).head(10))

print('#'*20)
print('feature importance entity：')
print(df_feature_importance_ent.sort_values(by='gain', ascending=False).head(10))

'''Cell 19'''
df_index = pd.DataFrame()
df_index['feature'] = features_slc

'''Cell 20'''
from sklearn.metrics import roc_curve
from scipy.stats import ks_2samp
print('#'*20)
print('variable_ks')
ks_lst = []
for col in features_slc:
    ks=ks_calc_auc(train_data,[col], ['label'])
#     print('{} ks :{}'.format(col,ks))
    ks_lst.append(ks)
df_index['ks'] = ks_lst
df_index.to_excel('../res/ks.xlsx', index=False)



