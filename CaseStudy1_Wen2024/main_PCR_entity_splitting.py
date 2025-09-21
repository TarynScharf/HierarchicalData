# coding=gbk
import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedGroupKFold, GroupKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.impute import KNNImputer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')
from custom_functions import plot_cv_results, plot_confusion_matrix,custom_stratified_group_kfold, custom_stratified_train_test_split,test_independence_of_entities

def knn_impute(df, label_col, n_neighbors=3, test_size=0.1):

    groups = df.groupby(label_col)

    imputer = KNNImputer(n_neighbors=n_neighbors)

    imputed_dfs = []
    for label, group in groups:
        for i, key in enumerate(group):
            #To skip over the Deposit column, change 0 to 1
            if i >1:#!= 0:
                if n_neighbors > 0:
                    try:
                        group[key] = imputer.fit_transform(np.array(group[key]).reshape(-1, 1))
                    except:
                        stop=1
                else:
                    group.loc[:, str(key)] = group.loc[:, str(key)].fillna(group.loc[:, str(key)].median())

        imputed_dfs.append(group)

    df = pd.concat(imputed_dfs, ignore_index=True)
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)

    return df

def replace_outliers_with_nan(data, threshold=1.5):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    data = np.where((data < lower_bound) | (data > upper_bound), np.nan, data)
    return data

if __name__ == '__main__':
    usehyparams_search = False
    useknn_impute = True

    # 'Deposit' is needed as an entity_ID. PCR_Database_with_deposit_name.csv is a copy of PCR_Database with a column ('Deposit')
    # into which deposit name has been copied from PCR Database Original.xlsx
    file_name = 'PCR_Database_with_deposit_name.csv'

    df = pd.read_csv(file_name, encoding='gbk')

    columns_to_drop = ['La', 'Pr']
    df = df.drop(columns=columns_to_drop)

    label_encoder = LabelEncoder()
    df['label'] = df['label'].astype(str)
    df['label'] = label_encoder.fit_transform(df['label'])

    for col in df.columns:
        #Updated to account for the additional alpha-column (Deposit) in the dataframe
        if col != 'label' and col !='Deposit':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = replace_outliers_with_nan(df[col])

    nan_percentage = df.isna().mean() * 100
    threshold = 30

    columns_to_drop = nan_percentage[nan_percentage > threshold].index.tolist()
    df = df.drop(columns=columns_to_drop)
    df = df.dropna(how='all')

    if useknn_impute:
        df = knn_impute(df, 'label')

        #test whether entities are dissimilar
        compositional_columns = ['Hf', 'Th', 'U', 'Y', 'Ti', 'Ce', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb',
                                 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'REE']
        test_independence_of_entities(df, entity_id='Deposit', target_column='label', compositional_columns=compositional_columns)
        #Note that I've changed the test_size from the original 0.1, to 0.3, to match the test size of the entity_split.
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.3, random_state=42)

        #Deposit must not be left in the train and test datasets.
        X_train.drop(columns='Deposit', inplace = True)
        X_test.drop(columns='Deposit', inplace = True)
        X, y = df.iloc[:, 1:], df['label']
        X.drop(columns = 'Deposit', inplace=True )

        # The previous train-test split is an observation-split (random splitting), as implemented by Wen et al. (2024).
        # Here I perform the entity_splitting, for comparison. I need each class to appear in both the train and test splits.
        # However, class V only has 3 entities. I thus need to ensure that 1 of these entities goes to the test split.
        # I've created a custom split function to do this, as I'm not aware of an existing function that will generate a train-test split that honours groups and is stratified
        # Note that I'm using a test size fraction of 0.7, instead of the 0.9 originally specified by Wen et al. (2024).
        # This is to achieve train/test splits that both contain an entity from class V. With only 3 entities, a 0.9 split places all of Class V
        # into the train subset
        X_train_entity, X_test_entity, y_train_entity, y_test_entity = custom_stratified_train_test_split(df, 'Deposit', 0.3)
        if any(item is None for item in [X_train_entity, X_test_entity, y_train_entity, y_test_entity]):
            exit()

    else:
        X, y = df.iloc[:, 1:], df['label']
        X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, 1:], df['label'], test_size=0.1,random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #scale entity_split datasets
    scaler_ent =StandardScaler()
    X_train_entity = scaler_ent.fit_transform(X_train_entity)
    X_test_entity = scaler_ent.transform(X_test_entity)

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    sample_weights = np.ones(len(X_train))
    for class_label, weight in enumerate(class_weights):
        sample_weights[y_train == class_label] = weight

    #Class_weights for entity_split datasets
    class_weights_entity = compute_class_weight('balanced', classes=np.unique(y_train_entity), y=y_train_entity)
    sample_weights_entity = np.ones(len(X_train_entity))
    for class_label_entity, weight_entity in enumerate(sample_weights_entity):
        sample_weights_entity[y_train_entity == class_label_entity] = weight_entity

    if not usehyparams_search:
        clf = XGBClassifier(
            objective='multi:softmax',
            num_class=5,
            max_depth=8,
            n_estimators=1000,
            random_state=42
        )
    else:
        clf = XGBClassifier(objective='multi:softmax', eval_metric='mlogloss', num_class=5, use_label_encoder=False,
                            random_state=42)

        param_space = {
            'learning_rate': Real(0.01, 0.5),
            'max_depth': Integer(1, 10),
            'n_estimators': Integer(50, 1200),
            'min_child_weight': Integer(1, 10),
            'subsample': Real(0.1, 1.0),
            'colsample_bytree': Real(0.1, 1.0),
            'gamma': Real(0, 1)
        }

        opt = BayesSearchCV(
            clf,
            param_space,
            n_iter=10,
            cv=5,
            scoring='f1_macro',
            n_jobs=1,
            random_state=42,
            verbose=0
        )

        opt.fit(np.row_stack((X_train, X_test)), np.concatenate((y_train, y_test)))

        search_res = np.column_stack(
            (np.array(opt.optimizer_results_[0]['x_iters']), -1 * opt.optimizer_results_[0]['func_vals']))

        column_names = ['colsample_bytree', 'gamma', "learning_rate", "max_depth", "min_child_weight", "n_estimators",
                        'subsample', "score"]
        search_res_df = pd.DataFrame(search_res, columns=column_names)

        search_res_df.to_csv(f'search_res.csv', index=False)

        clf.set_params(**opt.best_params_)
        print('best-f1:', opt.best_score_)
        print('best-params:', opt.best_params_)

    #moved up, because instantiation is not required inside the loop
    f1_scorer = make_scorer(f1_score, average='macro')

    # sanity check on entities per class
    print('Entities per label:')
    print(df.groupby('label')['Deposit'].unique())

    #Repeat cross-validation ten times
    repeated_xval_scores = []
    repeated_xval_scores_entity = []
    for repeat in range(10):
        #Updating the seed to change in each repeat. It used to be set to 42 in Wen et al. (2024).
        #Updating to limit to 3 splits, in line with kf_entity, below
        kf = KFold(n_splits=3, shuffle=True, random_state=repeat)
        scores = cross_val_score(clf, X, y, cv=kf, scoring=f1_scorer)
        repeated_xval_scores.append(scores)

        #entity_split kfold
        #There are only 3 entities in class V. Consequently, we're limited to 3 splits.
        #StratifiedGroupKFold struggles to both stratify and maintain groups, meaning that some folds return nan values
        #I could use GroupKFold, but this does not stratify, and stratification was part of the original methodology of Wen et al. (2024)
        #Consequently, I have created a custom kf generator that will stratify and honour groups
        kf_entity = custom_stratified_group_kfold(df, 'Deposit', n_splits=3, seed = repeat)#GroupKFold(n_splits=3, shuffle=True, random_state=repeat) #
        scores_entity = cross_val_score(clf,  X, y, cv=kf_entity, scoring=f1_scorer) #groups =X_groups,
        repeated_xval_scores_entity.append(scores_entity)

    average_scores_per_fold = np.mean(repeated_xval_scores, axis=0)
    for i, score in enumerate(average_scores_per_fold):
        print(f"Observation fold {i+1} average F1 Score: {score}")
    print(f"Observation mean F1 Score: {np.mean(repeated_xval_scores)}")
    observation_F1_array = np.concatenate(repeated_xval_scores)

    average_scores_per_fold_entity = np.mean(repeated_xval_scores_entity, axis=0)
    for i, score in enumerate(average_scores_per_fold_entity):
        print(f"Entity fold {i+1} average F1 Score: {score}")
    print(f"Entity mean F1 Score: {np.mean(repeated_xval_scores_entity)}")
    entity_F1_array = np.concatenate(repeated_xval_scores_entity)

    #Create KDE plot
    plot_cv_results(entity_F1_array,observation_F1_array, 'Comparison_of_F1')

    #Observation train and test, as per the original code
    clf_observation = XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        max_depth=8,
        n_estimators=1000,
        random_state=42
    )
    clf_observation.fit(X_train, y_train, sample_weights)
    y_pred = clf_observation.predict(X_test)
    feature_importance = clf_observation.feature_importances_
    accuracy = accuracy_score(y_test, y_pred)
    print('Observation classification report')
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(y_pred, y_test, 'observation', normalize=True, title=None, cmap=plt.cm.Blues)

    #Entity train and test
    clf_entity = XGBClassifier(
        objective='multi:softmax',
        num_class=5,
        max_depth=8,
        n_estimators=1000,
        random_state=42
    )
    clf_entity.fit(X_train_entity, y_train_entity, sample_weights_entity)
    y_pred_entity = clf_entity.predict(X_test_entity)
    feature_importance = clf_entity.feature_importances_
    accuracy = accuracy_score(y_test_entity, y_pred_entity)
    print('Entity classification report')
    print(classification_report(y_test_entity, y_pred_entity))
    plot_confusion_matrix(y_pred_entity, y_test_entity, 'entity', normalize=True,title=None,cmap=plt.cm.Reds)

    joblib.dump(scaler, 'cu_scaler.pkl')
    joblib.dump(clf, 'cu_model.pkl')

    df = pd.read_csv(file_name, encoding='gbk')
    #Remove the deposit column
    columns_to_drop1 = ['Deposit','La', 'Pr']
    df = df.drop(columns=columns_to_drop1)
    top_10_indices = np.argsort(feature_importance)[::-1][:]

    df = df.drop(columns=columns_to_drop)

    top_10_feature_names = [df.columns[1:][i] for i in top_10_indices]

    top_10_feature_importances = feature_importance[top_10_indices]/np.sum(feature_importance)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top_10_feature_names, top_10_feature_importances, color='skyblue')
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Feature Name')
    plt.title('Top 10 Feature Importances in XGBoost Model')
    plt.gca().invert_yaxis()

    for bar, importance_score in zip(bars, top_10_feature_importances):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance_score:.3f}', ha='left', va='center')

    plt.savefig('feature importance.png',dpi=600)
    plt.show()

    explainer = shap.Explainer(clf_observation)

    shap_values = explainer.shap_values(X_train)

    feature_importance = np.mean(np.sum(np.abs(shap_values), axis=0), axis=1) #This edit should provide each feature's mean contribution across all samples and all classes
    feature_importance = feature_importance / np.sum(feature_importance)

    feature_importance_df = pd.DataFrame({#Without the edit above, this will throw an error.
        'feature': df.columns[1:],
        'importance': feature_importance
    })

    top10_features = feature_importance_df.sort_values(by='importance', ascending=False)#.head(10)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(top10_features['feature'], top10_features['importance'], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('Top 10 Important Features based on shap')
    plt.gca().invert_yaxis()

    for bar, importance_score in zip(bars, top10_features['importance']):

        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f'{importance_score:.3f}', ha='left', va='center')

    plt.savefig('feature importance_shap.png', dpi=600)
    plt.show()
    plt.show()

    class_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))

    print("Class to Index Mapping:")
    for class_name, index in class_mapping.items():
        print(f"{class_name}: {index}")

    cm = confusion_matrix(y_test, y_pred)
    cm_entity = confusion_matrix(y_test_entity, y_pred_entity)
    #Wen at al. (2024) report a normalised confusion matrix. Consequently, I will normalise this matrix.
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    cm_entity_normalized = cm_entity.astype('float') / cm_entity.sum(axis=1, keepdims=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks(np.arange(len(np.unique(y_test))), class_mapping.keys())
    plt.yticks(np.arange(len(np.unique(y_test))), class_mapping.keys())
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    #I am flipping the y-axis so that the resulting matrix appears like that reported by Wen et al. (2024).
    plt.gca().invert_yaxis()

    for i in range(len(np.unique(y_test))):
        for j in range(len(np.unique(y_test))):
            #Wen et al. (2024) report values to 2 decimal places. I thus format the labels accordingly.
            plt.text(j, i, f"{cm_normalized[i, j]:.2f}", ha='center', va='center')
    plt.savefig('confusionmatrix.png', dpi=600)
    plt.show()

    class_probabilities = clf_observation.predict_proba(X_test)

    result_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Class_Probabilities': [list(probs) for probs in class_probabilities]
    })

    result_df.to_csv('probability_Cu.csv', index=False)