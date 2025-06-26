********************************************** 
Version: Python 3.8 
IDE: Jupyter
We recommend the readers to download Anaconda (https://www.anaconda.com/download/). The install time is typically less than one hour.
**********************************************
Library:
numpy, pandas, lightgbm, sklearn, matplotlib

Function：
main_lgb.ipynb reads the training dataset to train the LightGBM model and the optimal model is then applied to the application dataset to predict the carbonatite-alkaline zircons.

Data Set：
application data.xlsx: The application dataset containing detrital zircons with unknown source rock types.
training_raw data_1.xlsx: The training dataset containing positive samples (carbonatite-alkaline zircons).
training_raw data_0.xlsx: The training dataset containing negative samples (other zircons).

Parameters:
test_size = 0.3; random_state = 42; objective = binary; metric = binary_error; learning_rate = 0.05; subsample = 0.8; subsample_freq = 3; num_iterations = 2000; is_unbalance = True 

Output File:
application data_label: The predicted result of the application dataset with labels.
feature importance: The importance score of the selected features.
ks: The Kolmogorov-Smirnov score of the selected features.
training data_lable: The merged dataset of positive and negative samples with labels.

Notice:
In order to save run time, we input the selected features in In[14]. The code for feature selection is given in In[13].
You can change the filenames and parameters as you need.
It would take more than 2 hours for running the complete code on a "normal" computer. 
*********************************************

