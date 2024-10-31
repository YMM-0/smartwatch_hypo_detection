import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler


# define functions that create 'day' and 'night' datasets by extracting relevant features for each modeling state (daytime/nighttime)

def create_day_dataset(df):
    df_ml_awake = df[[
        'hr_10min_rolling','hr_30min_rolling', 'hr_60min_rolling',
        'last_measured_hrv_15min', 'hrv_change_15min', 
        'step_count_rollingsum_30min', 'step_count_rollingsum_60min', 
        'active_energy_rollingsum_30min', 'active_energy_rollingsum_60min',
        'time_since_lastmeal_ord','IOB', 
        'sin_hour', 'cos_hour',
        'hypoglycemia','sleep', 'day_night'
        ]]

    df_ml_awake = df_ml_awake[(df_ml_awake['hypoglycemia'].notna()) & (df_ml_awake['sleep']== 0)]
    
    return df_ml_awake

def create_night_dataset(df):

    df_ml_asleep = df[[ 
           'hr_10min_rolling','hr_30min_rolling', 'hr_60min_rolling',  
           'last_measured_hrv_15min', 'hrv_change_15min',
           'last_measured_ox','last_measured_rr',
           'time_since_lastmeal_ord','IOB',
           'sin_hour', 'cos_hour',
           'hypoglycemia','sleep', 'day_night'
            ]]

    df_ml_asleep = df_ml_asleep[(df_ml_asleep['hypoglycemia'].notna() & (df_ml_asleep['sleep']== 1))]
    
    return df_ml_asleep

    
def modeling(df, hypo_list):
    '''
    Creates custom train/test splits then fits and evaluates a model for each iteration.
    Random undersampling is performed using 10 random states to ensure reproducibility. 
    Returns evaluation metrics dataframe.
    'df' is the provided day/night dataset.
    'hypo_list' is the list of unique identifiers of days/nights where hypoglycemia was experienced.
    '''

    eval_metrics = dict()
    eval_metrics_list = []

    for i in hypo_list: 
        for random_state in range(10):
            # train test split
            train = df[(df.day_night != i)]
            train = train[train.hypo_duration <= 180] # for hypo events lasting longer than 180 minutes

            test = df[df.day_night == i]
            test = test[test.hypo_duration <= 180] # for hypo events lasting longer than 180 minutes

            X_train = train.drop(['hypoglycemia','sleep', 'day_night'], axis =1)
            X_test = test.drop(['hypoglycemia','sleep', 'day_night'], axis=1)
            y_train = train[['hypoglycemia']]
            y_test = test[['hypoglycemia']]

            # define undersample strategy
            sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=1)

            # undersampling train dataset
            X_train_under, y_train_under = sampler.fit_resample(X_train, y_train)

            # scaling train and test data
            robust_scaler = RobustScaler()

            X_train_ = robust_scaler.fit_transform(X_train_under)
            X_test_ = robust_scaler.transform(X_test)

            # create model
            model = xgb.XGBClassifier(reg_alpha=10, reg_lambda=10)

            # Fit the model to the training data
            model.fit(X_train_, y_train_under)

            # Make predictions on the testing data
            y_pred = model.predict(X_test_)

            # Evaluation metrics
            # Calculate precision
            precision = precision_score(y_test, y_pred)

            # Calculate recall
            recall = recall_score(y_test, y_pred)

            # Derive prediction probabilities 
            y_scores = model.predict_proba(X_test_)[:, 1]

            # Calculate the false positive rate (FPR), true positive rate (TPR), and thresholds
            fpr, tpr, thresholds = roc_curve(y_test, y_scores)

            # Calculate the area under the ROC curve (AUROC)
            roc_auc = auc(fpr, tpr)

            # populatin Evaluation metrics dict 
            eval_metrics['random_state'] = random_state
            eval_metrics['id'] = i
            eval_metrics['precision'] = precision
            eval_metrics['recall'] = recall
            eval_metrics['AUC'] = roc_auc
            eval_metrics_list.append(eval_metrics)
            eval_metrics = dict() #empty the dict for next set of values

            eval_metrics_df = pd.DataFrame(eval_metrics_list)
        
    return eval_metrics_df