# imports...
import pandas as pd
import numpy as np
from datetime import timedelta


# Define feature engineering/imputation functions

def hr_rolling_features(df):

    '''
    Returns dataframe with heart rate rolling features.
    '''
    df['hr_10min_rolling'] = df['heart_rate'].rolling(window=10, min_periods=1).apply(lambda x: x[x!= 0].mean()) # short term feature
    df['hr_30min_rolling'] = df['heart_rate'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].mean()) # medium term feature
    df['hr_60min_rolling'] = df['heart_rate'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].mean()) # long term feature
    
    return df

#############################################################################################################################################

def hrv_features(df):
    
    '''
    Returns a dataframe with heart rate variability engineered features.
    '''  
    df['last_measured_hrv_15min'] = df['heart_rate_variability'].fillna(method="ffill", limit=15)
    df['hrv_change_15min'] = df['last_measured_hrv_15min'].diff(periods=15)
   
    return df
    
#############################################################################################################################################

def bl_ox_rr_features(df):
    
    '''
    Returns dataframe with engineered features for blood oxygen saturation and respiratory rate.
    '''    
    # Blood oxygen saturation
    df['last_measured_ox'] = df['blood_oxygen_saturation'].fillna(method="ffill", limit=30)
  
    # Respiratory rate
    df['last_measured_rr'] = df['respiratory_rate'].fillna(method="ffill", limit=15)
  
    return df

#############################################################################################################################################

def steps_activity_features(df):
    
    '''
    Returns dataframe with step count and active energy engineered features.
    '''
    # fill NaNs with zeroes
    df['step_count'] = df['step_count'].fillna(0)
    df['active_energy'] = df['active_energy'].fillna(0)
    
    # derive rolling features
    df['step_count_rollingsum_30min'] = df['step_count'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].sum()) # medium term
    df['step_count_rollingsum_60min'] = df['step_count'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].sum()) # long term
    df['active_energy_rollingsum_30min'] = df['active_energy'].rolling(window=30, min_periods=1).apply(lambda x: x[x!= 0].sum()) # medium term
    df['active_energy_rollingsum_60min'] = df['active_energy'].rolling(window=60, min_periods=1).apply(lambda x: x[x!= 0].sum()) # long term

    return df
    
#############################################################################################################################################

def meal_features(df):
    
    '''
    Returns dataframe with meal engineered features.
    '''
    # creating a temp column 
    df['meal_temp'] = df['meal'].fillna(method="ffill")
    
    # reseting with a 0 every time a meal is ingested
    df['meal_temp'] = np.where(df['meal']==1, 0, df['meal_temp'])

    # cumulative sum that resets at the beginning of each meal 
    reset_mask = df['meal_temp'] == 0
    groups = reset_mask.cumsum()
    
    #calculating time since last meal 
    df['time_since_lastmeal'] = df.groupby(groups)['meal_temp'].cumsum()
    
    # creating an ordinal feature that refelcts how long it has been since the last meal 
    labels = [1,2,3,4,5,6]
    conditions =[
        (df['time_since_lastmeal'] >= 0) & (df['time_since_lastmeal'] <= 15),
        (df['time_since_lastmeal'] > 15) & (df['time_since_lastmeal'] <= 30),
        (df['time_since_lastmeal'] > 30) & (df['time_since_lastmeal'] <= 60),
        (df['time_since_lastmeal'] > 60) & (df['time_since_lastmeal'] <= 120),
        (df['time_since_lastmeal'] > 120) & (df['time_since_lastmeal'] <= 180),
        df['time_since_lastmeal'] > 180
        ]

    df['time_since_lastmeal_ord'] = np.select(conditions, labels, default= np.NaN)
    
    # dropping temp column
    df.drop('meal_temp', axis=1, inplace=True)

    return df

#############################################################################################################################################

def insulin_features(df):
    
    '''
    Returns dataframe with Insulin engineered features
    '''
    # subset short acting insulin column
    df_insulin = pd.DataFrame(df[['insulin_short']])

    # create a column for each short acting insulin shot, in case there are shots less than 4 hours apart
    counter = 0

    for index, row in df_insulin[['insulin_short']].iterrows():
        value = row['insulin_short']
        if value > 0:
            column_name = 'insulin_short_' + str(counter)
            df_insulin[column_name] = np.where(df_insulin.index == index, value, np.NaN)
            counter+=1

    # calculating insulin on board
    df_ins = df_insulin.drop('insulin_short', axis=1)

    for column in df_ins.columns:
        value = df_ins[column].loc[df_ins[column].first_valid_index()]
        idx = df_ins[column].first_valid_index()
        df_ins[column].loc[idx :(idx+timedelta(minutes=60))] = 1
        df_ins[column].loc[(idx+timedelta(minutes=60)):(idx+timedelta(minutes=120))] = 0.75
        df_ins[column].loc[(idx+timedelta(minutes=120)):(idx+timedelta(minutes=180))] = 0.5
        df_ins[column].loc[(idx+timedelta(minutes=180)):(idx+timedelta(minutes=240))] = 0.25

    # summing across each row
    df_ins['IOB'] = df_ins.sum(axis=1)
    
    df['IOB'] = df_ins['IOB']
    
    return df
    
#############################################################################################################################################

def glucose_features(df):

    '''
    Returns dataframe with cgm glucose features
    '''
    # fill in between CGM readings
    df['cgm'] = df['glucose'].fillna(method='bfill', limit=5)

    #create a temporary hypoglycemia column
    df['hypoglycemia_temp'] = np.where(df['cgm'] < 4, 1, 0).astype('int')
    df['hypoglycemia_temp'] = np.where(df['cgm'].isna(), np.NaN, df['hypoglycemia_temp']) # making sure NaNs remain NaNs
    
    # calculating duration of hypoglycemic episodes, to discard events lasting less than 15 minutes
    df["hypo_duration"] = df["hypoglycemia_temp"][df.hypoglycemia_temp.notna()].groupby((df["hypoglycemia_temp"] == 0).cumsum()).cumcount()
    df['hypo_duration_15more'] = np.where((df["hypo_duration"] < 15) | (df["hypo_duration"].isna()), np.NaN, 1)
    df['hypo_duration_15more'] = df['hypo_duration_15more'].fillna(method='bfill', limit=14)
    df['hypo_duration_15more'] = df['hypo_duration_15more'].fillna(0).astype('int')
    df['hypo_duration_15more'] = np.where(df['cgm'].isna(), np.NaN, df['hypo_duration_15more']) # making sure NaNs remain NaNs
    df['hypoglycemia'] = df['hypo_duration_15more']

    # creating a column for prehypoglycemia phase to evaluate percentage of false positives lying in this period 
    df['hypo_shift'] = df['hypo_duration_15more'].shift(-1)
    df['hypo_start'] = np.where((df['hypo_duration_15more'] == 0) & (df['hypo_shift'] == 1), 1, np.NaN)
    df['hypo_end'] = np.where((df['hypo_duration_15more'] == 1) & (df['hypo_shift'] == 0), 1, np.NaN)
    
    df['prehypoglycemia'] = df['hypo_start'].fillna(method='bfill', limit=15)
    df['prehypoglycemia'] = np.where(df['hypoglycemia']==1 , 0, df['prehypoglycemia']) # in case of coinciding hypoglycemic events    
    df['prehypoglycemia'] = df['prehypoglycemia'].fillna(0).astype('int')
    df['prehypoglycemia'] = np.where(df['cgm'].isna(), np.NaN, df['prehypoglycemia']) # making sure NaNs remain NaNs
    
    return df
    
#############################################################################################################################################

def feature_engineering(filename):
    '''
    Collective function for all feature engineering functions in addition to adding 'hour' and 'patient_code' features
    '''

    # read the data
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)

    ## Heart rate
    df = hr_rolling_features(df)

    ## Heart rate variability
    df = hrv_features(df)

    ## bl ox and rr
    df = bl_ox_rr_features(df)

    ## steps and activity
    df = steps_activity_features(df)

    ## meal
    df = meal_features(df)

    ## insulin
    df = insulin_features(df)

    ## cgm
    df = glucose_features(df)

    ## time of day (sine and cosine transformations)
    df['sin_hour'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['cos_hour'] = np.cos(2 * np.pi * df['hour'] / 24)

    ## patient code
    df['patient_code'] = filename[:5]
    
    return df

#############################################################################################################################################