# imports...
import pandas as pd

def cgm_preprocess(file):
    '''
    this function loads the data from the csv file, and then performs
    preprocessing steps to return a dataframe with timestamp and glucose data
    '''
    # load data from file
    cgm_df = pd.read_csv(file)
   
    # subset only relevant columns
    cgm_df = pd.DataFrame(cgm_df[['Zeitstempel (JJJJ-MM-TTThh:mm:ss)', 'Glukosewert (mmol/L)']])
    
    # rename the columns
    cgm_df = cgm_df.rename({'Zeitstempel (JJJJ-MM-TTThh:mm:ss)': 'timestamp', 'Glukosewert (mmol/L)':'glucose'}, axis=1)
    
    # fix the timestamp by removing the letter T and replacing it with a space
    cgm_df['timestamp'] = cgm_df['timestamp'].str.replace('T' , ' ')
    
    # coerce it to have a datatype of datetime
    cgm_df['timestamp'] = pd.to_datetime(cgm_df['timestamp'])
    
    # reset seconds to be zeroes, to make the timestamp consistent with other data sources 
    cgm_df['timestamp'] = cgm_df['timestamp'].apply(lambda x: x.replace(second=0))
    
    # set the timestamp to be the index
    cgm_df.set_index('timestamp', inplace=True)
    
    return cgm_df
