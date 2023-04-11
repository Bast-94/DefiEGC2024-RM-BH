import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# Generate external time series.
def generate_external_time_series(file_path, window=1) -> pd.DataFrame:
    # Load Dataframe 
    df = pd.read_csv(file_path)
    
    # Unuseful columns
    df = df.drop(['Unnamed: 0'],axis=1)
    
    df['date'] = pd.to_datetime(df['date'])
    df['HashRate'] = df['HashRate'].rolling(window=window,center=True).mean()
    df['PriceUSD'] = df['PriceUSD'].rolling(window=window,center=True).mean()
    df = df.set_index('date')
    return df

def generate_global_time_series(file_path) -> pd.DataFrame:
    # Load Dataframe 
    df = pd.read_csv(file_path)
    df["date"] = df["year"].astype(str) +"-" + df["month"].astype(str)+ "-" + df["day"].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    
    # Drop 'Unnamed: 0', 'year', 'month', 'day'
    df = df.drop(['Unnamed: 0', 'year', 'month', 'day'], axis=1)
    
    # Put date at first column
    date_col = df.pop('date')
    df.insert(0,'date',date_col)
    df = df.set_index('date')
    return df

def generate_blockchain_by_actor(file_path):
    df = pd.read_csv(filepath_or_buffer=file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date') 
    # Unuseful columns
    df = df.drop(['Unnamed: 0','year','day','month'],axis=1)
    df = df.set_index('date')
    return df

def min_max_norm(df):
    min_max_scaler = MinMaxScaler()
    df_normalized = min_max_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np.array(df_normalized, dtype=np.float64), columns=df.columns)
    return df_normalized

def std_scale(df):
    std_scaler = StandardScaler()
    df_normalized = std_scaler.fit_transform(df)
    df_normalized = pd.DataFrame(np.array(df_normalized, dtype=np.float64), columns=df.columns)
    return df_normalized