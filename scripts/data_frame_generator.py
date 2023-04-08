import pandas as pd
# Generate external time series.
def generate_external_time_series(file_path, window=1) -> pd.DataFrame:
    # Load Dataframe 
    df = pd.read_csv(file_path)
    
    # Unuseful columns
    df = df.drop(['Unnamed: 0'],axis=1)
    
    df['date'] = pd.to_datetime(df['date'])
    df['HashRate'] = df['HashRate'].rolling(window=window,center=True).mean()
    df['PriceUSD'] = df['PriceUSD'].rolling(window=window,center=True).mean()
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
    return df

def generate_blockchain_by_actor(file_path):
    df = pd.read_csv(filepath_or_buffer=file_path)
    df['date'] = pd.to_datetime(df['date']) 
    # Unuseful columns
    df = df.drop(['Unnamed: 0','year','day','month'],axis=1)
    
    return df