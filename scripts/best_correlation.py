import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def get_best_values(actor_correlation_matrix: pd.DataFrame) -> tuple:
    s = actor_correlation_matrix.stack()
    max_index = s.idxmax()[0]
    max_col = actor_correlation_matrix.loc[max_index].idxmax()
    max_value = actor_correlation_matrix[max_col].loc[max_index]
    return max_index,max_col, max_value

def compute_best_correlation_by_col(block_chain_df: pd.DataFrame,column_to_analyze: str ) -> tuple:
    df_unstacked = block_chain_df.groupby(['date', 'identity'])[column_to_analyze].mean().unstack()
    actor_correlation_matrix = df_unstacked.corr()
    np.fill_diagonal(actor_correlation_matrix.values,0)
    return get_best_values(actor_correlation_matrix)

def compute_best_correlation(block_chain_df: pd.DataFrame):
    best_index, best_col,best_val = None,None, 0
    best_col_to_analyse = None
    for column_to_analyze in block_chain_df.columns[1:]:
        cur_index,cur_col,cur_val = compute_best_correlation_by_col(block_chain_df,column_to_analyze)
        if(cur_val>best_val):
            best_index, best_col,best_val =cur_index,cur_col,cur_val
            best_col_to_analyse = column_to_analyze
    
    return best_index, best_col,best_val,best_col_to_analyse


def display_correlation_by_col(block_chain_df: pd.DataFrame,column_to_analyze: str =None, window_size: int = 14):
    if(column_to_analyze is None):
        
        max_index,max_col, _ ,column_to_analyze =compute_best_correlation(block_chain_df)
    else:
        max_index,max_col, _ = compute_best_correlation_by_col(block_chain_df,column_to_analyze)
    f1, ax1 = plt.subplots(figsize=(16,9))
    ax2 = ax1.twinx()
    temp_df = block_chain_df[block_chain_df['identity'] == max_col][column_to_analyze]
    ax1.plot(temp_df.rolling(window=window_size).mean(),color='orange')
    temp_df = block_chain_df[block_chain_df['identity'] == max_index][column_to_analyze]
    ax2.plot(temp_df.rolling(window=window_size).mean())
    plt.title(f"Comparison of '{column_to_analyze}' between {max_col} and {max_index}")