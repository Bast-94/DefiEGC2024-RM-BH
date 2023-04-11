import pandas as pd
import numpy as np

def compute_best_correlation_by_col(block_chain_df: pd.DataFrame,column_to_analyze: str ) -> tuple:
    block_chain_df.groupby('identity')[[column_to_analyze]].sum().sort_values(by=column_to_analyze, ascending=False).head()
    short_blochain_df = block_chain_df.copy()
    df_unstacked = short_blochain_df.groupby(['date', 'identity'])[column_to_analyze].mean().unstack()
    actor_correlation_matrix = df_unstacked.corr()
    np.fill_diagonal(actor_correlation_matrix.values,0)
    s = actor_correlation_matrix.stack()
    max_index = s.idxmax()[0]
    max_col = actor_correlation_matrix.loc[max_index].idxmax()
    max_value = actor_correlation_matrix[max_col].loc[max_index]
    return max_index,max_col, max_value

def compute_best_correlation(block_chain_df: pd.DataFrame):
    best_index, best_col,best_val = None,None, 0
    for column_to_analyze in block_chain_df.columns:
        cur_index,cur_col,cur_val = compute_best_correlation_by_col(block_chain_df,column_to_analyze)
        if(cur_val>best_val):
            best_index, best_col,best_val =cur_index,cur_col,cur_val
    
    return best_index, best_col,best_val

