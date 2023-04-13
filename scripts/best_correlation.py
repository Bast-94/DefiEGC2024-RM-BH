import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler

def get_best_values(actor_correlation_matrix: pd.DataFrame) -> tuple:
    """"
    Renvoie les indexes et la meilleure valeur de la matrice de corrélation
    """
    s = actor_correlation_matrix.stack()
    max_index = s.idxmax()[0]
    max_col = actor_correlation_matrix.loc[max_index].idxmax()
    max_value = actor_correlation_matrix[max_col].loc[max_index]
    return max_index,max_col, max_value

def get_corr_mat(block_chain_df: pd.DataFrame,column_to_analyze: str) -> pd.DataFrame: 
    """
    Renvoie la matrice de corrélation selon l'évolution d'une colonne.
    """
    df_unstacked = block_chain_df.groupby(['date', 'identity'])[column_to_analyze].mean().unstack()
    actor_correlation_matrix = df_unstacked.corr()
    np.fill_diagonal(actor_correlation_matrix.values,0) 
    return actor_correlation_matrix

def compute_best_correlation_by_col(block_chain_df: pd.DataFrame,column_to_analyze: str ) -> tuple:
    return get_best_values(get_corr_mat(block_chain_df,column_to_analyze))



def compute_best_correlation(block_chain_df: pd.DataFrame) -> tuple:
    """
    Renvoie le couple d'acteur, le meilleur coefficient de corrélation et 
    la colonne qui donne le meilleur coefficient de corrélation
    """
    actor1, actor2,best_val = None,None, 0
    best_col_to_analyse = None
    for column_to_analyze in block_chain_df.columns[1:]:
        cur_index,cur_col,cur_val = compute_best_correlation_by_col(block_chain_df,column_to_analyze)
        if(cur_val>best_val):
            actor1, actor2,best_val =cur_index,cur_col,cur_val
            best_col_to_analyse = column_to_analyze
    
    return actor1, actor2,best_val,best_col_to_analyse


def display_correlation_by_col(block_chain_df: pd.DataFrame,column_to_analyze: str =None, window_size: int = 14) -> None :
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
    


def get_correlation_pairs(df: pd.DataFrame, threshold : float) -> list:
    """
    Renvoie la liste des couples d'acteurs qui ont un coefficient supérieur à threshold.
    """
    corr_matrix = df.values
    np.fill_diagonal(corr_matrix, np.nan)
    idx = np.where(np.abs(corr_matrix) >= threshold)
    pairs = [(df.columns[i], df.columns[j], corr_matrix[i,j]) for i,j in zip(*idx)]
    return pairs

def get_unique_correlation_pairs(df: pd.DataFrame, threshold: float,col :str=None) -> list:
    """
    Renvoie une liste de listes qui contiennent:
        - 2 acteurs
        - un coefficient de corrélation dont la valeur absolue est supérieure à threshold
        - la colonne concernée par la corrélation
    """
    corr_pairs = get_correlation_pairs(df, threshold)
    pairs_list = []
    for pair in corr_pairs:
        if pair[0] < pair[1]:
            key = [pair[0], pair[1],df.loc[pair[0], pair[1]],col]
            pairs_list.append(key)
    return pairs_list

def best_corr_list(block_chain_by_actor_df: pd.DataFrame,thresh_old: float = 0.9) -> list:
    """
    Renvoie les meilleures corrélations en indiquant les acteurs et les colonnes concernées.
    """
    best_corr =[]
    for col in block_chain_by_actor_df.columns[1:]:
        corr_list_by_col = get_unique_correlation_pairs(get_corr_mat(block_chain_by_actor_df,col),thresh_old,col)
        if(corr_list_by_col != []):
            best_corr += (corr_list_by_col)
    return best_corr

def best_correlation_df(block_chain_by_actor_df: pd.DataFrame,thresh_old :float = 0) -> pd.DataFrame:
    """
    Renvoie la dataframe qui contient par défaut toute les corrélations entre tous les couples d'acteurs.
    """
    best_corr = best_corr_list(block_chain_by_actor_df,thresh_old)
    best_corr_df = pd.DataFrame.from_dict(best_corr)
    best_corr_df.columns = ['actor1','actor2','correlation_rate','related_col']
    return best_corr_df

def display_comparison(block_chain_by_actor_df: pd.DataFrame,actor1 : str,actor2 :str,column_to_analyze :str,window : int) -> None:
    """
    Affiche les évolutions d'une colonne `column_to_analyze` des acteurs `actor1` et `actor2` avec un lissage `window`.
    """
    std_scaler = StandardScaler()
    
    df1 = block_chain_by_actor_df[block_chain_by_actor_df['identity'] == actor1].copy()
    df1[column_to_analyze] = df1[column_to_analyze].rolling(window=window).mean()
    df1_normalized = std_scaler.fit_transform(df1[[column_to_analyze]])
    df1_normalized = pd.DataFrame(np.array(df1_normalized, dtype=np.float64), columns=[column_to_analyze],index=df1.index)
    
    df2 = block_chain_by_actor_df[block_chain_by_actor_df['identity'] == actor2].copy()
    df2[column_to_analyze] = df1[column_to_analyze].rolling(window=window).mean()
    df2_normalized = std_scaler.fit_transform(df2[[column_to_analyze]])
    df2_normalized = pd.DataFrame(np.array(df2_normalized, dtype=np.float64), columns=[column_to_analyze],index=df2.index)

    df = pd.concat([df1, df2])
    
    fig = px.line(df, x=df.index, y=column_to_analyze, color="identity",
                title=f"Comparison of '{column_to_analyze}' between {actor1} and {actor2} (Normalized)",
                labels={"value": f"{column_to_analyze} (rolling mean)", "index": "date"})
   
    fig.show()