import numpy as np
import pandas as pd


def load():
    features = pd.read_csv('../data/movie.features.txt', sep='\t', header=None, 
                           names=['feature', 'movie_id'], usecols=[0,2])
    box_office = pd.read_csv('../data/movie.box_office.txt',
                             sep='\t', header=None, names=['movie_id', 'hit'])
    return features, box_office

def merge(f, b):
    f = f[f.feature == 'John_Goodman'][['movie_id']]
    f.drop_duplicates(inplace=True)
    f['John_Goodman'] = 1
    movie = pd.merge(f, b, on='movie_id', how='outer')
    movie['John_Goodman'].fillna(0, inplace=True)
    return movie

def _to_arr(df, target):
    df = df.copy()
    df.sort_values(by=target, ascending=False, inplace=True)
    arr = df.hit.values
    n = df[target].sum()
    return arr, n

def _tstat(arr, n):
    return abs(np.mean(arr[:n]) - np.mean(arr[n:]))

def permute(df, target, permutations):
    np.random.seed(42)
    arr, n = _to_arr(df, target)
    baseline = _tstat(arr, n)
    v = []
    for _ in range(permutations):
        np.random.shuffle(arr)
        v.append(_tstat(arr, n))
    return (baseline <= np.array(v)).sum() / permutations
