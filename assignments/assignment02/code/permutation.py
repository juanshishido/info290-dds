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
