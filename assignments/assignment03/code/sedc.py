import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def load_coefficients():
    df = pd.read_csv('../data/logreg_coefficients.txt', sep='\t', header=None,
                     quoting=3, names=['term', 'weight'])
    return df

def load_reviews():
    reviews = []
    with open('../data/test_movies.txt', 'r') as movie_reviews:
        for review in movie_reviews:
            reviews.append(review)
    return np.array(reviews)

def features(reviews):
    cv = CountVectorizer()
    X = cv.fit_transform(reviews)
    X = (X > 0) * 1
    return X, cv.get_feature_names()

def df_tdm_w(coefficients, X, feature_names):
    df = pd.DataFrame(X.todense(), columns=feature_names).T.reset_index()
    df.rename(columns={'index' : 'term'}, inplace=True)
    tdm = pd.merge(coefficients, df, on='term', how='outer')
    tdm.fillna(0, inplace=True)
    return tdm

def logistic(weights, values):
    assert isinstance(weights, np.ndarray) and isinstance(values, np.ndarray)
    alpha = weights.dot(values)
    return np.exp(alpha) / (1 + np.exp(alpha))
