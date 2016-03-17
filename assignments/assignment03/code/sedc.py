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

def build_cases(df):
    df = df.copy()
    cases = {}
    for c in range(10):
        yhat = logistic(df.weight.values, df[c].values)
        cases[c] = {'yhat' : yhat, 'class' : yhat >= 0.5}
    return cases

def explanations(df, cases):
    df = df.copy()
    for c in range(10):
        positive_class = cases[c]['class']
        df.sort_values('weight', inplace=True, ascending=(not positive_class))
        df.reset_index(drop=True, inplace=True)
        E = []
        arr = df[c].values
        indices = np.where(arr==1)[0]
        for ind in indices:
            arr[ind] = 0
            E.append(ind)
            if positive_class and logistic(df.weight.values, arr) < 0.5:
                break
            elif not positive_class and logistic(df.weight.values, arr) >= 0.5:
                break
        terms = df.ix[E]['term'].tolist()
        cases[c]['explanations'] = terms
    return None


if __name__ == '__main__':
    coefficients = load_coefficients()
    X, feature_names = features(load_reviews())
    df = df_tdm_w(coefficients, X, feature_names)
    cases = build_cases(df)
    explanations(df, cases)
    for c in range(10):
        print('Case:', c, '| y_hat:', cases[c]['yhat'])
        print('Removed Terms:', ' '.join(cases[c]['explanations']), end='\n\n')
