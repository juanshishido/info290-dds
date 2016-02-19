import os

import bs4
import pandas as pd


def load_labels(fname, colnames, category=None):
    assert os.path.isfile(fname), 'File not found'
    df = pd.read_csv(fname, sep='\t', header=None)
    df.columns = colnames
    df['winner'] = df.winner.apply(lambda x:
                                   1 if x == 'winner'
                                   else 0)
    if isinstance(category, str):
        df['category'] = category
    return df

def load_features(fname, colnames):
    assert os.path.isfile(fname), 'File not found'
    df = pd.read_csv(fname, sep='\t', header=None)
    df.columns = colnames
    return df

def wiki_data(fname, wiki_id, table_id):
    assert os.path.isfile(fname), 'File not found'
    assert table_id in ['infobox vevent', 'infobox biography vcard']
    with open(fname, 'r') as wiki:
        film = bs4.BeautifulSoup(wiki, 'html.parser')
    infobox = film.findAll('table', class_=table_id)
    if len(infobox) > 0:
        x = infobox[0].find_all('th')
        y = infobox[0].find_all('td')
        if len(x) == len(y):
            film_info = [(z[0].text, z[1].text) for z in zip(x, y)]
            df = pd.DataFrame(film_info[1:]).T
            cols = ['_'.join(d.split()).lower() for d in df.iloc[0]]
            df.columns = cols
            df = df.reindex(df.index.drop(0))
            if table_id == 'infobox vevent':
                df['wiki_slug_film'] = wiki_id
            elif table_id == 'infobox biography vcard':
                df['wiki_slug_person'] = wiki_id
            return df
