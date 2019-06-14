import re
import unicodedata
from difflib import get_close_matches
import string
import pandas as pd

def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError): # unicode is a default on python 3 
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_id(text):
    """
    Convert input text to id.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.
    """
    text = strip_accents(text.lower())
    text = re.sub('[ ]+', '_', text)
    text = re.sub('[^0-9a-zA-Z_-]', '', text)
    return text

def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return text_to_id(s.strip())

def return_first_element(list_element):
    if list_element == []:
        return "not match"
    return list_element[0]

# add country location in the dataset
def add_country_location(dataset):
    dataset["country_location"] = dataset["source.location.country.label.eng"]
    for index, row in dataset.iterrows():
        if row["country_location"] == "-":
            dataset.loc[index, "country_location"] = row["source.location.label.eng"]
    return dataset

# add bias column to news data based on values of bias file.
# match_similarity is the minimum similarity between strings
# in order to match news outlets' names in news and in bias file.
def add_bias_information(news, bias, match_similarity=0.9):
    bias["source_id_bias"] = bias.outlet.apply(remove_punctuation)
    news['source_id_news'] = news['source.title'].apply(remove_punctuation)
    closest_to_source = bias.source_id_bias.apply(get_close_matches, args=(news.source_id_news.unique(), 1, match_similarity))
    matchs = closest_to_source.apply(return_first_element)
    base = pd.concat([bias.source_id_bias, matchs.rename("source_id_news")], axis=1)
    base = pd.merge(bias, base, on="source_id_bias", how='inner')
    base = pd.merge(news, base, on="source_id_news", how='left')
    base = base.drop(['source_id_news','outlet','source_id_bias'], axis=1)
    return base