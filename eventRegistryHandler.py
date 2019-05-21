from eventregistry import *
import pandas as pd
from pandas.io.json import json_normalize
import json

def get_news_iterator(er, keyword, language, loc, outlets_ids = None, n_items = -1): #n_items = -1 implies in retrieve all possible items

	politicsCat = er.getCategoryUri("politics")
	businessCat = er.getCategoryUri("business")

	if outlets_ids is not None:
		query = QueryArticlesIter(
			keywords = keyword,
            lang = language,
            keywordsLoc = loc,
            categoryUri = QueryItems.OR([politicsCat, businessCat]))
	else:
		query = QueryArticlesIter(
            keywords = keyword,
            lang = language,
            keywordsLoc = loc,
            sourceUri = outlets_ids,
            categoryUri = QueryItems.OR([politicsCat, businessCat]))

	articlesIter = query.execQuery(er, returnInfo = ReturnInfo(
        articleInfo=ArticleInfoFlags(duplicateList=True, concepts=True, categories=True, location=True),
        sourceInfo=SourceInfoFlags(location=True),
        locationInfo=LocationInfoFlags(wikiUri=True)
        ), maxItems = n_items)

	return articlesIter

def add_missing_columns(article_row, columns):
    if not set(columns).issubset(set(article_row.columns)): # if article_row doesn't contain all informations
        dif = set(columns) - set(article_row.columns)
        for newcol in dif:
            article_row[newcol]="-"
    return article_row

def format_row(article, columns):
    article_row = json_normalize(article)
    article_row = add_missing_columns(article_row, columns)
    row_list = article_row.loc[0,columns].tolist()
    return row_list

#this method is slower than retrieve_news_2, however it grants that
#aditional columns in different articles will be sucessifully added
def retrieve_news_1(news_iterator): 
    
    df = pd.DataFrame()
    for article in news_iterator:
        article_row = json_normalize(article)
        df = df.append(article_row, ignore_index=True)
    return df

#this method is faster than retrieve_news_1, however it limits
#the information retrieved to a predifined set of columns
def retrieve_news_2(news_iterator, columns):
    idx = -1
    for article in news_iterator:
        idx += 1
        if idx == 0: #create dataframe
            row_list = format_row(article, columns)
            df = pd.DataFrame(columns=columns)
            df.loc[idx] = row_list
            pass
    
        row_list = format_row(article, columns)
        df.loc[idx] = row_list
    return df
