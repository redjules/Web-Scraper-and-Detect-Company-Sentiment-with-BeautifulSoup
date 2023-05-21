#Import dependencies
import requests 
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from textblob import Word

#Define a get reviews
def get_reviews():
    links = [f'https://www.yelp.com/biz/mcdonalds-los-angeles-106?start={10+x*10}' for x in range(12)]
    links.insert(0,'https://www.yelp.com/biz/mcdonalds-los-angeles-106')
    regex = re.compile('raw__')


    reviews=[]
    for link in links:
        r=requests.get(link)
        soup=BeautifulSoup(r.text,'html.parser')
        results = soup.find_all('span',{'lang':'en'},class_=regex)
        reviews = [*reviews, *[result.text for result in results]]
    return reviews

#Preprocess collected reviews
def preprocess(reviews):
    df = pd.DataFrame(np.array(reviews), columns=['review'])
    stop_words = stopwords.words('english')
    
    # Lowercase
    df['review_lower'] = df['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # Strip punctuation
    df['review_nopunc'] = df['review_lower'].str.replace('[^\w\s]','')
    # Remove stopwords
    df['review_nostop'] = df['review_nopunc'].apply(lambda x:" ".join(x for x in x.split() if x not in stop_words))
    # Custom stopwords list
    other_stopwords = ['one','get','go','im','2','thru','tell','says','two']
    # Remove other stop words
    df['review_noother'] = df['review_nostop'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
    #Lemmatize
    df['cleaned_review'] = df['review_noother'].apply(lambda x: " ".join(Word(word).lemmatize() for word in x.split()))

    return df
    
    #Calculate sentiment
    def calculate_sentiment(df):
        df['polarity'] = df['cleaned_review'].apply(lambda x:TextBlob(x).sentiment[0])
        df['subjectivity'] = df['cleaned_review'].apply(lambda x:TextBlob(x).sentiment[1])
        # return the final dataframe
        return df

    if __name__ == "__main_":
        reviews = get_reviews()
        df = preprocess(reviews)
        sentiment_df = calculate_sentiment(df)
        sentiment_df.to_csv('results.csv')
    


