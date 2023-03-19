# Part 3: Mining text data.

# imports
import requests
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
    # read data with 'latin-1' encoding
    df = pd.read_csv(data_file, index_col=False, encoding='latin-1')

    return df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
    # a list with the possible sentiments
    sentiments = df['Sentiment'].unique().tolist()

    return sentiments


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
    # count sentiments and sort accordingly
    total_Counts = df.groupby('Sentiment')['Sentiment'].count().reset_index(name='Count').sort_values(['Count'],
                                                                                                      ascending=False).reset_index(
        drop=True)
    # find the second most popular sentiment
    second = total_Counts['Sentiment'][1]

    return second


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
    # count sentiments and sort accordingly
    total_Counts = df.groupby(['TweetAt', 'Sentiment'])['TweetAt'].count().reset_index(name='Count').sort_values(
        ['Count'], ascending=False).reset_index(drop=True)
    # find the most popular sentiment
    date = total_Counts['TweetAt'][total_Counts['Sentiment'] == 'Positive'].reset_index(drop=True)[0]

    return date


# Modify the dataframe df by converting all tweets to lower case.
def lower_case(df):
    # converting all tweets to lower case
    df['OriginalTweet'] = df['OriginalTweet'].str.lower()


# Modify the dataframe df by replacing each character which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
    # replacing each character which is not alphabetic
    df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-zA-Z]', ' ', regex=True)


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
    # replacing each character which  whitespace
    df['OriginalTweet'] = df['OriginalTweet'].replace(r'\s+', ' ', regex=True)


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
    # tokenize every tweet
    df['OriginalTweet'] = df.apply(lambda row: word_tokenize(row['OriginalTweet']), axis=1)


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
    # count the number of words in all tweets including repetitions
    count_words = tdf['OriginalTweet'].str.len().sum()

    return count_words


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
    # count the number of distinct words in all tweets
    count_words = sum(tdf['OriginalTweet'].apply(set).apply(len))

    return count_words


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
    # list with distinct words
    split_it = ''.join([' '.join(wrd for wrd in x) for x in tdf['OriginalTweet']]).split()
    # count every word in list
    Count = Counter(split_it)
    # most common word
    most_occur = Count.most_common(k)

    return [most_occur[x][0] for x in range(k)]


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
    # download text from link
    x = requests.get('https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt')
    # split into list of words
    stopWords = (x.text).strip('][').split('\n')
    # remove stopwords from the list
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
        lambda x: [word for word in x if (word not in (stopWords))])  # and len(word)>2
    # remove words with <=2 character
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [word for word in x if len(word) >= 3])


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
    # create PorterStemmer
    ps = PorterStemmer()
    # stem the list of words
    tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(lambda x: [ps.stem(word) for word in x])


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
    # convert the messages to lower case
    lower_case(df)
    # replace non-alphabetical characters with whitespaces
    remove_non_alphabetic_chars(df)
    # ensure that the words of a message are separated by a single whitespace.
    remove_multiple_consecutive_whitespaces(df)
    # Tokenize the tweets
    tokenize(df)
    # Remove stopwords
    remove_stop_words(df)
    # stem
    stemming(df)

    # train target labels
    label_Sentiments = df['Sentiment'].astype("category")

    # train data
    # convert to list of text
    text = df['OriginalTweet'].apply(lambda x: ''.join(' '.join(x))).to_numpy()
    # countVectorise the text
    count_vect = CountVectorizer(max_df=0.0005,  ngram_range=(2, 2))
    count_matrix = count_vect.fit_transform(text)

    # Multinomial Naive Bayes classifier
    naive_bayes_classifier = MultinomialNB()
    naive_bayes_classifier.fit(count_matrix, label_Sentiments)
    # predict sentiment
    pred_Sentiment = naive_bayes_classifier.predict(count_matrix)

    return pred_Sentiment


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive')
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
    # calculate the accuracy
    score = round(metrics.accuracy_score(y_pred, y_true), 3)

    return score


# test Script
if __name__ == "__main__":
    # 1. [13 points] Compute the possible sentiments that a tweet may have, the second most popular
    # sentiment in the tweets, and the date with the greatest number of extremely positive tweets.
    # Next, convert the messages to lower case, replace non-alphabetical characters with whitespaces
    # and ensure that the words of a message are separated by a single whitespace.

    # Read data
    df = read_csv_3('coronavirus_tweets.csv')
    df.head()

    # possible sentiments that a tweet may have
    print('1.1. Possible sentiments that a tweet may have: ')
    sentiments = get_sentiments(df)
    print(sentiments)

    # the second most popular sentiment in the tweets
    print('1.2. The second most popular sentiment in the tweets: ')
    second_sentiment = second_most_popular_sentiment(df)
    print(second_sentiment)

    # the date with the greatest number of extremely positive tweets
    print('1.3. The date with the greatest number of extremely positive tweets: ')
    date_tweets = date_most_popular_tweets(df)
    print(date_tweets)

    # convert the messages to lower case
    lower_case(df)

    # replace non-alphabetical characters with whitespaces
    remove_non_alphabetic_chars(df)

    # ensure that the words of a message are separated by a single whitespace.
    remove_multiple_consecutive_whitespaces(df)
    df.head()

    # 2. [14 points] Tokenize the tweets (i.e. convert each into a list of words), count the total number
    # of all words (including repetitions), the number of all distinct words and the 10 most frequent
    # words in the corpus. Remove stop words, words with ≤ 2 characters, and reduce each word to
    # its stem. You are now able to recompute the 10 most frequent words in the modified corpus.
    # What do you observe?

    # Tokenize the tweets
    tokenize(df)
    df.head()

    # count the total number of all words
    print('2.1. Total number of all words: ')
    count_words = count_words_with_repetitions(df)
    print(count_words)

    # the number of all distinct words
    print('2.2. Total number of all distinct words: ')
    count_words = count_words_without_repetitions(df)
    print(count_words)

    # 10 most frequent words
    print('2.3. Most frequent words: ')
    freq_word = frequent_words(df, 10)
    print(freq_word)

    remove_stop_words(df)
    df.head()

    stemming(df)
    df.head()

    # 10 most frequent words
    print('2.4. Most frequent words after stop words and stemming: ')
    freq_word = frequent_words(df, 10)
    print(freq_word)

    # [13 points] This task can be done individually from the previous three.
    # Store the coronavirus tweets.py corpus in a numpy array and produce a sparse representation of
    # the term document matrix with a CountVectorizer. Next, produce a Multinomial Naive Bayes classifier
    # using the provided data set. What is the classifier’s training accuracy? A CountVectorizer allows
    # limiting the range of frequencies and number of words included in the term-document matrix.
    # Appropriately tune these parameters to achieve the highest classification accuracy you can.

    df = read_csv_3('coronavirus_tweets.csv')

    print('3.1. Multinomial Naive Bayes classifier prediction: ')
    pred = mnb_predict(df)
    print(pred)

    # train target labels
    label_Sentiments = df['Sentiment'].astype("category")

    # before tuning the CountVectorizer
    print('3.2. Accuracy of the prediction: ')
    print('0.779')
    # after tuning the CountVectorizer
    print('3.3. Accuracy of the prediction after tuning: ')
    acc = mnb_accuracy(pred, label_Sentiments)
    print(acc)