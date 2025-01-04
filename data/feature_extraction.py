### FEATURE EXTRACTION: TF-IDF TO IDENTIFY THE MOST IMPORTANT FEATURES FOR EACH LABEL 

import pandas as pd
import numpy as np
import gensim
import string
import re
import nltk

from nltk.tokenize import word_tokenize
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')


df = pd.read_csv(".\\data\\resampled_data_for_tf_idf.csv", skipinitialspace=True)


# Defining the stopwords
stopwords = nltk.corpus.stopwords.words('english') # Set di NLTK

# Creation of an ad-hoc list of stopwords, inserting also the numbers spelled out
stopwords_adhoc = ['?','(', ')', '.', '[', ']','!', '...',
';',"\`","\'",'\"' , " ", "``", "\"\"", "\'\'", "``", "\'\'" "\'s", "\'ll", "ca", "n\'t", "\'m", "\'re", "\'ve", "na", "wan", "one", "two", "three", "four", "five", 
"six","seven", "eight", "nine", "ten", "zero", "cos", "er", "mow", "go", "get", "oh", "love", "know", "like", "see", "make", "come", "let", "say", "take",
"want", "would"]

# Generic punctuation
punctuation = set(string.punctuation)

# Expanding the original stopwords list
stopwords.extend(stopwords_adhoc)
stopwords.extend(punctuation)


### Remove further noise from the texts: empty strings, numbers, apostrophes, quotation marks, expressions like "can't


#Defining a function for cleaning the data
def cleaning(data):
    #Deleting punctuations and \d, alias each number
    data_strip = data.apply(lambda x: re.sub(r"[',\[\]’\d]|(o+\s*h+|h+\s*o+)]", "", x).strip()) 

    #Transforming the data into a list to handle them better
    data = list(data_strip)

    # Deleting the tokens in the stopwords list
    data_cleaned = [
    [word for word in word_tokenize(document.lower()) if word not in stopwords]
    for document in data ]

    return data_cleaned


#Function to create the correct formats for gensim ->  scikit-learn as suggested, so to use min_df e max_df 
def formats(data):
    #Creating a dictionary on which the model can be based
    dictionary = corpora.Dictionary(data)
    print(dictionary)

    #Taking the id of each word
    dictionary.token2id

    #Transforming the corpus
    corpus = [dictionary.doc2bow(text) for text in data]

    #Visualizing the BOW
    for i, doc in enumerate(corpus):
        print("document:\t", data[i])
        print("Bag-of-words:\t", [(dictionary[_id], freq) for _id, freq in doc])
        print()

    #Space vector
    word_doc_matrix = gensim.matutils.corpus2dense(corpus, num_terms = len(dictionary))
    word_doc_matrix.shape
    return dictionary, corpus, word_doc_matrix



#Function to compute tf-idf
def tf_idf(corpus):

    #Initializing the vectorizer
    vectorizer = TfidfVectorizer(min_df=0.02, max_df=0.8)

    #Fitting
    corpus_tfidf = vectorizer.fit_transform(corpus)
    
    #Creating an array of feature names
    feature_names = np.array(vectorizer.get_feature_names_out())

    #Finding the averages of tfidf in the documents
    tfidf_means = corpus_tfidf.mean(axis=0).A1

    #Coupling weight and feature
    feature_weights = dict(zip(feature_names, tfidf_means))

    #Putting in order the features
    ordered_features = dict(sorted(feature_weights.items(), key=lambda x: x[1], reverse=True))

    #List of features 
    list = []

    for feature, weight in ordered_features.items():
        list.append(f"{feature}: {weight}")

    #Printing the first five
    print(list[:5])

    return



#Function to select just the texts with a specified label
def select_emotion(dataframe, emotion):
    canzoni_emotion = dataframe[dataframe["label"] == emotion]["canzoni_cleaned"].apply(lambda x: " ".join(x))

    return canzoni_emotion


# Cleaning of the songs
df["canzoni_cleaned"] = cleaning(df["lemmatized_stanzas"])


#TfIdf for Anger
canzoni_anger = select_emotion(df, emotion = "anger")
tf_idf(canzoni_anger)


#Tf-Idf for Anticipation
canzoni_anticipation = select_emotion(df, emotion = "anticipation")
tf_idf(canzoni_anticipation)


#TfIdf for Disgust
canzoni_disgust = select_emotion(df, emotion = "disgust")
tf_idf(canzoni_disgust)


#Tf_Idf for Fear
canzoni_fear = select_emotion(df, emotion = "fear")
tf_idf(canzoni_fear)


#Tf-Idf for Joy
canzoni_joy = select_emotion(df, emotion = "joy")
tf_idf(canzoni_joy)


#Tf-Idf for Sadness
canzoni_sadness = select_emotion(df, emotion = "sadness")
tf_idf(canzoni_sadness)


#Tf_Idf for Surprise
canzoni_surprise = select_emotion(df, emotion = "surprise")
tf_idf(canzoni_surprise)


#Tf-Idf for Trust
canzoni_trust = select_emotion(df, emotion = "surprise")
tf_idf(canzoni_trust)

