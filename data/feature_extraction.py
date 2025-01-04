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


def cleaning(data):
    """
    Cleans the input text data by removing unwanted characters (punctuation, numbers, specific patterns),
    and filtering out stopwords. The text is tokenized into a list of words.

    Args:
        data (pandas.Series): A pandas Series where each element is a document (string) that needs to be cleaned.

    Returns:
        list of list of str: A list where each element is a list of cleaned tokens (words) from a document.
                              Stopwords and unwanted patterns (like punctuation and numbers) are removed.
    """
    #Deleting punctuations and \d, alias each number
    data_strip = data.apply(lambda x: re.sub(r"[',\[\]’\d]|(o+\s*h+|h+\s*o+)]", "", x).strip()) 

    #Transforming the data into a list to handle them better
    data = list(data_strip)

    # Deleting the tokens in the stopwords list
    data_cleaned = [
    [word for word in word_tokenize(document.lower()) if word not in stopwords]
    for document in data ]

    return data_cleaned



def formats(data):
    """
    Converts a given list of text data into a Bag-of-Words (BOW) representation using Gensim's Dictionary and corpus utilities.

    Args:
        data (list of list of str): A list where each element is a document represented as a list of words (tokens).

    Returns:
        tuple: A tuple containing:
            - dictionary (gensim.corpora.Dictionary): The dictionary mapping each unique word to a unique id.
            - corpus (list of list of tuples): A list of Bag-of-Words representations for each document, 
              where each document is a list of (word_id, frequency) pairs.
            - word_doc_matrix (numpy.ndarray): A dense matrix where each row corresponds to a document, 
              and each column corresponds to the frequency of a word from the dictionary in that document.

    Prints:
        - The Gensim dictionary object.
        - The Bag-of-Words (BOW) representation for each document.
    """
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
def tf_idf(corpus: list[str]):
    """
    Prints the 5 most important features in a document.

    Args:
        corpus (list[str]): The document to be inspected
    """

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
    lst = []

    for feature, weight in ordered_features.items():
        lst.append(f"{feature}: {weight}")

    #Printing the first five
    print(lst[:5])



#Function to select just the texts with a specified label
def select_emotion(dataframe: pd.Series, emotion) -> pd.DataFrame:
    """
    Selects only entries with label equal to the given emotion.

    Args:
        dataframe: The dataset.
        emotion: The emotion to filter the dataset with.

    Returns:
        pd.DataFrame: The filtered dataset.
    """
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

