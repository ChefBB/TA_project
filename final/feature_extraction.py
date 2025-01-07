
'''
This script further cleans the already lemmatized and labelled dataframe with a specified stopwords list. Then finds the most important
feature for each emotion. 

'''

import pandas as pd
import numpy as np
import string
import re
import nltk

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')


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


### Remove further noise from the texts: empty strings, numbers, apostrophes, quotation marks, expressions like "can't"

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

# Function to read and clean the dataframe
def preprocess_and_clean_data(file_path):
    """
    Reads and cleans the dataframe selected through file_path.

    Args:
        file_path (str): path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with cleaned texts.
    """
    df = pd.read_csv(file_path)
    df["canzoni_cleaned"] = cleaning(df["lemmatized_stanzas"])
    return df


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

# Run the script
if __name__ == "__main__":
    # Percorso al dataset
    file_path = ".\\models\\data\\resampled_data_for_tf_idf.csv"
    
    # Preprocessing e pulizia
    df = preprocess_and_clean_data(file_path)

    # Calcolo TF-IDF per ogni emozione
    emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
    for emotion in emotions:
        print(f"TF-IDF per {emotion}")
        canzoni = select_emotion(df, emotion)
        tf_idf(canzoni)

