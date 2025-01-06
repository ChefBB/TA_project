"""
preprocessing.py

This module contains functions and utilities for preprocessing data used in neural network models. 
The preprocessing steps include tokenization, padding, feature extraction, and scaling, ensuring 
the input data is in a suitable format for model training and evaluation.
"""

import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np
import joblib


vocab_size = 20000
max_seq_length = 150

num_topics = 8

emotion_mapping = {
    "joy": 0,
    "trust": 1,
    "fear": 2,
    "surprise": 3,
    "sadness": 4,
    "disgust": 5,
    "anger": 6,
    "anticipation": 7
}


def preprocess (data: dict, folder_path: str) -> dict:
    """
    Preprocesses the input data for training neural network models.

    This function handles preprocessing tasks for training, validation
    and test sets.
    It ensures that the data is properly formatted and 
    stored for model training and evaluation.

    Args:
        data (dict): A dictionary containing the raw input data.
        
        folder_path (str): Path to the folder where preprocessing tools are saved.

    Returns:
        dict: A dictionary containing the preprocessed data, including 
              tokenized and padded sequences, scaled features, and any 
              additional information needed for model training.
    """
    
    # title unsupervised learning
    preprocessed_data = {
        'train': {
            'booleans': data['train']['booleans'],
            'y': data['train']['y']
        },
        'val': {
            'booleans': data['val']['booleans'],
            'y': data['val']['y']
        },
        'test': {
            'booleans': data['test']['booleans'],
            'y': data['test']['y']
        }
    }
    
    # convert song titles to TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(data['train']['titles'])
    
    joblib.dump(tfidf_vectorizer, folder_path + '/tfidf_vectorizer.joblib')

    # NMF for topic modeling
    nmf_model = NMF(n_components=num_topics)
    preprocessed_data['train']['topic_distributions'] = nmf_model.fit_transform(tfidf_matrix_train)
    
    joblib.dump(nmf_model, folder_path + '/nmf_model.joblib')
    
    # apply to test, val sets
    preprocessed_data['val']['topic_distributions'] = nmf_model.transform(
        tfidf_vectorizer.transform(data['val']['titles'])
    )
    
    preprocessed_data['test']['topic_distributions'] = nmf_model.transform(
        tfidf_vectorizer.transform(data['test']['titles'])
    )
    
    # initialize the tokenizer
    tokenizer = Tokenizer(num_words= vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(data['train']['lemmatized_stanzas'])

    preprocessed_data['train']['sequences'] = tokenizer.texts_to_sequences(data['train']['lemmatized_stanzas'])
    
    preprocessed_data['val']['sequences'] = tokenizer.texts_to_sequences(data['val']['lemmatized_stanzas'])

    preprocessed_data['test']['sequences'] = tokenizer.texts_to_sequences(data['test']['lemmatized_stanzas'])

    
    preprocessed_data['train']['padded_sequences'] = pad_sequences(
        preprocessed_data['train']['sequences'], maxlen=max_seq_length,
        padding='post', truncating='post'
    )

    preprocessed_data['val']['padded_sequences'] = pad_sequences(
        preprocessed_data['val']['sequences'], maxlen=max_seq_length,
        padding='post', truncating='post'
    )

    preprocessed_data['test']['padded_sequences'] = pad_sequences(
        preprocessed_data['test']['sequences'], maxlen=max_seq_length,
        padding='post', truncating='post'
    )
    
    joblib.dump(tokenizer, folder_path + '/basic_tokenizer.joblib')
    
    
    # preprocess non-text data
    scaler = StandardScaler()
    preprocessed_data['train']['stanza_numbers'] = scaler.fit_transform(data['train']['stanza_numbers'])
    
    joblib.dump(scaler, folder_path + '/scaler.joblib')
    
    # apply on val, test sets
    preprocessed_data['val']['stanza_numbers'] = scaler.transform(data['val']['stanza_numbers'])

    
    preprocessed_data['test']['stanza_numbers'] = scaler.transform(data['test']['stanza_numbers'])
    
    
    return preprocessed_data
    

def preprocess_labels (labels: pd.Series | list) -> np.array:
    """
    Converts emotion labels into a categorical format suitable for training neural network models.

    Args:
        labels (pd.Series | list): A pandas Series or list containing emotion labels as strings.

    Returns:
        np.array: A NumPy array representing the one-hot encoded labels, with shape 
                  (num_samples, num_classes), where `num_classes` is set to 8.
    """
    labels_indices = np.array([emotion_mapping[label] for label in labels])

    return to_categorical(labels_indices, num_classes=8)


def preprocess_new_data(data: dict, path: str) -> dict:
    """
    Preprocesses new input data using pre-trained models and saved preprocessing objects.

    Args:
        data (dict): A dictionary containing the input data.

    Returns:
        dict: A dictionary containing the preprocessed data.
    """

    processed_data = {'booleans': data['booleans']}
    
    with open(path + '/tfidf_vectorizer.joblib', 'rb') as f:
        tfidf_vectorizer = joblib.load(f)
        
    with open(path + '/nmf_model.joblib', 'rb') as f:
        nmf_model = joblib.load(f)
    
    processed_data['topic_distributions'] = nmf_model.transform(
        tfidf_vectorizer.transform(data['titles'])
    )
    
    
    with open(path + '/basic_tokenizer.joblib', 'rb') as f:
        basic_tokenizer = joblib.load(f)
        
    processed_data['sequences'] = basic_tokenizer.texts_to_sequences(data['lemmatized_stanzas'])
    
    processed_data['padded_sequences'] = pad_sequences(
        processed_data['sequences'], maxlen=max_seq_length,
        padding='post', truncating='post'
    )
    
    
    with open(path + '/scaler.joblib', 'rb') as f:
        scaler = joblib.load(f)
    
    processed_data['stanza_numbers'] = scaler.transform(data['stanza_numbers'])
    
    
    return processed_data