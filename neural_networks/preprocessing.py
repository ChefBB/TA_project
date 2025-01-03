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
    #########################
    # title unsupervised learning
    #########################
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
    preprocessed_data['val']['topic_distributions'] = topic_analysis(
        tfidf_vectorizer= tfidf_vectorizer,
        nmf_model= nmf_model,
        titles= data['val']['titles']
    )
    
    preprocessed_data['test']['topic_distributions'] = topic_analysis(
        tfidf_vectorizer= tfidf_vectorizer,
        nmf_model= nmf_model,
        titles= data['test']['titles']
    )
    
    #########################
    # Initialize the tokenizer
    #########################
    tokenizer = Tokenizer(num_words= vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(data['train']['lemmatized_stanzas'])

    preprocessed_data['train']['sequences'] = tokenize(tokenizer, data['train']['lemmatized_stanzas'])
    
    preprocessed_data['val']['sequences'] = tokenize(tokenizer, data['val']['lemmatized_stanzas'])

    preprocessed_data['test']['sequences'] = tokenize(tokenizer, data['test']['lemmatized_stanzas'])

    
    preprocessed_data['train']['padded_sequences'] = padding(preprocessed_data['train']['sequences'])

    preprocessed_data['val']['padded_sequences'] = padding(preprocessed_data['val']['sequences'])

    preprocessed_data['test']['padded_sequences'] = padding(preprocessed_data['test']['sequences'])
    
    joblib.dump(tokenizer, folder_path + '/basic_tokenizer.joblib')
    
    
    #########################
    # preprocess non-text data
    #########################
    scaler = StandardScaler()
    preprocessed_data['train']['stanza_numbers'] = scaler.fit_transform(data['train']['stanza_numbers'])
    
    joblib.dump(scaler, folder_path + '/scaler.joblib')
    
    # apply on val, test sets
    preprocessed_data['val']['stanza_numbers'] = scale_stanza_numbers(
        scaler, data['val']['stanza_numbers'])

    
    preprocessed_data['test']['stanza_numbers'] = scale_stanza_numbers(
        scaler, data['test']['stanza_numbers'])
    
    
    return preprocessed_data
    

    
#########################
# obtain labels
#########################
def preprocess_labels (labels: pd.Series | list) -> np.array:
    labels_indices = np.array([emotion_mapping[label] for label in labels])

    return to_categorical(labels_indices, num_classes=8)


#########################
# tokenization
#########################
def tokenize (tokenizer: Tokenizer, lyrics: pd.Series | list) -> list:
    return tokenizer.texts_to_sequences(lyrics)


#########################
# add padding
#########################
def padding (sequences):
    return pad_sequences(
        sequences, maxlen=max_seq_length,
        padding='post', truncating='post'
    )
    
    
#########################
# convert booleans
#########################
def booleans_conv (df: pd.DataFrame) -> np.ndarray:
    return df[['is_country', 'is_pop', 'is_rap',
               'is_rb', 'is_rock', 'is_chorus']].astype(int).values


#########################
# title transformation
#########################
def topic_analysis (tfidf_vectorizer: TfidfVectorizer,
                    nmf_model: NMF,
                    titles: pd.Series | list) -> np.ndarray:
    
    tfidf_matrix = tfidf_vectorizer.transform(titles)
    return nmf_model.transform(tfidf_matrix)


#########################
# title transformation
#########################
def scale_stanza_numbers (scaler: StandardScaler,
                         stanza_numbers: pd.Series | list) -> np.ndarray:
    # stanza_numbers = stanza_numbers.reshape(-1, 1)
    return scaler.transform(stanza_numbers)


#########################
# preprocessing of unseen data
#########################
def preprocess_new_data(data: dict, path: str) -> dict:
    processed_data = {'booleans': data['booleans']}
    
    with open(path + '/tfidf_vectorizer.joblib', 'rb') as f:
        tfidf_vectorizer = joblib.load(f)
        
    with open(path + '/nmf_model.joblib', 'rb') as f:
        nmf_model = joblib.load(f)
    
    processed_data['topic_distributions'] = topic_analysis(
        tfidf_vectorizer,
        nmf_model,
        data['titles']
    )
    
    
    with open(path + '/basic_tokenizer.joblib', 'rb') as f:
        basic_tokenizer = joblib.load(f)
        
    processed_data['sequences'] = tokenize(basic_tokenizer, data['lemmatized_stanzas'])
    
    processed_data['padded_sequences'] = padding(processed_data['sequences'])
    
    
    with open(path + '/scaler.joblib', 'rb') as f:
        scaler = joblib.load(f)
    
    processed_data['stanza_numbers'] = scale_stanza_numbers(
        scaler, data['stanza_numbers']
    )
    
    
    return processed_data