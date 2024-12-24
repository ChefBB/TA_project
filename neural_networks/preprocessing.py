import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
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


def preprocess (df: pd.DataFrame, folder_path: str) -> tuple:
    #########################
    # undersampling
    #########################
    # !!! does not seem to help
    # from sklearn.utils import resample

    # min_size = df['label'].value_counts().min()

    # downsampled_samples = []
    # for class_label, group in df.groupby('label'):
    #     sampled_group = resample(
    #         group, replace=False, n_samples=min_size, random_state=42
    #     )
    #     downsampled_samples.append(sampled_group)

    # df = pd.concat(downsampled_samples)



    #########################
    # split data
    #########################
    (
        lemmatized_stanzas_train, lemmatized_stanzas_test,
        stanza_numbers_train, stanza_numbers_test,
        booleans_train, booleans_test,
        titles_train, titles_test,
        y_train, y_test
    ) = train_test_split(
        df['lemmatized_stanzas'], df[['stanza_number']], booleans_conv(df), df['title'],
        df['label'], test_size=0.3, random_state=42
    )
    
    
    #########################
    # title unsupervised learning
    #########################
    # convert song titles to TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix_train = tfidf_vectorizer.fit_transform(titles_train)
    
    joblib.dump(tfidf_vectorizer, folder_path + '/tfidf_vectorizer.joblib')

    # NMF for topic modeling
    nmf_model = NMF(n_components=num_topics, random_state=42)
    topic_distributions_train = nmf_model.fit_transform(tfidf_matrix_train)
    
    joblib.dump(nmf_model, folder_path + '/nmf_model.joblib')
    
    # apply to test set
    topic_distributions_test = topic_analysis(
        tfidf_vectorizer= tfidf_vectorizer,
        nmf_model= nmf_model,
        titles= titles_test
    )
    
    #########################
    # Initialize the tokenizer
    #########################
    tokenizer = Tokenizer(num_words= vocab_size, oov_token="<UNK>")
    tokenizer.fit_on_texts(lemmatized_stanzas_train)

    sequences_train = tokenize(tokenizer, lemmatized_stanzas_train)
    
    sequences_test = tokenize(tokenizer, lemmatized_stanzas_test)
    
    padded_sequences_train = padding(sequences_train)

    padded_sequences_test = padding(sequences_test)
    
    joblib.dump(tokenizer, folder_path + '/basic_tokenizer.pkl')
    
    
    #########################
    # preprocess non-text data
    #########################
    scaler = StandardScaler()
    scaled_stanza_numbers_train = scaler.fit_transform(stanza_numbers_train)
    
    joblib.dump(scaler, folder_path + '/scaler.joblib')
    
    # apply on test set
    scaled_stanza_numbers_test = scale_stanza_numbers(scaler, stanza_numbers_test)
    
    
    return (
        padded_sequences_train, padded_sequences_test,
        scaled_stanza_numbers_train, scaled_stanza_numbers_test,
        booleans_train, booleans_test,
        topic_distributions_train, topic_distributions_test,
        preprocess_labels(y_train), preprocess_labels(y_test)
    )
    
    
    
#########################
# obtain labels
#########################
def preprocess_labels (labels: pd.Series) -> np.array:
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
    return scaler.transform(stanza_numbers)