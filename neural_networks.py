import argparse
import pandas as pd

#########################
# arguments
#########################
parser = argparse.ArgumentParser(
    description= "Train or evaluate the emotion labeling model."
)

path= '/Users/brunobarbieri/Library/CloudStorage/OneDrive-UniversityofPisa/TA_Project/'

parser.add_argument(
    '--dataset', type=str, required= False,
    default= path + 'data/lab_lem_merge.csv'
)

# 1: 1DConvnet
# 2: RNN
parser.add_argument(
    '--type', type= int, required= False,
    default= 1
)

# number of epochs
parser.add_argument(
    '--epochs', type= int, required= False,
    default= 10
)

args = parser.parse_args()

df = pd.read_csv(args.dataset)


#########################
# create folder to save data
#########################
import os
from datetime import datetime

# Get the current date and time as a string in the format DD-MM-YYYY_HH-MM-SS
folder_name = datetime.now().strftime(
    ('1DConvnet' if args.type == 1 else 'RNN') +
    "_%d-%m-%Y_%H-%M-%S"
)

folder_path= path + 'neural_networks/' + folder_name

os.makedirs(folder_path)

#########################
# undersampling
#########################
# from sklearn.utils import resample

# # Find the smallest class size
# min_size = df['label'].value_counts().min()

# # Downsample each class to the smallest size
# downsampled_samples = []
# for class_label, group in df.groupby('label'):
#     sampled_group = resample(group, replace=False, n_samples=min_size, random_state=42)
#     downsampled_samples.append(sampled_group)

# # Combine the downsampled groups
# df = pd.concat(downsampled_samples)


#########################
# Initialize the tokenizer
#########################
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 20000
tokenizer = Tokenizer(num_words= vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(df['lemmatized_stanzas'])

sequences = tokenizer.texts_to_sequences(df['lemmatized_stanzas'])

# Save the tokenizer for future preprocessing
import joblib

joblib.dump(tokenizer, folder_path + '/basic_tokenizer.pkl')


#########################
# add padding
#########################
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_seq_length = 150  # Adjust based on your average stanza length
padded_sequences = pad_sequences(
    sequences, maxlen=max_seq_length,
    padding='post', truncating='post'
)


#########################
# preprocess non-text data
#########################
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
stanza_numbers = scaler.fit_transform(df[['stanza_number']])

booleans = df[['is_country', 'is_pop', 'is_rap',
               'is_rb', 'is_rock', 'is_chorus']].astype(int).values

# additional_data = [stanza_numbers] + booleans


#########################
# obtain labels
#########################
from tensorflow.keras.utils import to_categorical
import numpy as np

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

labels_indices = np.array([emotion_mapping[label] for label in df['label']])

labels = to_categorical(labels_indices, num_classes=8)


#########################
# split data
#########################
from sklearn.model_selection import train_test_split

(
    padded_sequences_train, padded_sequences_test,
    stanza_numbers_train, stanza_numbers_test,
    booleans_train, booleans_test,
    titles_train, titles_test,
    y_train, y_test
) = train_test_split(
    padded_sequences, stanza_numbers, booleans, df['title'],
    labels, test_size=0.2, random_state=42
)


#########################
# title unsupervised learning
#########################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import numpy as np

# Convert song titles to a TF-IDF matrix
tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix_train = tfidf_vectorizer.fit_transform(titles_train)

# Apply NMF for topic modeling
num_topics = 8
nmf_model = NMF(n_components=num_topics, random_state=42)
topic_distributions_train = nmf_model.fit_transform(tfidf_matrix_train)


#########################
# NNs
#########################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Concatenate, Dropout, GRU
)

lyrics_input = Input(shape=(max_seq_length,),name= 'text_input')

embedding_lyrics = Embedding(
    input_dim= vocab_size, output_dim= 128,
    input_length= max_seq_length
) (lyrics_input)

# CNN
if args.type == 1:
    conv1 = Conv1D(
        filters= 64, kernel_size= 5, activation= 'relu'
    ) (embedding_lyrics)

    pooling = GlobalMaxPooling1D()(conv1)
    
# RNN
if args.type == 2:
    gru1 = GRU(
        64, return_sequences= True,
        name= "recurrent_layer1"
    ) (embedding_lyrics)
    recurrent_layers = GRU(
        32, return_sequences= False,
        name= "recurrent_layer2"
    ) (gru1)
    
lyrics_dropout = Dropout(0.2) (pooling if args.type == 1 else recurrent_layers)

# Additional Features Branch
stanza_number_input = Input(shape=(1,), name='stanza_number')

bool_inputs = [
    Input(shape= (1,), name= name, dtype= 'int32')
    for name in [
        'is_country', 'is_pop', 'is_rap',
        'is_rb', 'is_rock', 'is_chorus'
    ]
]

# title topic
topic_input = Input(shape=(num_topics,), name="Topic_Input")

# Concatenate all inputs
additional_input = Concatenate(name= 'additional_input') (
    [stanza_number_input, topic_input] + bool_inputs
)

# Combine the Branches
combined = Concatenate()([
    lyrics_dropout, additional_input
])
dense1 = Dense(64, activation='relu')(combined)
dropout = Dropout(0.3)(dense1)
output = Dense(8, activation='softmax', name='output')(dropout)

# Define the Model
model = Model(
    inputs=[lyrics_input, stanza_number_input, topic_input] + bool_inputs,
    outputs=output
)

# Compile the Model
model.compile(
    optimizer='adam', loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    [
        padded_sequences_train, stanza_numbers_train, topic_distributions_train
    ] + list(booleans_train.T),
    y_train,
    validation_split= 0.2,
    epochs= args.epochs,
    batch_size= 32
)

# Model Summary
model.summary()

# get test topics
tfidf_matrix_test = tfidf_vectorizer.fit_transform(titles_test)

topic_distributions_test = nmf_model.transform(tfidf_matrix_test)

# Model evaluation on the test set
loss, accuracy = model.evaluate(
    [
        padded_sequences_test, stanza_numbers_test, topic_distributions_test
    ] + list(booleans_test.T), 
    y_test
)


#########################
# Save the Model
#########################
model.save(folder_path + '/multi_input_emotion_model.h5')


#########################
# roc curve
#########################
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarize labels for multi-class support
classes = df['label'].unique()
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)

# Get the predicted probabilities from the Keras model
y_score = model.predict(
    [
        padded_sequences_test, stanza_numbers_test, topic_distributions_test
    ] + list(booleans_test.T)
)

# Plot ROC curves for each class
plt.figure(figsize=(8, 6))
for i, class_name in enumerate(classes):
    # Compute ROC curve and AUC for each class
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {roc_auc:.2f})")

# Add a reference line (y = x)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Configure the plot
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for 1D Convolutional Neural Network")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.savefig(folder_path + '/roc_curve.png')


#########################
# Training and Validation Loss/Accuracy Curve
#########################
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(folder_path + '/accuracy.png')