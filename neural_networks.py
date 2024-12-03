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
    '--dataset', type=str, required=True,
    default= path + 'data/lab_lem_merge.csv')

args = parser.parse_args()

df = pd.read_csv(args.dataset)

#########################
# create folder to save data
#########################
import os
from datetime import datetime

# Get the current date and time as a string in the format DD-MM-YYYY_HH-MM-SS
folder_name = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

folder_path= path + 'neural_networks/' + folder_name

os.makedirs(folder_path)

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
# 1dconvnet model
#########################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Concatenate, Dropout, Flatten
)

text_input = Input(shape=(max_seq_length,), name='text_input')
embedding = Embedding(
    input_dim=vocab_size, output_dim= 128,
    input_length=max_seq_length) (text_input)
conv1 = Conv1D(filters=64, kernel_size=3, activation='relu')(embedding)
pooling = GlobalMaxPooling1D()(conv1)

# Additional Features Branch
additional_input = Input(shape=(6,), name='additional_input')

# Combine the Branches
combined = Concatenate()([pooling, additional_input])
dense1 = Dense(64, activation='relu')(combined)
dropout = Dropout(0.5)(dense1)
output = Dense(8, activation='softmax', name='output')(dropout)

# Define the Model
model = Model(inputs=[text_input, additional_input], outputs=output)

# Compile the Model
model.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Model Summary
model.summary()

# Save the Model
model.save(folder_path + '/multi_input_emotion_model.h5')