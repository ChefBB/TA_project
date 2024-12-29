import argparse
import pandas as pd
import preprocessing
import graphs
import numpy as np

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

folder_name = datetime.now().strftime(
    ('1DConvnet' if args.type == 1 else 'RNN') +
    "_%d-%m-%Y_%H-%M-%S"
)

folder_path= path + 'neural_networks/' + folder_name

os.makedirs(folder_path)


#########################
# preprocessing
#########################
(
    padded_sequences_train, padded_sequences_test,
    stanza_numbers_train, stanza_numbers_test,
    booleans_train, booleans_test,
    topic_distributions_train, topic_distributions_test,
    y_train, y_test
) = preprocessing.preprocess(df, folder_path)


#########################
# NNs
#########################
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D, GlobalMaxPooling1D,
    Dense, Concatenate, Dropout, GRU
)
from tensorflow.keras.optimizers import RMSprop

lyrics_input = Input(shape=(preprocessing.max_seq_length,),name= 'text_input')

embedding_lyrics = Embedding(
    input_dim= preprocessing.vocab_size, output_dim= 128,
    input_length= preprocessing.max_seq_length
) (lyrics_input)

# CNN
if args.type == 1:
    conv = Conv1D(
        filters= 12, kernel_size= 4, activation= 'relu',
        name= 'conv_layer1'
    ) (embedding_lyrics)
    
    # conv = Conv1D(
    #     filters= 64, kernel_size= 8, activation= 'relu',
    #     name= 'conv_layer2'
    # ) (conv)
    
    conv = Conv1D(
        filters= 8, kernel_size= 6, activation= 'relu',
        name= 'conv_layer3'
    ) (conv)
    
    conv = Conv1D(
        filters= 6, kernel_size= 8, activation= 'relu',
        name= 'conv_layer4'
    ) (conv)

    pooling = GlobalMaxPooling1D()(conv)
    
# RNN
if args.type == 2:
    recurrent_layer = GRU(
        128, return_sequences= True,
        name= "recurrent_layer1",
        activation='tanh', recurrent_activation='sigmoid',
        dropout= 0.4, recurrent_dropout= 0.3
    ) (embedding_lyrics)
    
    recurrent_layer = GRU(
        64, return_sequences= True,
        name= "recurrent_layer2",
        activation='tanh', recurrent_activation='sigmoid',
        dropout= 0.4, recurrent_dropout= 0.3
    ) (recurrent_layer)
    
    recurrent_layers = GRU(
        32, return_sequences= False,
        activation='tanh', recurrent_activation='sigmoid',
        name= "recurrent_layer3"
    ) (recurrent_layer)

    
lyrics_dropout = Dropout(0.3 if args.type == 1
                         else 0.5) (pooling if args.type == 1
                                    else recurrent_layers)

# additional features
stanza_number_input = Input(shape=(1,), name='stanza_number')

bool_inputs = [
    Input(shape= (1,), name= name, dtype= 'int32')
    for name in [
        'is_country', 'is_pop', 'is_rap',
        'is_rb', 'is_rock', 'is_chorus'
    ]
]

# title topic
topic_input = Input(shape=(preprocessing.num_topics,), name="Topic_Input")

# concatenate all inputs
additional_input = Concatenate() (
    [stanza_number_input, topic_input] + bool_inputs
)

additional_input = Dense(
    32, activation= 'relu',
    name= 'additional_input'
)(additional_input)

# combine branches
combined = Concatenate()([
    lyrics_dropout, additional_input
])
dense1 = Dense(32, activation='relu')(combined)
dropout = Dropout(0.3)(dense1)
output = Dense(8, activation='softmax', name='output')(dropout)

# define model
model = Model(
    inputs=[lyrics_input, stanza_number_input, topic_input] + bool_inputs,
    outputs=output
)


from tensorflow.keras.metrics import TopKCategoricalAccuracy

model.compile(
    optimizer=('adam' if args.type == 1 else RMSprop(learning_rate=0.001)),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

history = model.fit(
    [
        padded_sequences_train, stanza_numbers_train, topic_distributions_train
    ] + list(booleans_train.T),
    y_train,
    validation_split= 0.3,
    epochs= args.epochs,
    batch_size= 32
)

model.summary()

x_test = [
    padded_sequences_test, stanza_numbers_test, topic_distributions_test
] + list(booleans_test.T)

loss, accuracy = model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)

y_test_encoded = np.argmax(y_test, axis=1)
y_pred_encoded = np.argmax(y_pred, axis=1)


#########################
# Save the Model
#########################
model.save(folder_path + '/multi_input_emotion_model.h5')

classes = df['label'].unique()

graphs.roc_curve_graph(
    y_pred = y_pred,
    classes = classes,
    folder_path= folder_path,
    y_test= y_test
)

graphs.accuracy_curve(history, folder_path)

graphs.confusion_matrix_graph(
    y_test_encoded, y_pred_encoded, classes, folder_path
)

graphs.plot_class_wise_accuracy(
    y_test_encoded, y_pred_encoded, classes, folder_path
)