import argparse
import pandas as pd
import preprocessing
import graphs
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
from sklearn.utils import resample


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

parser.add_argument(
    '--semisupervised', type= int, required= False,
    default= 0
)

parser.add_argument(
    '--reset', action= 'store_true', 
)

parser.add_argument(
    '--even-labels', action= 'store_true', 
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
    ('SEMI_SUPERVISED' if args.semisupervised != 0 else '') +
    "_%d-%m-%Y_%H-%M-%S"
)

folder_path= path + 'neural_networks/' + folder_name

os.makedirs(folder_path)


with open(folder_path + "/model_summary.txt", "w") as f:
    with redirect_stdout(f):
        print(('1DConvnet' if args.type == 1 else 'RNN'))
        print(('semisupervised learning' if args.semisupervised != 0
               else 'supervised learning') + '\n')
        print(f'Epochs: {args.epochs}')
        if args.semisupervised!= 0:
            print(f'Epochs with pseudo-labeled data: {args.semisupervised}')
            print(f'Reset model weights after first training: {args.reset}')
        print('\n')


#########################
# preprocessing
#########################

#########################
# split data
#########################

#########################
# undersampling
#########################
# !!! does not seem to help
if args.even_labels:
    min_size = df['label'].value_counts().min()

    downsampled_samples = []
    for class_label, group in df.groupby('label'):
        sampled_group = resample(
            group, replace=False, n_samples=min_size, random_state=42
        )
        downsampled_samples.append(sampled_group)

    # could reuse for testing but seems problematic as pollutes the test set
    # cut_df = df[~df.isin(pd.concat(downsampled_samples)).all(axis=1)]

    df = pd.concat(downsampled_samples)

# test split
(
    lemmatized_stanzas_train, lemmatized_stanzas_test,
    stanza_numbers_train, stanza_numbers_test,
    booleans_train, booleans_test,
    titles_train, titles_test,
    y_train, y_test
) = train_test_split(
    df['lemmatized_stanzas'], df[['stanza_number']],
    df[['is_country', 'is_pop', 'is_rap',
               'is_rb', 'is_rock', 'is_chorus']].astype(int).values,
    df['title'],
    preprocessing.preprocess_labels(df['label']), test_size=0.3
)

# validation split
(
    lemmatized_stanzas_train, lemmatized_stanzas_val,
    stanza_numbers_train, stanza_numbers_val,
    booleans_train, booleans_val,
    titles_train, titles_val,
    y_train, y_val
) = (
    train_test_split(
        lemmatized_stanzas_train, stanza_numbers_train,
        booleans_train, titles_train,
        y_train, test_size=0.3)
)


data = {
    'train': {
        'lemmatized_stanzas': lemmatized_stanzas_train,
        'stanza_numbers': stanza_numbers_train,
        'booleans': booleans_train,
        'titles': titles_train,
        'y': y_train
    },
    'val': {
        'lemmatized_stanzas': lemmatized_stanzas_val,
        'stanza_numbers': stanza_numbers_val,
        'booleans': booleans_val,
        'titles': titles_val,
        'y': y_val
    },
    'test': {
        'lemmatized_stanzas': lemmatized_stanzas_test,
        'stanza_numbers': stanza_numbers_test,
        'booleans': booleans_test,
        'titles': titles_test,
        'y': y_test
    }
}


preprocessed_data = preprocessing.preprocess(data, folder_path)


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
    # input_length= preprocessing.max_seq_length
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

X_train = [
    preprocessed_data['train']['padded_sequences'],
    preprocessed_data['train']['stanza_numbers'],
    preprocessed_data['train']['topic_distributions']
] + list(preprocessed_data['train']['booleans'].T)

X_val = [
    preprocessed_data['val']['padded_sequences'],
    preprocessed_data['val']['stanza_numbers'],
    preprocessed_data['val']['topic_distributions']
] + list(preprocessed_data['val']['booleans'].T)


history = model.fit(
    X_train, preprocessed_data['train']['y'],
    validation_data=(X_val, preprocessed_data['val']['y']),
    epochs= args.epochs,
    batch_size= 32
)

if args.semisupervised != 0:
    X_unlabeled = pd.read_csv(
        path + 'data/def_lemmatized_df.csv'
    ).iloc[130001:].dropna().sample(frac= 0.05)
    
    X_unlabeled = {
        'lemmatized_stanzas': X_unlabeled['lemmatized_stanzas'],
        'stanza_numbers': X_unlabeled[['stanza_number']],
        'booleans': X_unlabeled[['is_country', 'is_pop', 'is_rap',
               'is_rb', 'is_rock', 'is_chorus']].astype(int).values,
        'titles': X_unlabeled['title']
    }

    X_unlabeled_preprocessed = preprocessing.preprocess_new_data(
        X_unlabeled, folder_path
    )
    
    X_list = [
        X_unlabeled_preprocessed['padded_sequences'],
        X_unlabeled_preprocessed['stanza_numbers'],
        X_unlabeled_preprocessed['topic_distributions']
    ] + list(X_unlabeled_preprocessed['booleans'].T)
    
    predictions = model.predict(X_list)

    max_indices = np.argmax(predictions, axis=1)

    one_hot_batch = np.zeros_like(predictions)
    one_hot_batch[np.arange(predictions.shape[0]), max_indices] = 1
    
    X_unlabeled['y'] = one_hot_batch
    
    data['train']['lemmatized_stanzas'] = pd.concat(
        [data['train']['lemmatized_stanzas'], X_unlabeled['lemmatized_stanzas']],
        ignore_index= True
    )
    
    data['train']['stanza_numbers'] = pd.concat(
        [data['train']['stanza_numbers'], X_unlabeled['stanza_numbers']],
        ignore_index= True
    )
    
    data['train']['booleans'] = np.concatenate((
        data['train']['booleans'], X_unlabeled['booleans']), axis= 0)

    data['train']['titles'] = pd.concat(
        [data['train']['titles'], X_unlabeled['titles']],
        ignore_index= True
    )
    
    data['train']['y'] = np.concatenate((
        data['train']['y'], X_unlabeled['y']), axis= 0)

    
    preprocessed_data = preprocessing.preprocess(data, folder_path)

    X_train = [
        preprocessed_data['train']['padded_sequences'],
        preprocessed_data['train']['stanza_numbers'],
        preprocessed_data['train']['topic_distributions']
    ] + list(preprocessed_data['train']['booleans'].T)

    X_val = [
        preprocessed_data['val']['padded_sequences'],
        preprocessed_data['val']['stanza_numbers'],
        preprocessed_data['val']['topic_distributions']
    ] + list(preprocessed_data['val']['booleans'].T)

    if args.reset:
        for layer in model.layers:
            if (
                hasattr(layer, 'kernel_initializer') and
                hasattr(layer, 'bias_initializer')
            ):
                layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                layer.bias.assign(layer.bias_initializer(layer.bias.shape))

    
    model.fit(
        X_train, preprocessed_data['train']['y'],
        validation_data=(X_val, preprocessed_data['val']['y']),
        epochs= args.semisupervised,
        batch_size=32
    )


# save summary
with open(folder_path + "/model_summary.txt", "w") as f:
    with redirect_stdout(f):
        model.summary()


x_test = [
    preprocessed_data['test']['padded_sequences'],
    preprocessed_data['test']['stanza_numbers'],
    preprocessed_data['test']['topic_distributions']
] + list(preprocessed_data['test']['booleans'].T)

loss, accuracy = model.evaluate(x_test, preprocessed_data['test']['y'])

y_pred = model.predict(x_test)

y_test_encoded = np.argmax(preprocessed_data['test']['y'], axis=1)
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
    y_test= preprocessed_data['test']['y']
)

graphs.accuracy_curve(history, folder_path)

graphs.confusion_matrix_graph(
    y_test_encoded, y_pred_encoded, classes, folder_path
)

graphs.plot_class_wise_accuracy(
    y_test_encoded, y_pred_encoded, classes, folder_path
)