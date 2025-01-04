"""
This script trains or evaluates an emotion labeling model for song stanzas using either 
a 1D Convolutional Network or a Recurrent Neural Network. The user can enable various 
training modes, such as semi-supervised learning, label balancing, and model weight resetting.

Arguments:
---------
--dataset : str, optional
    Path to the input dataset file (default: 
    '/Users/brunobarbieri/Library/CloudStorage/OneDrive-UniversityofPisa/TA_Project/data/lab_lem_merge.csv').
    This CSV file should contain labeled stanzas for model training.

--type : int, optional
    Type of model to train. Accepted values are:
        1 -> 1D Convolutional Network
        2 -> Recurrent Neural Network
    Default is 1.

--epochs : int, optional
    Number of epochs to train the model. Default is 10.

--semisupervised : int, optional
    Number of epochs to train the model using semi-supervised learning with pseudo-labeled data. 
    Set to 0 to disable semi-supervised learning and use supervised learning only. Default is 0.

--reset : flag, optional
    If specified, the model's weights will be reset before semi-supervised training.

--even-labels : flag, optional
    If specified, the dataset will be balanced by ensuring an even distribution of labels 
    across all classes before training.
"""

import argparse
import pandas as pd
import graphs
import numpy as np
from contextlib import redirect_stdout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model
import os
from datetime import datetime

import preprocessing
import data_splitting
from nn_architectures import build_model


# methods
def reset_weights():
    """
    Resets the trainable weights of each layer in the Keras model.
    """
    for layer in model.layers:
            if (
                hasattr(layer, 'kernel_initializer') and
                hasattr(layer, 'bias_initializer')
            ):
                layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
                layer.bias.assign(layer.bias_initializer(layer.bias.shape))



# arguments
parser = argparse.ArgumentParser(
    description= "Train or evaluate the emotion labeling model."
)

path= '/Users/brunobarbieri/Library/CloudStorage/OneDrive-UniversityofPisa/TA_Project/'

# Path to the dataset file
parser.add_argument(
    '--dataset', type=str, required=False,
    default=path + 'data/lab_lem_merge.csv',
    help='Path to the input dataset file. Defaults to "data/lab_lem_merge.csv".'
)

# Model type: 1 for 1D Convolutional Network, 2 for Recurrent Neural Network
parser.add_argument(
    '--type', type=int, required=False,
    default=1,
    help='Type of model to train. 1 for 1D Convolutional Network, 2 for Recurrent Neural Network. Defaults to 1.'
)

# Number of epochs for model training
parser.add_argument(
    '--epochs', type=int, required=False,
    default=10,
    help='Number of training epochs. Defaults to 10.'
)

# Enable semi-supervised learning
parser.add_argument(
    '--semisupervised', type=int, required=False,
    default=0,
    help=('Number of training epochs on the dataset with pseudo-labeled entries. '
          'Set to 0 for supervised learning only. Defaults to 0.')
)

# Reset model weights before training
parser.add_argument(
    '--reset', action='store_true',
    help='Reset the model weights before semi-supervised training.'
)

# Use even labels for downsampling the dataset
parser.add_argument(
    '--even-labels', action='store_true',
    help='Balance the dataset by ensuring an even distribution of labels.'
)


args = parser.parse_args()

df = pd.read_csv(args.dataset)


# create folder to save data
folder_name = datetime.now().strftime(
    ('1DConvnet' if args.type == 1 else 'RNN') +
    ('SEMI_SUPERVISED' if args.semisupervised != 0 else '') +
    "_%d-%m-%Y_%H-%M-%S"
)

folder_path= path + 'neural_networks/' + folder_name

os.makedirs(folder_path)

# save initial info into model_summary.txt file
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


# undersampling
if args.even_labels:
    data_splitting(df)

data = data_splitting.train_test_val_split(df)


preprocessed_data = preprocessing.preprocess(data, folder_path)


# NNs
# define model
(inputs, output) = build_model(args.type)
model = Model(
    inputs=inputs,
    outputs=output
)

model.compile(
    optimizer=('adam' if args.type == 1 else RMSprop(learning_rate=0.001)),
    loss='categorical_crossentropy',
    metrics=['categorical_accuracy']
)

X_train = data_splitting.get_data_as_list(preprocessed_data['train'])

X_val = data_splitting.get_data_as_list(preprocessed_data['val'])


history = model.fit(
    X_train, preprocessed_data['train']['y'],
    validation_data=(X_val, preprocessed_data['val']['y']),
    epochs= args.epochs,
    batch_size= 32
)


# Check if semi-supervised learning is enabled (i.e., not 0)
if args.semisupervised != 0:
    
    # Load a subset of the unlabeled data for semi-supervised learning
    # Load the CSV, start from the 130001st row, drop any missing values, and sample a number of entries equal
    # to half of the training set
    X_unlabeled = pd.read_csv(
        path + 'data/def_lemmatized_df.csv'
    ).iloc[130001:].dropna().sample(n= int(len(X_train) / 2))
    
    # Prepare the unlabeled data by extracting specific columns and organizing them in a dictionary
    X_unlabeled = {
        'lemmatized_stanzas': X_unlabeled['lemmatized_stanzas'],
        'stanza_numbers': X_unlabeled[['stanza_number']],
        'booleans': X_unlabeled[['is_country', 'is_pop', 'is_rap',
               'is_rb', 'is_rock', 'is_chorus']].astype(int).values,
        'titles': X_unlabeled['title']
    }

    # Preprocess the unlabeled data (e.g., tokenization, normalization, etc.)
    X_unlabeled_preprocessed = preprocessing.preprocess_new_data(
        X_unlabeled, folder_path
    )
    
    # Convert the preprocessed data into a list format suitable for the model
    X_list = data_splitting.get_data_as_list(
        X_unlabeled_preprocessed
    )
    
    # Use the model to predict the labels for the unlabeled data
    predictions = model.predict(X_list)

    # Get the index of the maximum predicted value (the most likely label for each sample)
    max_indices = np.argmax(predictions, axis=1)

    # Convert the predicted indices to a one-hot encoded format
    one_hot_batch = np.zeros_like(predictions)
    one_hot_batch[np.arange(predictions.shape[0]), max_indices] = 1
    
    # Add the pseudo-labels (predicted labels) to the unlabeled data
    X_unlabeled['y'] = one_hot_batch
    
    # Combine the original labeled training set with the pseudo-labeled data
    # Concatenate the lemmatized stanzas (lyrics), stanza numbers, boolean features, titles, and pseudo-labels
    data['train']['lemmatized_stanzas'] = pd.concat(
        [data['train']['lemmatized_stanzas'], X_unlabeled['lemmatized_stanzas']],
        ignore_index=True
    )
    
    data['train']['stanza_numbers'] = pd.concat(
        [data['train']['stanza_numbers'], X_unlabeled['stanza_numbers']],
        ignore_index=True
    )
    
    data['train']['booleans'] = np.concatenate((
        data['train']['booleans'], X_unlabeled['booleans']), axis=0)

    data['train']['titles'] = pd.concat(
        [data['train']['titles'], X_unlabeled['titles']],
        ignore_index=True
    )
    
    data['train']['y'] = np.concatenate((
        data['train']['y'], X_unlabeled['y']), axis=0)

    # Preprocess the entire training set (including pseudo-labeled data)
    preprocessed_data = preprocessing.preprocess(data, folder_path)

    # Convert the preprocessed training and validation data into a list format
    X_train = data_splitting.get_data_as_list(preprocessed_data['train'])

    X_val = data_splitting.get_data_as_list(preprocessed_data['val'])

    # Reset the model weights if specified
    if args.reset:
        reset_weights()
    
    # Fit the model using the updated training data (including pseudo-labels)
    # Validate on the validation set during training
    model.fit(
        X_train, preprocessed_data['train']['y'],
        validation_data=(X_val, preprocessed_data['val']['y']),
        epochs=args.semisupervised,
        batch_size=32
    )


# test data into list
X_test = data_splitting.get_data_as_list(preprocessed_data['test'])

# save summary
with open(folder_path + "/model_summary.txt", "w") as f:
    with redirect_stdout(f):
        model.summary()
        model.evaluate(X_test, preprocessed_data['test']['y'])

y_pred = model.predict(X_test)

y_test_encoded = np.argmax(preprocessed_data['test']['y'], axis=1)
y_pred_encoded = np.argmax(y_pred, axis=1)


# save the Model
model.save(folder_path + '/model.keras')

classes = df['label'].unique()

# graph generation
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