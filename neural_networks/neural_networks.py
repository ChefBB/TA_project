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
    
    X_list = data_splitting.get_data_as_list(
        X_unlabeled_preprocessed
    )
    
    predictions = model.predict(X_list)

    max_indices = np.argmax(predictions, axis=1)

    one_hot_batch = np.zeros_like(predictions)
    one_hot_batch[np.arange(predictions.shape[0]), max_indices] = 1
    
    X_unlabeled['y'] = one_hot_batch
    
    # concatenate the train set with the pseudo labeled training set
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

    X_train = data_splitting.get_data_as_list(preprocessed_data['train'])

    X_val = data_splitting.get_data_as_list(preprocessed_data['val'])

    if args.reset:
        reset_weights()
    
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


X_test = data_splitting.get_data_as_list(preprocessed_data['test'])

loss, accuracy = model.evaluate(X_test, preprocessed_data['test']['y'])

y_pred = model.predict(X_test)

y_test_encoded = np.argmax(preprocessed_data['test']['y'], axis=1)
y_pred_encoded = np.argmax(y_pred, axis=1)


# save the Model
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