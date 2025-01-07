"""
This script provides functionality to load a pretrained model and use it to classify new data.
The file also provides the classify method which can function as a very simple
API for external usage.

Arguments:
    --dataset (str, optional): Path to a CSV file containing the data to classify. If not provided, 
        a default test file can be used. If a DataFrame is directly passed, it will be used for classification.
    --model-folder (str): Path to the folder containing the trained models and utilities. This is a required argument.
    --type (int): Specifies the type of model to use for classification:
        - 1: Random Forest
        - 2: Support Vector Machine (SVM)
        - 3: 1D Convolutional Neural Network (CNN)
        - 4: Recurrent Neural Network (RNN)

From within the main directory of the project, run a demo using the following:

python models/classify.py --model-folder models/static_models/<model> --type <appropriate_type>

To test on random forest:
python3 models/classify.py --model-folder models/static_models/rf_model.pkl --type 1

To test on 1dconvnet:
python3 models/classify.py --model-folder models/1DConvnet/ --type 3

To test on rnn:
python3 models/classify.py --model-folder models/RNN/ --type 4

Note: this script was not tested for svm models.
"""

import preprocessing
import data_splitting

import argparse
import pandas as pd
import numpy as np
import joblib
import ast
from tensorflow.keras.models import load_model

# arguments
parser = argparse.ArgumentParser(
    description= "Utilize an emotion labeling model to label data."
)

parser.add_argument(
    '--dataset', type= str, required= False,
    default= 'models/data/unlabeled_sample.csv',
    help= 'Path to .csv file to classify.'
)

parser.add_argument(
    '--model-folder', type= str, required= True,
    help= 'Path to folder containing the models and utilities.'
)

parser.add_argument(
    '--type', type= int, required= True,
    help= 'Type of model to use.'
)

emotion_labels = [
    'anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust'
]


def convert_bool_to_int(x: np.ndarray) -> np.ndarray:
    """
    Converts bool to int

    Args:
        x (np.ndarray): a matrix of booleans

    Returns:
        np.ndarray: a matrix of integers
    """
    return x.astype(int)


def translate_into_labels(predictions: np.ndarray) -> pd.Series:
    """
    Translate the predicted probabilities into corresponding emotion labels.

    Args:
        predictions (numpy.ndarray): The predicted probabilities for each emotion class.

    Returns:
        pd.Series: A Pandas Series with emotion labels corresponding to the highest probability for each row.
    """
    predicted_indices = np.argmax(predictions, axis=1)
    
    return pd.Series([emotion_labels[i] for i in predicted_indices])


def classify(dataset: str | pd.DataFrame,
             model_folder: str,
             t: int) -> pd.Series:
    """
    Classifies data using a pretrained model.

    Args:
        dataset (str | pd.DataFrame): data to classify. If a file name is given,
        it must be a .csv file.
        
        model_folder (str): folder containing the model and its other tools.
        
        t (int): type of model to use. 
            - 1: Random Forest
            - 2: SVM
            - 3: 1d Convolutional neural Network
            - 4: Recurrent Neural Network

    Returns:
        pd.Series: series of labels.
        
    Note: this method was not tested for svm models.
    """
    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)
    
    if t == 1 or t == 2:
        # prepare data
        dataset['lemmatized_stanzas'] = dataset['lemmatized_stanzas'].apply(ast.literal_eval)

        dataset['text_str'] = dataset['lemmatized_stanzas'].apply(lambda x: ' '.join(x))
        
        # load model
        model = joblib.load(model_folder)
        
        # predict
        return pd.Series(model.predict(
            dataset[[
                'text_str', 'title', 'stanza_number', 'is_country',
                'is_pop', 'is_rap', 'is_rb', 'is_rock', 'is_chorus'
            ]]
        ))
    
    else:
        X_unlabeled = {
            'lemmatized_stanzas': dataset['lemmatized_stanzas'],
            'stanza_numbers': dataset[['stanza_number']],
            'booleans': dataset[['is_country', 'is_pop', 'is_rap',
                'is_rb', 'is_rock', 'is_chorus']].astype(int).values,
            'titles': dataset['title']
        }

        # Preprocess the unlabeled data (e.g., tokenization, normalization, etc.)
        X_unlabeled_preprocessed = preprocessing.preprocess_new_data(
            X_unlabeled, model_folder
        )
        
        # Convert the preprocessed data into a list format suitable for the model
        X_list = data_splitting.get_data_as_list(
            X_unlabeled_preprocessed
        )
        
         # load model
        model = load_model(model_folder + 'model.keras')
        
        # Use the model to predict the labels for the unlabeled data
        return translate_into_labels(model.predict(X_list))



args = parser.parse_args()

print(classify(
    dataset= args.dataset,
    model_folder= args.model_folder,
    t= args.type
))