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
"""

import preprocessing

import argparse
import pandas as pd
import joblib
from tensorflow.keras.model import load_model

# arguments
parser = argparse.ArgumentParser(
    description= "Utilize an emotion labeling model to label data."
)

parser.add_argument(
    '--dataset', type= str, required= False,
    default= None,                              # TODO add default test file
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
    """
    if isinstance(dataset, str):
        dataset = pd.read_csv(dataset)
    
    if t == 1 | t == 2:
        # prepare data
        dataset['text_str'] = dataset['lemmatized_stanzas'].apply(
            lambda x: ' '.join(x)
        )
        
        # load model
        clf_loaded = joblib.load(
            model_folder + 'best_' +
            ('rf' if t == 1 else 'svm') +
            'pipeline.pkl')
        
        # predict
        return clf_loaded.best_predictor.predict(
            dataset[[
                'text_str', 'title', 'stanza_number', 'is_country',
                'is_pop', 'is_rap', 'is_rb', 'is_rock', 'is_chorus'
            ]]
        )
    
    else:
        # convert data into dict
        X_unlabeled = {
            'lemmatized_stanzas': X_unlabeled['lemmatized_stanzas'],
            'stanza_numbers': X_unlabeled[['stanza_number']],
            'booleans': X_unlabeled[['is_country', 'is_pop', 'is_rap',
                'is_rb', 'is_rock', 'is_chorus']].astype(int).values,
            'titles': X_unlabeled['title']
        }
        
        # preprocess data
        X_unlabeled = preprocessing.preprocess_new_data(
            X_unlabeled, model_folder
        )
        
        # load model
        model = load_model(model_folder + 'model.keras')
        
        # predict
        return pd.Series(model.predict(X_unlabeled))



args = parser.parse_args()

print(classify(
    dataset= args.dataset,
    model_folder= args.model_folder,
    t= args.type
))