import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

from preprocessing import preprocess_labels

def downsample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs downsampling to balance the class distribution in the
    input DataFrame.

    This function groups the input DataFrame by the 'label' column and
    downsamples each group to match the size of the smallest class.
    The resulting downsampled DataFrame contains an equal number of
    samples for each label, which helps in mitigating the class
    imbalance problem during model training.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data to be
                           downsampled. 

    Returns:
        pd.DataFrame: A downsampled pandas DataFrame with balanced class
                      distribution.
    """

    min_size = df['label'].value_counts().min()

    downsampled_samples = []
    for _, group in df.groupby('label'):
        sampled_group = resample(
            group, replace=False, n_samples=min_size, random_state=42
        )
        downsampled_samples.append(sampled_group)

    # could reuse for testing but seems problematic as pollutes the test set
    # cut_df = df[~df.isin(pd.concat(downsampled_samples)).all(axis=1)]

    return pd.concat(downsampled_samples)


def train_test_val_split(df: pd.DataFrame) -> dict:
    """
    Splits the input DataFrame into training, validation, and test sets.


    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data.
    Returns:
        dict: A dictionary containing three splits (train, val, test). 
              Each split is represented as a dictionary with the same structure.
    """

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
        preprocess_labels(df['label']), test_size=0.3,
        random_state= 42
    )

    # validation split
    (
        lemmatized_stanzas_train, lemmatized_stanzas_val,
        stanza_numbers_train, stanza_numbers_val,
        booleans_train, booleans_val,
        titles_train, titles_val,
        y_train, y_val
    ) = train_test_split(
        lemmatized_stanzas_train, stanza_numbers_train,
        booleans_train, titles_train,
        y_train, test_size=0.3,
        random_state= 42
    )

    return {
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
    

def get_data_as_list (data: dict) -> list:
    """
    Converts processed data into a list of features for model input.

    Args:
        data (dict): A dictionary containing the data.

    Returns:
        list: A list containing the same data as the dictionary,
              refactored into a more manageable format.
    """
    return [
        data['padded_sequences'],
        data['stanza_numbers'],
        data['topic_distributions']
    ] + list(data['booleans'].T)