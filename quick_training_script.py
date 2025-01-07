from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib
from sklearn.metrics import classification_report



path = '/Users/brunobarbieri/Library/CloudStorage/OneDrive-UniversityofPisa/TA_Project/data/'
df = pd.read_csv(path + "lab_lem_merge.csv")

import ast

df['lemmatized_stanzas'] = df['lemmatized_stanzas'].apply(ast.literal_eval)

from sklearn.feature_extraction.text import TfidfVectorizer
# import numpy as np


# Step 1: Convert token lists back into space-separated strings
# (needed for vectorizer)
# texts_str = df['lemmatized_stanzas'].apply(
#     lambda tokens: " ".join(tokens)
# )
# print(texts_str)
df['text_str'] = df['lemmatized_stanzas'].apply(lambda x: ' '.join(x))


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[[
        'text_str', 'title', 'stanza_number', 'is_country',
        'is_pop', 'is_rap', 'is_rb', 'is_rock', 'is_chorus'
    ]],
    df['label'], test_size=0.3, random_state=42
)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    df[[
        'text_str', 'title', 'stanza_number', 'is_country',
        'is_pop', 'is_rap', 'is_rb', 'is_rock', 'is_chorus'
    ]],
    df['label'], test_size=0.3, random_state=42
)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer

def convert_bool_to_int(x):
    return x.astype(int)

preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'text_str'),
        ('title_topic', TfidfVectorizer(), 'title'),
        ('scaler', StandardScaler(), ['stanza_number']),
        (
            'bools', FunctionTransformer(
                convert_bool_to_int, validate=False
            ), [
                'is_country', 'is_pop', 'is_rap',
                'is_rb', 'is_rock', 'is_chorus'
            ]
        )

])



model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=1,
        max_depth=None,
        bootstrap=False,
        random_state=42
    ))
])

model.fit(X_train, y_train)

joblib.dump(model, 'final/models/static_models/rf_model.joblib')

# print(f"Best Parameters: {model.best_params_}")
# print(f"Best Cross-Validation Accuracy: {model.best_score_}")

# Predict on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))


print('done')