import pandas as pd
import numpy as np
import joblib
import ast
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FunctionTransformer, Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from graphs import confusion_matrix_graph, plot_class_wise_accuracy

# Reading the dataframe
path = ".\\data"
df = pd.read_csv(path + "\\lab_lem_merge.csv")

df['lemmatized_stanzas'] = df['lemmatized_stanzas'].apply(ast.literal_eval)

# Visualizing the dataframe
df

# Converting the token lists back into space-separated strings
# (needed for vectorizer)

df['text_str'] = df['lemmatized_stanzas'].apply(lambda x: ' '.join(x))

# Splitting protocol
X_train, X_test, y_train, y_test = train_test_split(
    df[[
        'text_str', 'title', 'stanza_number', 'is_country',
        'is_pop', 'is_rap', 'is_rb', 'is_rock', 'is_chorus'
    ]],
    df['label'], test_size=0.3, random_state=42
)

# Converting the boolean array into an integer object
def convert_bool_to_int(x):
    return x.astype(int)

"""
Convert a boolean array to integers.bjects:

   Parameters:
   - x: array-like
     An array of boolean values to be converted to integers.

   Returns:
   - array-like
     An array of integers (0 or 1) corresponding to the input booleans.

"""

# Defining the preprocessor

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

"""
The preprocessor is an element `ColumnTransformer` that preprocesses the various types of data we are working with:
   - Text columns are transformed using `TfidfVectorizer`.
   - Numerical columns are scaled using `StandardScaler`.
   - Boolean columns are converted to integers using a custom transformer.

Preprocessing steps:
- Text data:
  - The `text_str` column is transformed using `TfidfVectorizer`.
  - The `title` column is also transformed using `TfidfVectorizer`.
- Numerical data:
  - The `stanza_number` column is scaled using `StandardScaler`.
- Boolean data:
  - Columns `is_country`, `is_pop`, `is_rap`, `is_rb`, `is_rock`, and `is_chorus` 
    are converted to integers (0 or 1) using the custom `convert_bool_to_int` function 
    within a `FunctionTransformer`.

"""

### RANDOM FOREST ###

# Define the pipeline
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_distributions = {
    'preprocessor__text__max_features': [500, 1000, 5000, None],  # Max features for TF-IDF
    'preprocessor__text__ngram_range': [(1, 1), (1, 2)],          # Unigrams or bigrams
    'classifier__n_estimators': [50, 100, 200, 300],              # Number of trees
    'classifier__max_depth': [None, 10, 20, 30],                  # Tree depth
    'classifier__min_samples_split': [2, 5, 10],                  # Min samples per split
    'classifier__min_samples_leaf': [1, 2, 4],                    # Min samples per leaf
    'classifier__bootstrap': [True, False],                       # Bootstrap sampling
}

# RandomizedSearchCV setup
random_search_rf = RandomizedSearchCV(
    estimator= rf_pipeline,
    param_distributions= rf_param_distributions,
    n_iter=20,                                  # Number of random combinations to try
    cv=5,                                       # 5-fold cross-validation
    scoring='accuracy',                         # Metric to optimize
    verbose=2,
    random_state=42,
    n_jobs=-1                                   # Use all available cores
)

# Fit RandomizedSearchCV to the data
#random_search_rf.fit(X_train, y_train)

# Best parameters and cross-validation accuracy
# print(f"Best Parameters: {random_search_rf.best_params_}")
# print(f"Best Cross-Validation Accuracy: {random_search_rf.best_score_}")

# The model was trained, saved and loaded for further inspection using joblib

model_path = '/Users/brunobarbieri/Library/CloudStorage/OneDrive-UniversityofPisa/TA_Project/models/'

'''
import joblib
joblib.dump(
    random_search_rf.best_estimator_,
    model_path + 'RF_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + '.pkl'
)
'''

rf_path = "C:\\Users\\cinna\\Desktop\\static_models\\random_forest.pkl"
rf_loaded = joblib.load(rf_path)

# Predict on the test set
y_pred_rf = rf_loaded.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Binarization of the labels for the multi-class support
classes = df['label'].unique()
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)

# Probabilities of the model
y_score = rf_loaded.predict_proba(X_test)

# Plotting the confusion matrix
folder_path = "C:\\Users\\cinna\\Desktop\\progetto_TA\\Immagini"
classes_list = classes.tolist()

# Plotting the confusion matrix
confusion_matrix_graph(y_test, y_pred_rf, classes = classes_list, folder_path = folder_path)

# Plotting the class-wise accuracy
plot_class_wise_accuracy(y_test, y_pred_rf, classes = classes_list, folder_path= folder_path)


### SUPPORT VECTOR MACHINE ###
from sklearn.svm import SVC


svm_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))
])

# svm_param_distributions = {
#     'preprocessor__text__max_features': [500, 1000, 5000, None],
#     'preprocessor__text__ngram_range': [(1, 1), (1, 2)],
#     'classifier__estimator__C': [0.1, 1, 10, 100],
#     'classifier__estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#     'classifier__estimator__degree': [2, 3, 4],
#     'classifier__estimator__gamma': ['scale', 'auto'],
#     'classifier__estimator__class_weight': [None, 'balanced']
# }

svm_param_distributions = {
    'preprocessor__text__max_features': [500, 1000, 2500],
    'preprocessor__text__ngram_range': [(1, 1), (1, 2)],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__gamma': ['scale', 'auto'],
    'classifier__class_weight': [None, 'balanced']
}



# RandomizedSearchCV setup
random_search_svm = RandomizedSearchCV(
    estimator= svm_pipeline,
    param_distributions= svm_param_distributions,
    n_iter=2,
    cv=3,
    scoring='accuracy',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit RandomizedSearchCV to the data
#random_search_svm.fit(X_train, y_train)

# Best parameters and cross-validation accuracy
#print(f"Best Parameters: {random_search_svm.best_params_}")
#print(f"Best Cross-Validation Accuracy: {random_search_svm.best_score_}")

# Saving the models after training
'''
joblib.dump(
    random_search_svm.best_estimator_,
    model_path + "\\svc_model2.joblib"
)
'''
# Again loading the previously trained model for further inspection
SVC_loaded = joblib.load("C:\\Users\\cinna\\Desktop\\static_models\\svc_model2.joblib")

# Predict on the test set
y_pred_SVC = SVC_loaded.predict(X_test)

print(classification_report(y_test, y_pred_SVC))

# Ensure the `classes` are sorted consistently for reproducibility
classes = sorted(df['label'].unique())

# Binarize labels for multi-class support
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)

# Compute the decision function scores from the SVM model
y_score_SVC = SVC_loaded.predict_proba(X_test)

# Plotting the confusion matrix
confusion_matrix_graph(y_test, y_pred_SVC, classes= classes, folder_path= folder_path)

# Plotting the class-wise accuracy for SVC
plot_class_wise_accuracy(y_test, y_pred_SVC, classes= classes, folder_path= "C:\\Users\\cinna\\Desktop\\progetto_TA\\Immagini")
