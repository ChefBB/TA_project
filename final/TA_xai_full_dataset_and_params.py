
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.svm import SVC
from sklearn.tree import plot_tree
from sklearn.multiclass import OneVsOneClassifier

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from scikitplot.metrics import plot_roc
from sklearn.preprocessing import label_binarize, LabelEncoder

from lime.lime_text import LimeTextExplainer

df = pd.read_csv(r".\\models\\data\\lab_lem_merge.csv", skipinitialspace=True)

import ast

df['lemmatized_stanzas'] = df['lemmatized_stanzas'].apply(ast.literal_eval)

df['text_str'] = df['lemmatized_stanzas'].apply(lambda x: ' '.join(x))
X = df['text_str']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['label'])


np.unique(df['label'])

class_names = np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False)


vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)



### SVC

svc = SVC(C=1, class_weight='balanced', gamma='scale', kernel='linear', probability=True, random_state=42)

svc.fit(train_vectors, y_train)

y_pred = svc.predict(test_vectors)

print('Accuracy %s' % accuracy_score(y_test, y_pred))
print('F1-score %s' % f1_score(y_test, y_pred, average=None))
print(classification_report(y_test, y_pred))


# Save the trained SVC model
from joblib import dump, load
import os
model_file_path = r".\\data\\\svc_model.joblib"
dump(svc, model_file_path)
print(f"Model saved to {os.path.abspath(model_file_path)}")


# Binarization of labels for multi-class support
classes = np.unique(y)
y_train_bin = label_binarize(y_train, classes=classes)
y_test_bin = label_binarize(y_test, classes=classes)

# Computation of model probabilities
y_score = svc.decision_function(test_vectors)

# ROC curve plots for each class
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(classes):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {class_name} (AUC = {roc_auc:.2f})")

# Add the reference line (y=x)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Plot configuration
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Multi-Class SVM (One-vs-One)")
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


### EXPLAINABILITY WITH LIME

from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


def explain_x(vectorizer, bb, class_names, idx):
    """
    Generate and display a LIME explanation for a specific instance in a test dataset.

    Parameters:
    -----------
    vectorizer : 
        The text vectorizer used to transform raw text into feature vectors.

    bb : 
        The black-box model (e.g., classifier) whose predictions are being explained.

    class_names : list of str
        List of class labels corresponding to the model's output classes.

    idx : int
        Index of the instance in the test dataset to explain.

    Returns:
    --------
    None
        The function generates an explanation using LIME, prints information about the 
        instance, and displays the explanation in a notebook environment.
    """
    c = make_pipeline(vectorizer, bb)
    explainer = LimeTextExplainer(class_names=class_names, bow=True) # initialize a LIME explainer

    element = test_vectors[idx, :] # feature vector for the test instance at the given index

    # Generate an explanation for the instance in X_test at the given index
    exp = explainer.explain_instance(
        X_test.iloc[idx], # raw text data for the instance
        c.predict_proba, # prediction probability function
        num_features=20, # number of features to include in the explanation
        top_labels=1)    # how many top labels to explain, just one for readability
    
    print('Document id: %d' % idx)
    print('Predicted class =', class_names[bb.predict(element).reshape(1, -1)[0, 0]])
    print('True class: %s' % class_names[y_test[idx]])

    exp.show_in_notebook(text=X_test.iloc[idx])
    return


# The left part of the graphS below represent the predicted probability of the model for each class, in the center part
# there are the feature importances from the most to the least relevant one, divided in two groups; on the right
# the ones with positive influence on the prediction of the label, on the left the ones with negative influence that
# suggest the model to chose other classes, and on the right part of the graph the value of the most important
# features, with the bright color ones have positive influence.

explain_x(vectorizer, svc, class_names, 0) # predicted fear, but true class was sadness

explain_x(vectorizer, svc, class_names, 1) # predicted sadness, but true class was disgust

explain_x(vectorizer, svc, class_names, 2) # predicted fear, but true class was sadness

explain_x(vectorizer, svc, class_names, 3) # predicted anticipation, but true class was joy

explain_x(vectorizer, svc, class_names, 4) # predicted joy, but true class was sadness
# here actually 'smile' make people think of a happy song, but the correct label was sadness

explain_x(vectorizer, svc, class_names, 5) # # predicted anger, but true class was joy

explain_x(vectorizer, svc, class_names, 6) # predicted joy, but true class was surprise

explain_x(vectorizer, svc, class_names, 7) # predicted trust, but true class was fear
# this is plausible as both trust and fear

explain_x(vectorizer, svc, class_names, 8) # predicted joy, and got it right!
# indeed love and flame are associated with joy

explain_x(vectorizer, svc, class_names, 9) # predicted fear, but true class was joy

explain_x(vectorizer, svc, class_names, 10) # predicted trust, but true class was joy
# it is plausible that trust and joy are confused

explain_x(vectorizer, svc, class_names, 11) # predicted sadness, but true class was joy

