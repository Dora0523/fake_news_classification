
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import inf
import itertools
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import warnings
import statistics


"""#Text Classification


"""

import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV

"""## Preprocessing"""

fake_news_train = pd.read_csv("fake_news/fake_news_train.csv",index_col=False)
fake_news_test = pd.read_csv("fake_news/fake_news_test.csv",index_col=False)
fake_news_validation = pd.read_csv("fake_news/fake_news_val.csv",index_col=False)

# no missing value 
fake_news_train

"""### Process text

"""

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    text = re.sub(r'[^\w\s]','', text)
    text = text.lower()
    return text

fake_news_train['text'] = fake_news_train['text'].apply(preprocessor)

x_train = fake_news_train['text']
y_train = fake_news_train['label']
x_test = fake_news_validation['text']
y_test = fake_news_validation['label']
#x_train,x_test,y_train,y_test = train_test_split(fake_news_train['text'], fake_news_train.label, test_size=0.2, random_state=1)

"""## Apply Logistic Regression

### Get Best Regularization Strength C
"""

C_list=[]
val_accuracy_list=[]
train_accuracy_list=[]
for c in [0.01, 0.1, 1, 10, 100]:
  print('--------------------------------')
  print(f"c={c}")
  pipe1 = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression(C=c, max_iter=200))])
  model_lr = pipe1.fit(x_train, y_train)
  predicted = model_lr.predict(x_test)
  validation_accuracy = np.mean(predicted == y_test)
  predicted_train = model_lr.predict(x_train)
  train_accuracy = np.mean(predicted_train == y_train)
  C_list.append(c)
  val_accuracy_list.append(validation_accuracy)
  train_accuracy_list.append(train_accuracy)

  print("Accuracy of Logistic Regression Classifier: {}%".format(round(validation_accuracy*100,2)))
  print("\nConfusion Matrix of Logistic Regression Classifier:\n")
  print(confusion_matrix(y_test, predicted))
  print("\nCLassification Report of Logistic Regression Classifier:\n")
  print(classification_report(y_test, predicted))

def showPlot(c_list, val_accuracy_list, train_accuracy_list):
    plt.plot(c_list, val_accuracy_list, label='Validation Accuracy') 
    plt.plot(c_list, train_accuracy_list, label='Training Accuracy')
    plt.xscale("log")
    plt.xlabel("Inverse Regularization Strength")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

showPlot(C_list, val_accuracy_list, train_accuracy_list)

"""### Test Accuracy"""

X_test = fake_news_test['text']
Y_test = fake_news_test['label']
bestC = C_list[val_accuracy_list.index(max(val_accuracy_list))]
pipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('model', LogisticRegression(C=bestC, max_iter=200))])
model_lr = pipe.fit(x_train, y_train)
lr_pred = model_lr.predict(X_test)
validation_accuracy = np.mean(lr_pred == Y_test)
print(f"Apply Logistic Regression on test dataset with C={bestC}")
print('----------------------------------------------------------')
print("Accuracy of Logistic Regression Classifier: {}%".format(round(validation_accuracy*100,2)))
print("\nConfusion Matrix of Logistic Regression Classifier:\n")
print(confusion_matrix(Y_test, lr_pred))
print("\nCLassification Report of Logistic Regression Classifier:\n")
print(classification_report(Y_test, lr_pred))

"""### Parameter Tuning with Random Search

*   SGD



"""

from sklearn.linear_model import SGDClassifier
text_sgd = Pipeline([
                     ('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier())
                     ])

sgd = text_sgd.fit(x_train, y_train)
predicted_sgd = sgd.predict(x_test)
np.mean(predicted_sgd == y_test)

parameters = {
    'clf__loss':["hinge", "log", "squared_hinge", "modified_huber", "perceptron"],
    'clf__alpha': [1e-7, 1e-5, 1e-3, 1e-2, 1e-1],
    'clf__penalty': ["l2", "l1"],
    }
gs_clf_sgd = RandomizedSearchCV(sgd, parameters, scoring='accuracy', n_jobs=-1, cv=5)
gs_clf_sgd.fit(x_train, y_train)
gs_clf_sgd.predict(x_test)

print('Best Score: %s' % gs_clf_sgd.best_score_)
print('Best Hyperparameters: %s' % gs_clf_sgd.best_params_)

#Best Score: 0.732
#Best Hyperparameters: {'clf__penalty': 'none', 'clf__loss': 'log', 'clf__alpha': 1e-05}

"""* Logistic Regression"""

lg_search = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('lg', LogisticRegression(max_iter=200, random_state=0, warm_start=True))])
parameters = {
        'lg__penalty': ['l1', 'l2'],
        'lg__C':np.logspace(-3,3,20),
        'lg__solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
        }
gs_clf = RandomizedSearchCV(lg_search, parameters, scoring='accuracy', n_jobs=-1, cv=5)
gs_clf = gs_clf.fit(x_train, y_train)
gs_clf.predict(x_test)

print('Best Score: %s' % gs_clf.best_score_)
print('Best Hyperparameters: %s' % gs_clf.best_params_)

#Best score: 0.736, l2, solver=lbfgs, class_weight=none, C=26.3

"""### Test Accuracy with best parameters"""

pipeline_bestParam_sgd = Pipeline([
                     ('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('lg', SGDClassifier(
                                               alpha=gs_clf_sgd.best_params_['clf__alpha'],
                                               loss=gs_clf_sgd.best_params_['clf__loss'],
                                               penalty = gs_clf_sgd.best_params_['clf__penalty'],
                                               ),
                     )
                     ])
model_best_sgd = pipeline_bestParam_sgd.fit(x_train, y_train)
sgd_pred = pipeline_bestParam_sgd.predict(X_test)
validation_accuracy = np.mean(sgd_pred == Y_test)

print(f"Apply SGD on test dataset with best parameters from Random Search")
print('----------------------------------------------------------')
print("Accuracy of SGD Classifier: {}%".format(round(validation_accuracy*100,2)))
print("\nConfusion Matrix of SGD Classifier:\n")
print(confusion_matrix(Y_test, lr_pred))
print("\nCLassification Report of SGD Classifier:\n")
print(classification_report(Y_test, lr_pred))

pipeline_bestParam_lr = Pipeline([
                     ('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('lg', LogisticRegression(
                                               C=gs_clf.best_params_['lg__C'],
                                               solver=gs_clf.best_params_['lg__solver'],
                                               penalty = gs_clf.best_params_['lg__penalty'],
                                               max_iter=500,
                                               random_state=0, 
                                               warm_start=True
                                               ),
                     )
                     ])
model_best_lr = pipeline_bestParam_lr.fit(x_train, y_train)
lr_pred = model_best_lr.predict(X_test)
validation_accuracy = np.mean(lr_pred == Y_test)

print(f"Apply Regression on test dataset with best parameters from Random Search")
print('----------------------------------------------------------')
print("Accuracy of Regression Classifier: {}%".format(round(validation_accuracy*100,2)))
print("\nConfusion Matrix of Logistic Regression Classifier:\n")
print(confusion_matrix(Y_test, lr_pred))
print("\nCLassification Report of Logistic Regression Classifier:\n")
print(classification_report(Y_test, lr_pred))