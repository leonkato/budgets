
# coding: utf-8

# ### Do one model.

# In[1]:


#### Imports/setup

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 60)

from timeit import default_timer as timer

# for the pipeline
from sklearn.pipeline import Pipeline
# for the selectors
from sklearn.preprocessing import FunctionTransformer, StandardScaler
# for gluing preprocessed text and numbers together
from sklearn.pipeline import FeatureUnion
# for nans in the numeric data
from sklearn.preprocessing import Imputer

# Import classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Import CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# metrics
from sklearn.metrics import f1_score, accuracy_score, classification_report

# unflattener
import python.flat_to_labels as ftl

#### Set up a train-test split making sure we have all labels in both splits
from python.multilabel import multilabel_train_test_split

from python.dd_mmll import multi_multi_log_loss, BOX_PLOTS_COLUMN_INDICES


# #### Load the data

# In[2]:


# Get data
the_data = pd.read_csv('data/TrainingData.csv', index_col=0)

# take a look
the_data.head()


# ####  Encode the targets as categorical variables

# In[3]:


### bind variable LABELS - these are actually the targets and we're going to one-hot encode them...
LABELS = ['Function',  'Use',  'Sharing',  'Reporting',  'Student_Type',  'Position_Type', 
          'Object_Type',  'Pre_K',  'Operating_Status']

### This turns out to be key.  Submission requires the dummy versions of these vars to be in this order.
LABELS.sort()

# Define the lambda function: categorize_label
categorize_label = lambda x: x.astype('category')

# Convert df[LABELS] to a categorical type
the_data[LABELS] = the_data[LABELS].apply(categorize_label, axis=0)

# Print the converted dtypes
print(the_data[LABELS].dtypes)


# #### Save the unique labels for each output (category)

# In[4]:


# build a dictionary
the_labels = {col : the_data[col].unique().tolist() for col in the_data[LABELS].columns}
# take a look at one entry
the_labels['Use']


# #### Change fraction to suit.
# Note: small fractions will have a hard time ensuring labels in both splits.

# In[5]:


# downsize it or not
df = the_data.sample(frac=0.98)
# df = the_data


# #### Get targets as set of one-hot encoded columns

# In[6]:


# name these columns
NUMERIC_COLUMNS = ['FTE', 'Total']

# Get labels and convert to dummy variables: label_dummies
label_dummies = pd.get_dummies(df[LABELS])


# #### Setting up a train-test split  for modeling

# #### ======================== Begin Mod1_1_1; add bigrams ===================================

# Some things to note about the default CountVectorizer:
# 1) All strings are downcased
# 2) The default setting selects tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a token separator).  This means single letter or digit tokens are ignored.
# 3) If the vectorizer is used to transform another input (e.g. test), any tokens not in the original corpus are ignored.

# #### One way to work around bug exposed with CountVectorizer/OneVsRest/Logistic is to replace all the numeric values with 0.  The classifiers  should ignore (and might work with n_jobs=-1).
# 
# Yes, this works well and uses all processors yielding the same results as the slower, 1-processor version above.  Fits in 464 sec instead of 827 sec.

# In[7]:


# define combine_text_columns()
def combine_text_columns(df, to_drop=NUMERIC_COLUMNS + LABELS):
    """ converts all text columns in each row of df to single string """
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(df.columns.tolist())
    text_data = df.drop(to_drop, axis=1)  
    # Replace nans with blanks
    text_data.fillna('', inplace=True)    
    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


# In[8]:


# Import FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

# Get the dummy encoding of the labels
dummy_labels = pd.get_dummies(df[LABELS])

# Get the features in the data
NON_LABELS = [c for c in df.columns if c not in LABELS]

# Split into training and test sets
X_train, X_test, y_train, y_test = multilabel_train_test_split(df[NON_LABELS],
                                                               dummy_labels,
                                                               0.2, 
                                                               seed=123)
# Preprocess the text data: get_text_data
get_text_data = FunctionTransformer(combine_text_columns, validate=False)

# Use all 0s instead of noise: get_numeric_data
get_numeric_data_hack = FunctionTransformer(lambda x: np.zeros(x[NUMERIC_COLUMNS].shape, dtype=np.float), validate=False)


# In[ ]:


#### Build the pipeline
mod_1_1_1 = Pipeline([
        ('union', FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([('selector', get_numeric_data_hack),
                                               ('imputer', Imputer())])),
                ('text_features', Pipeline([('selector', get_text_data),
                                            ('vectorizer', CountVectorizer(ngram_range=(1,2)))]))
             ])),
        ('clf', OneVsRestClassifier(LogisticRegression(), n_jobs=-1))
    ])

start = timer()
# Fit to the training data
mod_1_1_1.fit(X_train, y_train)
end = timer()
print('fit time: {:0.2f} seconds'.format(end - start))


# #### For log loss we need the probabilities, not the predicted labels

# In[ ]:


# get probas
start = timer()
mod_1_1_1_yhat_train_probas = mod_1_1_1.predict_proba(X_train)
mod_1_1_1_yhat_test_probas = mod_1_1_1.predict_proba(X_test)
end = timer()
print('Predict.proba time: {:0.2f} seconds'.format(end - start))


# In[ ]:


print('log loss on training set: {:0.4f}'.format(multi_multi_log_loss(mod_1_1_1_yhat_train_probas, 
                                                                      y_train.values, BOX_PLOTS_COLUMN_INDICES)))
print('log loss on training set: {:0.4f}'.format(multi_multi_log_loss(mod_1_1_1_yhat_test_probas, 
                                                                      y_test.values, BOX_PLOTS_COLUMN_INDICES)))


# In[ ]:


def report_f1(true, pred):
    the_scores = []
    for target in range(len(LABELS)):
        the_score = f1_score(true[:, target], pred[:, target], average='weighted')
        print('F1 score for target {}: {:.3f}'.format(LABELS[target], the_score))
        the_scores.append(the_score)
    print('Average F1 score for all targets : {:.3f}'.format(np.mean(the_scores)))

def report_accuracy(true, pred):
    the_scores = []
    for target in range(len(LABELS)):
        the_score = accuracy_score(true[:, target], pred[:, target])
        print('Accuracy score for target {}: {:.3f}'.format(LABELS[target], the_score))
        the_scores.append(the_score)
    print('Average accuracy score for all targets : {:.3f}'.format(np.mean(the_scores)))


# In[ ]:


# ftl wants ndarray, not pd.Dataframe
the_ys = ftl.flat_to_labels(y_test.values)


# In[ ]:


report_f1(the_ys, ftl.flat_to_labels(mod_1_1_1_yhat_test_probas))

report_accuracy(the_ys, ftl.flat_to_labels(mod_1_1_1_yhat_test_probas))


# #### =============================== End of mod_1_1_1 ============================================

# ***
