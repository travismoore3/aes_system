import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

## Load the output file saved by feature_extractor.py
open = pd.read_table('/Users/output_file.tsv', encoding='utf-8')

df = open.fillna(0)
 
# Remove the fields from the data set that we don't want to include in our model
del df['Sex']
del df['Age']
del df['ID']
del df['Essay']
del df['Level']
del df['Language']
# del df['prn_density']
# del df['prn_noun_ratio']
# del df['neg_usage']
# del df['s3a']
# del df['s2c']
# del df['s1b']
# del df['s2a']
# del df['s1c']
# del df['s1a']
# del df['s2b']
# del df['s1']
# del df['s4']
# del df['s2']
# del df['s4b']
# del df['s3c']
# del df['s4a']
# del df['s3b']
# del df['s3']
# del df['s4c']
# del df['conjunctions']
# del df['determiners']
# del df['n_bigram_lemma_types']
# del df['avg_len_word']
# del df['num_types']
# del df['num_tokens']
# del df['ntypes_low_np']
# del df['ncontent_types']
# del df['ncontent_tokens']
# del df['nfunction_types']
# del df['nfunction_tokens']
# del df['sent_density']
# del df['ttr']
# del df['english_usage']
# del df['pct_rel_trigrams']
# del df['fre']
# del df['fkg']
# del df['cli']
# del df['ari']
# del df['dcrs']
# del df['dw']
# del df['lwf']
# del df['gf']
# del df['pct_transitions']
# del df['grammar_chk']
# del df['n_trigram_lemma_types']
# del df['nlemma_types']
# del df['nlemmas']
# del df['n_bigram_lemmas']
# del df['n_trigram_lemmas']
# del df['ncontent_words']
# del df['noun_ttr']
# del df['nfunction_words']
# del df['function_ttr']
del df['transition_word']
del df['grammar_error']

# Replace categorical data with one-hot encoded data
# features_df = pd.get_dummies(df, columns=['grammar_error', 'transition_word'])
features_df = pd.get_dummies(df)
 
# Remove the score from the feature data
del features_df['Score']

print(list(features_df))

# Create the X and y arrays
X = features_df.as_matrix()
y = df['Score'].as_matrix()
             
## Split the data set in a training set (80%) and a test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scatter_x = []
scatter_y = []
    
## K-fold split (just another option)
# kf = KFold(n_splits=12, shuffle=True)
#         
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
             
## Fit regression model

model = ensemble.GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=6,
    min_samples_leaf=17,
    max_features=0.1,
    loss='ls',
    random_state=0
)
model.fit(X_train, y_train)
             
# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_essay_scoring_model.pkl')

## Evaluations
# Find the error rate on the training and test sets
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

predicted_x = model.predict(X_test)

mse1 = mean_absolute_error(y_test, predicted_x)
print("Test Set Mean Absolute Error: %.4f" % mse1)

def mape_vectorized_v2(a, b): 
    mask = a != 0
    return (np.fabs(a[mask] - b[mask])/a[mask]).mean()

print("MAPE:", mape_vectorized_v2(y_test, predicted_x).round(2)*100)
                  
# Percent within 0.5 and 1.0

exact_std = []
half_pt_std = []
one_pt_std = []
happy = abs(predicted_x - y_test)

for error in happy:
    if error < .099:
        exact_std.append(error)
    if error < .544:
        half_pt_std.append(error)
    if error < 1.044:
        one_pt_std.append(error)

print("Pct exact:", round(len(exact_std)*100/len(happy), 2))
print("Within 0.5:", round(len(half_pt_std)*100/len(happy), 2))
print("Within 1.0:", round(len(one_pt_std)*100/len(happy), 2))
    
# r2 is the proportion of the variance in the scores that is predictable from the features
artoo = metrics.r2_score(y_train, model.predict(X_train))
print("Train Set coefficient of determination: %.4f" % artoo)
      
artoo = metrics.r2_score(y_test, model.predict(X_test))
print("Test Set coefficient of determination: %.4f" % artoo)

## Visualizations
#                 
# scatter_x.extend(list(predicted_x))
# scatter_y.extend(list(y_test))
#                 
# plt.scatter(scatter_x, scatter_y, color='b', s=30, alpha=.4)
# # plt.title("Current results", fontsize=14, fontweight='bold')
# plt.xlabel("Predicted score")
# plt.ylabel("Actual score")
#               
# plt.show()
#               
# plt.boxplot([y_test, predicted_x])
# # plt.title("Current results", fontsize=14, fontweight='bold')
# plt.xlabel("Actual scores                                   Predicted scores")
# plt.ylabel("Score")
# x1, x2, y1, y2 = plt.axis()
# plt.axis([x1,x2,0,9]) 
#               
# plt.show()
