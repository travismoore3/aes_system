import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble, metrics
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt

# Load the data set
open = pd.read_table('/Users/travismoore/Documents/workspace/AES system/AES system/Feature extractors/output_file.tsv', encoding='utf-8')

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

# del features_df['grammar_error_ACTUAL_EXPERIENCE']
# del features_df['grammar_error_THEM_SELVES']
# del features_df['grammar_error_MAY_COULD_POSSIBLY']
# del features_df['grammar_error_MAY_BE']
# del features_df['grammar_error_THERE_RE_MANY']
# del features_df['grammar_error_MANY_FEW_UNCOUNTABLE']
# del features_df['grammar_error_LOTS_OF_NN']
# del features_df['grammar_error_THERE_S_MANY']
# del features_df['grammar_error_LIFE_TIME']
# del features_df['grammar_error_LESS_DOLLARSMINUTESHOURS']
# del features_df['grammar_error_LESS_COMPARATIVE']
# del features_df['grammar_error_LARGE_NUMBER_OF']
# del features_df['grammar_error_KNOW_NOW']
# del features_df['grammar_error_KIND_OF_A']
# del features_df['grammar_error_THESE_ONES']
# del features_df['grammar_error_I_AM']
# del features_df['grammar_error_YOUR_S']
# del features_df['grammar_error_IT_IS']
# del features_df['grammar_error_IS_WERE']
# del features_df['grammar_error_IS_SHOULD']
# del features_df['grammar_error_IN_THE_MOMENT']
# del features_df['grammar_error_IN_PRINCIPAL']
# del features_df['grammar_error_IN_PAST']
# del features_df['grammar_error_IN_NOWADAYS']
# del features_df['grammar_error_IN_A_X_MANNER']
# del features_df['grammar_error_ILL_I_LL']
# del features_df['grammar_error_THE_SUPERLATIVE']
# del features_df['grammar_error_HE_THE']
# del features_df['grammar_error_MOST_COMPARATIVE']
# del features_df['grammar_error_HERE_HEAR']
# del features_df['grammar_error_MOST_SOME_OF_NNS']
# del features_df['grammar_error_MUCH_COUNTABLE']
# del features_df['grammar_error_SENTENCE_WHITESPACE']
# del features_df['grammar_error_SOME_OF_THE']
# del features_df['grammar_error_SAY_TELL']
# del features_df['grammar_error_REASON_IS_BECAUSE']
# del features_df['grammar_error_RATHER_THEN']
# del features_df['grammar_error_SO_AS_TO']
# del features_df['grammar_error_PROGRESSIVE_VERBS']
# del features_df['grammar_error_PREFER_TO_VBG']
# del features_df['grammar_error_POSSESSIVE_APOSTROPHE']
# del features_df['grammar_error_PM_IN_THE_EVENING']
# del features_df['grammar_error_PHRASE_REPETITION']
# del features_df['grammar_error_PERS_PRONOUN_AGREEMENT_SENT_START']
# del features_df['grammar_error_PERSONAL_OPINION_FRIENDSHIP']
# del features_df['grammar_error_PERIOD_OF_TIME']
# del features_df['grammar_error_PAST_EXPERIENCE_MEMORY']
# del features_df['grammar_error_OUT_COME']
# del features_df['grammar_error_ONES']
# del features_df['grammar_error_SUPERIOR_THAN']
# del features_df['grammar_error_NOW_A_DAYS']
# del features_df['grammar_error_NOW']
# del features_df['grammar_error_NOUN_AROUND_IT']
# del features_df['grammar_error_NOT_NOTHING']
# del features_df['grammar_error_NON_ACTION_CONTINUOUS']
# del features_df['grammar_error_NON3PRS_VERB']
# del features_df['grammar_error_NODT_DOZEN']
# del features_df['grammar_error_NEEDS_FIXED']
# del features_df['grammar_error_SUPPOSE_TO']
# del features_df['grammar_error_MOST_SUPERLATIVE']
# del features_df['grammar_error_SOME_NN_VBP']
# del features_df['grammar_error_HELP_TO_FIND']
# del features_df['grammar_error_THIS_NNS']
# del features_df['grammar_error_BELIEF_BELIEVE']
# del features_df['grammar_error_WHO_NOUN']
# del features_df['grammar_error_WITH_OUT']
# del features_df['grammar_error_A_RB_NN']
# del features_df['grammar_error_A_PLURAL']
# del features_df['grammar_error_A_MUCH_NN1']
# del features_df['grammar_error_WOMAN_WOMEN']
# del features_df['grammar_error_WORLD_WIDE']
# del features_df['grammar_error_A_HUNDREDS']
# del features_df['grammar_error_AS_ADJ_AS']
# del features_df['grammar_error_APOS_ARE']
# del features_df['grammar_error_APOSTROPHE_PLURAL']
# del features_df['grammar_error_AN_OTHER']
# del features_df['grammar_error_ANY_MORE']
# del features_df['grammar_error_ANY_BODY']
# del features_df['grammar_error_AND_ETC']
# del features_df['grammar_error_ALSO_SENT_END']
# del features_df['grammar_error_ALL_OF_THE']
# del features_df['grammar_error_ALLTHOUGH']
# del features_df['grammar_error_WORRY_FOR']
# del features_df['grammar_error_ALLOT_OF']
# del features_df['grammar_error_AGREEMENT_SENT_START']
# del features_df['grammar_error_AFFORD_VBG']
# del features_df['grammar_error_YOUR']
# del features_df['grammar_error_AFFECT_EFFECT']
# del features_df['grammar_error_ADVISE_VBG']
# del features_df['grammar_error_ADMIT_ENJOY_VB']
# del features_df['grammar_error_BE_USE_TO_DO']
# del features_df['grammar_error_HELL']
# del features_df['grammar_error_BORED_OF']
# del features_df['grammar_error_WHITESPACE_RULE']
# del features_df['grammar_error_HAVE_CD_YEARS']
# del features_df['grammar_error_HASNT_IRREGULAR_VERB']
# del features_df['grammar_error_GOING_TO_VBD']
# del features_df['grammar_error_GIVE_ADVISE']
# del features_df['grammar_error_GENERAL_XX']
# del features_df['grammar_error_FEWER_LESS']
# del features_df['grammar_error_EVERY_WHERE']
# del features_df['grammar_error_THROUGH_OUT']
# del features_df['grammar_error_TODAY_MORNING']
# del features_df['grammar_error_TOO_TO']
# del features_df['grammar_error_TO_NON_BASE']
# del features_df['grammar_error_EN_COMPOUNDS']
# del features_df['grammar_error_EN_A_VS_AN']
# del features_df['grammar_error_CALENDER']
# del features_df['grammar_error_ENGLISH_WORD_REPEAT_BEGINNING_RULE']
# del features_df['grammar_error_EACH_EVERY_NNS']
# del features_df['grammar_error_DT_PRP']
# del features_df['grammar_error_DT_JJ_NO_NOUN']
# del features_df['grammar_error_USE_TO_VERB']
# del features_df['grammar_error_DONT_NEEDS']
# del features_df['grammar_error_DOES_NP_VBZ']
# del features_df['grammar_error_CURRENCY']
# del features_df['grammar_error_CONFUSION_OF_OUR_OUT']
# del features_df['grammar_error_COMP_THAN']
# del features_df['grammar_error_WANT_THAT_I']
# del features_df['grammar_error_CD_NN']
# del features_df['grammar_error_CD_DOZENS_OF']
# del features_df['grammar_error_ECONOMIC_ECONOMICAL']
# del features_df['grammar_error_SHOULD_BE_DO']
# del features_df['grammar_error_ADVERB_WORD_ORDER']
# del features_df['grammar_error_MANY_NN']
# del features_df['grammar_error_YOU_THING']
# del features_df['grammar_error_EN_UNPAIRED_BRACKETS']
# del features_df['grammar_error_A_UNCOUNTABLE']
# del features_df['grammar_error_EVERYDAY_EVERY_DAY']
# del features_df['grammar_error_A_LOT_OF_NN']
# del features_df['grammar_error_AFFORD_VB']
# del features_df['grammar_error_MUST_HAVE_TO']
# del features_df['grammar_error_PRP_RB_NO_VB']
# del features_df['grammar_error_BEEN_PART_AGREEMENT']
# del features_df['grammar_error_SENTENCE_FRAGMENT']
# del features_df['grammar_error_IT_VBZ']
# del features_df['grammar_error_HAVE_PART_AGREEMENT']
# del features_df['grammar_error_EN_QUOTES']
# del features_df['grammar_error_ALLOW_TO']
# del features_df['grammar_error_HE_VERB_AGR']
# del features_df['grammar_error_MORFOLOGIK_RULE_EN_US']
# del features_df['grammar_error_SENT_START_CONJUNCTIVE_LINKING_ADVERB_COMMA']
# del features_df['grammar_error_COMMA_PARENTHESIS_WHITESPACE']
# del features_df['grammar_error_DT_DT']
# del features_df['grammar_error_CANT']
# del features_df['grammar_error_LITTLE_BIT']
# del features_df['grammar_error_MASS_AGREEMENT']
# del features_df['grammar_error_A_INFINITVE']
# del features_df['grammar_error_NUMEROUS_DIFFERENT']
# del features_df['grammar_error_EN_CONTRACTION_SPELLING']
# del features_df['grammar_error_UPPERCASE_SENTENCE_START']
# del features_df['grammar_error_I_LOWERCASE']
 
# Remove the score from the feature data
del features_df['Score']

print(list(features_df))

# Create the X and y arrays
X = features_df.as_matrix()
y = df['Score'].as_matrix()
             
# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

scatter_x = []
scatter_y = []
    
# # K-fold split - unfortunately useless unless data is in order of semesters; 12 splits is currently the best
# kf = KFold(n_splits=12, shuffle=True)
#         
# for train_index, test_index in kf.split(X):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
             
# Fit regression model
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
# print(one_pt_std)
print("Pct exact:", round(len(exact_std)*100/len(happy), 2))
print("Within 0.5:", round(len(half_pt_std)*100/len(happy), 2))
print("Within 1.0:", round(len(one_pt_std)*100/len(happy), 2))
    
# r2 is the proportion of the variance in the scores that is predictable from the features
artoo = metrics.r2_score(y_train, model.predict(X_train))
print("Train Set coefficient of determination: %.4f" % artoo)
      
artoo = metrics.r2_score(y_test, model.predict(X_test))
print("Test Set coefficient of determination: %.4f" % artoo)

# # Visualizations
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
