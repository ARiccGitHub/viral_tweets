'''
###################################################################################
#                                  Viral Tweets                                   #
#               Twitter Classification Cumulative Project Part-1                  #
###################################################################################

My Codecademy Challenging Part-1 Project From The Data Scientist Path Foundations of Machine Learning:
Supervised Learning Course, Advance Classification Models Section.

Overview
In this project, Twitter Classification Cumulative Project,
I use real tweets to find patterns in the way people use social media. There are two parts to this project:

Part-1: Viral Tweets (This Section)
Part-2: Classifying Tweets, using Naive Bayes classifiers to predict whether a tweet was sent from New York City, London, or Paris.

+ Project Goal
Viral Tweets, using a K-Nearest Neighbor classifier to predict whether or not a tweet will go viral.

+ Project Requirements

    Be familiar with:
        -Python3
        -Machine Learning: Supervised Learning
        -The Python Libraries:
            Pandas
            NumPy
            Matplotlib
            SKlearn

'''
#
####################################################################################  Libraries
#
# Data manipulation tool
import pandas as pd
# Scientific computing, array
import numpy as np
# Data visualization
from matplotlib import pyplot as plt
# Theme to use with matplotlib
from jupyterthemes import jtplot
jtplot.style(theme='chesterish')
# Data scaler
from sklearn.preprocessing import scale
# Data splitter
from sklearn.model_selection import train_test_split
# K-Nearest Neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
# Model evaluation scores
from sklearn.metrics import accuracy_score, recall_score, precision_score
# Model Calibration
from sklearn.calibration import CalibratedClassifierCV
# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
# ------------ Import Files
from best_k_value import best_k_value
#
###################################################################################  Exploring the data
#
all_tweets = pd.read_json("data_json/random_tweets.json", lines=True)
print(f'\nNumber of tweets in the data: {len(all_tweets)}\n')
# Features and features' data type
features_type = all_tweets[all_tweets.columns].dtypes.to_frame().rename(columns={0:'dtype'})
print(f'--------- Tweets\' Features Dtype\n\n{features_type}')
# Some of the features are objects, for example "user", let's explore and find out what kind of object those features are
features_type['type'] = [type(all_tweets.loc[0][col]).__name__ for col in all_tweets.columns]
print(f'\n--------- Tweets\' Features with Dtype and type\n\n{features_type}')
# "user" dictionary sample.
print(f"\n--------- 'user' dictionary sample.\n\n{pd.Series(all_tweets.loc[0]['user'])}")
# "retweeted_status" dictionary sample
print(f"\n--------- 'retweeted_status' dictionary sample.\n\n{pd.Series(all_tweets.loc[5584]['retweeted_status'])}")
#
#################################################################################### Defining Viral Tweets¶
#
#-------------------------------------------- Exploring the feature "retweet_count"
#
# The highest "retweet_count" value
print(f"\n\n--------- The highest 'retweet_count' value: {all_tweets['retweet_count'].max()}\n")
# The tweet with the highest "retweet_count" value
print("\n--------- The tweet with the highest 'retweet_count' value\n\n")
print(all_tweets.loc[all_tweets['retweet_count'] == all_tweets['retweet_count'].max()].T)
# Average "retweet_count"
avg_retweet_count = all_tweets['retweet_count'].median()
print(f"\n\n--------- Average 'retweet_count': {avg_retweet_count}\n")
#
# ------------------------------------------  "retweet_count" scatter plots
#
plt.figure(figsize=(5, 5))
ax = plt.subplot()
plt.scatter(range(len(all_tweets)), all_tweets['retweet_count'])
# Average line
ax.plot(range(len(all_tweets)), [avg_retweet_count for i in range(len(all_tweets))], linewidth=2, color="red")
# Labels
plt.tick_params(labelbottom=False) # labels along the bottom edge are off
plt.title('Tweets')
plt.ylabel('Retweeted Count')
plt.legend(['Average retweet_count'], loc=[0.48, 0.8], prop={'size': 10})

plt.savefig('graph/retweet_count.png')
plt.show()
#
# The highest "retweet_count" value is an outlier,
# --------- let's remove it from the graph and try to improve the readability of the average line.
plt.figure(figsize=(5, 10))
plt.scatter(range(len(all_tweets)), all_tweets['retweet_count'])
plt.yticks(np.arange(0, all_tweets['retweet_count'].max() + 10000, step=5000))
# Removes the highest retweet_count value outlier, set y axis bottom to -1 and upper to 180000
plt.ylim(-5000 ,180000)
# Average line
plt.plot(range(len(all_tweets)), [avg_retweet_count for i in range(len(all_tweets))], linewidth=2, color="red")
# Labels
plt.legend(['Average retweet_count'], prop={'size': 10}, loc='upper left')
plt.tick_params(labelbottom=False) # labels along the bottom edge are off
plt.title('Tweets')
plt.ylabel('Retweeted Count')
plt.tick_params(labelsize=9.5)

plt.savefig('graph/retweet_count_no_outliers.png')
plt.show()
#
#  --------- extra benchmark at the 10'000 retweets value count.
plt.figure(figsize=(5, 10))
plt.scatter(range(len(all_tweets)), all_tweets['retweet_count'])
plt.yticks(np.arange(0, all_tweets['retweet_count'].max() + 10000, step=5000))
# Removes the highest retweet_count value outlier, set y axis bottom to -1 and upper to 180000
plt.ylim(-5000 ,180000)
# Average line
plt.plot(range(len(all_tweets)), [avg_retweet_count for i in range(len(all_tweets))], linewidth=2, color="red")
# Benchmark line
plt.plot(range(len(all_tweets)), [10000 for i in range(len(all_tweets))], linewidth=2, color="cyan")
# Labels
plt.legend(['Average retweet_count', 'Benchmark viral retweet'], loc='upper left', prop={'size': 10})
plt.tick_params(labelbottom=False) # labels along the bottom edge are off
plt.title('Tweets')
plt.ylabel('Retweeted Count')
plt.tick_params(labelsize=9.5)

plt.savefig('graph/retweet_count_benchmark.png')
plt.show()
#
################################################################################### Viral retweets' label
#
# Viral retweets' classes:
#
#     - 0 stand for
#     it is Not a viral retweet class
#
#     - The 1 stand for
#     it is a viral retweet class
#
# 5 count benchmark
all_tweets['is_viral_retweet_b5'] = np.where(all_tweets['retweet_count'] >= 5, 1, 0)
print("\n--------- 5 count benchmark\n\n")
print(all_tweets['is_viral_retweet_b5'].value_counts())
# Average count benchmark
all_tweets['is_viral_retweet_bavg'] = np.where(all_tweets['retweet_count'] >= avg_retweet_count, 1, 0)
print("\n--------- Average count benchmark\n\n")
print(all_tweets['is_viral_retweet_bavg'].value_counts())
# 10000 retweets count benchmark
all_tweets['is_viral_retweet_b10000'] = np.where(all_tweets['retweet_count'] >= 10000, 1, 0)
print("\n--------- 10'000 count benchmark\n\n")
print(all_tweets['is_viral_retweet_b10000'].value_counts())
#
############################################################################### Exploring the feature "favorite_count"
#
# Adding the feature to the "favorite_count" to the "all_tweet" dataframe
all_tweets['favorite_count'] = [0 if pd.isnull(all_tweets.loc[i]['retweeted_status']) \
                                  else all_tweets.loc[i]['retweeted_status']['favorite_count'] \
                                  for i in range(len(all_tweets))]
print("\n--------- 'favorite_count' values\n\n")
print(all_tweets['favorite_count'])
# The highest "favorite_count" value
print(f"\n\n--------- The highest 'favorite_count' value: {all_tweets['favorite_count'].max()}\n")
# The tweet with the highest "favorite_count" value
print("\n--------- The tweet with the highest 'favorite_count' value\n\n")
print(all_tweets.loc[all_tweets['favorite_count'] == all_tweets['favorite_count'].max()].T)
# Average "favorite_count"
avg_favorite_count = all_tweets['favorite_count'].median()
print(f"\n\n--------- Average 'favorite_count': {avg_favorite_count}\n")
#
################################################################################### "favorite_count" scatter plots
#
plt.figure(figsize=(7, 14))
plt.scatter(range(len(all_tweets)), all_tweets['favorite_count'])
plt.yticks(np.arange(0, all_tweets['favorite_count'].max() + 10000, step=5000))
# Removes the highest favorite_count values outliers, set y axis bottom to -1 and upper to 320000
plt.ylim(-5000, 320000)
# Average line
plt.plot(range(len(all_tweets)), [avg_favorite_count for i in range(len(all_tweets))], linewidth=2, color="red")
# Labels
plt.legend(['Average favorite_count'], prop={'size': 10}, loc='upper left')
plt.tick_params(labelbottom=False) # labels along the bottom edge are off
plt.title('Tweets')
plt.ylabel('Liked Count')
plt.tick_params(labelsize=9.5)

plt.savefig('graph/favorite_count.png')
plt.show()
#
# --------- let's remove outliners from the graph and try to improve the readability of the average line.
# Adds an extra benchmark at the 10'000 favorites value count
plt.figure(figsize=(7, 14))
plt.scatter(range(len(all_tweets)), all_tweets['favorite_count'])
plt.yticks(np.arange(0, all_tweets['favorite_count'].max() + 10000, step=5000))
# Removes the highest favorite_count values outliers, set y axis bottom to -1 and upper to 320000
plt.ylim(-2000, 320000)
# Average line
plt.plot(range(len(all_tweets)), [avg_favorite_count for i in range(len(all_tweets))], linewidth=2, color="red")
# Benchmark line
plt.plot(range(len(all_tweets)), [10000 for i in range(len(all_tweets))], linewidth=2, color="cyan")
# Labels
plt.legend(['Average favorite_count', 'Benchmark viral favorite'], prop={'size': 10})
plt.tick_params(labelbottom=False) # labels along the bottom edge are off
plt.title('Tweets')
plt.ylabel('Liked Count')
plt.tick_params(labelsize=9.5)

plt.savefig('graph/favorite_count_benchmark.png')
plt.show()
#
################################################################################### Viral favorite' label
#
# Viral favorites' classes:
#
#     - 0 stand for
#     it is Not a viral favorite class
#
#     - The 1 stand for
#     it is a viral favorite class
#
# 5 count benchmark
all_tweets['is_viral_favorite_b5'] = np.where(all_tweets['favorite_count'] >= 5, 1, 0)

print("\n--------- 5 count benchmark\n\n")
print(all_tweets['is_viral_favorite_b5'].value_counts())

# Average count benchmark
all_tweets['is_viral_favorite_bavg'] = np.where(all_tweets['favorite_count'] >= avg_favorite_count, 1, 0)

print("\n--------- Average count benchmark\n\n")
print(all_tweets['is_viral_favorite_bavg'].value_counts())

# 10000 favorite count benchmark
all_tweets['is_viral_favorite_b10000'] = np.where(all_tweets['favorite_count'] >= 10000, 1, 0)

print("\n--------- 10'000 count benchmark\n\n")
print(all_tweets['is_viral_favorite_b10000'].value_counts())
#
################################################################################## Labels Sets
#
#               The sets of labels defining a viral tweet:
#
#                       5 counts
#                           "is_viral_retweet_b5"
#                           "is_viral_favorite_b5"
#                      average counts
#                           "is_viral_retweet_bavg"
#                           "is_viral_favorite_bavg"
#                       10000 counts
#                           "is_viral_retweet_b10000"
#                           "is_viral_favorite_b10000"
#
# labels 5 count
labels_b5 = all_tweets[['is_viral_retweet_b5', 'is_viral_favorite_b5']]
# labels average count
labels_bavg = all_tweets[['is_viral_retweet_bavg', 'is_viral_favorite_bavg']]
# labels 10000 count
labels_b10000 = all_tweets[['is_viral_retweet_b10000', 'is_viral_favorite_b10000']]

#
##################################################################################  "is viral tweet" Labels
# Defining a 'viral tweet'

#       "is_viral_tweet_b5", 5 "retweet_count" and "favorite_count" benchmark
#       "is_viral_tweet_bavg", Average "retweet_count" and "favorite_count" benchmark
#       "is_viral_tweet_b10000", 10'000 "retweet_count" and "favorite_count" benchmark
#
#  Are combinations of four classes:
#
#       [1, 1] 'is a viral tweet', 'is a viral retweet' and 'is a viral favorite'
#       [1, 0] 'is a viral a tweet', 'is a viral retweet' and 'is not-viral a favorite'
#       [0, 1] 'is a viral a tweet', 'is a viral favorite' and 'is not-viral a retweet'
#       [0, 0] 'is not-viral a tweet', 'is not-viral a retweet', and 'is not-viral a favorite'
#
all_tweets['is_viral_tweet_b5'] = [[all_tweets.loc[i]['is_viral_retweet_b5'],
                                    all_tweets.loc[i]['is_viral_favorite_b5']] for i in range(len(all_tweets))]
all_tweets['is_viral_tweet_bavg'] = [[all_tweets.loc[i]['is_viral_retweet_bavg'],
                                      all_tweets.loc[i]['is_viral_favorite_bavg']] for i in range(len(all_tweets))]
all_tweets['is_viral_tweet_b10000'] = [[all_tweets.loc[i]['is_viral_retweet_b10000'],
                                        all_tweets.loc[i]['is_viral_favorite_b10000']] for i in range(len(all_tweets))]

print("\n--------- 'is viral tweet' Labels\n\n")
print(all_tweets['is_viral_tweet_b10000'])
#
################################################################################## Making Features
#
#        The length of the tweet.
#        The number followers
#        The friend count
#        The number of hashtags in the tweet.
#        The number of links in the tweet.
#        The number of words in the tweet.
#
#
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['hastags_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis=1)
all_tweets['links_count'] = all_tweets.apply(lambda tweet: tweet['text'].count('http'), axis=1)
all_tweets['words_count'] = all_tweets.apply(lambda tweet: len(tweet['text'].split()), axis=1)
# The following features are found in the user dictionary
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
# saving
all_tweets.to_csv('data/all_tweets.csv')
#
################################################################################## Model Data
#
data = all_tweets[['tweet_length','followers_count','friends_count', 'tweet_length', 'hastags_count', 'links_count', 'words_count']]
#
############################################################################## Creating the Training Sets and Test Sets
#
# 5 count benchmark
train_data_b5, test_data_b5, train_labels_b5, test_labels_b5 = train_test_split(data,
                                                                                labels_b5,
                                                                                test_size = 0.2,
                                                                                random_state = 1)
# avg count benchmark
train_data_bavg, test_data_bavg, train_labels_bavg, test_labels_bavg = train_test_split(data,
                                                                                        labels_bavg,
                                                                                        test_size = 0.2,
                                                                                        random_state = 1)
# avg count benchmark
train_data_b10000, test_data_b10000, train_labels_b10000, test_labels_b10000 = train_test_split(data,
                                                                                                labels_b10000,
                                                                                                test_size = 0.2,
                                                                                                random_state = 1)
# Sample from test_labels_b5
print("\n--------- Sample from test_labels_b5\n\n")
print(test_labels_b5)
#
############################################################################## Normalizing The Model Data
#
# Using the sklearn.preprocessing library
# 5 count benchmark
scaled_train_data_b5 = scale(train_data_b5, axis=0)
scaled_test_data_b5 = scale(test_data_b5, axis=0)
# Average count benchmark
scaled_train_data_bavg = scale(train_data_bavg, axis=0)
scaled_test_data_bavg = scale(test_data_bavg, axis=0)
# 10000 count benchmark
scaled_train_data_b10000 = scale(train_data_b10000, axis=0)
scaled_test_data_b10000 = scale(test_data_b10000, axis=0)
# Scaled data Sample
print("\n--------- Scaled data Sample\n\n")
print(test_labels_b5)
print(scaled_train_data_b5[0])
#
############################################################################## Classifiers Models and Accuracy scores
## Model evalution dataframe
classifiers_eval = pd.DataFrame()
# 5 count benchmark
classifier_b5 = KNeighborsClassifier(n_neighbors = 5)
classifier_b5.fit(scaled_train_data_b5, train_labels_b5)
# Accuracy scores
accuracy_test_b5 = classifier_b5.score(scaled_test_data_b5, test_labels_b5)

# avg count benchmark
classifier_bavg = KNeighborsClassifier(n_neighbors = 5)
classifier_bavg.fit(scaled_train_data_bavg, train_labels_bavg)
# Accuracy scores
accuracy_test_bavg = classifier_bavg.score(scaled_test_data_bavg, test_labels_bavg)

# 10000 count benchmark
classifier_b10000 = KNeighborsClassifier(n_neighbors = 5)
classifier_b10000.fit(scaled_train_data_b10000, train_labels_b10000)
# Accuracy scores
accuracy_test_b10000 = classifier_b10000.score(scaled_test_data_b10000, test_labels_b10000)

# Storing Scores
classifiers_eval['Classifiers'] = ['5 count benchmark', 'avg count benchmark', '10000 count benchmark']
classifiers_eval['Test Accuracy'] = [accuracy_test_b5, accuracy_test_bavg, accuracy_test_b10000]

classifiers_eval.style.set_properties(**{'text-align': 'right'})

print("\n--------- classifiers_eval DataFrame with Accuracy scores\n\n")
print(classifiers_eval)
#
############################################################################## Models Evaluations
#
#
# ------------------------------------------ Predictions
#
predictions_b5 = classifier_b5.predict(scaled_test_data_b5)
predictions_bavg = classifier_bavg.predict(scaled_test_data_bavg)
predictions_b10000 = classifier_b10000.predict(scaled_test_data_b10000)
# Predictions Sample
print("\n--------- Predictions Sample\n\n")
print(predictions_b5)
#
# ------------------------------------------  Precision
#
# calculating precision scores
# The argument "average='weighted'" returns the averaged precision score of the two classes precision scores
precision_b5 = precision_score(test_labels_b5, predictions_b5, average='weighted')
precision_bavg = precision_score(test_labels_bavg, predictions_bavg, average='weighted')
precision_b10000 = precision_score(test_labels_b10000, predictions_b10000, average='weighted')
# Saving scores
classifiers_eval['Precision'] = [precision_b5, precision_bavg, precision_b10000]

print("\n--------- classifiers_eval DataFrame with Precision scores\n\n")
print(classifiers_eval)
#
# ------------------------------------------  Recall
#
# calculating recall scores
# The argument "average='weighted'" returns the averaged recall score of the two classes recall scores
recall_b5 = recall_score(test_labels_b5, predictions_b5, average='weighted')
recall_bavg = recall_score(test_labels_bavg, predictions_bavg, average='weighted')
recall_b10000 = recall_score(test_labels_b10000, predictions_b10000, average='weighted')
# Saving scores
classifiers_eval['Recall'] = [recall_b5, recall_bavg, recall_b10000]
classifiers_eval.to_csv('data/classifiers_eval.csv')
print("\n--------- classifiers_eval DataFrame with Recall scores\n\n")
print(classifiers_eval)
#
# ------------------------------------- Test  "is a viral tweet" verses vs predicted "is a viral tweet"
#
# ----------- Test sets Viral Retweets, Favorites and Tweets
# Creates a Predictions vs test DataFrame
prediction_vs_test = pd.DataFrame({'Classifiers':['5 count benchmark', 'avg count benchmark', '10000 count benchmark']})
# Test viral retweets
test_viral_retweets_b5 = test_labels_b5['is_viral_retweet_b5'].sum()
test_viral_retweets_bavg = test_labels_bavg['is_viral_retweet_bavg'].sum()
test_viral_retweets_b10000 = test_labels_b10000['is_viral_retweet_b10000'].sum()
# Test viral favorites
test_viral_favorites_b5 = test_labels_b5['is_viral_favorite_b5'].sum()
test_viral_favorites_bavg = test_labels_bavg['is_viral_favorite_bavg'].sum()
test_viral_favorites_b10000 = test_labels_b10000['is_viral_favorite_b10000'].sum()
# Test viral tweets
test_viral_tweets_b5 = len(test_labels_b5[(test_labels_b5['is_viral_retweet_b5'] == 1) & \
                                          (test_labels_b5['is_viral_favorite_b5'] == 1)])
test_viral_tweets_bavg = len(test_labels_bavg[(test_labels_bavg['is_viral_retweet_bavg'] == 1) & \
                                              (test_labels_bavg['is_viral_favorite_bavg'] == 1)])
test_viral_tweets_b10000 = len(test_labels_b10000[(test_labels_b10000['is_viral_retweet_b10000'] == 1) & \
                                                  (test_labels_b10000['is_viral_favorite_b10000'] == 1)])

# ----------- Predicted Viral Retweets, Favorites and Tweets
predicted_viral_retweets_favorites_b5 = np.sum(predictions_b5, axis=0)
predicted_viral_retweets_favorites_bavg = np.sum(predictions_bavg, axis=0)
predicted_viral_retweets_favorites_b10000 = np.sum(predictions_b10000, axis=0)
# ----------- Predicted Viral Tweets
predicted_viral_tweets_b5 = len([tweet for tweet in predictions_b5 if tweet[0] == 1 and tweet[1] == 1])
predicted_viral_tweets_bavg = len([tweet for tweet in predictions_bavg if tweet[0] == 1 and tweet[1] == 1])
predicted_viral_tweets_b10000 = len([tweet for tweet in predictions_b10000 if tweet[0] == 1 and tweet[1] == 1])

# Saving viral retweets
prediction_vs_test['Predicted Viral Retweets'] = [predicted_viral_retweets_favorites_b5[0],
                                                predicted_viral_retweets_favorites_bavg[0],
                                                predicted_viral_retweets_favorites_b10000[0]]
prediction_vs_test['Test Set Viral Retweets'] = [test_viral_retweets_b5,
                                               test_viral_retweets_bavg,
                                               test_viral_retweets_b10000]
# Saving viral favorites
prediction_vs_test['Predicted Viral Favorites'] = [predicted_viral_retweets_favorites_b5[1],
                                                 predicted_viral_retweets_favorites_bavg[1],
                                                 predicted_viral_retweets_favorites_b10000[1]]
prediction_vs_test['Test Set Viral Favorites'] = [test_viral_favorites_b5,
                                                test_viral_favorites_bavg,
                                                test_viral_favorites_b10000]
# Saving viral tweets
prediction_vs_test['Predicted Viral Tweets'] = [predicted_viral_tweets_b5,
                                              predicted_viral_tweets_bavg,
                                              predicted_viral_tweets_b10000]
prediction_vs_test['Test Set Viral Tweets'] = [test_viral_tweets_b5,
                                             test_viral_favorites_bavg,
                                             test_viral_tweets_b10000]

prediction_vs_test.to_csv('data/prediction_vs_test.csv')

print("\n--------- Test  'is a viral tweet' verses vs predicted 'is a viral tweet'\n\n")
print(prediction_vs_test)
#
############################################################################## Improving Models
#
#
# ------------------------------------- Calibrating KNN Classifier 10'000 Benchmark
#
# Initialing KNN model
model = KNeighborsClassifier()
# Calibrating KNN model
cal_classifier_b10000 = CalibratedClassifierCV(model, cv=3, method='sigmoid')
# Processes train and test labels before using it with the calibrated KNN classifier model
p_train_labels = np.ravel(train_labels_b10000.iloc[:, 0].values.reshape(-1, 1))
p_test_labels = np.ravel(test_labels_b10000.iloc[:, 0].values.reshape(-1, 1))
# Training model
cal_classifier_b10000.fit(scaled_train_data_b10000, p_train_labels)
# Model Predictions
cal_predictions_b1000 = cal_classifier_b10000.predict(scaled_test_data_b10000)
# Model Eval
print('calibrated Classifier 10\'000 Benchmark Evaluation\n')
print(f'Accuracy: {cal_classifier_b10000.score(scaled_test_data_b10000, p_test_labels)}')
print(f'Precision: {precision_score(p_test_labels, cal_predictions_b1000)}')
print(f'Recall: {recall_score(p_test_labels, cal_predictions_b1000)}')
#
# ------------------------------------- Logistic Regression Model
#
# Initialing Logistic Regression model
Logreg_classifier_b10000 = LogisticRegression(class_weight='balanced')
# Training model
Logreg_classifier_b10000.fit(scaled_train_data_b10000, p_train_labels)
# Model Predictions
Logreg_predictions_b10000 = Logreg_classifier_b10000.predict(scaled_test_data_b10000)
# Model Eval
print('Logistic Regression Model 10\'000 Benchmark Evaluation\n')
print(f'Accuracy: {Logreg_classifier_b10000.score(scaled_test_data_b10000, p_test_labels)}')
print(f'Precision: {precision_score(p_test_labels, Logreg_predictions_b10000)}')
print(f'Recall: {recall_score(p_test_labels, Logreg_predictions_b10000)}')
#
# ------------------------------------- Choosing K
#
#  ------ Best k for the KNN 5 count Benchmark Classifier
name ='Classifier 5 count Benchmark'
best_k_b5, k_eval_b5 = best_k_value(scaled_train_data_b5, train_labels_b5,
                                    scaled_test_data_b5, test_labels_b5,
                                    100, 'Classifier 5 count Benchmark')

best_k_b5.to_csv('data/best_k_b5.csv')
k_eval_b5.to_csv('data/k_eval_b5.csv')
#  ------ Best k for the KNN average count Benchmark Classifier
best_k_bavg, k_eval_bavg = best_k_value(scaled_train_data_bavg, train_labels_bavg,
                                        scaled_test_data_bavg, test_labels_bavg,
                                        50, 'Classifier Average count Benchmark')

best_k_bavg.to_csv('data/best_k_bavg.csv')
k_eval_bavg.to_csv('data/k_eval_bavg.csv')
#  ------ Best k for the KNN 10'000 count Benchmark Classifier
best_k_b10000, k_eval_b10000 = best_k_value(scaled_train_data_b10000, train_labels_b10000,
                                           scaled_test_data_b10000, test_labels_b10000,
                                           30, 'Classifier 10000 Count Benchmark')

best_k_b10000.to_csv('data/best_k_b10000.csv')
k_eval_b10000.to_csv('data/k_eval_b10000.csv')