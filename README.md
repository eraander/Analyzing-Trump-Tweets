# Analyzing Trump Tweets
Course: COSI140 (Natural Language Annotation for Machine Learning)

### Overview ###
- This project contains a logistic regression classifier that predicts the factuality score of annotated Trump tweets based on the emotion toward the tweets. The corpus consisted of 200 tweets.
- The model had difficulty predicting the correct factuality in test data, especially when the factuality was negative.

### Contents ###
* data - includes annotated data and gold corpus
* docs - includes guidelines for annotating tweets, final report, dtd for MAE
* generator - includes script to convert JSON of tweets to txt, MAE XML to txt, and script to calculate IAA
* model - logistic regression classifier
