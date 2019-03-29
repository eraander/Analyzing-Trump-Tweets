import json
import nltk
import os

tweets = {}

with open('condensed_2018.json') as json_file:
    data = json.load(json_file)
    for tweet_whole in data:
        if len(tweet_whole['text']) > 60:
            tweets[tweet_whole['id_str']] = [tweet_whole['created_at'], tweet_whole['text']]


for tweet_id, tweet_content in tweets.items():
    file_name = os.path.join(os.curdir, 'Tweets/trump_tweet_' + tweet_id + '.txt')
    with open(file_name, 'w') as f:
            f.write(tweet_content[0] + '\n')
            sentences = nltk.sent_tokenize(tweet_content[1])
            f.write("[Trump]\n")
            f.write('\n'.join(sentences))
