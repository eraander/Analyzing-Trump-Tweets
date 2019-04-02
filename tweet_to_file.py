import json
import nltk
import os

tweets = {}

with open('condensed_2018.json') as json_file:
    data = json.load(json_file)
    for tweet_whole in data:
        if len(tweet_whole['text']) > 60:
            tweets[tweet_whole['id_str']] = [tweet_whole['created_at'], tweet_whole['text']]

for index, key in enumerate(tweets):
    if index == 0:
        file_name = os.path.join(os.curdir, 'tweets_xml/2018_trump_tweets_' + tweets[key][0][:10] + '.xml')
        with open(file_name, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" ?>\n\n<EmotionalFactuality_v1.0>\n<TEXT><![CDATA[\n\n')
    elif index % 20 == 0:
        with open(file_name, 'a') as f:
            f.write(']]></TEXT>\n<TAGS>\n</TAGS>\n</EmotionalFactuality_v1.0>')
        file_name = os.path.join(os.curdir, 'tweets_xml/2018_trump_tweets_' + tweets[key][0][:10] + '.xml')
        with open(file_name, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" ?>\n\n<EmotionalFactuality_v1.0>\n<TEXT><![CDATA[\n\n')
    elif index == len(tweets)-1:
        with open(file_name, 'a') as f:
            f.write(']]></TEXT>\n<TAGS>\n</TAGS>\n</EmotionalFactuality_v1.0>')
    with open(file_name, 'a') as f:
        f.write(key + '\n')
        f.write(tweets[key][0] + '\n')
        f.write("[Trump]\n")
        sentences = nltk.sent_tokenize(tweets[key][1])
        f.write('\n'.join(sentences))
        f.write('\n\n')
