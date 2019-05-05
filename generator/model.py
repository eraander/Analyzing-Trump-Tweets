'''
Name: model.py
COSI 140B
30 Apr 2019
'''


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from operator import itemgetter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import os, re

#
negative_stems = set(['no', 'not', 'never', 't', 'don\'t', 'can\'t', 'arent', 'didnt', 'wouldnt', 'ever', 'doesnt',
                      'without', 'lack', 'miss', 'forget', 'lose', 'least', 'less', 'waste'])
# hyp_words = set(['would', 'if', 'could', 'couldve', 'wouldve', 'might', 'mightve', 'may'])
# intensifiers = set(['most', 'very', 'extremely'])

def keep_only_alpha(text):
    #text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?:\/\/.+', ' ', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    text.lower()
    # stop_words = set([word for word in stopwords.words('english') if word not in negative_stems.union(hyp_words)])
    return [token for token in text.split()]

def change_data(data_frame):
    data_frame.event_confidence[data_frame.event_confidence == '+2'] = 'pos'
    data_frame.event_confidence[data_frame.event_confidence == '+1'] = 'pos'
    data_frame.event_confidence[data_frame.event_confidence == '-1'] = 'neg'
    data_frame.event_confidence[data_frame.event_confidence == '-2'] = 'neg'
    data_frame = data_frame[data_frame['event_confidence'] != '0']
    return data_frame

def make_data_frame(dir_path):
    df1 = pd.DataFrame()
    for file in os.listdir(dir_path):
        if file.endswith('txt'):
            f = open(file)
            file_content = f.read().strip('\n')
            file_annotations = file_content.split('\n\n')
            all_fields = [annotation.split('|') for annotation in file_annotations]
            data = all_fields[1:]
            cols = all_fields[0]
            df2 = pd.DataFrame(data=data, columns=cols, )
            df1 = pd.concat([df1, df2])
    df1.reset_index(inplace=True)
    df1['event_text'].str.lower()
    df1['type'] = 'train'
    df1['type'][200:] = 'test'
    print(df1)

    return df1

def extract_feat_vocab(data_frame):
    feat_vocab = dict()
    for index, row in data_frame[data_frame['type'] == 'train'].iterrows():
        negative = 0
        for token in keep_only_alpha(row['tweet_content']):
            if token in row['event_text'].split():
                feat_vocab['event_' + token] = feat_vocab.get(token, 0) + 1
                continue
            # main modifications conducted here
            if token in negative_stems:
                negative = 1
                continue
            if token in row['emotion']:
                feat_vocab['emotion_' + token] = feat_vocab.get(token, 0) + 1
                emotion_value = row['emotion_value']
                feat_vocab[emotion_value] = feat_vocab.get(token, 0) + 1
                continue

            if 0 < negative <= 5:
                feat_vocab['not_' + token] = feat_vocab.get(token, 0) + 1
                negative += 1
            else:
                feat_vocab[token] = feat_vocab.get(token, 0) + 1
                negative = 0
    return feat_vocab



def select_features(feat_vocab, most_freq=3, least_freq=500):
    sorted_feat_vocab = sorted(feat_vocab.items(), key=itemgetter(1), reverse=True)
    print(sorted_feat_vocab)
    feat_dict = dict(sorted_feat_vocab[most_freq:len(sorted_feat_vocab)-least_freq])
    print(feat_dict)
    return set(feat_dict.keys())

def featurize(data_frame, feat_vocab):
    cols = ['_type_', '_confidence_']
    cols.extend(list(feat_vocab))
    row_count = data_frame.shape[0]
    print(row_count)
    feat_data_frame = pd.DataFrame(index=np.arange(row_count), columns=cols)
    feat_data_frame.fillna(0, inplace=True) #inplace: mutable
    for index, row in data_frame.iterrows():
        print(index)
        print(row)
        feat_data_frame.loc[index, '_type_'] = row['type']
        feat_data_frame.loc[index, '_confidence_'] = row['event_confidence']
        for token in keep_only_alpha(row['tweet_content']):
            if token in feat_vocab:
                feat_data_frame.loc[index, token] += 1
    return feat_data_frame


def vectorize(feature_csv, split='train'):
    '''
    note: the code to flip 20% of the labels is commented out
    also includes code to slice the df (current code selects 500 rows)
    '''
    df = pd.read_csv(feature_csv, encoding='latin1')
    df = df[df['_type_'] == split]
    if split == 'train':
        df = df.iloc[:200]
        # to_update = df.sample(frac=0.2)
        # for index, row in to_update.iterrows():
        #    if row['_label_'] == 'pos':
        #        row['_label_'] = 'neg'
        #        print(row['_label_'])
        #    else:
        #        row['_label_'] == 'pos'
        # df.update(to_update)
    df.fillna(0, inplace=True)
    data = list()
    for index, row in df.iterrows():
        datum = dict()
        datum['bias'] = 1
        for col in df.columns:
            if not (col == "_type_" or col == "_confidence_" or col == 'index'):
                datum[col] = row[col]
        data.append(datum)
    vec = DictVectorizer()
    data = vec.fit_transform(data).toarray()
    print(data.shape)
    labels = df._confidence_.as_matrix()
    print(labels.shape)
    return data, labels

def train_model(X_train, y_train, model):
    model.fit(X_train, y_train)
    print ('Shape of model coefficients and intercepts: {} {}'.format(model.coef_.shape, model.intercept_.shape))
    return model

def test_model(X_test, y_test, model):
    predictions = model.predict(X_test)
    report = classification_report(predictions, y_test)
    accuracy = accuracy_score(predictions, y_test)
    return accuracy, report

def classify(feat_csv):
    X_train, y_train = vectorize(feat_csv)
    X_test, y_test = vectorize(feat_csv, split='test')
    model = LogisticRegression(multi_class='multinomial', penalty='l2', solver='lbfgs', max_iter=200, verbose=1)
    model = train_model(X_train, y_train, model)
    accuracy, report = test_model(X_test, y_test, model)
    print (report)

if __name__ == '__main__':
    model_path = os.path.curdir
    df = make_data_frame(model_path)
    df = change_data(df)
    feat_vocab = extract_feat_vocab(df)
    print(feat_vocab)
    selected_feat_vocab = select_features(feat_vocab)
    '''
    ps = PorterStemmer()
    sent_dict = load_sentiment_dictionary(ps, 'subjclueslen1-HLTEMNLP05.tff')
    feat_vocab = extract_feat_vocab(hw1_path, ps, sent_dict)
    print(len(feat_vocab))
    selected_feat_vocab = select_features(feat_vocab, 100, 10000)
    print(len(selected_feat_vocab))
    '''
    feat_data_frame = featurize(df, selected_feat_vocab)
    featfile = os.path.join(os.path.curdir, 'features.csv')
    feat_data_frame.to_csv(featfile, encoding='latin1', index=False)
    classify('features.csv')
