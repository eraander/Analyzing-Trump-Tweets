'''
Name: model.py
COSI 140B
30 Apr 2019
'''


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from operator import itemgetter
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import os, re

#
negative_stems = set(['no', 'not', 't', 'never', 'don\'t', 'can\'t', 'aren', 'didnt', 'wouldn', 'ever', 'shouldn',
                      'without', 'lack', 'miss', 'couldn', 'isn', 'cannot', 'nt', 'didn', 'don'])
auxiliaries = set(['will', 'shall', 'should', 'can', 'must', 'may', 'did', 'do', 'could', 'would', 'might', 'would',
                   'want', 'if'])

def keep_only_alpha(text, stemmer):
    #text = re.sub(r'\d+', '', text)
    text = re.sub(r'https?:\/\/.+', ' ', text)
    text = re.sub(r'[^A-Za-z]+', ' ', text)
    text = text.lower()
    # bigrams = nltk.bigrams(word_t)
    # print(bigrams)
    return [token for token in word_tokenize(text)]

def change_data(data_frame):
    data_frame.is_copy = False
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
    # df1.set_value('0', 'event_text',
    df1['type'] = 'train'
    return df1

def train_to_test(df1):
    negative_entries = df1[df1['event_confidence'] == 'neg']
    len_neg = len(negative_entries)
    train_test_spl = int(0.23*len_neg)
    negative_entries['type'][:train_test_spl] = 'test'
    pos_sample = df1[df1['event_confidence'] == 'pos']
    len_pos = len(pos_sample)
    train_test_pos = int(0.27*len_pos)
    pos_sample['type'][:train_test_pos] = 'test'

    df2 = negative_entries.append(pos_sample)
    negative_entries = df1[df1['event_confidence'] == 'neg']
    neg_train = negative_entries[negative_entries['type'] == 'train']
    return df2

def extract_feat_vocab(data_frame, stemmer):
    feat_vocab = dict()
    for index, row in data_frame[data_frame['type'] == 'train'].iterrows():
        negative = 0
        event = keep_only_alpha(row['event_text'], stemmer)
        tokens = keep_only_alpha(row['tweet_content'], stemmer)
        # print(tokens)
        # print(event)
        event_index = tokens.index(event[0])
        event = tokens[event_index]
        # print(event_index)
        # print(event[0])
        event_start = event_index - 6
        if event_start < 0:
            event_start = 0
        tokens = tokens[event_start:event_index+2]
        bigrams = nltk.bigrams(tokens)
        key_tokens = [t for t in tokens if t in negative_stems or t in auxiliaries]
        if not key_tokens:
            tokens.append('likely_posit')
        for token in tokens:
            if token == event:
                feat_vocab['ends_with_' + event[-2:]] = feat_vocab.get('ends_with_' + event[-2:], 0) + 1
            unstemmed = token
            token = stemmer.stem(token)
            if token in negative_stems:
                feat_vocab['neg_' + token] = feat_vocab.get('neg_' + token, 0) + 10
            elif token in auxiliaries:
                feat_vocab['aux_' + token] = feat_vocab.get('aux_' + token, 0) + 10
            elif token == 'likely_posit':
                feat_vocab['aux_null'] = feat_vocab.get('aux_null', 0) + 10
            else:
                feat_vocab[token] = feat_vocab.get(token, 0) + 1
            if token in row['emotion'].lower().split() or unstemmed in row['emotion'].lower().split():
                feat_vocab['emotion_' + token] = feat_vocab.get('emotion_' + token, 0) + 1
                emotion_value = row['emotion_value']
                feat_vocab[emotion_value] = feat_vocab.get(emotion_value, 0) + 1
        # if not row['emotion']:
        #    feat_vocab['no_emotion'] = feat_vocab.get('no_emotion', 0) + 1
        for (token_a, token_b) in bigrams:
            if token_a in negative_stems or token_b in negative_stems:
                if token_b in negative_stems:
                    feat_vocab[token_a + '_' + token_b] = feat_vocab.get(token_a + '_' + token_b, 0) + 1
                elif token_a in negative_stems:
                    feat_vocab[token_a + '_' + token_b] = feat_vocab.get(token_a + '_' + token_b, 0) + 1
                else:
                    feat_vocab[token_a + '_' + token_b] = feat_vocab.get(token_a + '_' + token_b, 0) + 1
            elif token_b in negative_stems:
                feat_vocab[token_a + '_' + token_b] = feat_vocab.get(token_a + '_' + token_b, 0) + 1
            else:
                feat_vocab[token_a + '_' + token_b] = feat_vocab.get(token_a + '_' + token_b, 0) + 1
            negative = 0
    return feat_vocab



def select_features(feat_vocab, most_freq=1, least_freq=100):
    sorted_feat_vocab = sorted(feat_vocab.items(), key=itemgetter(1), reverse=True)
    feat_dict = dict(sorted_feat_vocab[most_freq:len(sorted_feat_vocab)-least_freq])
    return set(feat_dict.keys())

def featurize(data_frame, feat_vocab, stemmer):
    cols = ['_type_', '_confidence_']
    cols.extend(list(feat_vocab))
    row_count = data_frame.shape[0]
    feat_data_frame = pd.DataFrame(index=np.arange(row_count), columns=cols)
    feat_data_frame.fillna(0, inplace=True) #inplace: mutable
    for index, row in data_frame.iterrows():
        feat_data_frame.loc[index, '_type_'] = row['type']
        feat_data_frame.loc[index, '_confidence_'] = row['event_confidence']
        for token in keep_only_alpha(row['tweet_content'], stemmer):
            if token in feat_vocab:
                feat_data_frame.loc[index, token] += 1
    return feat_data_frame


def vectorize(feature_csv, split='train'):
    df = pd.read_csv(feature_csv, encoding='latin1')
    df = df[df['_type_'] == split]
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
    model = LogisticRegression(multi_class='multinomial', penalty='l2', solver='lbfgs', max_iter=500, verbose=1,
                             class_weight='balanced')
    # model = SVC(C=1.0, gamma='auto', class_weight='balanced')
    # model = LogisticRegressionCV(cv=5, multi_class='multinomial', max_iter=800)
    model = train_model(X_train, y_train, model)
    accuracy, report = test_model(X_test, y_test, model)
    print (report)

if __name__ == '__main__':
    model_path = os.path.curdir
    df = make_data_frame(model_path)
    df = change_data(df)
    df = train_to_test(df)
    ps = PorterStemmer()
    feat_vocab = extract_feat_vocab(df, ps)
    # print(feat_vocab)
    selected_feat_vocab = select_features(feat_vocab)
    feat_data_frame = featurize(df, selected_feat_vocab, ps)
    featfile = os.path.join(os.path.curdir, 'features.csv')
    feat_data_frame.to_csv(featfile, encoding='latin1', index=False)
    classify('features.csv')
