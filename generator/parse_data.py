#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script parses the output xml file from Mae
ingests a gold standard annotated corpus
The output of the file contains following fileds:
    1. the original tweet
    2. the event (text)
    3. the source
    4. the confidence of the event with respect to the source
    5. emotion (text)
    6. emotion polarity
"""

import xml.etree.ElementTree as et
import os
from collections import defaultdict

class Parse():
    
    def __init__(self):
        """
        :param data_dir: the directory of the data files
        :type data_dir: A directory
        
        """
        self.output = []
        
    def parse(self, file):
        """
        return a result dictionary with the key event_id 
        and values in a list containing event_text, source_id, 
        source, emotion_id, emotion, emotion_value
        
        :rtype: dictionary
        
        """
        
        events = defaultdict(dict)
        emotions = defaultdict(dict)
        link_src_evnt = defaultdict(dict)
        link_emtn_evnt = defaultdict(dict)
        results = defaultdict(dict)

        tree = et.parse(file)
        root = tree.getroot()
        text = root.find('TEXT').text
        
        for event in root.iter('Event'):
            event_id = event.get('id')
            event_span = event.get('spans')
            event_text = event.get('text')
            confidence = event.get('value')
            events[event_id] = [event_span, event_text, confidence]
        
        for s in root.iter('Source'):
            source_id = s.get('id')
            source_span = s.get('spans')
            source_text = s.get('text')
            
        for emotion in root.iter('Emotion'):
            emotion_id = emotion.get('id')
            emotion_text = emotion.get('text')
            emotion_value = emotion.get('value')
            emotions[emotion_id] = [emotion_text, emotion_value]
            
        for link_s_e in root.iter('Link_Source_Event'):
            #link_id_s_e = link_s_e.get('id')
            #source_id = link_s_e.get('fromID')
            source = link_s_e.get('fromText')
            event = link_s_e.get('toText')
            event_id = link_s_e.get('toID')
            link_src_evnt[event_id] = source
            
        for link_e_e in root.iter('Link_Emotion_Event'):
            #link_id_e_e = link_e_e.get('id')
            emotion_id = link_e_e.get('fromID')
            event_id = link_e_e.get('toID')
            emotion = link_e_e.get('fromText')
            event = link_e_e.get('toText')
            #add emotion text, span, and value from emotions dict
            link_emtn_evnt[event_id] = emotions[emotion_id]
           
        #put everything into the result list
        for event_id in events:
            results[event_id]['event_text'] = events[event_id]
            results[event_id]['source'] = link_src_evnt[event_id]
            results[event_id]['emotion'] = link_emtn_evnt[event_id]
                
        annotation = self._get_tweet(text, results)
        
        outfile = file.split('.')[0] + '.txt'
        self.write_file(annotation, outfile)
            
    
    def write_file(self, annotation, outfile):
        """
        write the annotation result to a file
        
        """
        with open(outfile, 'w') as f:
            #write header
            f.write('annotation_id|tweet_content|event_text|event_confidence|source|emotion|emotion_value\n')
            for tweet_id in annotation:
                tweet_content = annotation[tweet_id]['TEXT']
                for event in annotation[tweet_id]:
                    if event != 'TEXT':
                        event_text = annotation[tweet_id][event]['event_text'][1]
                        event_confidence = annotation[tweet_id][event]['event_text'][2]
                        source = annotation[tweet_id][event]['source']
                        if not source:
                            source = 'Trump'

                        emotion = annotation[tweet_id][event]['emotion']
                        emotion_value = ''
                        
                        if emotion:
                            emotion = annotation[tweet_id][event]['emotion'][0]
                            emotion_value = annotation[tweet_id][event]['emotion'][1]
                        else:
                            emotion = ''
                            
                        f.write(tweet_id + '|' + tweet_content + '|' + event_text + '|' + event_confidence + '|' + source + '|' + emotion + '|' + emotion_value + '\n\n')
        
        
    
    def _get_tweet(self, text, events):
        """
        construct a dictionary result that contains the shared tweet id,
        the actual tweet text, and annotated results
        
        :param text: actual tweet text
        :type text: string
        :param events: annotated corpus in a dictionary
        :type events: dictionary
        
        :rtype: dictionary
        
        """
        
        results = defaultdict(dict)
        tweets = text.split('\n\n')
        tweets_span = defaultdict(dict)
        start_span, end_span = 0, 0

        for tweet in tweets:
            start_span += 1 #each line break brings in 1 span
            tweet_id = tweet.split('\n')[0]
            if tweet != '':
                tweet_len = sum(len(line)+1 for line in tweet.split('\n'))
            else:
                tweet_len = 0
            end_span = start_span + tweet_len 
            tweets_span[tweet_id] = [tweet, start_span, end_span] 
            start_span = end_span
        #print(tweets_span)    
        for event in events:
            span = events[event]['event_text'][0].split('~')
            
            for tweet_id in tweets_span:
                start = tweets_span[tweet_id][1]
                end = tweets_span[tweet_id][2]
                
                if int(span[1]) <= end and int(span[0]) >= start:
                    results[tweet_id]['TEXT'] = tweets_span[tweet_id][0]
                    results[tweet_id][event] = events[event]
        return results
        
if __name__ == '__main__':
    file_dir = 'data/annotator3'
    file_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', file_dir))
    
    parser = Parse()
    for root, dirs, files in os.walk(file_dir):
        files = [os.path.join(root, name) for name in files]
        for file in files:
            if file[-3:] == 'xml':
                parser.parse(file)
        