#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from generator.parse_data import Parse
import unittest, os

FILE_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/annotator2'))

tweet1 = '''967023714268319744
Fri Feb 23 13:09:54 +0000 2018
[Trump]
For those of you who are still interested, the Democrats have totally forgotten about DACA.
Not a lot of interest on this subject from them!'''
            
tweet2 = '''966657362789568512
Thu Feb 22 12:54:09 +0000 2018
[Trump]
....History shows that a school shooting lasts, on average, 3 minutes.
It takes police &amp; first responders approximately 5 to 8 minutes to get to site of crime.
Highly trained, gun adept, teachers/coaches would solve the problem instantly, before police arrive.
GREAT DETERRENT!'''
            
tweet3 = '''966653833047887874
Thu Feb 22 12:40:08 +0000 2018
[Trump]
....immediately fire back if a savage sicko came to a school with bad intentions.
Highly trained teachers would also serve as a deterrent to the cowards that do this.
Far more assets at much less cost than guards.
A “gun free” school is a magnet for bad people.
ATTACKS WOULD END!'''
            
tweet4 = """966687491616059392
Thu Feb 22 14:53:52 +0000 2018
[Trump]
Will be meeting with Lawmakers today at 11:30 A.M. to discuss School Safety.
Next week it will be with our Nation’s Governors.
It’s been many years of all talk, no action.
We’ll get it done!"""

class TestEvents(unittest.TestCase):
    
    def test_tweet(self):
        parser = Parse(FILE_DIR)
        text = parser.parse()
        
        self.assertEqual(text['967023714268319744']['TEXT'], tweet1)
        self.assertEqual(text['966657362789568512']['TEXT'], tweet2)
        self.assertEqual(text['966653833047887874']['TEXT'], tweet3)
        self.assertEqual(text['966687491616059392']['TEXT'], tweet4)
        
    def test_events(self):
        parser = Parse(FILE_DIR)
        text = parser.parse()
        self.assertEqual(text['967023714268319744']['E10']['event_text'][1], 'forgotten')
        self.assertEqual(text['967023714268319744']['E11']['event_text'][1], 'interest')
        self.assertEqual(text['966711641378381826']['E18']['event_text'][1], 'winning')
        self.assertEqual(text['967083810981597185']['E8']['event_text'][1], 'STAND')
        self.assertEqual(text['967083810981597185']['E7']['event_text'][1], 'salute')
        
            
    def test_event_value(self):
        parser = Parse(FILE_DIR)
        text = parser.parse()
        print(text)
        self.assertEqual(text['967023714268319744']['E10']['event_text'][-1], '0')
        self.assertEqual(text['967023714268319744']['E11']['event_text'][-1], '-1')
        self.assertEqual(text['966711641378381826']['E18']['event_text'][-1], '+2')
        self.assertEqual(text['967083810981597185']['E8']['event_text'][-1], '+1')
        self.assertEqual(text['967083810981597185']['E7']['event_text'][-1], '+1')
        
        
    def test_emotion(self):
        parser = Parse(FILE_DIR)
        text = parser.parse()
        self.assertEqual(text['967023714268319744']['E10']['emotion'][0], 'totally')
        self.assertEqual(text['967023714268319744']['E11']['emotion'][0], 'Not')
        self.assertEqual(text['967083810981597185']['E8']['emotion'][0], 'PROUDLY')
        self.assertEqual(text['967083810981597185']['E7']['emotion'][0], 'great')
        
    def test_emotion_value(self):
        parser = Parse(FILE_DIR)
        text = parser.parse()
        self.assertEqual(text['967023714268319744']['E10']['emotion'][1], 'negative')
        self.assertEqual(text['967023714268319744']['E11']['emotion'][1], 'negative')
        self.assertEqual(text['967083810981597185']['E8']['emotion'][1], 'positive')
        self.assertEqual(text['967083810981597185']['E7']['emotion'][1], 'positive')
    
    def test_source(self):
        parser = Parse(FILE_DIR)
        text = parser.parse()
        self.assertEqual(text['967083810981597185']['E8']['source'], 'Trump')
        self.assertEqual(text['967083810981597185']['E7']['source'], 'Trump')
        self.assertEqual(text['967023714268319744']['E10']['source'], 'Trump')
        self.assertEqual(text['967023714268319744']['E11']['source'], 'Trump')
        
if __name__ == '__main__':

    unittest.main()