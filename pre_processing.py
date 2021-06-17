# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 12:05:05 2021

@author: Asus
"""

# =============================================================================
# required libraries
# =============================================================================
import string 
import re
'''
import nltk
nltk.download('stopwords')
nltk.download('punkt')
'''
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


class PreProcessing:
    def __init__(self, text):
        self.text = text
        
    def get(self):
        return self.text
        
    def lowercase(self):
        self.text = self.text.lower()
        
    'punctuation: !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    def remove_punctuation(self):
        translator = str.maketrans(string.punctuation, ' '*len(string.punctuation)) 
        self.text = self.text.translate(translator)
        
    def remove_numbers(self): 
        self.text = re.sub(r'\d+', '', self.text) 
    
    def remove_under3char_words(self):
        self.text = ' '.join([word for word in self.text.split() if len(word) > 2])
    
    def remove_stopwords(self): 
        stop_words = set(stopwords.words("english")) 
        word_tokens = word_tokenize(self.text) 
        self.text = [word for word in word_tokens if word not in stop_words] 