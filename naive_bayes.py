# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:51:21 2020

@author: Asus
"""

import numpy as np
from pre_processing import PreProcessing

# =============================================================================
# STEP 1: pre-processing of Train and Test files
# =============================================================================
file_list = [[38229, 38329], [74735, 74835], [101566, 101666], [15183, 15283], [20502, 20602], [54131, 54231]]
preprocess_train_file_list = [[], [], [], [], [], []]
for index in range(0, 6):
    for file in range(file_list[index][0], file_list[index][1]):
        f = open("dataset\\" + str(index+1) + "\\Train\\" + str(file), "r")
        
        'PreProcessing'
        t = PreProcessing(f.read())
        t.lowercase()
        t.remove_punctuation()
        t.remove_numbers()
        t.remove_under3char_words()
        t.remove_stopwords()
        ''
        
        preprocess_train_file_list[index].append(t.get())
        f.close()

file_list = [[38214, 38229], [74720, 74735], [101551, 101566], [15168, 15183], [20487, 20502], [54116, 54131]]
preprocess_test_file_list = [[], [], [], [], [], []]
for index in range(0, 6):
    for file in range(file_list[index][0], file_list[index][1]):
        f = open("dataset\\" + str(index+1) + "\\Test\\" + str(file), "r")
        
        'PreProcessing'
        t = PreProcessing(f.read())
        t.lowercase()
        t.remove_punctuation()
        t.remove_numbers()
        t.remove_under3char_words()
        t.remove_stopwords()
        ''
        
        preprocess_test_file_list[index].append(t.get())
        f.close()  
        

# =============================================================================
# STEP 2: finding words frequency
# =============================================================================
word_frequency_list = [[], [], [], [], [], []]
def word_frequency(word_list, index):
    for i in range(0, len(word_list)):
        if word_list[i] in word_frequency_list[index]:
            word_frequency_list[index][word_frequency_list[index].index(word_list[i]) + 1] += 1
        else:
            word_frequency_list[index].append(word_list[i])
            word_frequency_list[index].append(1)

for i in range(0, 6): 
    for j in range(0, 100): 
        word_frequency(preprocess_train_file_list[i][j], i)

'SAVE WORDS FREQUENCY'
for index in range(0, 6):
    f = open("dataset\\" + str(index+1) + "\\Train\\word_frequency_list_" + str(index+1) + ".txt", "w+")
    for i in range(0, len(word_frequency_list[index]), 2):
        f.write("%s = " % word_frequency_list[index][i])
        f.write("%s\n" % word_frequency_list[index][i+1])
    f.close()  


# =============================================================================
# STEP 3: finding best attributes
# =============================================================================
best_attributes_list = [[], [], [], [], [], []]
def best_attributes(word_list, index):
    for i in range(0, len(word_list), 2):
        for j in range(0, 6):
            if j != index:
                if word_list[i] in word_frequency_list[j]:
                    word_list[i+1] -= word_frequency_list[j][word_frequency_list[j].index(word_list[i]) + 1]
        if word_list[i+1] > 0:
            best_attributes_list[index].append(word_list[i])
            best_attributes_list[index].append(word_list[i+1])
            
def define_best_attributes():
    for i in range(0, 6):
        l = np.array(best_attributes_list[i])
        l.shape = (int(len(l)/2),2)
        l = l[l[:,1].astype(np.int32).argsort()][::-1]
        best_attributes_list[i] = l[0:20,0]
                 
for i in range(0, 6): 
    best_attributes(word_frequency_list[i].copy(), i)
define_best_attributes() 

'SAVE BEST ATTRIBUTES'
for index in range(0, 6):
    f = open("dataset\\" + str(index+1) + "\\Train\\best_attributes_list_" + str(index+1) + ".txt", "w+")
    for i in range(0, 20):
        f.write("%s\n" % best_attributes_list[index][i])
    f.close()    


# =============================================================================
# STEP 4: combining all attributes and build a table
# =============================================================================
'TRAIN TABLE'
attributes_train_table = np.empty((601, 121), dtype="<U20")
for i in range(0, 6):
    attributes_train_table[0, 20*i:20*i+20] = best_attributes_list[i]

for i in range(0, 6):
    for j in range(0, 100):
        for k in range(0, 120):
            attributes_train_table[100*i+1 + j][120] = i+1
            if attributes_train_table[0][k] in preprocess_train_file_list[i][j]:
                attributes_train_table[100*i+1 + j][k] = 1
            else:
                attributes_train_table[100*i+1 + j][k] = 0


'TEST TABLE'
attributes_test_table = np.empty((91, 121), dtype="<U20")
for i in range(0, 6):
    attributes_test_table[0, 20*i:20*i+20] = best_attributes_list[i]

for i in range(0, 6):
    for j in range(0, 15):
        for k in range(0, 120):
            attributes_test_table[15*i+1 + j][120] = i+1
            if attributes_test_table[0][k] in preprocess_train_file_list[i][j]:
                attributes_test_table[15*i+1 + j][k] = 1
            else:
                attributes_test_table[15*i+1 + j][k] = 0


# =============================================================================
# STEP 5: implementation of Naive Bayes algorithm
# =============================================================================
'''
    naive bayes formula(using m-estimate):
    n_k = number of word(w) in class(i)
    n = number of all words in class(i)
    |vocabulary| = 120
    
    (n_k + 1) / (n + |vocabulary|)
'''
vocabulary = 120

n = []
for i in range(0, 6):
    c = 0
    for j in range(0, 100):
          (unique, counts) = np.unique(attributes_train_table[100*i+1 + j, 0:120], return_counts=True)
          c += counts[1] 
    n.append(c)

def n_k(index, c):
    (unique, counts) = np.unique(attributes_train_table[c*100+1:c*100+101, index], return_counts=True)
    if len(counts) == 1:
        if unique[0] == '0':
            return 0
        else:
            return counts[0] 
    elif len(counts) == 2:
        return counts[1]

def classification(test_attributes, c):
    'prior probability = 1/6 = 0.16'
    h = 0.16 
    for i in range(0, 120): 
        if test_attributes[i] == '1':
            h = h * ((n_k(i, c) + 1) / (n[c] + vocabulary))
    return h
    

# =============================================================================
# STEP 6: classification of test and train data
# =============================================================================
'TRAIN DATA'
classified_train_list = np.zeros((600, 7))

for i in range(0, 600):
    for c in range(0, 6):
          classified_train_list[i][c] = classification(attributes_train_table[i + 1], c)
    classified_train_list[i][6] = attributes_train_table[i + 1][120]
    hold = 0
    for c in range(0, 6):
          hold += classified_train_list[i][c]
    for c in range(0, 6):
          classified_train_list[i][c] = float("{:.4f}".format(classified_train_list[i][c] / hold))
          
correctly_classified_percent_train = 0
for i in range(0, 600):
    maximum = np.amax(classified_train_list[i, 0:6])
    index = np.where(classified_train_list[i, 0:6] == maximum)[0]  + 1
    if index == classified_train_list[i][6]:
        correctly_classified_percent_train += 1
correctly_classified_percent_train /= 600
correctly_classified_percent_train *= 100


'TEST DATA'
classified_test_list = np.zeros((90, 7))

for i in range(0, 90):
    for c in range(0, 6):
          classified_test_list[i][c] = classification(attributes_test_table[i + 1], c)
    classified_test_list[i][6] = attributes_test_table[i + 1][120]
    hold = 0
    for c in range(0, 6):
          hold += classified_test_list[i][c]
    for c in range(0, 6):
          classified_test_list[i][c] = float("{:.4f}".format(classified_test_list[i][c] / hold))
  
correctly_classified_percent_test = 0
for i in range(0, 90):
    maximum = np.amax(classified_test_list[i, 0:6])
    index = np.where(classified_test_list[i, 0:6] == maximum)[0]  + 1
    if index == classified_test_list[i][6]:
        correctly_classified_percent_test += 1
correctly_classified_percent_test /= 90
correctly_classified_percent_test *= 100               