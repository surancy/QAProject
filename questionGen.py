import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import os
def pos(sent):
    sent = word_tokenize(sent)
    sent = pos_tag(sent)
    return sent

train = ['set1','set2','set3','set4','set5']
articals=[]
for directory in train:
    for f in os.listdir(directory):
        if f.endWith(".txt"):
            with open(directory+"/"+f):
                
for open()

tokens = word_tokenize(sent)


