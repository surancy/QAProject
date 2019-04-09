import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
import os
import sys
from collections import Counter
import math
import en_core_web_sm

# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.parse import stanford

os.environ['STANFORD_PARSER'] = 'C:/Users/geyiyang/OneDrive/CMU/2019 Spring/NLP/team project/QAProject/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = 'C:/Users/geyiyang/OneDrive/CMU/2019 Spring/NLP/team project/QAProject/stanford-parser-3.9.2-models.jar'
java_path = "D:/Java/bin/java.exe"
os.environ['JAVA_HOME'] = java_path
nltk.internals.config_java(java_path)
# nltk.download("punkt")

def computeTFIDF(none_len, freq_dict):
    scores = []
    for id, d in enumerate(freq_dict):
        l = none_len[id]
        c = d.copy()
        for k,v in c.items():
            tf_score = v/l
            count = sum([k in temp for temp in freq_dict])
            idf_score = math.log(len(none_len)/count)
            tfidf_score = tf_score*idf_score
            c[k] = tfidf_score
        scores.append(c)
    return scores

# train = ['set1','set2','set3','set4','set5']
train = []
train.append( sys.argv[1])
n_questions = sys.argv[2]
#----------------------------------------read docs----------------------------------------------------------------
# list of 50 articles,each is a dict whose key is the title and value is a list of sentences splited from the whole body
articles=[]
for directory in train:
    for f in os.listdir(directory):
        if f.endswith(".txt"):
            d = dict()
            s = ""
            with open(directory+"/"+f,encoding='utf-8',mode = 'r') as _f:
                for i, line in enumerate(_f):
                    if i == 0:
                        title = line.strip().lower()
                    # delete content after "see also"
                    elif line.strip().lower() in set(["see also",'references']):
                        break
                    else:
                        s += line.strip().lower()
            d[title] = sent_tokenize(s)
            articles.append(d)


#----------------------------------------parse tree----------------------------------------------------------------
# parse tree : select NP-VP structured sentence
candidateSent = []
parser = stanford.StanfordParser(model_path="C:/Users/geyiyang/OneDrive/CMU/2019 Spring/NLP/team project/QAProject/englishPCFG.ser.gz",encoding='utf8')
# sentences = parser.raw_parse_sents(('Hello,My name is completely Melro.','Are you ok?'))

TOTAL = 50 #try small number <50 to debug
# for article in articles[:TOTAL]:
for article in articles[:TOTAL]:
    candidate = []
    for v in article.values():
        sentences = parser.raw_parse_sents(v)
        for line in sentences:
            for sentence in line:
                if sentence[0].label()=="S": # start 
                    subtree = sentence[0]
                    for i in range(len(subtree) - 1):
                        if subtree[i].label() == "NP" and subtree[i+1].label() == "VP":
                            candidate.append(sentence) #save this NP-VP sentence as a tree structure
                            break
    candidateSent.append(candidate)


#----------------------------------------TF-IDF----------------------------------------------------------------
Nones = set(["NN","NNS","NNP","NNPS"])
#extract None
#  a list of freq dict for each doc
freq_dict = []
# number of none tokens for each doc
none_len = []

for doc in candidateSent:
    t = []
    for sent in doc: #sent is a tree  
        for word, tag in sent.pos(): # POS
            if tag in Nones:
                t.append(word)
    none_len.append(len(t))
    freq_dict.append(Counter(t))


# a list of dicts, each dict is a word:tf_idf weight
tf_idf = computeTFIDF(none_len, freq_dict)


#----------------------------------------NER tag----------------------------------------------------------------
# this is all the NER tags, delete unnecessary ones to looking for only what we need
NER = set(["PERSON","NORP","FAC","ORG","GPE","LOC","PRODUCT","EVENT",\
    "WORK_OF_ART","LAW","LANGUAGE","DATE","TIME","PERCENT","MONEY","QUANTITY","ORDINAL","CARDINAL"])
#compute NER for all candidate sentences

# candidateSent2 : list of length TOTAL(50)
# each element is a list sentences structured as a pair (sentence tree, NER tag dict(key, tag) )
candidateSent2 = []
nlp = en_core_web_sm.load()
for doc in candidateSent:
    temp = []
    for sent in doc:
        str = " ".join(sent.leaves())
        x = nlp(str)
        # pprint([(X.text, X.label_) for X in x.ents])
        for X in x.ents:
            label = X.label_
            if label in NER:# contains NER tag that we want
                #each sentence is store as a pair (sent tree, NER tag dict)
                temp.append((sent,dict([(X.text, X.label_) for X in x.ents])))
                break
    candidateSent2.append(temp)


#----------------------------------------question generation----------------------------------------------------------------
# go through all the sentence (candidateSent2) in each doc, combine tf-idf dict to assign a score to every sentence and do ranking